/**
 *  Copyright (c) 2021 by Contributors
 * @file array/cuda/rowwise_sampling.cu
 * @brief uniform rowwise sampling
 */

#include <curand_kernel.h>
#include <dgl/random.h>
#include <dgl/runtime/device_api.h>

#include <stdgpu/unordered_set.cuh>
#include <stdgpu/bitset.cuh>
#include <stdgpu/queue.cuh>
#include <stdgpu/deque.cuh>
#include <stdgpu/vector.cuh>

#include <numeric>
#include <nvtx3/nvToolsExt.h>

#include "../../array/cuda/atomic.cuh"
#include "../../runtime/cuda/cuda_common.h"
#include "./dgl_cub.cuh"
#include "./utils.h"

namespace dgl {
using namespace cuda;
using namespace aten::cuda;
namespace aten {
namespace impl {

namespace {

constexpr int BLOCK_SIZE = 128;

/**
 * @brief Compute the size of each row in the sampled CSR, without replacement.
 *
 * @tparam IdType The type of node and edge indexes.
 * @param num_picks The number of non-zero entries to pick per row.
 * @param num_rows The number of rows to pick.
 * @param in_rows The set of rows to pick.
 * @param in_ptr The index where each row's edges start.
 * @param out_deg The size of each row in the sampled matrix, as indexed by
 * `in_rows` (output).
 */
template <typename IdType>
__global__ void _CSRRowWiseSampleDegreeKernel(
    const int64_t num_picks, const int64_t num_rows,
    const IdType* const in_rows, const IdType* const in_ptr,
    IdType* const out_deg) {
  const int tIdx = threadIdx.x + blockIdx.x * blockDim.x;

  if (tIdx < num_rows) {
    const int in_row = in_rows[tIdx];
    const int out_row = tIdx;
    out_deg[out_row] = min(
        static_cast<IdType>(num_picks), in_ptr[in_row + 1] - in_ptr[in_row]);

    if (out_row == num_rows - 1) {
      // make the prefixsum work
      out_deg[num_rows] = 0;
    }
  }
}

/**
 * @brief Compute the size of each row in the sampled CSR, with replacement.
 *
 * @tparam IdType The type of node and edge indexes.
 * @param num_picks The number of non-zero entries to pick per row.
 * @param num_rows The number of rows to pick.
 * @param in_rows The set of rows to pick.
 * @param in_ptr The index where each row's edges start.
 * @param out_deg The size of each row in the sampled matrix, as indexed by
 * `in_rows` (output).
 */
template <typename IdType>
__global__ void _CSRRowWiseSampleDegreeReplaceKernel(
    const int64_t num_picks, const int64_t num_rows,
    const IdType* const in_rows, const IdType* const in_ptr,
    IdType* const out_deg) {
  const int tIdx = threadIdx.x + blockIdx.x * blockDim.x;

  if (tIdx < num_rows) {
    const int64_t in_row = in_rows[tIdx];
    const int64_t out_row = tIdx;

    if (in_ptr[in_row + 1] - in_ptr[in_row] == 0) {
      out_deg[out_row] = 0;
    } else {
      out_deg[out_row] = static_cast<IdType>(num_picks);
    }

    if (out_row == num_rows - 1) {
      // make the prefixsum work
      out_deg[num_rows] = 0;
    }
  }
}

// the sample result of a hop
struct selectedEdgeInfo {
    short hop;
    int64_t row_num;
    int64_t col_num;
    int64_t data_num;
};

struct CheckHopNum {
    int hop_num;
    CheckHopNum(int num) : hop_num(num) {};
    __host__ __device__
    // num from stencil
    bool operator()(const int64_t &num) {
        return num == hop_num;
    }
};

struct RowTrans : public thrust::unary_function<selectedEdgeInfo, int64_t> {
    __host__ __device__
    int64_t operator()(const selectedEdgeInfo &tuple) {
        return tuple.row_num;
    }
};

struct ColTrans : public thrust::unary_function<selectedEdgeInfo, int64_t> {
    __host__ __device__
    int64_t operator()(const selectedEdgeInfo &tuple) {
        return tuple.col_num;
    }
};

struct DataTrans : public thrust::unary_function<selectedEdgeInfo, int64_t> {
    __host__ __device__
    int64_t operator()(const selectedEdgeInfo &tuple) {
        return tuple.data_num;
    }
};

struct HopTrans : public thrust::unary_function<selectedEdgeInfo, int> {
    __host__ __device__
    int operator()(const selectedEdgeInfo &tuple) {
        return tuple.hop;
    }
};

// should try to push the node with more edges first
__global__ void queue_init(
        stdgpu::queue<thrust::pair<short, int64_t>> queue, uint* bits, const int64_t* const in_rows,
        const int64_t num_rows, const int hops, const int64_t total_num_rows) {
    const int tIdx = threadIdx.x + blockIdx.x * blockDim.x;
//    if (tIdx < num_rows * hops) {
//        int hop_num = 1 + tIdx / num_rows;
//        queue.push({hop_num, in_rows[tIdx % num_rows]});
//        if (hop_num > 1)
//            bits[in_rows[tIdx % num_rows] + total_num_rows * (hop_num - 2)] = 1;
//    }
    if (tIdx < num_rows) {
        queue.push({1, in_rows[tIdx]});
    }
}

__launch_bounds__(128) __global__ void _CSRRowWiseSampleUniformTaskParallelismKernel(
        const uint64_t rand_seed, const int64_t * num_picks, const int64_t * const in_rows,
        const int64_t num_rows, const int hops, const int64_t total_num_rows,
        const int64_t * const in_ptr, const int64_t * const in_index, const int64_t * const data,
        stdgpu::queue<thrust::pair<short, int64_t>> task_queue,
        uint* bits,
        stdgpu::vector<selectedEdgeInfo> result
        ) {
    __shared__ int64_t blockTask[2];
    __shared__ bool sharedRes[1];
    // num_pick cannot be larger than 128
    // any better solution?
    __shared__ int64_t permList[128];

    // do not use separate init kernel maybe faster? the result seems correct although only block level sync
    // not correct! convergence will be changed(accuracy~0.6, correct accuracy~0.7)
//    const int tIdx = threadIdx.x + blockIdx.x * blockDim.x;
//    if (tIdx < num_rows) {
//        task_queue.push({1, in_rows[tIdx]});
//    }
//    __syncthreads();

    curandStatePhilox4_32_10_t rng;
    // different block has different seed
    // different thread in block has different (sub)sequence
    curand_init(rand_seed * gridDim.x + blockIdx.x, threadIdx.x, 0, &rng);
//    curand_init(rand_seed, 0, 0, &rng);

    while (true) {
        if (threadIdx.x == 0) {
            auto pop_res = task_queue.pop();
            sharedRes[0] = pop_res.second;
            if (pop_res.second) {
                auto task = pop_res.first;
                // hop num
                blockTask[0] = task.first;
                // row_num
                blockTask[1] = task.second;
            }
        }
        __syncthreads();
//        if (!sharedRes[0] && task_queue.empty())
        if (!sharedRes[0])
            break;
        // result.size() > num_rows * 5 just for test, should have a better check policy
//        if (!sharedRes[0] && result.size() > num_rows * 5)
//            break;
//        else if (!sharedRes[0])
//            continue;
        // run task, same block threads have same task(hop_num, row_num)
        const short hop_num = blockTask[0];
        const int64_t row = blockTask[1];
        const int64_t in_row_start = in_ptr[row];
        const int64_t deg = in_ptr[row + 1] - in_row_start;

        if (deg <= num_picks[hop_num - 1]) {
//            std::printf("row: %ld, deg: %ld, num_picks: %ld\n", row, deg, num_picks[hop_num - 1]);
            // just copy row when there is not enough nodes to sample
            for (int idx = threadIdx.x; idx < deg; idx += BLOCK_SIZE) {
                const int64_t in_idx = in_row_start + idx;
                result.push_back({hop_num, row, in_index[in_idx], data ? data[in_idx] : in_idx});
//                std::printf("result push hop_num: %d, row: %ld, col: %ld, data: %ld\n", hop_num, row, in_index[in_idx], data ? data[in_idx] : in_idx);
                // last hop don't need to push task
                if (hop_num < hops) {
                    if (!bits[in_index[in_idx] + total_num_rows * (hop_num - 1)]) {
//                        auto old = bits.set(in_index[in_idx] * hop_num);
                        auto old = atomicOr(&bits[in_index[in_idx] + total_num_rows * (hop_num - 1)], 1);
                        if (!old) {
                            task_queue.push({hop_num + 1, in_index[in_idx]});
                        }
                    }
                }
            }
        } else {
            // generate permutation list via reservoir algorithm
            // reservoir init
            for (int idx = threadIdx.x; idx < num_picks[hop_num - 1]; idx += BLOCK_SIZE) {
                permList[idx] = idx;
            }
            __syncthreads();

            for (int idx = num_picks[hop_num - 1] + threadIdx.x; idx < deg; idx += BLOCK_SIZE) {
                const int num = curand(&rng) % (idx + 1);
                if (num < num_picks[hop_num - 1]) {
                    // use shared memory, faster than DGL?
                    AtomicMax(permList + num, idx);
                }
            }
            __syncthreads();

            for (int idx = threadIdx.x; idx < num_picks[hop_num - 1]; idx += BLOCK_SIZE) {
                // permList[idx] is the idx of the sampled edge, from 0 to deg-1, should be added with in_row_start
                const int64_t perm_idx = permList[idx] + in_row_start;
                result.push_back({hop_num, row, in_index[perm_idx], data ? data[perm_idx] : perm_idx});
//                std::printf("result push hop_num: %d, row: %ld, col: %ld, data: %ld\n", hop_num, row, in_index[perm_idx], data ? data[perm_idx] : perm_idx);
                // last hop don't need to push task
                if (hop_num < hops) {
                    if (!bits[in_index[perm_idx] + total_num_rows * (hop_num - 1)]) {
                        auto old = atomicOr(&bits[in_index[perm_idx] + total_num_rows * (hop_num - 1)], 1);
                        if (!old) {
                            task_queue.push({hop_num + 1, in_index[perm_idx]});
                        }
                    }
                }
            }
        }
        // push self
        if (threadIdx.x == 0 && hop_num < hops) {
            if (!bits[row + total_num_rows * (hop_num - 1)]) {
                auto old = atomicOr(&bits[row + total_num_rows * (hop_num - 1)], 1);
                if (!old) {
                    task_queue.push({hop_num + 1, row});
                }
            }
        }
//        __syncthreads();
    }
}

/**
 * @brief Perform row-wise uniform sampling on a CSR matrix,
 * and generate a COO matrix, without replacement.
 *
 * @tparam IdType The ID type used for matrices.
 * @tparam TILE_SIZE The number of rows covered by each threadblock.
 * @param rand_seed The random seed to use.
 * @param num_picks The number of non-zeros to pick per row.
 * @param num_rows The number of rows to pick.
 * @param in_rows The set of rows to pick.
 * @param in_ptr The indptr array of the input CSR.
 * @param in_index The indices array of the input CSR.
 * @param data The data array of the input CSR.
 * @param out_ptr The offset to write each row to in the output COO.
 * @param out_rows The rows of the output COO (output).
 * @param out_cols The columns of the output COO (output).
 * @param out_idxs The data array of the output COO (output).
 */
template <typename IdType, int TILE_SIZE>
__global__ void _CSRRowWiseSampleUniformKernel(
    const uint64_t rand_seed, const int64_t num_picks, const int64_t num_rows,
    const IdType* const in_rows, const IdType* const in_ptr,
    const IdType* const in_index, const IdType* const data,
    const IdType* const out_ptr, IdType* const out_rows, IdType* const out_cols,
    IdType* const out_idxs) {
  // we assign one warp per row
  assert(blockDim.x == BLOCK_SIZE);

  int64_t out_row = blockIdx.x * TILE_SIZE;
  const int64_t last_row =
      min(static_cast<int64_t>(blockIdx.x + 1) * TILE_SIZE, num_rows);

  curandStatePhilox4_32_10_t rng;
  curand_init(rand_seed * gridDim.x + blockIdx.x, threadIdx.x, 0, &rng);
//    curand_init(rand_seed, 0, 0, &rng);

  while (out_row < last_row) {
    const int64_t row = in_rows[out_row];
    const int64_t in_row_start = in_ptr[row];
    const int64_t deg = in_ptr[row + 1] - in_row_start;
    const int64_t out_row_start = out_ptr[out_row];

    if (deg <= num_picks) {
      // just copy row when there is not enough nodes to sample.
      for (int idx = threadIdx.x; idx < deg; idx += BLOCK_SIZE) {
        const IdType in_idx = in_row_start + idx;
        out_rows[out_row_start + idx] = row;
        out_cols[out_row_start + idx] = in_index[in_idx];
        out_idxs[out_row_start + idx] = data ? data[in_idx] : in_idx;
      }
    } else {
      // generate permutation list via reservoir algorithm
      for (int idx = threadIdx.x; idx < num_picks; idx += BLOCK_SIZE) {
        out_idxs[out_row_start + idx] = idx;
      }
      __syncthreads();

      for (int idx = num_picks + threadIdx.x; idx < deg; idx += BLOCK_SIZE) {
        const int num = curand(&rng) % (idx + 1);
        if (num < num_picks) {
          // use max so as to achieve the replacement order the serial
          // algorithm would have
          AtomicMax(out_idxs + out_row_start + num, idx);
        }
      }
      __syncthreads();

      // copy permutation over
      for (int idx = threadIdx.x; idx < num_picks; idx += BLOCK_SIZE) {
        const IdType perm_idx = out_idxs[out_row_start + idx] + in_row_start;
        out_rows[out_row_start + idx] = row;
        out_cols[out_row_start + idx] = in_index[perm_idx];
        out_idxs[out_row_start + idx] = data ? data[perm_idx] : perm_idx;
      }
    }
    out_row += 1;
  }
}

/**
 * @brief Perform row-wise uniform sampling on a CSR matrix,
 * and generate a COO matrix, with replacement.
 *
 * @tparam IdType The ID type used for matrices.
 * @tparam TILE_SIZE The number of rows covered by each threadblock.
 * @param rand_seed The random seed to use.
 * @param num_picks The number of non-zeros to pick per row.
 * @param num_rows The number of rows to pick.
 * @param in_rows The set of rows to pick.
 * @param in_ptr The indptr array of the input CSR.
 * @param in_index The indices array of the input CSR.
 * @param data The data array of the input CSR.
 * @param out_ptr The offset to write each row to in the output COO.
 * @param out_rows The rows of the output COO (output).
 * @param out_cols The columns of the output COO (output).
 * @param out_idxs The data array of the output COO (output).
 */
template <typename IdType, int TILE_SIZE>
__global__ void _CSRRowWiseSampleUniformReplaceKernel(
    const uint64_t rand_seed, const int64_t num_picks, const int64_t num_rows,
    const IdType* const in_rows, const IdType* const in_ptr,
    const IdType* const in_index, const IdType* const data,
    const IdType* const out_ptr, IdType* const out_rows, IdType* const out_cols,
    IdType* const out_idxs) {
  // we assign one warp per row
  assert(blockDim.x == BLOCK_SIZE);

  int64_t out_row = blockIdx.x * TILE_SIZE;
  const int64_t last_row =
      min(static_cast<int64_t>(blockIdx.x + 1) * TILE_SIZE, num_rows);

  curandStatePhilox4_32_10_t rng;
  curand_init(rand_seed * gridDim.x + blockIdx.x, threadIdx.x, 0, &rng);

  while (out_row < last_row) {
    const int64_t row = in_rows[out_row];
    const int64_t in_row_start = in_ptr[row];
    const int64_t out_row_start = out_ptr[out_row];
    const int64_t deg = in_ptr[row + 1] - in_row_start;

    if (deg > 0) {
      // each thread then blindly copies in rows only if deg > 0.
      for (int idx = threadIdx.x; idx < num_picks; idx += BLOCK_SIZE) {
        const int64_t edge = curand(&rng) % deg;
        const int64_t out_idx = out_row_start + idx;
        out_rows[out_idx] = row;
        out_cols[out_idx] = in_index[in_row_start + edge];
        out_idxs[out_idx] =
            data ? data[in_row_start + edge] : in_row_start + edge;
      }
    }
    out_row += 1;
  }
}

}  // namespace

///////////////////////////// CSR sampling //////////////////////////

template <DGLDeviceType XPU, typename IdType>
COOMatrix _CSRRowWiseSamplingUniform(
    CSRMatrix mat, IdArray rows, const int64_t num_picks, const bool replace) {
  const auto& ctx = rows->ctx;
  auto device = runtime::DeviceAPI::Get(ctx);
  cudaStream_t stream = runtime::getCurrentCUDAStream();

  const int64_t num_rows = rows->shape[0];
  const IdType* const slice_rows = static_cast<const IdType*>(rows->data);

  IdArray picked_row =
      NewIdArray(num_rows * num_picks, ctx, sizeof(IdType) * 8);
  IdArray picked_col =
      NewIdArray(num_rows * num_picks, ctx, sizeof(IdType) * 8);
  IdArray picked_idx =
      NewIdArray(num_rows * num_picks, ctx, sizeof(IdType) * 8);
  IdType* const out_rows = static_cast<IdType*>(picked_row->data);
  IdType* const out_cols = static_cast<IdType*>(picked_col->data);
  IdType* const out_idxs = static_cast<IdType*>(picked_idx->data);

  const IdType* in_ptr = static_cast<IdType*>(GetDevicePointer(mat.indptr));
  const IdType* in_cols = static_cast<IdType*>(GetDevicePointer(mat.indices));
  const IdType* data = CSRHasData(mat)
                           ? static_cast<IdType*>(GetDevicePointer(mat.data))
                           : nullptr;

  // compute degree
  IdType* out_deg = static_cast<IdType*>(
      device->AllocWorkspace(ctx, (num_rows + 1) * sizeof(IdType)));
  if (replace) {
    const dim3 block(512);
    const dim3 grid((num_rows + block.x - 1) / block.x);
    CUDA_KERNEL_CALL(
        _CSRRowWiseSampleDegreeReplaceKernel, grid, block, 0, stream, num_picks,
        num_rows, slice_rows, in_ptr, out_deg);
  } else {
    const dim3 block(512);
    const dim3 grid((num_rows + block.x - 1) / block.x);
    CUDA_KERNEL_CALL(
        _CSRRowWiseSampleDegreeKernel, grid, block, 0, stream, num_picks,
        num_rows, slice_rows, in_ptr, out_deg);
  }

  // fill out_ptr
  IdType* out_ptr = static_cast<IdType*>(
      device->AllocWorkspace(ctx, (num_rows + 1) * sizeof(IdType)));
  size_t prefix_temp_size = 0;
  CUDA_CALL(cub::DeviceScan::ExclusiveSum(
      nullptr, prefix_temp_size, out_deg, out_ptr, num_rows + 1, stream));
  void* prefix_temp = device->AllocWorkspace(ctx, prefix_temp_size);
  CUDA_CALL(cub::DeviceScan::ExclusiveSum(
      prefix_temp, prefix_temp_size, out_deg, out_ptr, num_rows + 1, stream));
  device->FreeWorkspace(ctx, prefix_temp);
  device->FreeWorkspace(ctx, out_deg);

  cudaEvent_t copyEvent;
  CUDA_CALL(cudaEventCreate(&copyEvent));

  // TODO(dlasalle): use pinned memory to overlap with the actual sampling, and
  // wait on a cudaevent
  IdType new_len;
  // copy using the internal current stream
  device->CopyDataFromTo(
      out_ptr, num_rows * sizeof(new_len), &new_len, 0, sizeof(new_len), ctx,
      DGLContext{kDGLCPU, 0}, mat.indptr->dtype);
  CUDA_CALL(cudaEventRecord(copyEvent, stream));

//  const uint64_t random_seed = RandomEngine::ThreadLocal()->RandInt(1000000000);
  // fix only for reproduce!
  const uint64_t random_seed = 1234;

  // select edges
  // the number of rows each thread block will cover
  constexpr int TILE_SIZE = 128 / BLOCK_SIZE;
  if (replace) {  // with replacement
    const dim3 block(BLOCK_SIZE);
    const dim3 grid((num_rows + TILE_SIZE - 1) / TILE_SIZE);
    CUDA_KERNEL_CALL(
        (_CSRRowWiseSampleUniformReplaceKernel<IdType, TILE_SIZE>), grid, block,
        0, stream, random_seed, num_picks, num_rows, slice_rows, in_ptr,
        in_cols, data, out_ptr, out_rows, out_cols, out_idxs);
  } else {  // without replacement
    const dim3 block(BLOCK_SIZE);
    const dim3 grid((num_rows + TILE_SIZE - 1) / TILE_SIZE);
    CUDA_KERNEL_CALL(
        (_CSRRowWiseSampleUniformKernel<IdType, TILE_SIZE>), grid, block, 0,
        stream, random_seed, num_picks, num_rows, slice_rows, in_ptr, in_cols,
        data, out_ptr, out_rows, out_cols, out_idxs);
  }
  device->FreeWorkspace(ctx, out_ptr);

  // wait for copying `new_len` to finish
  CUDA_CALL(cudaEventSynchronize(copyEvent));
  CUDA_CALL(cudaEventDestroy(copyEvent));

  // from shape{num_rows * num_picks} to shape{new_len}
  // I guess: if wait the copy of `new_len`, then new IdArray picked_row, picked_col, picked_idx with shape `new_len`, not need to reshape here
  // DGL use async copy, overlap copy with the sampling kernel execution
  // and the kernel need the data of picked_row, picked_col, picked_idx, so need to new them with enough shape before and reshape here
  picked_row = picked_row.CreateView({new_len}, picked_row->dtype);
  picked_col = picked_col.CreateView({new_len}, picked_col->dtype);
  picked_idx = picked_idx.CreateView({new_len}, picked_idx->dtype);

  return COOMatrix(
      mat.num_rows, mat.num_cols, picked_row, picked_col, picked_idx);
}

template <DGLDeviceType XPU, typename IdType>
COOMatrix CSRRowWiseSamplingUniform(
    CSRMatrix mat, IdArray rows, const int64_t num_picks, const bool replace) {
  if (num_picks == -1) {
    // Basically this is UnitGraph::InEdges().
    COOMatrix coo = CSRToCOO(CSRSliceRows(mat, rows), false);
    IdArray sliced_rows = IndexSelect(rows, coo.row);
    return COOMatrix(
        mat.num_rows, mat.num_cols, sliced_rows, coo.col, coo.data);
  } else {
    return _CSRRowWiseSamplingUniform<XPU, IdType>(
        mat, rows, num_picks, replace);
  }
}

stdgpu::vector<selectedEdgeInfo> res_vector;
//stdgpu::queue<thrust::pair<short, int64_t>> task_queue;
const int RES_VEC_CAP = 1e8;
stdgpu::index_t vector_cap = 0;
bool first_time = true;
int64_t old_size = 0;

std::vector<COOMatrix> CustomCSRRowWiseSamplingUniformTaskParallelism(
        CSRMatrix mat, IdArray rows, const IdArray &num_picks) {
//    std::printf("CustomCSRRowWiseSamplingUniformTaskParallelism run here\n");
    const auto& ctx = rows->ctx;
    const auto& num_picks_vec = num_picks.ToVector<int64_t>();
    auto device = runtime::DeviceAPI::Get(ctx);
    cudaStream_t stream = runtime::getCurrentCUDAStream();

    // 1-hop seed nodes number
    const int64_t num_rows = rows->shape[0];
    const auto hops = num_picks->shape[0];

    // rows(i.e. batch nodes id) static cast to int64_t array, also a device ptr
    const int64_t* const sliced_rows = static_cast<const int64_t*>(rows->data);

    // for COO result
    // picked_row, picked_col, picked_idx: IdArray(NdArray)
    // out_row, out_col, out_idx: The data pointer points to the allocated data space(here is device ptr)
    std::vector<IdArray> picked_rows(hops), picked_cols(hops), picked_indices(hops);
    const int64_t* in_ptr = static_cast<int64_t*>(GetDevicePointer(mat.indptr));
    const int64_t* in_cols = static_cast<int64_t*>(GetDevicePointer(mat.indices));
    const int64_t* data = CSRHasData(mat)
                          ? static_cast<int64_t*>(GetDevicePointer(mat.data))
                          : nullptr;
    const int64_t* num_picks_ptr = static_cast<int64_t*>(GetDevicePointer(num_picks));

    // allocate space for stdgpu container
    stdgpu::index_t queue_cap = num_rows;
    // last hop sample result do not need to enqueue
    for (int i = 0; i < hops - 1; i++)
        queue_cap += queue_cap * (num_picks_vec[i] + 1);
    // pair(hop_num, src_node_id)
    nvtxRangePushA("create task_queue");
    auto task_queue = stdgpu::queue<thrust::pair<short, int64_t>>::createDeviceObject(queue_cap);
    nvtxRangePop();

    nvtxRangePushA("create bits");
    uint* bool_arr = static_cast<uint *>(device->AllocWorkspace(ctx, mat.num_rows * (hops - 1) * sizeof(uint)));
    CUDA_CALL(cudaMemset(bool_arr, 0, mat.num_rows * (hops - 1) * sizeof(uint)));
    nvtxRangePop();

    // init
    const dim3 init_block(512);
    const dim3 init_grid((num_rows + init_block.x - 1) / init_block.x);
//    const dim3 init_grid((num_rows * hops + init_block.x - 1) / init_block.x);
    CUDA_KERNEL_CALL((queue_init), init_grid, init_block, 0, stream, task_queue, bool_arr, sliced_rows, num_rows, hops, mat.num_rows);
//    assert(task_queue.size() == num_rows);

    nvtxRangePushA("create or clear container");
    if (first_time) {
        int64_t node_num = num_rows;
//        vector_cap = num_rows * num_picks_vec[0];
        // last hop sample result need to push into result vector
        for (int i = 0; i < hops; i++) {
            // the dstnodes of current hop should be the srcnodes of next hop
            vector_cap += node_num * num_picks_vec[i];
            node_num += vector_cap;
        }
        res_vector = stdgpu::vector<selectedEdgeInfo>::createDeviceObject(RES_VEC_CAP);

//        nvtxRangePushA("create bits");
//        bits = stdgpu::bitset<>::createDeviceObject(static_cast<stdgpu::index_t>(mat.num_rows * (hops - 1)));
//        nvtxRangePop();

        first_time = false;
    }
        // task queue will be "clear" when sample kernel finished
    else{
        if (old_size + vector_cap >= RES_VEC_CAP){
            res_vector.clear();
            old_size = 0;
        }
    }
    nvtxRangePop();

//    const uint64_t random_seed = RandomEngine::ThreadLocal()->RandInt(1000000000);
    // fix only for reproduce!
    const uint64_t random_seed = 1234;

    const dim3 block(BLOCK_SIZE);
    // should gird num be max?
    // best performance:arxiv, [25,10]
    const dim3 grid(num_rows);
//    const dim3 grid(num_rows * hops);
//    const dim3 grid(1);

//    std::printf("queue valid:%d\n", task_queue.valid());
    CUDA_KERNEL_CALL((_CSRRowWiseSampleUniformTaskParallelismKernel), grid, block, 0, stream,
                     random_seed, num_picks_ptr, sliced_rows, num_rows, hops, mat.num_rows, in_ptr, in_cols, data,
                     task_queue, bool_arr, res_vector);
//    std::printf("queue valid:%d\n", task_queue.valid());
//    assert(task_queue.empty());
//    std::printf("cuda kernel finished\n");

    // 传多个COO res的row, col, idx的指针的指针，用res_vector取fill，逻辑上最直观. 传指针的指针要写个demo试一下
    nvtxRangePushA("get result");
    int64_t node_num = num_rows;
    nvtxRangePushA("NewIdArray");
    for(int i = 0; i < hops; i++) {
        int64_t hop_res_num = node_num * num_picks_vec[i];
        picked_rows[i] = NewIdArray(hop_res_num, ctx, sizeof(int64_t) * 8);
        picked_cols[i] = NewIdArray(hop_res_num, ctx, sizeof(int64_t) * 8);
        picked_indices[i] = NewIdArray(hop_res_num, ctx, sizeof(int64_t) * 8);
        node_num += hop_res_num;
    }
    nvtxRangePop();

    std::vector<COOMatrix> ret_coo(hops);

    auto range_vec = res_vector.device_range();
    auto zip_begin = thrust::make_zip_iterator(thrust::make_tuple(
            thrust::make_transform_iterator(range_vec.begin(), RowTrans()),
            thrust::make_transform_iterator(range_vec.begin(), ColTrans()),
            thrust::make_transform_iterator(range_vec.begin(), DataTrans())
    ));
    auto zip_end = thrust::make_zip_iterator(thrust::make_tuple(
            thrust::make_transform_iterator(range_vec.end(), RowTrans()),
            thrust::make_transform_iterator(range_vec.end(), ColTrans()),
            thrust::make_transform_iterator(range_vec.end(), DataTrans())
    ));
//    cudaPointerAttributes attributes;
    for(int i = 0; i < hops; i++) {
        auto out_rows = static_cast<int64_t*>(picked_rows[i]->data);
        auto out_cols = static_cast<int64_t*>(picked_cols[i]->data);
        auto out_idx = static_cast<int64_t*>(picked_indices[i]->data);
        nvtxRangePushA("thrust process");
//        CUDA_CALL(cudaPointerGetAttributes(&attributes, out_rows));
//        if (attributes.type == cudaMemoryTypeDevice)
//        {
//            std::cout << "Device pointer is a valid device pointer." << std::endl;
//        }

        auto zip_res = thrust::make_zip_iterator(thrust::make_tuple(
                out_rows,
                out_cols,
                out_idx
        ));
        int64_t new_size = thrust::copy_if(
                thrust::device,
                zip_begin + old_size,
                zip_end,
                thrust::make_transform_iterator(range_vec.begin() + old_size, HopTrans()),
                zip_res,
                CheckHopNum(i + 1)
                ) - zip_res;
        nvtxRangePop();

        nvtxRangePushA("CreateView");
        picked_rows[i] = picked_rows[i].CreateView({new_size}, picked_rows[i]->dtype);
        picked_cols[i] = picked_cols[i].CreateView({new_size}, picked_cols[i]->dtype);
        picked_indices[i] = picked_indices[i].CreateView({new_size}, picked_indices[i]->dtype);
        nvtxRangePop();
        ret_coo[i] = COOMatrix(mat.num_rows, mat.num_cols, picked_rows[i], picked_cols[i], picked_indices[i]);
    }
    nvtxRangePop();

    old_size = res_vector.size();

//    nvtxRangePushA("free res_vector");
    //TODO: should free them once when program ends(last iteration)... but I have not found an easy way to do that.
//    stdgpu::vector<selectedEdgeInfo>::destroyDeviceObject(res_vector);
//    nvtxRangePop();
//    std::printf("CustomCSRRowWiseSamplingUniformTaskParallelism finished here\n");
    device->FreeWorkspace(ctx, bool_arr);
    stdgpu::queue<thrust::pair<short, int64_t>>::destroyDeviceObject(task_queue);
//    stdgpu::bitset<>::destroyDeviceObject(bits);
    return ret_coo;
}

template COOMatrix CSRRowWiseSamplingUniform<kDGLCUDA, int32_t>(
    CSRMatrix, IdArray, int64_t, bool);
template COOMatrix CSRRowWiseSamplingUniform<kDGLCUDA, int64_t>(
    CSRMatrix, IdArray, int64_t, bool);

}  // namespace impl
}  // namespace aten
}  // namespace dgl
