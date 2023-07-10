/**
 *  Copyright (c) 2021 by Contributors
 * @file array/cuda/rowwise_sampling.cu
 * @brief uniform rowwise sampling
 */

#include <curand_kernel.h>
#include <dgl/random.h>
#include <dgl/runtime/device_api.h>

#include <stdgpu/unordered_set.cuh>
#include <stdgpu/queue.cuh>
#include <stdgpu/deque.cuh>
#include <stdgpu/vector.cuh>

#include <numeric>

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
    int hop;
    int64_t row_num;
    int64_t col_num;
    int64_t data_num;
};

struct PairHash {
    inline STDGPU_HOST_DEVICE std::size_t
    operator()(const thrust::pair<int, int64_t>& pair) const
    {
        std::size_t x = pair.first;
        std::size_t y = pair.second;

        // Cantor pairing function
        // 数太大会溢出，考虑用别的hash函数
        std::size_t hashValue = ((x + y) * (x + y + 1)) / 2 + y;
//        std::printf("x: %d, y: %ld, hash: %ld\n", x, y, hashValue);
        return hashValue;
    }
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
        stdgpu::queue<thrust::pair<int, int64_t>> queue, const int64_t* const in_rows, const int64_t num_rows){
    const int tIdx = threadIdx.x + blockIdx.x * blockDim.x;
    if (tIdx < num_rows) {
        queue.push({1, in_rows[tIdx]});
    }
}

__global__ void _CSRRowWiseSampleUniformTaskParallelismKernel(
        const uint64_t rand_seed, const int64_t * num_picks, int64_t * res_num,
        const int64_t * const in_rows, const int64_t num_rows, const int hops,
        const int64_t * const in_ptr, const int64_t * const in_index, const int64_t * const data,
        stdgpu::queue<thrust::pair<int, int64_t>> task_queue,
        stdgpu::unordered_set<thrust::pair<int, int64_t>, PairHash> set,
        stdgpu::vector<selectedEdgeInfo> result
        ) {
    __shared__ int64_t blockTask[2];
    __shared__ bool sharedRes[1];
    // num_pick cannot be larger than 128
    // any better solution?
    __shared__ int64_t permList[128];
    extern __shared__ int64_t local_res_num[];

    // do not use separate init kernel maybe faster? the result seems correct although only block level sync
    // not correct! convergence will be changed(accuracy~0.6, correct accuracy~0.7)
//    const int tIdx = threadIdx.x + blockIdx.x * blockDim.x;
//    if (tIdx < num_rows) {
//        task_queue.push({1, in_rows[tIdx]});
//    }
//    __syncthreads();

    // init local_res_num
    if (threadIdx.x < hops - 1) {
        local_res_num[threadIdx.x] = 0;
    }
//    __syncthreads();

    curandStatePhilox4_32_10_t rng;
    // different block has different seed
    // different thread in block has different (sub)sequence
    curand_init(rand_seed * gridDim.x + blockIdx.x, threadIdx.x, 0, &rng);
//    curand_init(rand_seed, 0, 0, &rng);

    while (true){
        if (threadIdx.x == 0){
            auto pop_res = task_queue.pop();
            if (!pop_res.second)
                sharedRes[0] = pop_res.second;
            else{
                auto task = pop_res.first;
                // hop num
                blockTask[0] = task.first;
                // row_num
                blockTask[1] = task.second;
                sharedRes[0] = pop_res.second;
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
        const int hop_num = blockTask[0];
        const int64_t row = blockTask[1];
        const int64_t in_row_start = in_ptr[row];
        const int64_t deg = in_ptr[row + 1] - in_row_start;

        if (deg <= num_picks[hop_num - 1]) {
//            if (threadIdx.x == 0)
//                AtomicAdd<int64_t>(&local_res_num[hop_num - 1], deg);
//            std::printf("row: %ld, deg: %ld, num_picks: %ld\n", row, deg, num_picks[hop_num - 1]);
            // just copy row when there is not enough nodes to sample
            for (int idx = threadIdx.x; idx < deg; idx += BLOCK_SIZE){
                const int64_t in_idx = in_row_start + idx;
                result.push_back({hop_num, row, in_index[in_idx], data ? data[in_idx] : in_idx});
//                std::printf("result push hop_num: %d, row: %ld, col: %ld, data: %ld\n", hop_num, row, in_index[in_idx], data ? data[in_idx] : in_idx);
//                AtomicAdd<int64_t>(&local_res_num[hop_num - 1], 1);
//                AtomicAdd<int64_t>(&res_num[hop_num - 1], 1);
                // last hop don't need to push task
                if (hop_num < hops){
                    auto dup_res = set.insert({hop_num + 1, in_index[in_idx]});
                    // if insert successfully, means no duplication, push task
                    if (dup_res.second){
//                        std::printf("insert hop_num: %d, row: %ld success\n", hop_num + 1, in_index[in_idx]);
                        task_queue.push({hop_num + 1, in_index[in_idx]});
//                        std::printf("task queue push hop_num: %d, row: %ld\n", hop_num + 1, in_index[in_idx]);
                    }
//                    else
//                        std::printf("insert hop_num: %d, row: %ld false\n", hop_num + 1, in_index[in_idx]);
                }
            }
        }
        else {
//            if (threadIdx.x == 0)
//                AtomicAdd<int64_t>(&local_res_num[hop_num - 1], num_picks[hop_num - 1]);
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
//                AtomicAdd<int64_t>(&local_res_num[hop_num - 1], 1);
//                AtomicAdd<int64_t>(&res_num[hop_num - 1], 1);
                // last hop don't need to push task
                if (hop_num < hops){
                    auto dup_res = set.insert({hop_num + 1, in_index[perm_idx]});
                    // if insert successfully, means no duplication, push task
                    if (dup_res.second){
//                        std::printf("insert hop_num: %d, row: %ld success\n", hop_num + 1, in_index[perm_idx]);
                        task_queue.push({hop_num + 1, in_index[perm_idx]});
//                        std::printf("task queue push hop_num: %d, row: %ld\n", hop_num + 1, in_index[perm_idx]);
                    }
//                    else
//                        std::printf("insert hop_num: %d, row: %ld false\n", hop_num + 1, in_index[perm_idx]);
                }
            }
        }
        // update local_res_num and push self
        // we can get res_num[hop_num - 1] by vector.size - res_num[0...hop_num - 2]
        if (threadIdx.x == 0 && hop_num < hops) {
//            AtomicAdd<int64_t>(&local_res_num[hop_num - 1],
//                               deg <= num_picks[hop_num - 1] ? deg : num_picks[hop_num - 1]);
            local_res_num[hop_num - 1] += deg <= num_picks[hop_num - 1] ? deg : num_picks[hop_num - 1];
            auto dup_res = set.insert({hop_num + 1, row});
            if (dup_res.second) {
//                std::printf("insert hop_num: %d, row: %ld success\n", hop_num + 1, row);
                task_queue.push({hop_num + 1, row});
//                std::printf("task queue push hop_num: %d, row: %ld\n", hop_num + 1, row);
            }
//            else
//                std::printf("insert hop_num: %d, row: %ld false\n", hop_num + 1, row);
        }
//        __syncthreads();
    }
    __syncthreads();

    // write local_res_num to global
    if (threadIdx.x < hops){
        AtomicAdd<int64_t>(&res_num[threadIdx.x], local_res_num[threadIdx.x]);
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

std::vector<COOMatrix> CustomCSRRowWiseSamplingUniformTaskParallelism(
        CSRMatrix mat, IdArray rows, const IdArray &num_picks){
//    std::printf("CustomCSRRowWiseSamplingUniformTaskParallelism run here\n");
    const auto& ctx = rows->ctx;
    const auto& num_picks_vec = num_picks.ToVector<int64_t>();
    auto device = runtime::DeviceAPI::Get(ctx);
    cudaStream_t stream = runtime::getCurrentCUDAStream();

    // 1-hop seed nodes number
    const int64_t num_rows = rows->shape[0];
    const auto hops = num_picks->shape[0];

//    int64_t * d_num_picks = static_cast<int64_t *>(device->AllocWorkspace(ctx, hops * sizeof(int64_t)));
    // res num per hop
    int64_t * d_res_num = static_cast<int64_t *>(device->AllocWorkspace(ctx, (hops - 1) * sizeof(int64_t)));

    // hops must > 1?
    auto res_num = (int64_t *)malloc((hops - 1) * sizeof(int64_t));
//    int64_t * res_num = static_cast<int64_t *>(device->AllocWorkspace(DGLContext{kDGLCPU, 0}, hops * sizeof(int64_t)));
    CUDA_CALL(cudaMemset(d_res_num, 0, (hops - 1) * sizeof(int64_t)));

//    cudaEvent_t copyEvent;
//    CUDA_CALL(cudaEventCreate(&copyEvent));
//    device->CopyDataFromTo(
//            num_picks.data(), 0, d_num_picks, 0, hops * sizeof(int64_t),
//            DGLContext{kDGLCPU, 0}, ctx, mat.indptr->dtype);
//    CUDA_CALL(cudaEventRecord(copyEvent, stream));

    // rows(i.e. batch nodes id) static cast to int64_t array, also a device ptr
    const int64_t* const sliced_rows = static_cast<const int64_t*>(rows->data);

    // for COO result
    // picked_row, picked_col, picked_idx: IdArray(NdArray)
    // out_row, out_col, out_idx: The data pointer points to the allocated data space(here is device ptr)
    std::vector<IdArray> picked_rows(hops), picked_cols(hops), picked_indices(hops);
//    std::vector<int64_t*> out_rows(hops), out_cols(hops), out_indices(hops), out_ptrs(hops),
//    out_degrees(hops), new_lens(hops);

//    cudaEvent_t copyEvents[hops];

    // allocate space for stdgpu container
    stdgpu::index_t cap = num_rows;
    // last hop sample result do not need to enqueue
    for (int i = 0; i < num_picks->shape[0] - 1; i++)
        cap += cap * (num_picks_vec[i] + 1);
    // pair(hop_num, src_node_id)
    auto task_queue = stdgpu::queue<thrust::pair<int, int64_t>>::createDeviceObject(cap);
    auto set = stdgpu::unordered_set
            <
            thrust::pair<int, int64_t>,
            PairHash
            >::createDeviceObject(cap);
    // init
    const dim3 init_block(512);
    const dim3 init_grid((num_rows + init_block.x - 1) / init_block.x);
    CUDA_KERNEL_CALL((queue_init), init_grid, init_block, 0, stream, task_queue, sliced_rows, num_rows);
    assert(task_queue.size() == num_rows);

    stdgpu::index_t vector_cap = num_rows * num_picks_vec[0];
    // last hop sample result need to push into result vector
    for (int i = 1; i < num_picks->shape[0]; i++)
        // the dstnodes of current hop should be the srcnodes of next hop
        vector_cap += vector_cap * (num_picks_vec[i] + 1);
    auto res_vector = stdgpu::vector<selectedEdgeInfo>::createDeviceObject(vector_cap);

    const int64_t* in_ptr = static_cast<int64_t*>(GetDevicePointer(mat.indptr));
    const int64_t* in_cols = static_cast<int64_t*>(GetDevicePointer(mat.indices));
    const int64_t* data = CSRHasData(mat)
                          ? static_cast<int64_t*>(GetDevicePointer(mat.data))
                          : nullptr;
    const int64_t* num_picks_ptr = static_cast<int64_t*>(GetDevicePointer(num_picks));

//    const uint64_t random_seed = RandomEngine::ThreadLocal()->RandInt(1000000000);
    // fix only for reproduce!
    const uint64_t random_seed = 1234;

    const dim3 block(BLOCK_SIZE);
//    const dim3 block(32);
    // should gird num be max?
    // best performance:arxiv, [25,10]
    const dim3 grid(num_rows);
//    const dim3 grid(256);
//    const dim3 grid(128);
//    const dim3 grid(64);
//    const dim3 grid(1);
    // wait for copying `d_num_picks` to finish
//    CUDA_CALL(cudaEventSynchronize(copyEvent));
//    CUDA_CALL(cudaEventDestroy(copyEvent));
    CUDA_KERNEL_CALL((_CSRRowWiseSampleUniformTaskParallelismKernel), grid, block, (hops - 1) * sizeof(int64_t), stream,
                     random_seed, num_picks_ptr, d_res_num, sliced_rows, num_rows, hops, in_ptr, in_cols, data,
                     task_queue, set, res_vector);
//    CUDA_KERNEL_CALL((_CSRRowWiseSampleUniformTaskParallelismKernel), grid, block, 0, stream,
//                     random_seed, d_num_picks, d_res_num, sliced_rows, num_rows, hops, in_ptr, in_cols, data,
//                     task_queue, set, res_vector);
    assert(task_queue.empty());
//    std::printf("cuda kernel finished\n");

    cudaEvent_t copyBackEvent;
    CUDA_CALL(cudaEventCreate(&copyBackEvent));
    device->CopyDataFromTo(
            d_res_num, 0, res_num, 0, (hops - 1) * sizeof(int64_t),
            ctx, DGLContext{kDGLCPU, 0}, mat.indptr->dtype);
    CUDA_CALL(cudaEventRecord(copyBackEvent, stream));
    // wait for copying back `d_res_num` to finish
    CUDA_CALL(cudaEventSynchronize(copyBackEvent));
    CUDA_CALL(cudaEventDestroy(copyBackEvent));
//    device->FreeWorkspace(ctx, d_num_picks);
    device->FreeWorkspace(ctx, d_res_num);

    stdgpu::queue<thrust::pair<int, int64_t>>::destroyDeviceObject(task_queue);
    stdgpu::unordered_set<thrust::pair<int, int64_t>, PairHash > ::destroyDeviceObject(set);

    // 传多个COO res的row, col, idx的指针的指针，用res_vector取fill，逻辑上最直观. 传指针的指针要写个demo试一下
    auto range_vec = res_vector.device_range();
    std::vector<COOMatrix> ret_coo(hops);
//    cudaPointerAttributes attributes;
//    std::printf("result vector size: %ld\n", res_vector.size());
    int64_t last_hop_size = res_vector.size();
    for(int i = 0; i < hops; i++) {
//        std::printf("res_num[%d]: %ld\n", i, res_num[i]);
        if (i < hops - 1)
            last_hop_size -= res_num[i];
        picked_rows[i] = NewIdArray(i < hops - 1 ? res_num[i] : last_hop_size, ctx, sizeof(int64_t) * 8);
        picked_cols[i] = NewIdArray(i < hops - 1 ? res_num[i] : last_hop_size, ctx, sizeof(int64_t) * 8);
        picked_indices[i] = NewIdArray(i < hops - 1 ? res_num[i] : last_hop_size, ctx, sizeof(int64_t) * 8);
        auto out_rows = static_cast<int64_t*>(picked_rows[i]->data);
        auto out_cols = static_cast<int64_t*>(picked_cols[i]->data);
        auto out_idx = static_cast<int64_t*>(picked_indices[i]->data);
//        CUDA_CALL(cudaPointerGetAttributes(&attributes, out_rows));
//        if (attributes.type == cudaMemoryTypeDevice)
//        {
//            std::cout << "Device pointer is a valid device pointer." << std::endl;
//        }
        thrust::copy_if(
                thrust::device,
                thrust::make_transform_iterator(range_vec.begin(), RowTrans()),
                thrust::make_transform_iterator(range_vec.end(), RowTrans()),
                thrust::make_transform_iterator(range_vec.begin(), HopTrans()),
                out_rows,
                CheckHopNum(i + 1)
                );
        thrust::copy_if(
                thrust::device,
                thrust::make_transform_iterator(range_vec.begin(), ColTrans()),
                thrust::make_transform_iterator(range_vec.end(), ColTrans()),
                thrust::make_transform_iterator(range_vec.begin(), HopTrans()),
                out_cols,
                CheckHopNum(i + 1)
        );
        thrust::copy_if(
                thrust::device,
                thrust::make_transform_iterator(range_vec.begin(), DataTrans()),
                thrust::make_transform_iterator(range_vec.end(), DataTrans()),
                thrust::make_transform_iterator(range_vec.begin(), HopTrans()),
                out_idx,
                CheckHopNum(i + 1)
        );
        // inplace
//        thrust::transform(out_rows, out_rows + res_num[i], out_rows, RowTrans());
//        thrust::transform(out_cols, out_rows + res_num[i], out_cols, ColTrans());
//        thrust::transform(out_idx, out_rows + res_num[i], out_idx, DataTrans());
        ret_coo[i] = COOMatrix(mat.num_rows, mat.num_cols, picked_rows[i], picked_cols[i], picked_indices[i]);
    }

    free(res_num);
    stdgpu::vector<selectedEdgeInfo>::destroyDeviceObject(res_vector);
//    std::printf("CustomCSRRowWiseSamplingUniformTaskParallelism finished here\n");
    return ret_coo;

//    for(int i = 0; i < hops; i++){
//        // allocate space for COO result on device
//        picked_rows[i] = NewIdArray(num_rows * num_picks[i], ctx, sizeof(int64_t) * 8);
//        picked_cols[i] = NewIdArray(num_rows * num_picks[i], ctx, sizeof(int64_t) * 8);
//        picked_indices[i] = NewIdArray(num_rows * num_picks[i], ctx, sizeof(int64_t) * 8);
//        out_rows[i] = static_cast<int64_t*>(picked_rows[i]->data);
//        out_cols[i] = static_cast<int64_t*>(picked_cols[i]->data);
//        out_indices[i] = static_cast<int64_t*>(picked_indices[i]->data);
//
//        // compute degree
//        out_degrees[i] = static_cast<int64_t*>(
//                device->AllocWorkspace(ctx, (num_rows + 1) * sizeof(int64_t)));
//        const dim3 block(512);
//        const dim3 grid((num_rows + block.x - 1) / block.x);
//        CUDA_KERNEL_CALL(
//                _CSRRowWiseSampleDegreeKernel, grid, block, 0, stream, num_picks[i],
//                num_rows, slice_rows, in_ptr, out_degrees[i]);
//
//        // fill out_ptrs
//        out_ptrs[i] = static_cast<int64_t*>(
//                device->AllocWorkspace(ctx, (num_rows + 1) * sizeof(int64_t)));
//        size_t prefix_temp_size = 0;
//        CUDA_CALL(cub::DeviceScan::ExclusiveSum(
//                nullptr, prefix_temp_size, out_degrees[i], out_ptrs[i], num_rows + 1, stream));
//        void* prefix_temp = device->AllocWorkspace(ctx, prefix_temp_size);
//        CUDA_CALL(cub::DeviceScan::ExclusiveSum(
//                prefix_temp, prefix_temp_size, out_degrees[i], out_ptrs[i], num_rows + 1, stream));
//        device->FreeWorkspace(ctx, prefix_temp);
//        device->FreeWorkspace(ctx, out_degrees[i]);
//
//        CUDA_CALL(cudaEventCreate(&copyEvents[i]));
//        // new_len, i.e. the number of sampled edges, last element of out_ptr
//        // copy using the internal current stream
//        device->CopyDataFromTo(
//            out_ptrs[i], num_rows * sizeof(new_lens[i]), &new_lens[i], 0, sizeof(new_lens[i]),
//            ctx,DGLContext{kDGLCPU, 0}, mat.indptr->dtype);
//        CUDA_CALL(cudaEventRecord(copyEvents[i], stream));
//    }
//
//    const uint64_t random_seed = RandomEngine::ThreadLocal()->RandInt(1000000000);
//
//    const dim3 block(BLOCK_SIZE);
//    const dim3 grid(cudaDevAttrMaxGridDimX);
//    CUDA_KERNEL_CALL(
//            (_CSRRowWiseSampleUniformTaskParallelismKernel), grid, block, 0,
//            stream, random_seed, num_picks, num_rows, slice_rows, in_ptr, in_cols,
//            data, out_ptrs, out_rows, out_cols, out_indices);
//
//    for (auto& copyEvent:copyEvents) {
//        // wait for copying `new_len` to finish
//        CUDA_CALL(cudaEventSynchronize(copyEvent));
//        CUDA_CALL(cudaEventDestroy(copyEvent));
//    }

    // need picked_row, picked_col, picked_idx with proper data(out_row, out_col, out_idx), remember these are device ptrs
//    return COOMatrix(
//            mat.num_rows, mat.num_cols, picked_row, picked_col, picked_idx);
}

template COOMatrix CSRRowWiseSamplingUniform<kDGLCUDA, int32_t>(
    CSRMatrix, IdArray, int64_t, bool);
template COOMatrix CSRRowWiseSamplingUniform<kDGLCUDA, int64_t>(
    CSRMatrix, IdArray, int64_t, bool);

}  // namespace impl
}  // namespace aten
}  // namespace dgl
