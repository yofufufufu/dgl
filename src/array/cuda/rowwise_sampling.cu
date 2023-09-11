/**
 *  Copyright (c) 2021 by Contributors
 * @file array/cuda/rowwise_sampling.cu
 * @brief uniform rowwise sampling
 */

#include <curand_kernel.h>
#include <dgl/random.h>
#include <dgl/runtime/device_api.h>

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
constexpr int BLOCK_SIZE_CUSTOM = 96;

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
//struct selectedEdgeInfo {
//    short hop;
//    int64_t row_num;
//    int64_t col_num;
//    int64_t data_num;
//};
struct selectedEdgeInfo {
    int64_t* rows;
    int64_t* cols;
    int64_t* datas;
};

// must be volatile!
// __device__ volatile uint tail_index;
 __device__ uint tail_index;
// __device__ volatile uint finished_block_num;
 __device__ uint task_idx;
// should try to push the node with more edges first
//TODO: if let init_kernel kernel do more things, try overlap it with more host codes
__global__ void init_kernel(
        thrust::pair<short, int64_t>* queue, uint* bits, const int64_t* const in_rows,
        const int64_t num_rows, const int hops, const int64_t queue_cap) {
    const int tIdx = threadIdx.x + blockIdx.x * blockDim.x;
    if (tIdx == 0) {
        tail_index = num_rows;
//        finished_block_num = 0;
        task_idx = 0;
    }
//    if (tIdx < num_rows * hops) {
//        int hop_num = 1 + tIdx / num_rows;
//        queue.push({hop_num, in_rows[tIdx % num_rows]});
//        if (hop_num > 1)
//            bits[in_rows[tIdx % num_rows] + total_num_rows * (hop_num - 2)] = 1;
//    }
    if (tIdx < num_rows) {
        queue[tIdx] = {1, in_rows[tIdx]};
    }
    // thread num > queue_cap, should restrict else condition
    else if (tIdx < queue_cap) {
        // avoid read uninitialized value, which can cause strange error
        queue[tIdx] = {0, -1};
    }
}

__launch_bounds__(BLOCK_SIZE_CUSTOM) __global__ void _CSRRowWiseSampleUniformTaskParallelismKernel(
        const uint64_t rand_seed, const int64_t * num_picks, uint* vector_lens,
        const int64_t num_rows, const int hops, const int64_t total_num_rows,
        const int64_t * const in_ptr, const int64_t * const in_index, const int64_t * const data,
        // must be volatile!
        volatile thrust::pair<short, int64_t>* task_queue,
        uint* bits,
        selectedEdgeInfo* result
        ) {
//    __shared__ int64_t blockTask[2];
    __shared__ bool sharedRes[1];
    __shared__ uint sharedTask[1];
    // num_pick cannot be larger than 128
    // any better solution?
    __shared__ int64_t permList[128];

    // writes to tail_index and finished_block_num is not visible(at least within a period of time) to other blocks
    // if variables are not volatile, the value will be cached?
    // so other blocks will loop infinite time in a very short time, which causes program hung.
    // `printf` can change data visibility by some ways...
    // see: https://forums.developer.nvidia.com/t/printf-in-cuda-kernel-changes-program-behavior/234524

//    if (threadIdx.x == 0) {
//        sharedRes[0] = false;
//        while (finished_block_num != tail_index) {
//            if (blockIdx.x < tail_index) {
//                sharedRes[0] = true;
//                break;
//            }
//        }
//    }
//    __syncthreads();

    if (threadIdx.x == 0) {
        auto taskid = atomicAdd(&task_idx, 1);
        sharedTask[0] = taskid;
        if (taskid >= tail_index) {
            // theoretical max block resident on 3090 when block size 96, and training need drop last when batchSize < 1312
            if (taskid < 1312)
                sharedRes[0] = true;
            else
                sharedRes[0] = false;
        }
        else
            sharedRes[0] = true;
    }
    __syncthreads();

    if (sharedRes[0]) {
        curandStatePhilox4_32_10_t rng;
        // different block has different seed
        // different thread in block has different (sub)sequence
        curand_init(rand_seed * gridDim.x + blockIdx.x, threadIdx.x, 0, &rng);
        // curand_init(rand_seed, 0, 0, &rng);

        // run task, same block threads have same task(hop_num, row_num)
        // write to task queue may not visible, so task will be init_kernel value, i.e.{0,-1}
        // just do again, use while loop(any better solution?).
        // `task_queue` also must be volatile, or it will be cached, causing loop infinite time
        auto const taskid = sharedTask[0];
        short hop_num = task_queue[taskid].first;
        int64_t row = task_queue[taskid].second;
        while (hop_num == 0 || row == -1) {
            hop_num = task_queue[taskid].first;
            row = task_queue[taskid].second;
        }

        const int64_t in_row_start = in_ptr[row];
        const int64_t deg = in_ptr[row + 1] - in_row_start;
        const int64_t num_pick = num_picks[hop_num - 1];

        if (deg <= num_pick) {
            // just copy row when there is not enough nodes to sample
            for (int idx = threadIdx.x; idx < deg; idx += BLOCK_SIZE_CUSTOM) {
                const int64_t in_idx = in_row_start + idx;
                const int64_t neighbor = in_index[in_idx];
                auto index = atomicAdd(vector_lens + (hop_num - 1), 1);
                result[hop_num - 1].rows[index] = row;
                result[hop_num - 1].cols[index] = neighbor;
                result[hop_num - 1].datas[index] = data ? data[in_idx] : in_idx;
                // last hop don't need to push task
                // TODO: can this atomic operation be optimized?
                if (hop_num < hops) {
                    const int64_t bits_offset = neighbor + total_num_rows * (hop_num - 1);
                    if (!bits[bits_offset]) {
                        //  auto old = bits.set(in_index[in_idx] * hop_num);
                        auto old = atomicOr(bits + bits_offset, 1);
                        if (!old) {
                            auto tail = atomicAdd(&tail_index, 1);
                            task_queue[tail].first = hop_num + 1;
                            task_queue[tail].second = neighbor;
//                                task_queue[tail] = {hop_num + 1, in_index[in_idx]};
                        }
                    }
                }
            }
        } else {
            // generate permutation list via reservoir algorithm
            // reservoir init
            for (int idx = threadIdx.x; idx < num_pick; idx += BLOCK_SIZE_CUSTOM) {
                permList[idx] = idx;
            }
            __syncthreads();

            for (int idx = num_pick + threadIdx.x; idx < deg; idx += BLOCK_SIZE_CUSTOM) {
                const int num = curand(&rng) % (idx + 1);
                if (num < num_pick) {
                    // use shared memory, faster than DGL?
                    AtomicMax(permList + num, idx);
                }
            }
            __syncthreads();

            for (int idx = threadIdx.x; idx < num_pick; idx += BLOCK_SIZE_CUSTOM) {
                // permList[idx] is the idx of the sampled edge, from 0 to deg-1, should be added with in_row_start
                const int64_t perm_idx = permList[idx] + in_row_start;
                const int64_t neighbor = in_index[perm_idx];
                auto index = atomicAdd(vector_lens + (hop_num - 1), 1);
                result[hop_num - 1].rows[index] = row;
                result[hop_num - 1].cols[index] = neighbor;
                result[hop_num - 1].datas[index] = data ? data[perm_idx] : perm_idx;
                // last hop don't need to push task
                if (hop_num < hops) {
                    const int64_t bits_offset = neighbor + total_num_rows * (hop_num - 1);
                    if (!bits[bits_offset]) {
                        auto old = atomicOr(bits + bits_offset, 1);
                        if (!old) {
                            auto tail = atomicAdd(&tail_index, 1);
//                                task_queue[tail] = {hop_num + 1, in_index[perm_idx]};
                            task_queue[tail].first = hop_num + 1;
                            task_queue[tail].second = neighbor;
                        }
                    }
                }
            }
        }
        // push self
        if (threadIdx.x == 0 && hop_num < hops) {
            const int64_t bits_offset = row + total_num_rows * (hop_num - 1);
            if (!bits[bits_offset]) {
                auto old = atomicOr(bits + bits_offset, 1);
                if (!old) {
                    auto tail = atomicAdd(&tail_index, 1);
//                        task_queue[tail] = {hop_num + 1, row};
                    task_queue[tail].first = hop_num + 1;
                    task_queue[tail].second = row;
                }
            }
        }
        // enough blocks, every block loop once.
    } else {
        return;
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

  cudaDeviceSynchronize();
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

uint historical_max_queue_size = 0;
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
    uint queue_cap = num_rows;
    // last hop sample result do not need to enqueue
    for (int i = 0; i < hops - 1; i++)
        queue_cap += queue_cap * (num_picks_vec[i] + 1);

    nvtxRangePushA("create task_queue");
    auto task_queue = static_cast<thrust::pair<short, int64_t> *>(device->AllocWorkspace(ctx, queue_cap * sizeof(thrust::pair<short, int64_t>)));
//    auto task_queue = stdgpu::queue<thrust::pair<short, int64_t>>::createDeviceObject(queue_cap);
    nvtxRangePop();

    nvtxRangePushA("create bits");
    const auto bits_size = mat.num_rows * (hops - 1);
    uint* bool_arr = static_cast<uint *>(device->AllocWorkspace(ctx, bits_size * sizeof(uint)));
    //TODO: reset in init_kernel kernel(maybe better)
    CUDA_CALL(cudaMemset(bool_arr, 0, bits_size * sizeof(uint)));
    nvtxRangePop();

    // init
    const dim3 init_block(512);
//    const dim3 init_grid((num_rows + init_block.x - 1) / init_block.x);
//    const dim3 init_grid((num_rows * hops + init_block.x - 1) / init_block.x);
    const dim3 init_grid((queue_cap + init_block.x - 1) / init_block.x);
    CUDA_KERNEL_CALL((init_kernel), init_grid, init_block, 0, stream, task_queue, bool_arr, sliced_rows, num_rows, hops, queue_cap);
//    assert(task_queue.size() == num_rows);

    nvtxRangePushA("create array of structs of array");
    // use SoA instead of AoS, which do not need to use thrust::copy_if
    selectedEdgeInfo* struct_arr_h = (selectedEdgeInfo*)malloc(hops * sizeof(selectedEdgeInfo));
    int64_t node_num = num_rows;
    for(int i = 0; i < hops; i++) {
        int64_t hop_res_num = node_num * num_picks_vec[i];
        picked_rows[i] = NewIdArray(hop_res_num, ctx, sizeof(int64_t) * 8);
        picked_cols[i] = NewIdArray(hop_res_num, ctx, sizeof(int64_t) * 8);
        picked_indices[i] = NewIdArray(hop_res_num, ctx, sizeof(int64_t) * 8);
        struct_arr_h[i].rows = static_cast<int64_t*>(picked_rows[i]->data);
        struct_arr_h[i].cols = static_cast<int64_t*>(picked_cols[i]->data);
        struct_arr_h[i].datas = static_cast<int64_t*>(picked_indices[i]->data);
        node_num += hop_res_num;
    }
    //TODO: res_vector can be reused, and need not to reset everytime because we "push" the result by atomic index!
    // and then using `CreateView` to get the corrct result
    auto struct_arr_d = static_cast<selectedEdgeInfo *>(device->AllocWorkspace(ctx, hops * sizeof(selectedEdgeInfo)));
    CUDA_CALL(cudaMemcpy(struct_arr_d, struct_arr_h, sizeof(selectedEdgeInfo) * hops, cudaMemcpyHostToDevice));
    nvtxRangePop();

//    const uint64_t random_seed = RandomEngine::ThreadLocal()->RandInt(1000000000);
    // fix only for reproduce!
    const uint64_t random_seed = 1234;

    const dim3 block(BLOCK_SIZE_CUSTOM);
//    const dim3 grid(num_rows);
    uint est_queue_cap;
    if (historical_max_queue_size == 0)
        est_queue_cap = queue_cap;
    else
        // more extreme
        est_queue_cap = historical_max_queue_size + num_rows;
    const dim3 grid(est_queue_cap);
//    const dim3 grid(num_rows * hops);
//    const dim3 grid(1);

    uint* vector_lens = static_cast<uint *>(device->AllocWorkspace(ctx, hops * sizeof(uint)));
    CUDA_CALL(cudaMemset(vector_lens, 0, hops * sizeof(uint)));
    uint* vector_lens_h = (uint *) malloc(hops * sizeof(uint));
//    std::printf("cuda kernel launched\n");
    CUDA_KERNEL_CALL((_CSRRowWiseSampleUniformTaskParallelismKernel), grid, block, 0, stream,
                     random_seed, num_picks_ptr, vector_lens, num_rows, hops, mat.num_rows, in_ptr, in_cols, data,
                     task_queue, bool_arr, struct_arr_d);
//    assert(task_queue.empty());
//    CUDA_CALL(cudaDeviceSynchronize());

//    std::printf("cuda kernel finished\n");

    // 传多个COO res的row, col, idx的指针的指针，用res_vector取fill，逻辑上最直观. 传指针的指针要写个demo试一下
    nvtxRangePushA("get result");
    //TODO: cannot overlap with host(DGL can)
    CUDA_CALL(cudaMemcpy(vector_lens_h, vector_lens, sizeof(uint) * hops, cudaMemcpyDeviceToHost));
    std::vector<COOMatrix> ret_coo(hops);
    for(int i = 0; i < hops; i++) {
        picked_rows[i] = picked_rows[i].CreateView({vector_lens_h[i]}, picked_rows[i]->dtype);
        picked_cols[i] = picked_cols[i].CreateView({vector_lens_h[i]}, picked_cols[i]->dtype);
        picked_indices[i] = picked_indices[i].CreateView({vector_lens_h[i]}, picked_indices[i]->dtype);
        ret_coo[i] = COOMatrix(mat.num_rows, mat.num_cols, picked_rows[i], picked_cols[i], picked_indices[i]);
    }
    nvtxRangePop();

    //TODO: we can do this only one epoch
    uint actual_queue_size;
    CUDA_CALL(cudaMemcpyFromSymbol(&actual_queue_size, tail_index, sizeof(uint), 0));
    // for correctness
    // (if not all producer has finished their job, actual_queue_size is not the corrct value too)
    assert(actual_queue_size <= est_queue_cap);
    historical_max_queue_size = std::max(historical_max_queue_size, actual_queue_size);

    nvtxRangePushA("free container");
    free(struct_arr_h);
    free(vector_lens_h);
    device->FreeWorkspace(ctx, bool_arr);
    device->FreeWorkspace(ctx, struct_arr_d);
    device->FreeWorkspace(ctx, vector_lens);
    device->FreeWorkspace(ctx, task_queue);
    nvtxRangePop();

//    std::printf("CustomCSRRowWiseSamplingUniformTaskParallelism finished here\n");
    return ret_coo;
}

template COOMatrix CSRRowWiseSamplingUniform<kDGLCUDA, int32_t>(
    CSRMatrix, IdArray, int64_t, bool);
template COOMatrix CSRRowWiseSamplingUniform<kDGLCUDA, int64_t>(
    CSRMatrix, IdArray, int64_t, bool);

}  // namespace impl
}  // namespace aten
}  // namespace dgl
