// CUDA implementation of bucket_optimizer.py, variant="bundle", allow_neutral=False.
//
// A half warp owns one token.  The eight active (token, expert) pairs are loaded once,
// then warp ballots form the token's ownership and executability masks.  The
// 64 directed rank pairs are distributed across the 16 lanes.  Every token
// emits at most one proposal, so different accepted proposals cannot conflict.
//
// We stably sort the bucket bits of packed (bucket, token ID) keys, rather
// than using atomic queue insertion order.  CUB's stability preserves input
// token order without spending radix passes on the token ID bits.  Thus the
// CPU's deterministic order is preserved in every round.  Workspace is supplied
// by the caller; no allocation or CPU synchronization occurs during a round.

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cub/device/device_radix_sort.cuh>
#include <cstdint>

namespace {

constexpr int kRanks = 8;
constexpr int kTopK = 8;
constexpr int kTokenLanes = 16;
constexpr int kBucketBits = 11;
constexpr int kStride = kTopK + 1;
constexpr int kBuckets = kRanks * kRanks * 2 * kStride;
constexpr int kThreads = 256;
using SortKey = unsigned long long;

#define CHECK_I32_CUDA(x) TORCH_CHECK((x).is_cuda() && (x).is_contiguous() && \
    (x).scalar_type() == at::kInt, #x " must be contiguous CUDA int32")

// C/C++ remainder is signed; the CPU uses Python's non-negative modulo.
__device__ __forceinline__ int positive_mod(int value, int divisor) {
  const int result = value % divisor;
  return result < 0 ? result + divisor : result;
}

__device__ __forceinline__ int edge_tie(int a, int b, int token, int step) {
  const int lo = min(a, b), hi = max(a, b);
  const int pair = lo * (2 * kRanks - lo - 1) / 2 + hi - lo - 1;
  const int priority = 27 - positive_mod(pair - token - 17 * step, 28);
  const bool prefer_ascending = ((token / 28 + step) & 1) == 0;
  return 2 * priority + int((a < b) == prefer_ascending);
}

__device__ __forceinline__ unsigned first_bits(unsigned mask, int size) {
  unsigned result = 0;
  // Called only for sizes 1/2/4; a full old destination reuses its mask.
#pragma unroll 4
  for (int i = 0; i < 4; ++i) {
    if (i < size) {
      const unsigned bit = mask & (0u - mask);
      result |= bit;
      mask ^= bit;
    }
  }
  return result;
}

__global__ void support_mask_kernel(const bool* support, int* resident, int experts) {
  const int expert = blockIdx.x * blockDim.x + threadIdx.x;
  if (expert >= experts) return;
  unsigned mask = 0;
#pragma unroll
  for (int rank = 0; rank < kRanks; ++rank)
    mask |= unsigned(support[rank * experts + expert]) << rank;
  resident[expert] = int(mask);
}

__global__ void propose_kernel(const int* experts, const int* owner,
                               const int* resident, SortKey* packed,
                               int* move_masks, int tokens, int token_bits,
                               int per_source, int global_offset, int step) {
  const int linear = blockIdx.x * blockDim.x + threadIdx.x;
  const int token = linear / kTokenLanes;
  const int lane = threadIdx.x & 15;
  const int half_shift = (threadIdx.x & 16) ? 16 : 0;
  const bool valid = token < tokens;
  const bool valid_pair = valid && lane < kTopK;
  const int expert = valid_pair ? experts[token * kTopK + lane] : 0;
  const int old = valid_pair ? owner[token * kTopK + lane] : -1;
  const unsigned executable = valid_pair ? unsigned(resident[expert]) : 0;
  const int global_token = global_offset + token;
  const int source = global_token / per_source;

  // Each lane stores just one rank's two masks.  Dynamic rank access uses
  // shuffles, avoiding per-thread arrays that would spill to local memory.
  unsigned my_old_mask = 0, my_executable_mask = 0;
#pragma unroll
  for (int rank = 0; rank < kRanks; ++rank) {
    const unsigned old_mask = (__ballot_sync(0xffffffffu, valid_pair && old == rank)
                               >> half_shift) & 0xffffu;
    const unsigned executable_mask = (__ballot_sync(
        0xffffffffu, valid_pair && ((executable >> rank) & 1u)) >> half_shift) & 0xffffu;
    if ((lane & 7) == rank) {
      my_old_mask = old_mask;
      my_executable_mask = executable_mask;
    }
  }

  int best_score = -1;
  int best_key = kBuckets;
  unsigned best_mask = 0;
  const int preferred = (global_token / 28 + step) & 3;

#pragma unroll
  for (int iteration = 0; iteration < 4; ++iteration) {
    const int edge = lane + 16 * iteration;
    const int a = edge / kRanks, b = edge % kRanks;
    // These shuffles are unconditional: all 32 lanes named in the mask must
    // participate, including an invalid token in the final partial warp.
    const unsigned old_a = __shfl_sync(0xffffffffu, my_old_mask, a, 16);
    const unsigned old_b = __shfl_sync(0xffffffffu, my_old_mask, b, 16);
    const unsigned target_b = __shfl_sync(0xffffffffu, my_executable_mask, b, 16);
    const unsigned available = old_a & target_b;
    if (a == b || available == 0) continue;
    const int occupancy_a = __popc(old_a), occupancy_b = __popc(old_b);
    const int available_size = __popc(available);
    const int tie = edge_tie(a, b, global_token, step);

    // Canonical subsets: first 1, 2, 4 eligible choices, then the complete
    // old destination if all its pairs are executable and size is new.
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const int size = q < 3 ? (1 << q) : occupancy_a;
      if (q < 3) {
        if (size > available_size) continue;
      } else if (available_size != occupancy_a || size == 1 || size == 2 || size == 4) {
        continue;
      }
      const int gain = int(a != source && size == occupancy_a)
                     - int(b != source && occupancy_b == 0);
      if (gain < 0) continue;
      const int size_class = q;  // q==3 is precisely the other full sizes.
      const int size_tie = 3 - ((size_class - preferred + 4) & 3);
      const int score = ((gain * 4 + size_tie) * kStride + occupancy_b) * 56 + tie;
      // The low six bits explicitly preserve ascending (a,b) on an exact
      // score tie, matching the strict ">" comparison in the reference.
      const int ordered_score = score * 64 + (63 - edge);
      if (ordered_score > best_score) {
        best_score = ordered_score;
        best_key = ((edge * 2 + gain) * kStride) + size;
        best_mask = q == 3 ? available : first_bits(available, size);
      }
    }
  }

  // Half-warp reduction.  The winner's canonical subset and bucket travel
  // with its score; no global reduction or synchronization is necessary.
#pragma unroll
  for (int delta = 8; delta > 0; delta /= 2) {
    const int score = __shfl_down_sync(0xffffffffu, best_score, delta, 16);
    const int key = __shfl_down_sync(0xffffffffu, best_key, delta, 16);
    const unsigned mask = __shfl_down_sync(0xffffffffu, best_mask, delta, 16);
    if (lane + delta < 16 && score > best_score) {
      best_score = score;
      best_key = key;
      best_mask = mask;
    }
  }
  if (valid && lane == 0) {
    packed[token] = (SortKey(best_key) << token_bits) | SortKey(token);
    move_masks[token] = int(best_mask);
  }
}

__global__ void bucket_boundaries_kernel(const SortKey* sorted, int* starts,
                                         int* ends, int tokens, int token_bits) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= tokens) return;
  const int key = int(sorted[i] >> token_bits);
  if (i == 0 || int(sorted[i - 1] >> token_bits) != key) starts[key] = i;
  if (i == tokens - 1 || int(sorted[i + 1] >> token_bits) != key) ends[key] = i + 1;
}

__global__ void quota_kernel(const int* counts, int* accept,
                             unsigned long long* stats) {
  const int item = blockIdx.x * blockDim.x + threadIdx.x;
  if (item >= kRanks * kRanks * kTopK) return;
  const int size = item % kTopK + 1;
  const int edge = item / kTopK, a = edge / kRanks, b = edge % kRanks;
  if (a >= b) return;
  const int ab0 = ((a * kRanks + b) * 2) * kStride + size;
  const int ab1 = ab0 + kStride;
  const int ba0 = ((b * kRanks + a) * 2) * kStride + size;
  const int ba1 = ba0 + kStride;
  const int x0 = counts[ab0], x1 = counts[ab1];
  const int y0 = counts[ba0], y1 = counts[ba1];
  const int n = min(min(x0 + x1, y0 + y1), max(x1, y1));
  const int p = min(n, x1), q = min(n, y1);
  accept[ab1] = p;
  accept[ab0] = n - p;
  accept[ba1] = q;
  accept[ba0] = n - q;
  if (n) {
    atomicAdd(stats + 0, (unsigned long long)n);              // exchanges
    atomicAdd(stats + 1, (unsigned long long)(2 * n * size));  // moved pairs
    atomicAdd(stats + 2, (unsigned long long)(2 * n));         // accepted tokens
    atomicAdd(stats + 3, (unsigned long long)(p + q));         // predicted gain
  }
}

__global__ void commit_kernel(const SortKey* sorted, const int* starts,
                              const int* accept, const int* prefix, const int* move_masks,
                              int* owner, int tokens, int token_bits) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= tokens) return;
  const SortKey packed = sorted[i];
  const int key = int(packed >> token_bits);
  if (key >= kBuckets || i - starts[key] + prefix[key] >= accept[key]) return;
  const int token = int(packed & ((SortKey(1) << token_bits) - 1));
  const int destination = (key / kStride / 2) % kRanks;
  unsigned mask = unsigned(move_masks[token]);
  while (mask) {
    const int choice = __ffs(mask) - 1;
    mask &= mask - 1;
    owner[token * kTopK + choice] = destination;
  }
}

__global__ void counts_kernel(const int* starts, const int* ends, int* counts) {
  const int bucket = blockIdx.x * blockDim.x + threadIdx.x;
  if (bucket < kBuckets) counts[bucket] = ends[bucket] - starts[bucket];
}

__global__ void fold_counts_kernel(const int* gathered, int* total, int* prefix,
                                  int rank, int world) {
  const int bucket = blockIdx.x * blockDim.x + threadIdx.x;
  if (bucket >= kBuckets) return;
  int sum = 0, before = 0;
  for (int r = 0; r < world; ++r) {
    int n = gathered[r * kBuckets + bucket];
    sum += n;
    if (r < rank) before += n;
  }
  total[bucket] = sum;
  prefix[bucket] = before;
}

} // namespace

int64_t workspace_size(const at::Tensor& input, const at::Tensor& output,
                       int64_t token_bits) {
  c10::cuda::CUDAGuard guard(input.device());
  size_t bytes = 0;
  C10_CUDA_CHECK(cub::DeviceRadixSort::SortKeys(
      nullptr, bytes, reinterpret_cast<const SortKey*>(input.data_ptr<int64_t>()),
      reinterpret_cast<SortKey*>(output.data_ptr<int64_t>()), input.numel(),
      int(token_bits), int(token_bits) + kBucketBits, c10::cuda::getCurrentCUDAStream()));
  return int64_t(bytes);
}

void set_support(const at::Tensor& support, at::Tensor& resident) {
  TORCH_CHECK(support.is_cuda() && support.is_contiguous() && support.scalar_type() == at::kBool,
              "support must be contiguous CUDA bool");
  CHECK_I32_CUDA(resident);
  TORCH_CHECK(support.dim() == 2 && support.size(0) == kRanks &&
              support.size(1) == resident.numel(), "support must have shape [8,E]");
  c10::cuda::CUDAGuard guard(resident.device());
  const int experts = resident.numel();
  support_mask_kernel<<<(experts + 255) / 256, 256, 0, c10::cuda::getCurrentCUDAStream()>>>(
      support.data_ptr<bool>(), resident.data_ptr<int>(), experts);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void propose_round(const at::Tensor& experts, at::Tensor& owner,
                   const at::Tensor& resident, at::Tensor& keys,
                   at::Tensor& sorted, at::Tensor& move_masks,
                   at::Tensor& starts, at::Tensor& ends, at::Tensor& counts,
                   at::Tensor& workspace, int64_t token_bits,
                   int64_t per_source, int64_t global_offset, int64_t step) {
  CHECK_I32_CUDA(experts); CHECK_I32_CUDA(owner); CHECK_I32_CUDA(resident);
  CHECK_I32_CUDA(move_masks); CHECK_I32_CUDA(starts); CHECK_I32_CUDA(ends);
  CHECK_I32_CUDA(counts);
  TORCH_CHECK(experts.dim() == 2 && experts.size(1) == kTopK &&
              owner.sizes() == experts.sizes(), "routes and owner require [T,8]");
  TORCH_CHECK(starts.numel() == kBuckets + 1 && ends.numel() == kBuckets + 1 &&
              counts.numel() == kBuckets && move_masks.numel() == experts.size(0),
              "wrong bucket or proposal buffer sizes");
  TORCH_CHECK(keys.is_cuda() && sorted.is_cuda() && keys.is_contiguous() && sorted.is_contiguous() &&
              keys.scalar_type() == at::kLong && sorted.scalar_type() == at::kLong &&
              keys.numel() == experts.size(0) && sorted.numel() == experts.size(0),
              "keys must be int64[T]");
  c10::cuda::CUDAGuard guard(experts.device());
  const auto stream = c10::cuda::getCurrentCUDAStream();
  const int tokens = experts.size(0);
  auto* key_ptr = reinterpret_cast<SortKey*>(keys.data_ptr<int64_t>());
  auto* sorted_ptr = reinterpret_cast<SortKey*>(sorted.data_ptr<int64_t>());
  C10_CUDA_CHECK(cudaMemsetAsync(starts.data_ptr<int>(), 0, (kBuckets + 1) * sizeof(int), stream));
  C10_CUDA_CHECK(cudaMemsetAsync(ends.data_ptr<int>(), 0, (kBuckets + 1) * sizeof(int), stream));
  propose_kernel<<<(tokens * kTokenLanes + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
      experts.data_ptr<int>(), owner.data_ptr<int>(), resident.data_ptr<int>(),
      key_ptr, move_masks.data_ptr<int>(), tokens, token_bits, per_source, global_offset, step);
  size_t bytes = workspace.numel();
  C10_CUDA_CHECK(cub::DeviceRadixSort::SortKeys(
      workspace.data_ptr<uint8_t>(), bytes, key_ptr, sorted_ptr, tokens,
      int(token_bits), int(token_bits) + kBucketBits, stream));
  bucket_boundaries_kernel<<<(tokens + 255) / 256, 256, 0, stream>>>(
      sorted_ptr, starts.data_ptr<int>(), ends.data_ptr<int>(), tokens, token_bits);
  counts_kernel<<<(kBuckets + 255) / 256, 256, 0, stream>>>(
      starts.data_ptr<int>(), ends.data_ptr<int>(), counts.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void finish_round(const at::Tensor& sorted, const at::Tensor& move_masks,
                  const at::Tensor& starts, const at::Tensor& gathered,
                  at::Tensor& total, at::Tensor& prefix, at::Tensor& accept,
                  at::Tensor& owner, at::Tensor& stats, int64_t token_bits,
                  int64_t rank, int64_t world, int64_t step) {
  CHECK_I32_CUDA(move_masks); CHECK_I32_CUDA(starts); CHECK_I32_CUDA(gathered);
  CHECK_I32_CUDA(total); CHECK_I32_CUDA(prefix); CHECK_I32_CUDA(accept); CHECK_I32_CUDA(owner);
  TORCH_CHECK(world >= 1 && world <= 8 && rank >= 0 && rank < world,
              "gather rank/world must describe up to eight contiguous token shards");
  TORCH_CHECK(gathered.numel() == world * kBuckets && total.numel() == kBuckets &&
              prefix.numel() == kBuckets && accept.numel() == kBuckets + 1,
              "wrong distributed bucket buffer sizes");
  TORCH_CHECK(step >= 0 && step < stats.size(0) && stats.size(1) == 4 &&
              stats.scalar_type() == at::kLong && stats.is_cuda(), "invalid stats buffer");
  c10::cuda::CUDAGuard guard(owner.device());
  const auto stream = c10::cuda::getCurrentCUDAStream();
  auto* stat_ptr = reinterpret_cast<unsigned long long*>(stats.data_ptr<int64_t>() + step * 4);
  C10_CUDA_CHECK(cudaMemsetAsync(stat_ptr, 0, 4 * sizeof(int64_t), stream));
  fold_counts_kernel<<<(kBuckets + 255) / 256, 256, 0, stream>>>(
      gathered.data_ptr<int>(), total.data_ptr<int>(), prefix.data_ptr<int>(), rank, world);
  quota_kernel<<<(kRanks * kRanks * kTopK + 255) / 256, 256, 0, stream>>>(
      total.data_ptr<int>(), accept.data_ptr<int>(), stat_ptr);
  const int tokens = owner.size(0);
  commit_kernel<<<(tokens + 255) / 256, 256, 0, stream>>>(
      reinterpret_cast<const SortKey*>(sorted.data_ptr<int64_t>()), starts.data_ptr<int>(),
      accept.data_ptr<int>(), prefix.data_ptr<int>(), move_masks.data_ptr<int>(),
      owner.data_ptr<int>(), tokens, token_bits);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("workspace_size", &workspace_size, "CUB temporary storage size");
  m.def("set_support", &set_support, "Pack support into expert rank masks");
  m.def("propose_round", &propose_round, "Local proposal, stable sort and counts");
  m.def("finish_round", &finish_round, "Global count quotas and prefix-aware commit");
}
