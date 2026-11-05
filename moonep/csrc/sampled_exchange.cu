// R8 source-local exchange kernels. Live rank collectives are in sampled_group.py.
// Proposal/tie semantics copied from the audited bundle_lane8; prefix mechanism
// adapted from two_gpu_overlap, with R8 dimensions and source offsets preserved.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cub/block/block_scan.cuh>
#include <cstdint>

namespace {

constexpr int kRanks = 8;
constexpr int kTopK = 8;
constexpr int kTokenLanes = 8;
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

__global__ void propose_kernel(const int* old_masks, const int* exec_masks,
                               int* packed,
                               int* move_masks, int tokens, int token_bits,
                               int per_source, int global_offset, int step) {
  const int linear = blockIdx.x * blockDim.x + threadIdx.x;
  const int token = linear / kTokenLanes;
  const int lane = threadIdx.x & (kTokenLanes - 1);
  const int half_shift = (threadIdx.x & 31) & ~(kTokenLanes - 1);
  const bool valid = token < tokens;
  const int global_token = global_offset + token;
  const int source = global_token / per_source;
  const unsigned my_old_mask = valid ? unsigned(old_masks[token*8+lane]) : 0;
  const unsigned my_executable_mask = valid ? unsigned(exec_masks[token*8+lane]) : 0;

  int best_score = -1;
  int best_key = kBuckets;
  unsigned best_mask = 0;
  const int preferred = (global_token / 28 + step) & 3;

#pragma unroll
  for (int iteration = 0; iteration < 64 / kTokenLanes; ++iteration) {
    const int edge = lane + kTokenLanes * iteration;
    const int a = edge / kRanks, b = edge % kRanks;
    // These shuffles are unconditional: all 32 lanes named in the mask must
    // participate, including an invalid token in the final partial warp.
    const unsigned old_a = __shfl_sync(0xffffffffu, my_old_mask, a, kTokenLanes);
    const unsigned old_b = __shfl_sync(0xffffffffu, my_old_mask, b, kTokenLanes);
    const unsigned target_b = __shfl_sync(0xffffffffu, my_executable_mask, b, kTokenLanes);
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

  // Eight-lane subgroup reduction.  The winner's canonical subset and bucket travel
  // with its score; no global reduction or synchronization is necessary.
#pragma unroll
  for (int delta = kTokenLanes / 2; delta > 0; delta /= 2) {
    const int score = __shfl_down_sync(0xffffffffu, best_score, delta, kTokenLanes);
    const int key = __shfl_down_sync(0xffffffffu, best_key, delta, kTokenLanes);
    const unsigned mask = __shfl_down_sync(0xffffffffu, best_mask, delta, kTokenLanes);
    if (lane + delta < kTokenLanes && score > best_score) {
      best_score = score;
      best_key = key;
      best_mask = mask;
    }
  }
  if (valid && lane == 0) {
    packed[token] = best_key;
    move_masks[token] = int(best_mask);
  }
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

__global__ void cache_exec(const int* routes,const int* resident,int* exec_masks,int S) {
 int linear=blockIdx.x*blockDim.x+threadIdx.x,t=linear/8,lane=threadIdx.x&7;
 int shift=(threadIdx.x&31)&~7;bool valid=t<S;unsigned ex=valid?resident[routes[t*8+lane]]:0;
 #pragma unroll
 for(int r=0;r<8;r++) {
  unsigned bits=(__ballot_sync(0xffffffffu,valid&&((ex>>r)&1))>>shift)&255;
  if(valid&&lane==r)exec_masks[t*8+lane]=bits;
 }
}
__global__ void reset_masks(const int* seed,int* owner,int* masks,int S) {
 int linear=blockIdx.x*blockDim.x+threadIdx.x,t=linear/8,lane=threadIdx.x&7;
 int shift=(threadIdx.x&31)&~7;bool valid=t<S;int x=valid?seed[t*8+lane]:-1;
 if(valid)owner[t*8+lane]=x;
 #pragma unroll
 for(int r=0;r<8;r++) {
  unsigned bits=(__ballot_sync(0xffffffffu,valid&&x==r)>>shift)&255;
  if(valid&&lane==r)masks[t*8+lane]=bits;
 }
}
__global__ void stable_ordinals(const int* keys,int* ordinals,int* block_counts,int S,int blocks) {
 __shared__ int wh[8][kBuckets+1];
 int lane=threadIdx.x&31,warp=threadIdx.x/32;
 for(int i=threadIdx.x;i<8*(kBuckets+1);i+=256)reinterpret_cast<int*>(wh)[i]=0;
 __syncthreads();
 int t=blockIdx.x*256+threadIdx.x,key=t<S?keys[t]:kBuckets;
 unsigned peers=__match_any_sync(0xffffffffu,key),lower=(1u<<lane)-1u;
 int local=__popc(peers&lower);
 if(lane==__ffs(peers)-1)wh[warp][key]=__popc(peers);
 __syncthreads();
 if(t<S) {
  int ordinal=local;
  #pragma unroll
  for(int w=0;w<8;w++)if(w<warp)ordinal+=wh[w][key];
  ordinals[t]=ordinal;
 }
 for(int b=threadIdx.x;b<kBuckets;b+=256) {
  int count=0;
  #pragma unroll
  for(int w=0;w<8;w++)count+=wh[w][b];
  block_counts[b*blocks+blockIdx.x]=count;
 }
}
__global__ void scan_blocks(const int* block_counts,int* block_prefix,int* counts,int blocks) {
 using Scan=cub::BlockScan<int,256>;__shared__ typename Scan::TempStorage temp;__shared__ int carry;
 int b=blockIdx.x;if(threadIdx.x==0)carry=0;__syncthreads();
 for(int begin=0;begin<blocks;begin+=256) {
  int i=begin+threadIdx.x,value=i<blocks?block_counts[b*blocks+i]:0,prefix,total;
  Scan(temp).ExclusiveSum(value,prefix,total);int prior=carry;
  if(i<blocks)block_prefix[b*blocks+i]=prior+prefix;
  __syncthreads();if(threadIdx.x==0)carry=prior+total;__syncthreads();
 }
 if(threadIdx.x==0)counts[b]=carry;
}
__global__ void apply_prefix(const int* keys,const int* move,const int* ordinal,const int* block_prefix,
 const int* prefix,const int* accept,int* old_masks,int* owner,int S,int blocks) {
 int t=blockIdx.x*blockDim.x+threadIdx.x;if(t>=S)return;int key=keys[t];if(key>=kBuckets)return;
 int pos=ordinal[t]+block_prefix[key*blocks+t/256]+prefix[key];if(pos>=accept[key])return;
 int edge=key/9/2,a=edge/8,b=edge%8;unsigned bits=move[t];
 old_masks[t*8+a]&=~bits;old_masks[t*8+b]|=bits;
 while(bits){int k=__ffs(bits)-1;bits&=bits-1;owner[t*8+k]=b;}
}
} // namespace

void prepare(torch::Tensor routes,torch::Tensor support,torch::Tensor resident,torch::Tensor exec_masks) {
 CHECK_I32_CUDA(routes);CHECK_I32_CUDA(resident);CHECK_I32_CUDA(exec_masks);
 TORCH_CHECK(support.is_cuda()&&support.scalar_type()==at::kBool&&support.is_contiguous()&&support.dim()==2&&support.size(0)==8);
 c10::cuda::CUDAGuard guard(routes.device());auto s=c10::cuda::getCurrentCUDAStream();int E=support.size(1),S=routes.size(0);
 support_mask_kernel<<<(E+255)/256,256,0,s>>>(support.data_ptr<bool>(),resident.data_ptr<int>(),E);
 cache_exec<<<(S*8+255)/256,256,0,s>>>(routes.data_ptr<int>(),resident.data_ptr<int>(),exec_masks.data_ptr<int>(),S);
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void reset(torch::Tensor seed,torch::Tensor owner,torch::Tensor masks) {
 CHECK_I32_CUDA(seed);CHECK_I32_CUDA(owner);CHECK_I32_CUDA(masks);
 TORCH_CHECK(seed.sizes()==owner.sizes()&&seed.sizes()==masks.sizes());
 c10::cuda::CUDAGuard guard(seed.device());auto s=c10::cuda::getCurrentCUDAStream();int S=seed.size(0);
 reset_masks<<<(S*8+255)/256,256,0,s>>>(seed.data_ptr<int>(),owner.data_ptr<int>(),masks.data_ptr<int>(),S);C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void propose(torch::Tensor masks,torch::Tensor exec_masks,torch::Tensor keys,torch::Tensor moves,
 torch::Tensor ordinals,torch::Tensor block_counts,torch::Tensor block_prefix,torch::Tensor counts,int per_source,int offset,int step) {
 c10::cuda::CUDAGuard guard(masks.device());auto s=c10::cuda::getCurrentCUDAStream();int S=masks.size(0),blocks=(S+255)/256;
 propose_kernel<<<(S*8+255)/256,256,0,s>>>(masks.data_ptr<int>(),exec_masks.data_ptr<int>(),keys.data_ptr<int>(),moves.data_ptr<int>(),S,0,per_source,offset,step);
 stable_ordinals<<<blocks,256,0,s>>>(keys.data_ptr<int>(),ordinals.data_ptr<int>(),block_counts.data_ptr<int>(),S,blocks);
 scan_blocks<<<kBuckets,256,0,s>>>(block_counts.data_ptr<int>(),block_prefix.data_ptr<int>(),counts.data_ptr<int>(),blocks);C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void quota(torch::Tensor gathered,torch::Tensor total,torch::Tensor prefix,torch::Tensor accept,torch::Tensor stats,int source,int step) {
 c10::cuda::CUDAGuard guard(gathered.device());auto s=c10::cuda::getCurrentCUDAStream();
 TORCH_CHECK(gathered.numel()==8*kBuckets&&source>=0&&source<8);
 auto st=reinterpret_cast<unsigned long long*>(stats.data_ptr<int64_t>())+step*4;
 C10_CUDA_CHECK(cudaMemsetAsync(st,0,4*sizeof(int64_t),s));
 C10_CUDA_CHECK(cudaMemsetAsync(accept.data_ptr<int>(),0,kBuckets*sizeof(int),s));
 fold_counts_kernel<<<(kBuckets+255)/256,256,0,s>>>(gathered.data_ptr<int>(),total.data_ptr<int>(),prefix.data_ptr<int>(),source,8);
 quota_kernel<<<2,256,0,s>>>(total.data_ptr<int>(),accept.data_ptr<int>(),st);C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void apply(torch::Tensor keys,torch::Tensor moves,torch::Tensor ordinals,torch::Tensor block_prefix,
 torch::Tensor prefix,torch::Tensor accept,torch::Tensor masks,torch::Tensor owner) {
 c10::cuda::CUDAGuard guard(owner.device());auto s=c10::cuda::getCurrentCUDAStream();int S=owner.size(0);
 apply_prefix<<<(S+255)/256,256,0,s>>>(keys.data_ptr<int>(),moves.data_ptr<int>(),ordinals.data_ptr<int>(),block_prefix.data_ptr<int>(),prefix.data_ptr<int>(),accept.data_ptr<int>(),masks.data_ptr<int>(),owner.data_ptr<int>(),S,(S+255)/256);C10_CUDA_KERNEL_LAUNCH_CHECK();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("prepare",&prepare);m.def("reset",&reset);m.def("propose",&propose);m.def("quota",&quota);m.def("apply",&apply);}

