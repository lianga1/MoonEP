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
#include <climits>

#ifndef MOONEP_EXCHANGE_TOP_K
#define MOONEP_EXCHANGE_TOP_K 8
#endif

namespace {

constexpr int kRanks = 8;
constexpr int kTopK = MOONEP_EXCHANGE_TOP_K;
static_assert(kTopK >= 1 && kTopK <= 32, "K must fit a uint32 choice mask");
constexpr int kTokenLanes = 8;
constexpr int kStride = kTopK + 1;
constexpr int kBuckets = kRanks * kRanks * 2 * kStride;
constexpr int kThreads = 256;
constexpr int power_classes() {
  int count = 0;
  for (int size = 1; size < kTopK; size *= 2) ++count;
  return count;
}
constexpr int kPowerClasses = power_classes();
constexpr int kSizeClasses = kPowerClasses + 1;
using Mask = uint32_t;

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
  // Reduce periodic terms first so large valid token/round indices cannot
  // overflow signed arithmetic. These are the original modulo/parity rules.
  const int priority = 27 - positive_mod(pair - token % 28 - 17 * (step % 28), 28);
  const bool prefer_ascending = (((token / 28) & 1) ^ (step & 1)) == 0;
  return 2 * priority + int((a < b) == prefer_ascending);
}

__device__ __forceinline__ unsigned first_bits(unsigned mask, int size) {
  unsigned result = 0;
  // size is a power of two strictly below K (at most 16). Full groups
  // reuse their existing mask, including all 32 choices without 1u << 32.
  for (int i = 0; i < size; ++i) {
    const unsigned bit = mask & (0u - mask);
    result |= bit;
    mask ^= bit;
  }
  return result;
}

__device__ __forceinline__ void support_mask_item(const bool* support, int* resident, int experts, int expert) {
  if (expert >= experts) return;
  unsigned mask = 0;
#pragma unroll
  for (int rank = 0; rank < kRanks; ++rank)
    mask |= unsigned(support[rank * experts + expert]) << rank;
  resident[expert] = int(mask);
}

__device__ __forceinline__ void propose_tile(const Mask* old_masks, const Mask* exec_masks,
                               int* packed,
                               Mask* move_masks, int tokens,
                               int per_source, int global_offset, int step, int tile) {
  const int64_t linear = int64_t(tile) * blockDim.x + threadIdx.x;
  const int token = linear / kTokenLanes;
  const int lane = threadIdx.x & (kTokenLanes - 1);
  const bool valid = token < tokens;
  const int global_token = global_offset + (valid ? token : 0);
  const int source = global_token / per_source;
  const unsigned my_old_mask = valid ? old_masks[int64_t(token)*kRanks+lane] : 0;
  const unsigned my_executable_mask = valid ? exec_masks[int64_t(token)*kRanks+lane] : 0;

  int best_score = -1;
  int best_key = kBuckets;
  unsigned best_mask = 0;
  const int preferred = ((global_token / 28) % kSizeClasses + step % kSizeClasses) % kSizeClasses;

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

    // Canonical subsets: powers of two below K, then the complete old
    // destination if all its pairs are executable and its size is new.
    // At K=8 this is exactly the original 1/2/4/full candidate order.
#pragma unroll
    for (int q = 0; q < kSizeClasses; ++q) {
      const bool full = q == kPowerClasses;
      const int size = full ? occupancy_a : (1 << q);
      if (!full) {
        if (size > available_size) continue;
      } else if (available_size != occupancy_a ||
                 (size < kTopK && (size & (size - 1)) == 0)) {
        continue;
      }
      const int gain = int(a != source && size == occupancy_a)
                     - int(b != source && occupancy_b == 0);
      if (gain < 0) continue;
      const int size_tie = kSizeClasses - 1 - positive_mod(q - preferred, kSizeClasses);
      const int score = ((gain * kSizeClasses + size_tie) * kStride + occupancy_b) * 56 + tie;
      // The low six bits explicitly preserve ascending (a,b) on an exact
      // score tie, matching the strict ">" comparison in the reference.
      const int ordered_score = score * 64 + (63 - edge);
      if (ordered_score > best_score) {
        best_score = ordered_score;
        best_key = ((edge * 2 + gain) * kStride) + size;
        best_mask = full ? available : first_bits(available, size);
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
    move_masks[token] = best_mask;
  }
}

__device__ __forceinline__ void quota_item(const int* counts, int* accept,
                             unsigned long long* stats, int item) {
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
    atomicAdd(stats + 1, 2ull * n * size);                    // moved pairs
    atomicAdd(stats + 2, 2ull * n);                           // accepted tokens
    atomicAdd(stats + 3, (unsigned long long)(p + q));         // predicted gain
  }
}

__device__ __forceinline__ void fold_item(const int* gathered, int* total, int* prefix,
                                  int rank, int world, int bucket) {
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

__device__ __forceinline__ void cache_tile(const int* routes,const int* resident,Mask* exec_masks,int S, int tile) {
 const int t=(int64_t(tile)*blockDim.x+threadIdx.x)/kTokenLanes;
 const int lane=threadIdx.x&(kTokenLanes-1),shift=(threadIdx.x&31)&~(kTokenLanes-1);
 const bool valid=t<S;Mask rank_mask=0;
 // Each subgroup still has R=8 lanes. It consumes ceil(K/8) choice chunks;
 // the last chunk predicates its loads but all lanes execute every ballot.
 #pragma unroll
 for(int base=0;base<kTopK;base+=kTokenLanes) {
  const int k=base+lane;const bool choice_valid=valid&&k<kTopK;
  const unsigned ex=choice_valid?unsigned(resident[routes[int64_t(t)*kTopK+k]]):0;
  #pragma unroll
  for(int r=0;r<kRanks;r++) {
   const Mask bits=(__ballot_sync(0xffffffffu,choice_valid&&((ex>>r)&1u))>>shift)&255u;
   if(lane==r)rank_mask|=bits<<base;  // base is 0/8/16/24, never 32.
  }
 }
 if(valid)exec_masks[int64_t(t)*kRanks+lane]=rank_mask;
}
__device__ __forceinline__ void reset_tile(const int* seed,int* owner,Mask* masks,int S, int tile) {
 const int t=(int64_t(tile)*blockDim.x+threadIdx.x)/kTokenLanes;
 const int lane=threadIdx.x&(kTokenLanes-1),shift=(threadIdx.x&31)&~(kTokenLanes-1);
 const bool valid=t<S;Mask rank_mask=0;
 #pragma unroll
 for(int base=0;base<kTopK;base+=kTokenLanes) {
  const int k=base+lane;const bool choice_valid=valid&&k<kTopK;
  const int x=choice_valid?seed[int64_t(t)*kTopK+k]:-1;
  if(choice_valid)owner[int64_t(t)*kTopK+k]=x;
  #pragma unroll
  for(int r=0;r<kRanks;r++) {
   const Mask bits=(__ballot_sync(0xffffffffu,choice_valid&&x==r)>>shift)&255u;
   if(lane==r)rank_mask|=bits<<base;
  }
 }
 if(valid)masks[int64_t(t)*kRanks+lane]=rank_mask;
}
template<int TileTokens=256>
__device__ __forceinline__ void stable_tile(const int* keys,int* ordinals,int* block_counts,int S,int blocks, int tile) {
 static_assert(TileTokens==64||TileTokens==128||TileTokens==256,"supported histogram tile");
 constexpr int tile_warps=TileTokens/32;
 if constexpr(kTopK<=8) {
  // Only warps with tokens need a private histogram. The default retains the
  // audited 8*(1152+1)*4 = 36,896-byte K=8 layout; smaller tiles use 2/4 warps.
  __shared__ int wh[tile_warps][kBuckets+1];
  int lane=threadIdx.x&31,warp=threadIdx.x/32;
  for(int i=threadIdx.x;i<tile_warps*(kBuckets+1);i+=kThreads)reinterpret_cast<int*>(wh)[i]=0;
  __syncthreads();
  int t=tile*TileTokens+threadIdx.x;
  const bool valid=threadIdx.x<TileTokens&&t<S;
  int key=valid?keys[t]:kBuckets;
  unsigned peers=__match_any_sync(0xffffffffu,key),lower=(1u<<lane)-1u;
  int local=__popc(peers&lower);
  if(warp<tile_warps&&lane==__ffs(peers)-1)wh[warp][key]=__popc(peers);
  __syncthreads();
  if(valid) {
   int ordinal=local;
   #pragma unroll
   for(int w=0;w<tile_warps;w++)if(w<warp)ordinal+=wh[w][key];
   ordinals[t]=ordinal;
  }
  for(int b=threadIdx.x;b<kBuckets;b+=kThreads) {
   int count=0;
   #pragma unroll
   for(int w=0;w<tile_warps;w++)count+=wh[w][b];
   block_counts[int64_t(b)*blocks+tile]=count;
  }
 } else {
 // One table per block, not per warp: max 4*(4224+1)=16,900 bytes at K=32.
 // Warps publish in token order; each match_any group has one unique writer.
 __shared__ int histogram[kBuckets+1];
 int lane=threadIdx.x&31,warp=threadIdx.x/32;
 for(int i=threadIdx.x;i<kBuckets+1;i+=kThreads)histogram[i]=0;
 __syncthreads();
 int t=tile*TileTokens+threadIdx.x;
 const bool valid=threadIdx.x<TileTokens&&t<S;
 int key=valid?keys[t]:kBuckets;
 unsigned peers=__match_any_sync(0xffffffffu,key),lower=(1u<<lane)-1u;
 int local=__popc(peers&lower);
 #pragma unroll
 for(int w=0;w<tile_warps;w++) {
  if(warp==w) {
   const int prior=histogram[key];
   // A leader must not publish until every peer has read the old count.
   __syncwarp(0xffffffffu);
   if(valid)ordinals[t]=prior+local;
   if(lane==__ffs(peers)-1)histogram[key]=prior+__popc(peers);
  }
  __syncthreads();
 }
 for(int b=threadIdx.x;b<kBuckets;b+=kThreads)
  block_counts[int64_t(b)*blocks+tile]=histogram[b];
 }
}
__device__ __forceinline__ void scan_bucket(const int* block_counts,int* block_prefix,int* counts,int blocks, int b) {
 using Scan=cub::BlockScan<int,256>;__shared__ typename Scan::TempStorage temp;__shared__ int carry;
 if(threadIdx.x==0)carry=0;__syncthreads();
 for(int begin=0;begin<blocks;begin+=256) {
  int i=begin+threadIdx.x,value=i<blocks?block_counts[int64_t(b)*blocks+i]:0,prefix,total;
  Scan(temp).ExclusiveSum(value,prefix,total);int prior=carry;
  if(i<blocks)block_prefix[int64_t(b)*blocks+i]=prior+prefix;
  __syncthreads();if(threadIdx.x==0)carry=prior+total;__syncthreads();
 }
 if(threadIdx.x==0)counts[b]=carry;
}
template<int TileTokens=256>
__device__ __forceinline__ void apply_item(const int* keys,const Mask* move,const int* ordinal,const int* block_prefix,
 const int* prefix,const int* accept,Mask* old_masks,int* owner,int S,int blocks, int t) {
 if(t>=S)return;int key=keys[t];if(key>=kBuckets)return;
 int pos=ordinal[t]+block_prefix[int64_t(key)*blocks+t/TileTokens]+prefix[key];if(pos>=accept[key])return;
 int edge=key/kStride/2,a=edge/kRanks,b=edge%kRanks;Mask bits=move[t];
 old_masks[int64_t(t)*kRanks+a]&=~bits;old_masks[int64_t(t)*kRanks+b]|=bits;
 while(bits){int k=__ffs(static_cast<int>(bits))-1;bits&=bits-1;owner[int64_t(t)*kTopK+k]=b;}
}

// A physical block may process multiple virtual tiles. The virtual IDs remain
// the original IDs in block_counts/block_prefix, so scheduling cannot affect
// either token ordinals or the selected exchange quota.
__global__ void support_mask_kernel(const bool* support,int* resident,int experts) {
 for(int expert=blockIdx.x*blockDim.x+threadIdx.x;expert<experts;expert+=gridDim.x*blockDim.x)
  support_mask_item(support,resident,experts,expert);
}
__global__ void propose_kernel(const Mask* old_masks,const Mask* exec_masks,int* packed,
 Mask* move_masks,int tokens,int per_source,int global_offset,int step) {
 const int tiles=(int64_t(tokens)*kTokenLanes+kThreads-1)/kThreads;
 for(int tile=blockIdx.x;tile<tiles;tile+=gridDim.x)
  propose_tile(old_masks,exec_masks,packed,move_masks,tokens,per_source,global_offset,step,tile);
}
__global__ void quota_kernel(const int* counts,int* accept,unsigned long long* stats) {
 for(int item=blockIdx.x*blockDim.x+threadIdx.x;item<kRanks*kRanks*kTopK;item+=gridDim.x*blockDim.x)
  quota_item(counts,accept,stats,item);
}
__global__ void fold_counts_kernel(const int* gathered,int* total,int* prefix,int rank,int world) {
 for(int bucket=blockIdx.x*blockDim.x+threadIdx.x;bucket<kBuckets;bucket+=gridDim.x*blockDim.x)
  fold_item(gathered,total,prefix,rank,world,bucket);
}
__global__ void cache_exec(const int* routes,const int* resident,Mask* exec_masks,int S) {
 const int tiles=(int64_t(S)*kTokenLanes+kThreads-1)/kThreads;
 for(int tile=blockIdx.x;tile<tiles;tile+=gridDim.x)
  cache_tile(routes,resident,exec_masks,S,tile);
}
__global__ void reset_masks(const int* seed,int* owner,Mask* masks,int S) {
 const int tiles=(int64_t(S)*kTokenLanes+kThreads-1)/kThreads;
 for(int tile=blockIdx.x;tile<tiles;tile+=gridDim.x)
  reset_tile(seed,owner,masks,S,tile);
}
template<int TileTokens=256>
__global__ void stable_ordinals(const int* keys,int* ordinals,int* block_counts,int S,int blocks) {
 for(int tile=blockIdx.x;tile<blocks;tile+=gridDim.x) {
  stable_tile<TileTokens>(keys,ordinals,block_counts,S,blocks,tile);
  // All consumers of this tile's shared histogram finish before it is reused.
  __syncthreads();
 }
}
__global__ void scan_blocks(const int* block_counts,int* block_prefix,int* counts,int blocks) {
 for(int bucket=blockIdx.x;bucket<kBuckets;bucket+=gridDim.x) {
  scan_bucket(block_counts,block_prefix,counts,blocks,bucket);
  __syncthreads();
 }
}
// The bounded path assigns a complete bucket to one warp. Eight independent
// buckets advance per CTA, avoiding a CTA-wide scan/barrier for every bucket.
// Segments still visit virtual tile IDs in increasing order; the carry is the
// exact sum of all preceding segments, including arbitrary non-32 tails.
__global__ void scan_blocks_warp(const int* block_counts,int* block_prefix,int* counts,int blocks) {
 const int lane=threadIdx.x&31,warp=threadIdx.x/32;
 constexpr int warps=kThreads/32;
 for(int bucket=blockIdx.x*warps+warp;bucket<kBuckets;bucket+=gridDim.x*warps) {
  int carry=0;
  for(int begin=0;begin<blocks;begin+=32) {
   const int tile=begin+lane;
   const int value=tile<blocks?block_counts[int64_t(bucket)*blocks+tile]:0;
   int inclusive=value;
   #pragma unroll
   for(int delta=1;delta<32;delta*=2) {
    const int prior=__shfl_up_sync(0xffffffffu,inclusive,delta);
    if(lane>=delta)inclusive+=prior;
   }
   if(tile<blocks)block_prefix[int64_t(bucket)*blocks+tile]=carry+inclusive-value;
   carry+=__shfl_sync(0xffffffffu,inclusive,31);
  }
  if(lane==0)counts[bucket]=carry;
 }
}
template<int TileTokens=256>
__global__ void apply_prefix(const int* keys,const Mask* move,const int* ordinal,const int* block_prefix,
 const int* prefix,const int* accept,Mask* old_masks,int* owner,int S,int blocks) {
 for(int token=blockIdx.x*blockDim.x+threadIdx.x;token<S;token+=gridDim.x*blockDim.x)
  apply_item<TileTokens>(keys,move,ordinal,block_prefix,prefix,accept,old_masks,owner,S,blocks,token);
}
__global__ void zero_quota(int* accept,unsigned long long* stats) {
 for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<kBuckets;i+=gridDim.x*blockDim.x)accept[i]=0;
 if(blockIdx.x==0&&threadIdx.x<4)stats[threadIdx.x]=0;
}

// A smaller histogram tile gives the bounded grid finer work division. Each
// tile still covers a contiguous source-token interval. Its local ordinal plus
// the exclusive prefix of all preceding tiles therefore preserves exactly the
// same global ordinal, irrespective of TileTokens or physical CTA scheduling.
template<int TileTokens=256>
__global__ void propose_ordinals_fused(const Mask* old_masks,const Mask* exec_masks,
 int* keys,Mask* moves,int* ordinals,int* block_counts,int S,int blocks,
 int per_source,int global_offset,int step) {
 for(int tile=blockIdx.x;tile<blocks;tile+=gridDim.x) {
  constexpr int parts=TileTokens/(kThreads/kTokenLanes);
  // Reuse the frozen proposal body; never duplicate or reorder its candidates.
  #pragma unroll 1
  for(int part=0;part<parts;++part)
   propose_tile(old_masks,exec_masks,keys,moves,S,per_source,global_offset,step,tile*parts+part);
  __syncthreads();
  stable_tile<TileTokens>(keys,ordinals,block_counts,S,blocks,tile);
  __syncthreads();
 }
}

// This small table fits one CTA. Combining reset/fold/quota avoids launching
// a memset kernel and preserves the exact per-source prefix and pairing rules.
__global__ void fold_quota_fused(const int* gathered,int* total,int* prefix,
 int* accept,unsigned long long* stats,int source) {
 for(int bucket=threadIdx.x;bucket<kBuckets;bucket+=blockDim.x)accept[bucket]=0;
 if(threadIdx.x<4)stats[threadIdx.x]=0;
 for(int bucket=threadIdx.x;bucket<kBuckets;bucket+=blockDim.x)
  fold_item(gathered,total,prefix,source,kRanks,bucket);
 __syncthreads();
 for(int item=threadIdx.x;item<kRanks*kRanks*kTopK;item+=blockDim.x)
  quota_item(total,accept,stats,item);
}

int limited_grid(int64_t work,int sm_limit) {
 TORCH_CHECK(sm_limit>=0,"sm_limit must be nonnegative");
 TORCH_CHECK(work>0&&work<=INT_MAX,"invalid launch work size");
 return int(sm_limit>0&&work>sm_limit?sm_limit:work);
}

// stage is a private, measurement-only split: 0=normal pipeline, 1=proposal,
// 2=ordinal, 3=prefix, 4=fused proposal+ordinal. A stage never prepares its
// inputs; the caller must first run stage 0 on the same state.
template<int TileTokens>
void launch_proposals(const Mask* masks,const Mask* exec_masks,int* keys,Mask* moves,
 int* ordinals,int* block_counts,int* block_prefix,int* counts,int S,
 int per_source,int offset,int step,int sm_limit,bool fuse,int stage,cudaStream_t stream) {
 const int blocks=(S+TileTokens-1)/TileTokens;
 if((stage==0&&fuse)||stage==4) {
  propose_ordinals_fused<TileTokens><<<limited_grid(blocks,sm_limit),256,0,stream>>>(masks,exec_masks,keys,moves,ordinals,block_counts,S,blocks,per_source,offset,step);
 } else {
  if(stage==0||stage==1)
   propose_kernel<<<limited_grid((int64_t(S)*kTokenLanes+255)/256,sm_limit),256,0,stream>>>(masks,exec_masks,keys,moves,S,per_source,offset,step);
  if(stage==0||stage==2)
   stable_ordinals<TileTokens><<<limited_grid(blocks,sm_limit),256,0,stream>>>(keys,ordinals,block_counts,S,blocks);
 }
 if(stage==0||stage==3) {
  if(sm_limit>0)
   scan_blocks_warp<<<limited_grid((kBuckets+kThreads/32-1)/(kThreads/32),sm_limit),kThreads,0,stream>>>(block_counts,block_prefix,counts,blocks);
  else
   scan_blocks<<<kBuckets,256,0,stream>>>(block_counts,block_prefix,counts,blocks);
 }
}

template<typename Kernel>
pybind11::dict resource_info(Kernel kernel) {
 cudaFuncAttributes attr{};
 C10_CUDA_CHECK(cudaFuncGetAttributes(&attr,kernel));
 int resident_blocks=0;
 C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident_blocks,kernel,kThreads,0));
 pybind11::dict result;
 result["numRegs"]=attr.numRegs;
 result["localSizeBytes"]=attr.localSizeBytes;
 result["sharedSizeBytes"]=attr.sharedSizeBytes;
 result["maxThreadsPerBlock"]=attr.maxThreadsPerBlock;
 result["maxActiveBlocksPerSM"]=resident_blocks;
 result["threadsPerBlock"]=kThreads;
 return result;
}

template<int TileTokens>
pybind11::dict tile_resources() {
 pybind11::dict result;
 result["tile_tokens"]=TileTokens;
 result["proposal"]=resource_info(propose_kernel);
 result["ordinal"]=resource_info(stable_ordinals<TileTokens>);
 result["fused"]=resource_info(propose_ordinals_fused<TileTokens>);
 result["prefix"]=resource_info(scan_blocks_warp);
 return result;
}

} // namespace

void prepare(torch::Tensor routes,torch::Tensor support,torch::Tensor resident,torch::Tensor exec_masks,int sm_limit) {
 CHECK_I32_CUDA(routes);CHECK_I32_CUDA(resident);CHECK_I32_CUDA(exec_masks);
 TORCH_CHECK(routes.dim()==2&&routes.size(0)>0&&routes.size(0)<=INT_MAX/kRanks&&routes.size(1)==kTopK,"routes must match the compiled [S,K]");
 TORCH_CHECK(support.is_cuda()&&support.scalar_type()==at::kBool&&support.is_contiguous()&&support.dim()==2&&support.size(0)==kRanks);
 TORCH_CHECK(support.size(1)>=8&&support.size(1)<=4096&&support.size(1)%kRanks==0,"E must be divisible by 8 and in [8,4096]");
 TORCH_CHECK(support.device()==routes.device()&&resident.device()==routes.device()&&exec_masks.device()==routes.device());
 TORCH_CHECK(resident.numel()==support.size(1)&&exec_masks.dim()==2&&exec_masks.size(0)==routes.size(0)&&exec_masks.size(1)==kRanks);
 c10::cuda::CUDAGuard guard(routes.device());auto s=c10::cuda::getCurrentCUDAStream();int E=support.size(1),S=routes.size(0);
 support_mask_kernel<<<limited_grid((E+255)/256,sm_limit),256,0,s>>>(support.data_ptr<bool>(),resident.data_ptr<int>(),E);
 cache_exec<<<limited_grid((int64_t(S)*kTokenLanes+255)/256,sm_limit),256,0,s>>>(routes.data_ptr<int>(),resident.data_ptr<int>(),reinterpret_cast<Mask*>(exec_masks.data_ptr<int>()),S);
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void reset(torch::Tensor seed,torch::Tensor owner,torch::Tensor masks,int sm_limit) {
 CHECK_I32_CUDA(seed);CHECK_I32_CUDA(owner);CHECK_I32_CUDA(masks);
 TORCH_CHECK(seed.dim()==2&&seed.size(0)>0&&seed.size(0)<=INT_MAX/kRanks&&seed.size(1)==kTopK&&seed.sizes()==owner.sizes());
 TORCH_CHECK(masks.dim()==2&&masks.size(0)==seed.size(0)&&masks.size(1)==kRanks,"masks must be [S,R], independent of K");
 TORCH_CHECK(seed.device()==owner.device()&&seed.device()==masks.device());
 c10::cuda::CUDAGuard guard(seed.device());auto s=c10::cuda::getCurrentCUDAStream();int S=seed.size(0);
 reset_masks<<<limited_grid((int64_t(S)*kTokenLanes+255)/256,sm_limit),256,0,s>>>(seed.data_ptr<int>(),owner.data_ptr<int>(),reinterpret_cast<Mask*>(masks.data_ptr<int>()),S);C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void propose(torch::Tensor masks,torch::Tensor exec_masks,torch::Tensor keys,torch::Tensor moves,
 torch::Tensor ordinals,torch::Tensor block_counts,torch::Tensor block_prefix,torch::Tensor counts,int per_source,int offset,int step,int sm_limit,bool fuse,int tile_tokens,int stage) {
 CHECK_I32_CUDA(masks);CHECK_I32_CUDA(exec_masks);CHECK_I32_CUDA(keys);CHECK_I32_CUDA(moves);CHECK_I32_CUDA(ordinals);
 CHECK_I32_CUDA(block_counts);CHECK_I32_CUDA(block_prefix);CHECK_I32_CUDA(counts);
 TORCH_CHECK(masks.dim()==2&&masks.size(0)>0&&masks.size(0)<=INT_MAX/kRanks&&masks.size(1)==kRanks&&masks.sizes()==exec_masks.sizes());
 TORCH_CHECK(tile_tokens==64||tile_tokens==128||tile_tokens==256,"histogram tile must be 64, 128 or 256");
 TORCH_CHECK(sm_limit>0||tile_tokens==256,"private histogram tiling requires a positive sm_limit");
 TORCH_CHECK(stage>=0&&stage<=4,"private proposal stage must be 0..4");
 TORCH_CHECK(stage!=4||fuse,"fused measurement stage requires fusion");
 c10::cuda::CUDAGuard guard(masks.device());auto s=c10::cuda::getCurrentCUDAStream();int S=masks.size(0),blocks=(S+tile_tokens-1)/tile_tokens;
 TORCH_CHECK(per_source==S&&offset>=0&&offset<=int64_t(kRanks-1)*S&&offset%S==0&&step>=0);
 TORCH_CHECK(keys.numel()==S&&moves.numel()==S&&ordinals.numel()==S&&counts.numel()==kBuckets);
 TORCH_CHECK(block_counts.numel()==int64_t(kBuckets)*blocks&&block_prefix.sizes()==block_counts.sizes());
 for(const auto& t:{exec_masks,keys,moves,ordinals,block_counts,block_prefix,counts})TORCH_CHECK(t.device()==masks.device());
 TORCH_CHECK(!fuse||sm_limit>0,"fusion requires a positive sm_limit");
 #define LAUNCH_PROPOSALS(TILE) launch_proposals<TILE>(reinterpret_cast<const Mask*>(masks.data_ptr<int>()),reinterpret_cast<const Mask*>(exec_masks.data_ptr<int>()),keys.data_ptr<int>(),reinterpret_cast<Mask*>(moves.data_ptr<int>()),ordinals.data_ptr<int>(),block_counts.data_ptr<int>(),block_prefix.data_ptr<int>(),counts.data_ptr<int>(),S,per_source,offset,step,sm_limit,fuse,stage,s)
 if(tile_tokens==64) { LAUNCH_PROPOSALS(64); }
 else if(tile_tokens==128) { LAUNCH_PROPOSALS(128); }
 else { LAUNCH_PROPOSALS(256); }
 #undef LAUNCH_PROPOSALS
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void quota(torch::Tensor gathered,torch::Tensor total,torch::Tensor prefix,torch::Tensor accept,torch::Tensor stats,int source,int step,int sm_limit,bool fuse) {
 CHECK_I32_CUDA(gathered);CHECK_I32_CUDA(total);CHECK_I32_CUDA(prefix);CHECK_I32_CUDA(accept);
 TORCH_CHECK(stats.is_cuda()&&stats.is_contiguous()&&stats.scalar_type()==at::kLong&&stats.dim()==2&&stats.size(1)==4&&step>=0&&step<stats.size(0));
 TORCH_CHECK(total.numel()==kBuckets&&prefix.numel()==kBuckets&&accept.numel()==kBuckets);
 for(const auto& t:{total,prefix,accept,stats})TORCH_CHECK(t.device()==gathered.device());
 c10::cuda::CUDAGuard guard(gathered.device());auto s=c10::cuda::getCurrentCUDAStream();
 TORCH_CHECK(gathered.numel()==kRanks*kBuckets&&source>=0&&source<kRanks);
 auto st=reinterpret_cast<unsigned long long*>(stats.data_ptr<int64_t>())+step*4;
 TORCH_CHECK(sm_limit>=0&&(!fuse||sm_limit>0),"fusion requires a positive sm_limit");
 if(fuse) {
  fold_quota_fused<<<1,256,0,s>>>(gathered.data_ptr<int>(),total.data_ptr<int>(),prefix.data_ptr<int>(),accept.data_ptr<int>(),st,source);
 } else {
  if(sm_limit>0)zero_quota<<<limited_grid((kBuckets+255)/256,sm_limit),256,0,s>>>(accept.data_ptr<int>(),st);
  else {
   C10_CUDA_CHECK(cudaMemsetAsync(st,0,4*sizeof(int64_t),s));
   C10_CUDA_CHECK(cudaMemsetAsync(accept.data_ptr<int>(),0,kBuckets*sizeof(int),s));
  }
  fold_counts_kernel<<<limited_grid((kBuckets+255)/256,sm_limit),256,0,s>>>(gathered.data_ptr<int>(),total.data_ptr<int>(),prefix.data_ptr<int>(),source,kRanks);
  quota_kernel<<<limited_grid((kRanks*kRanks*kTopK+255)/256,sm_limit),256,0,s>>>(total.data_ptr<int>(),accept.data_ptr<int>(),st);
 }
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void apply(torch::Tensor keys,torch::Tensor moves,torch::Tensor ordinals,torch::Tensor block_prefix,
 torch::Tensor prefix,torch::Tensor accept,torch::Tensor masks,torch::Tensor owner,int sm_limit,int tile_tokens) {
 CHECK_I32_CUDA(keys);CHECK_I32_CUDA(moves);CHECK_I32_CUDA(ordinals);CHECK_I32_CUDA(block_prefix);
 CHECK_I32_CUDA(prefix);CHECK_I32_CUDA(accept);CHECK_I32_CUDA(masks);CHECK_I32_CUDA(owner);
 TORCH_CHECK(owner.dim()==2&&owner.size(0)>0&&owner.size(0)<=INT_MAX/kRanks&&owner.size(1)==kTopK);
 TORCH_CHECK(masks.dim()==2&&masks.size(0)==owner.size(0)&&masks.size(1)==kRanks);
 c10::cuda::CUDAGuard guard(owner.device());auto s=c10::cuda::getCurrentCUDAStream();int S=owner.size(0);
 TORCH_CHECK(tile_tokens==64||tile_tokens==128||tile_tokens==256,"histogram tile must be 64, 128 or 256");
 TORCH_CHECK(sm_limit>0||tile_tokens==256,"private histogram tiling requires a positive sm_limit");
 const int blocks=(S+tile_tokens-1)/tile_tokens;
 TORCH_CHECK(keys.numel()==S&&moves.numel()==S&&ordinals.numel()==S&&prefix.numel()==kBuckets&&accept.numel()==kBuckets&&block_prefix.numel()==int64_t(kBuckets)*blocks);
 for(const auto& t:{keys,moves,ordinals,block_prefix,prefix,accept,masks})TORCH_CHECK(t.device()==owner.device());
 #define LAUNCH_APPLY(TILE) apply_prefix<TILE><<<limited_grid((S+255)/256,sm_limit),256,0,s>>>(keys.data_ptr<int>(),reinterpret_cast<const Mask*>(moves.data_ptr<int>()),ordinals.data_ptr<int>(),block_prefix.data_ptr<int>(),prefix.data_ptr<int>(),accept.data_ptr<int>(),reinterpret_cast<Mask*>(masks.data_ptr<int>()),owner.data_ptr<int>(),S,blocks)
 if(tile_tokens==64) { LAUNCH_APPLY(64); }
 else if(tile_tokens==128) { LAUNCH_APPLY(128); }
 else { LAUNCH_APPLY(256); }
 #undef LAUNCH_APPLY
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
pybind11::dict kernel_resources(int tile_tokens) {
 TORCH_CHECK(tile_tokens==64||tile_tokens==128||tile_tokens==256,"histogram tile must be 64, 128 or 256");
 if(tile_tokens==64)return tile_resources<64>();
 if(tile_tokens==128)return tile_resources<128>();
 return tile_resources<256>();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
 namespace py=pybind11;
 m.def("prepare",&prepare,py::arg("routes"),py::arg("support"),py::arg("resident"),py::arg("exec_masks"),py::arg("sm_limit")=0);
 m.def("reset",&reset,py::arg("seed"),py::arg("owner"),py::arg("masks"),py::arg("sm_limit")=0);
 m.def("propose",&propose,py::arg("masks"),py::arg("exec_masks"),py::arg("keys"),py::arg("moves"),py::arg("ordinals"),py::arg("block_counts"),py::arg("block_prefix"),py::arg("counts"),py::arg("per_source"),py::arg("offset"),py::arg("step"),py::arg("sm_limit")=0,py::arg("fuse")=false,py::arg("tile_tokens")=256,py::arg("stage")=0);
 m.def("quota",&quota,py::arg("gathered"),py::arg("total"),py::arg("prefix"),py::arg("accept"),py::arg("stats"),py::arg("source"),py::arg("step"),py::arg("sm_limit")=0,py::arg("fuse")=false);
 m.def("apply",&apply,py::arg("keys"),py::arg("moves"),py::arg("ordinals"),py::arg("block_prefix"),py::arg("prefix"),py::arg("accept"),py::arg("masks"),py::arg("owner"),py::arg("sm_limit")=0,py::arg("tile_tokens")=256);
 m.def("kernel_resources",&kernel_resources,py::arg("tile_tokens")=256);
}
