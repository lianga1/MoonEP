// This parent's plan ABI: signed dst, copies, cu, zero-fill, remote stats.
// Never publish a candidate here: Python/caller makes the global decision first.
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cub/block/block_scan.cuh>
#include <cub/block/block_reduce.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cuda_runtime.h>
#include <algorithm>

using at::Tensor;
#define PTR(t) (t).data_ptr<int>()
#define CHECK(t) TORCH_CHECK((t).is_cuda() && (t).is_contiguous() && (t).scalar_type()==at::kInt, #t " must be contiguous CUDA int32")

__global__ void decode_kernel(const int* raw, int* owner, int N, int nvs) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<N) { long long v=raw[i]; owner[i]=int((v<0?-v-1:v)/nvs); }
}

template<class Support>
__global__ void support_kernel(const Support* support,int* lut,int* copies,int* status,int R,int E,int B) {
  using Scan=cub::BlockScan<int,512>;
  __shared__ typename Scan::TempStorage temp;
  __shared__ int carry;
  int r=blockIdx.x,tid=threadIdx.x,epn=E/R;
  if(tid==0)carry=0;
  for(int b=tid;b<B;b+=512)copies[r*B+b]=-1;
  __syncthreads();
  for(int base=0;base<E;base+=512) {
    int e=base+tid,value=e<E?int(support[r*E+e]):0;
    bool home=e<E&&e/epn==r;
    if(e<E&&((value!=0&&value!=1)||(home&&!value)))atomicOr(status,1);
    int active=e<E&&!home&&value==1,local=0,total=0;
    Scan(temp).ExclusiveSum(active,local,total);
    int slot=carry+local;
    if(e<E) {
      lut[r*E+e]=home&&value==1?e:(active&&slot<B?E+slot:-1);
      if(active&&slot<B)copies[r*B+slot]=e;
    }
    // Every lane must finish using the previous carry and scan storage.
    __syncthreads();
    if(tid==0)carry+=total;
    __syncthreads();
  }
  if(tid==0&&carry>B)atomicOr(status,2);
}

// Fixed-size subwarps permit non-power-of-two K without crossing token groups.
// All 32 lanes execute both match_any calls, including padded/inactive lanes.
// Peers are restricted to this token's subwarp and to earlier top-k lanes.
__global__ void token_metadata_kernel(const int* routes,const int* owner,unsigned char* duplicates,
    int* status,int S,int K,int width) {
  int global=blockIdx.x*blockDim.x+threadIdx.x,lane=threadIdx.x&31;
  int token=global/width,k=lane&(width-1),group_start=lane-k;
  bool valid=token<S&&k<K;
  int p=token*K+k;
  int e=valid?routes[p]:-1-lane,d=valid?owner[p]:-1-lane;
  unsigned group=(0xffffffffu>>(32-width))<<group_start;
  unsigned earlier=(1u<<lane)-1u;
  unsigned same_expert=__match_any_sync(0xffffffffu,e)&group;
  unsigned same_dest=__match_any_sync(0xffffffffu,d)&group;
  if(valid) {
    if(same_expert&earlier)atomicOr(status,8);
    duplicates[p]=(same_dest&earlier)?1:0;
  }
}

__global__ void input_kernel(const int* routes,const int* owner,const int* seed,const int* order,
    const int* lut,int* keys,int* ids,int* counts,int* status,int N,int R,int E,int K,bool reuse) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=N)return;
  int p=reuse?order[i]:i;
  if(p<0||p>=N) {atomicOr(status,128);p=i;}
  int e=routes[p],d=owner[p],s=seed[p];
  bool valid_e=e>=0&&e<E,valid_d=d>=0&&d<R;
  if(!valid_e) atomicOr(status,8);
  if(!valid_d) atomicOr(status,16);
  if(valid_e) {
    if(s<0||s>=R||lut[s*E+e]<0) atomicOr(status,4);
    if(valid_d&&lut[d*E+e]<0) atomicOr(status,16);
  }
  if(reuse&&i>0) {
    int prev=order[i-1];
    if(prev<0||prev>=N||routes[prev]>e) atomicOr(status,128);
  }
  // Native C1 is trusted to be a permutation; range and sortedness are checked.
  // Per-token duplicate checks are performed once by token_metadata_kernel.
  bool valid=valid_e&&valid_d;
  keys[i]=valid?(reuse?d:d*E+e):0;
  ids[i]=p;
  if(valid) atomicAdd(counts+d*E+e,1);
}

__global__ void starts_kernel(const int* routes,const int* owner,const int* ids,
    int* starts,int* status,int N,int R,int E) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=N||*status)return;
  int p=ids[i],e=routes[p],d=owner[p];
  if(e<0||e>=E||d<0||d>=R)return;
  int key=d*E+e,previous=-1;
  if(i>0) {int q=ids[i-1],pe=routes[q],pd=owner[q];if(pe>=0&&pe<E&&pd>=0&&pd<R)previous=pd*E+pe;}
  if(key!=previous) starts[key]=i;
  if(previous>key) atomicOr(status,128);
}

__global__ void layout_kernel(const int* gathered,const int* copies,int* offsets,int* prefix,int* totals,
    int* cu,int* zeros,int* status,int R,int E,int B,int source,int padding,int nvs,int N) {
  using Scan=cub::BlockScan<long long,512>;
  __shared__ typename Scan::TempStorage temp;
  __shared__ long long carry_pad,carry_tasks;
  int dest=blockIdx.x,tid=threadIdx.x,Q=E+B,epn=E/R;
  if(tid==0){carry_pad=0;carry_tasks=0;}
  __syncthreads();
  for(int base=0;base<Q;base+=512) {
    int g=base+tid;
    int e=g<E?(g/epn==dest?g:-1):(g<Q?copies[dest*B+g-E]:-1);
    long long count=0,prior=0;
    if(e>=0)for(int s=0;s<R;s++) {
      int n=gathered[(s*R+dest)*E+e];count+=n;if(s<source)prior+=n;
    }
    long long padded=count?((count+padding-1)/padding)*padding:0,local=0,tilepad=0;
    Scan(temp).ExclusiveSum(padded,local,tilepad);
    long long offset=carry_pad+local;
    bool fits=offset>=0&&offset+padded<=nvs;
    if(e>=0) {
      offsets[dest*E+e]=fits?int(offset):0;
      prefix[dest*E+e]=int(prior);totals[dest*E+e]=int(count);
    }
    if(dest==source&&g<Q) {
      cu[g]=fits?int(offset+padded):0;
      zeros[g*2]=fits&&padded>count?int(offset+count):0;
      zeros[g*2+1]=fits?int(padded-count):0;
    }
    __syncthreads();
    long long ignored=0,tilecount=0;
    Scan(temp).ExclusiveSum(count,ignored,tilecount);
    __syncthreads();
    if(tid==0){carry_pad+=tilepad;carry_tasks+=tilecount;}
    __syncthreads();
  }
  if(tid==0) {
    if(carry_pad>nvs)atomicOr(status,64);
    if(carry_tasks!=N)atomicOr(status,32);
  }
}

__global__ void encode_kernel(const int* routes,const int* owner,const int* ids,const int* starts,
    const int* lut,const int* offsets,const int* prefix,const int* totals,const unsigned char* duplicates,int* dst,int* status,
    int N,int R,int E,int K,int nvs) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=N||*status)return;
  int p=ids[i],e=routes[p],d=owner[p];
  if(e<0||e>=E||d<0||d>=R) return;
  int key=d*E+e;
  if(lut[key]<0)return;
  int ordinal=i-starts[key],within=prefix[key]+ordinal,row=offsets[key]+within;
  if(ordinal<0||within<0||within>=totals[key]||row<0||row>=nvs) {atomicOr(status,64);return;}
  int v=d*nvs+row;dst[p]=duplicates[p]?-v-1:v;
}

__global__ void stats_kernel(const int* totals,const int* copies,int* stats,int R,int E,int B,int rank) {
  using Reduce=cub::BlockReduce<int,256>;
  __shared__ typename Reduce::TempStorage temp;
  int tid=threadIdx.x;
  int incoming=0,outgoing=0,epn=E/R;
  for(int e=tid;e<E;e+=256) if(e/epn!=rank&&totals[rank*E+e]>0)incoming++;
  // Native stats[1] counts planned prefetches from this home rank. A chosen
  // but presently idle copy still incurs a prefetch and therefore counts.
  for(int q=tid;q<R*B;q+=256) {int e=copies[q];if(e>=0&&e/epn==rank)outgoing++;}
  int totalin=Reduce(temp).Sum(incoming);
  if(tid==0)stats[0]=totalin;
  __syncthreads();
  int totalout=Reduce(temp).Sum(outgoing);
  if(tid==0)stats[1]=totalout;
}

__global__ void publish_kernel(const int* dst,int* meta,int N,int rank,int nvs,int stride,int src_info) {
  int p=blockIdx.x*blockDim.x+threadIdx.x;
  if(p<N) {
    long long raw=dst[p],v=raw<0?-raw-1:raw;
    int dest=int(v/nvs),row=int(v%nvs);
    meta[static_cast<long long>(dest)*stride+src_info+row]=rank*nvs+p;
    __threadfence_system();
  }
}

size_t workspace(Tensor kin,Tensor kout,Tensor vin,Tensor vout,int bits) {
  CHECK(kin);CHECK(kout);CHECK(vin);CHECK(vout);
  c10::cuda::CUDAGuard guard(kin.device());size_t bytes=0;
  C10_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(nullptr,bytes,PTR(kin),PTR(kout),PTR(vin),PTR(vout),kin.numel(),0,bits,c10::cuda::getCurrentCUDAStream()));
  return bytes;
}

void decode(Tensor raw,Tensor owner,int nvs) {
  CHECK(raw);CHECK(owner);c10::cuda::CUDAGuard guard(raw.device());
  decode_kernel<<<(raw.numel()+255)/256,256,0,c10::cuda::getCurrentCUDAStream()>>>(PTR(raw),PTR(owner),raw.numel(),nvs);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void support(Tensor input,Tensor lut,Tensor copies,Tensor status,int R,int E,int B) {
  TORCH_CHECK(input.is_cuda()&&input.is_contiguous()&&(input.scalar_type()==at::kBool||input.scalar_type()==at::kInt));
  CHECK(lut);CHECK(copies);CHECK(status);c10::cuda::CUDAGuard guard(input.device());auto stream=c10::cuda::getCurrentCUDAStream();
  C10_CUDA_CHECK(cudaMemsetAsync(PTR(status),0,sizeof(int),stream));
  if(input.scalar_type()==at::kBool) support_kernel<<<R,512,0,stream>>>(input.data_ptr<bool>(),PTR(lut),PTR(copies),PTR(status),R,E,B);
  else support_kernel<<<R,512,0,stream>>>(PTR(input),PTR(lut),PTR(copies),PTR(status),R,E,B);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void prepare(Tensor routes,Tensor owner,Tensor seed,Tensor order,Tensor lut,Tensor kin,Tensor kout,
    Tensor vin,Tensor vout,Tensor starts,Tensor counts,Tensor tmp,Tensor duplicates,Tensor status,int R,int E,int K,bool reuse,int bits) {
  CHECK(routes);CHECK(owner);CHECK(seed);CHECK(order);CHECK(lut);
  TORCH_CHECK(duplicates.is_cuda()&&duplicates.is_contiguous()&&duplicates.scalar_type()==at::kByte&&duplicates.numel()==routes.numel());
  c10::cuda::CUDAGuard guard(routes.device());auto stream=c10::cuda::getCurrentCUDAStream();int N=routes.numel();
  C10_CUDA_CHECK(cudaMemsetAsync(PTR(counts),0,R*E*sizeof(int),stream));
  C10_CUDA_CHECK(cudaMemsetAsync(PTR(starts),0,R*E*sizeof(int),stream));
  int width=1;while(width<K)width*=2;
  token_metadata_kernel<<<((N/K)*width+255)/256,256,0,stream>>>(PTR(routes),PTR(owner),duplicates.data_ptr<unsigned char>(),PTR(status),N/K,K,width);
  input_kernel<<<(N+255)/256,256,0,stream>>>(PTR(routes),PTR(owner),PTR(seed),PTR(order),PTR(lut),PTR(kin),PTR(vin),PTR(counts),PTR(status),N,R,E,K,reuse);
  size_t bytes=tmp.numel();
  C10_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(tmp.data_ptr(),bytes,PTR(kin),PTR(kout),PTR(vin),PTR(vout),N,0,bits,stream));
  starts_kernel<<<(N+255)/256,256,0,stream>>>(PTR(routes),PTR(owner),PTR(vout),PTR(starts),PTR(status),N,R,E);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void layout_encode(Tensor routes,Tensor owner,Tensor ids,Tensor starts,Tensor gathered,Tensor lut,
    Tensor copies,Tensor offsets,Tensor prefix,Tensor totals,Tensor duplicates,Tensor dst,Tensor cu,Tensor zeros,
    Tensor stats,Tensor status,int R,int E,int B,int K,int rank,int padding,int nvs) {
  CHECK(routes);CHECK(owner);c10::cuda::CUDAGuard guard(routes.device());auto stream=c10::cuda::getCurrentCUDAStream();int N=routes.numel();
  C10_CUDA_CHECK(cudaMemsetAsync(PTR(offsets),0,R*E*sizeof(int),stream));
  C10_CUDA_CHECK(cudaMemsetAsync(PTR(prefix),0,R*E*sizeof(int),stream));
  C10_CUDA_CHECK(cudaMemsetAsync(PTR(totals),0,R*E*sizeof(int),stream));
  C10_CUDA_CHECK(cudaMemsetAsync(PTR(dst),0,N*sizeof(int),stream));
  layout_kernel<<<R,512,0,stream>>>(PTR(gathered),PTR(copies),PTR(offsets),PTR(prefix),PTR(totals),PTR(cu),PTR(zeros),PTR(status),R,E,B,rank,padding,nvs,N);
  encode_kernel<<<(N+255)/256,256,0,stream>>>(PTR(routes),PTR(owner),PTR(ids),PTR(starts),PTR(lut),PTR(offsets),PTR(prefix),PTR(totals),duplicates.data_ptr<unsigned char>(),PTR(dst),PTR(status),N,R,E,K,nvs);
  stats_kernel<<<1,256,0,stream>>>(PTR(totals),PTR(copies),PTR(stats),R,E,B,rank);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void publish(Tensor dst,Tensor meta,int rank,int nvs,int stride,int src_info) {
  CHECK(dst);CHECK(meta);c10::cuda::CUDAGuard guard(dst.device());
  publish_kernel<<<(dst.numel()+255)/256,256,0,c10::cuda::getCurrentCUDAStream()>>>(PTR(dst),PTR(meta),dst.numel(),rank,nvs,stride,src_info);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// The bounded path uses only kernels launched with at most `limit` CTAs.
// Virtual tile IDs, rather than physical CTA IDs or completion order, determine
// all stable ordering. It intentionally avoids DeviceRadixSort and CUDA memset
// because their implementation kernels are outside our CTA-budget contract.
namespace bounded_materialize {
constexpr int THREADS=256,RANKS=8;

int grid(int64_t items,int limit,int threads=THREADS) {
  TORCH_CHECK(limit>0,"bounded materialization needs a positive CTA limit");
  return int(std::min<int64_t>(limit,std::max<int64_t>(1,(items+threads-1)/threads)));
}

__global__ void fill(int* out,int64_t n,int value) {
  for(int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=int64_t(gridDim.x)*blockDim.x)
    out[i]=value;
}

__global__ void clear_prepare(int* counts,int* starts,int n) {
  for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=gridDim.x*blockDim.x)
    counts[i]=starts[i]=0;
}

__global__ void clear_layout(int* offsets,int* prefix,int* totals,int* dst,int entries,int N) {
  for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<entries;i+=gridDim.x*blockDim.x)
    offsets[i]=prefix[i]=totals[i]=0;
  for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<N;i+=gridDim.x*blockDim.x)dst[i]=0;
}

__global__ void decode(const int* raw,int* owner,int N,int nvs) {
  for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<N;i+=gridDim.x*blockDim.x) {
    long long value=raw[i];owner[i]=int((value<0?-value-1:value)/nvs);
  }
}

template<class Support>
__global__ void support(const Support* values,int* lut,int* copies,int* status,int R,int E,int B) {
  using Scan=cub::BlockScan<int,512>;
  __shared__ typename Scan::TempStorage temp;
  __shared__ int carry;
  int tid=threadIdx.x,epn=E/R;
  for(int r=blockIdx.x;r<R;r+=gridDim.x) {
    if(tid==0)carry=0;
    for(int b=tid;b<B;b+=512)copies[r*B+b]=-1;
    __syncthreads();
    for(int base=0;base<E;base+=512) {
      int e=base+tid,value=e<E?int(values[r*E+e]):0;
      bool home=e<E&&e/epn==r;
      if(e<E&&((value!=0&&value!=1)||(home&&!value)))atomicOr(status,1);
      int active=e<E&&!home&&value==1,local=0,total=0;
      Scan(temp).ExclusiveSum(active,local,total);
      int slot=carry+local;
      if(e<E) {
        lut[r*E+e]=home&&value==1?e:(active&&slot<B?E+slot:-1);
        if(active&&slot<B)copies[r*B+slot]=e;
      }
      __syncthreads();
      if(tid==0)carry+=total;
      __syncthreads();
    }
    if(tid==0&&carry>B)atomicOr(status,2);
    __syncthreads();
  }
}

__device__ __forceinline__ void metadata_item(int global,const int* routes,const int* owner,
    unsigned char* duplicates,int* status,int S,int K,int width) {
  int lane=threadIdx.x&31,token=global/width,k=lane&(width-1),group_start=lane-k;
  bool valid=token<S&&k<K;
  int p=token*K+k,e=valid?routes[p]:-1-lane,d=valid?owner[p]:-1-lane;
  unsigned group=(0xffffffffu>>(32-width))<<group_start,earlier=(1u<<lane)-1u;
  unsigned same_expert=__match_any_sync(0xffffffffu,e)&group;
  unsigned same_dest=__match_any_sync(0xffffffffu,d)&group;
  if(valid) {
    if(same_expert&earlier)atomicOr(status,8);
    duplicates[p]=(same_dest&earlier)?1:0;
  }
}

// Return the sanitized eight-way sort key. Invalid entries still receive a
// safe key/id so all ranks can finish the collective/fallback protocol.
__device__ __forceinline__ int input_item(int i,const int* routes,const int* owner,const int* seed,
    const int* order,const int* lut,int* keys,int* ids,int* counts,int* status,int N,int E) {
  if(i>=N)return RANKS;
  int p=order[i];
  if(p<0||p>=N){atomicOr(status,128);p=i;}
  int e=routes[p],d=owner[p],s=seed[p];
  bool valid_e=e>=0&&e<E,valid_d=d>=0&&d<RANKS;
  if(!valid_e)atomicOr(status,8);
  if(!valid_d)atomicOr(status,16);
  if(valid_e) {
    if(s<0||s>=RANKS||lut[s*E+e]<0)atomicOr(status,4);
    if(valid_d&&lut[d*E+e]<0)atomicOr(status,16);
  }
  if(i>0) {
    int previous=order[i-1];
    if(previous<0||previous>=N||routes[previous]>e)atomicOr(status,128);
  }
  bool valid=valid_e&&valid_d;
  int key=valid?d:0;
  keys[i]=key;ids[i]=p;
  if(valid)atomicAdd(counts+d*E+e,1);
  return key;
}

// Exactly one warp leader writes each (warp,bucket), and every lane named in
// match_any participates even in the partial final tile. Return the stable
// ordinal within this 256-pair tile, independent of the physical CTA budget.
__device__ __forceinline__ int tile_histogram(int key,int* warp_counts) {
  int tid=threadIdx.x,lane=tid&31,warp=tid>>5;
  if(tid<64)warp_counts[tid]=0;
  __syncthreads();
  unsigned peers=__match_any_sync(0xffffffffu,key);
  if(key<RANKS&&lane==__ffs(peers)-1)warp_counts[warp*RANKS+key]=__popc(peers);
  __syncthreads();
  int ordinal=__popc(peers&((1u<<lane)-1u));
  if(key<RANKS)for(int w=0;w<warp;++w)ordinal+=warp_counts[w*RANKS+key];
  return ordinal;
}

__global__ void metadata(const int* routes,const int* owner,unsigned char* duplicates,
    int* status,int S,int K,int width) {
  int tiles=(S*width+THREADS-1)/THREADS;
  for(int tile=blockIdx.x;tile<tiles;tile+=gridDim.x)
    metadata_item(tile*THREADS+threadIdx.x,routes,owner,duplicates,status,S,K,width);
}

__global__ void input(const int* routes,const int* owner,const int* seed,const int* order,const int* lut,
    int* keys,int* ids,int* counts,int* status,int N,int E) {
  for(int i=blockIdx.x*THREADS+threadIdx.x;i<N;i+=gridDim.x*THREADS)
    input_item(i,routes,owner,seed,order,lut,keys,ids,counts,status,N,E);
}

__global__ void histogram(const int* keys,int* hist,int N,int tiles) {
  __shared__ int warp_counts[64];
  for(int tile=blockIdx.x;tile<tiles;tile+=gridDim.x) {
    int i=tile*THREADS+threadIdx.x,key=i<N?keys[i]:RANKS;
    tile_histogram(key,warp_counts);
    if(threadIdx.x<RANKS) {
      int total=0;for(int w=0;w<8;++w)total+=warp_counts[w*RANKS+threadIdx.x];
      hist[threadIdx.x*tiles+tile]=total;
    }
    __syncthreads();
  }
}

// Segment fusion: token metadata, input validation/expert counts and the
// eight-way tile histogram share one launch. They write disjoint arrays and
// only read immutable inputs; no inter-CTA barrier is necessary.
__global__ void prepare_fused(const int* routes,const int* owner,const int* seed,const int* order,
    const int* lut,int* keys,int* ids,int* counts,int* hist,unsigned char* duplicates,
    int* status,int N,int E,int K,int width,int sort_tiles) {
  __shared__ int warp_counts[64];
  int all_tiles=((N/K)*width+THREADS-1)/THREADS;
  for(int tile=blockIdx.x;tile<all_tiles;tile+=gridDim.x) {
    int i=tile*THREADS+threadIdx.x;
    metadata_item(i,routes,owner,duplicates,status,N/K,K,width);
    int key=input_item(i,routes,owner,seed,order,lut,keys,ids,counts,status,N,E);
    tile_histogram(key,warp_counts);
    if(tile<sort_tiles&&threadIdx.x<RANKS) {
      int total=0;for(int w=0;w<8;++w)total+=warp_counts[w*RANKS+threadIdx.x];
      hist[threadIdx.x*sort_tiles+tile]=total;
    }
    __syncthreads();
  }
}

__global__ void histogram_prefix(const int* hist,int* prefix,int tiles) {
  using Scan=cub::BlockScan<int,THREADS>;
  __shared__ typename Scan::TempStorage temp;
  __shared__ int carry;
  if(threadIdx.x==0)carry=0;
  __syncthreads();
  // Processing buckets in rank order gives the global bucket base and the
  // within-bucket tile prefix in one scan. Tile indices remain monotonically
  // ordered regardless of which CTA produced each histogram.
  for(int rank=0;rank<RANKS;++rank)for(int base=0;base<tiles;base+=THREADS) {
    int tile=base+threadIdx.x,value=tile<tiles?hist[rank*tiles+tile]:0;
    int local=0,total=0;
    Scan(temp).ExclusiveSum(value,local,total);
    if(tile<tiles)prefix[rank*tiles+tile]=carry+local;
    __syncthreads();
    if(threadIdx.x==0)carry+=total;
    __syncthreads();
  }
}

__global__ void scatter(const int* keys,const int* ids,const int* prefix,
    int* out_keys,int* out_ids,int N,int tiles) {
  __shared__ int warp_counts[64];
  for(int tile=blockIdx.x;tile<tiles;tile+=gridDim.x) {
    int i=tile*THREADS+threadIdx.x,key=i<N?keys[i]:RANKS;
    int ordinal=tile_histogram(key,warp_counts);
    if(i<N) {
      int position=prefix[key*tiles+tile]+ordinal;
      out_keys[position]=key;out_ids[position]=ids[i];
    }
    __syncthreads();
  }
}

__global__ void starts(const int* routes,const int* owner,const int* ids,int* starts,int* status,int N,int E) {
  for(int i=blockIdx.x*THREADS+threadIdx.x;i<N;i+=gridDim.x*THREADS) {
    if(*status)continue;
    int p=ids[i],e=routes[p],d=owner[p];
    if(e<0||e>=E||d<0||d>=RANKS)continue;
    int key=d*E+e,previous=-1;
    if(i>0){int q=ids[i-1],pe=routes[q],pd=owner[q];if(pe>=0&&pe<E&&pd>=0&&pd<RANKS)previous=pd*E+pe;}
    if(key!=previous)starts[key]=i;
    if(previous>key)atomicOr(status,128);
  }
}

__global__ void layout(const int* gathered,const int* copies,int* offsets,int* prefix,int* totals,
    int* cu,int* zeros,int* status,int R,int E,int B,int source,int padding,int nvs,int N) {
  using Scan=cub::BlockScan<long long,512>;
  __shared__ typename Scan::TempStorage temp;
  __shared__ long long carry_pad,carry_tasks;
  int tid=threadIdx.x,Q=E+B,epn=E/R;
  for(int dest=blockIdx.x;dest<R;dest+=gridDim.x) {
    if(tid==0){carry_pad=0;carry_tasks=0;}
    __syncthreads();
    for(int base=0;base<Q;base+=512) {
      int g=base+tid,e=g<E?(g/epn==dest?g:-1):(g<Q?copies[dest*B+g-E]:-1);
      long long count=0,prior=0;
      if(e>=0)for(int s=0;s<R;++s){int n=gathered[(s*R+dest)*E+e];count+=n;if(s<source)prior+=n;}
      long long padded=count?((count+padding-1)/padding)*padding:0,local=0,tilepad=0;
      Scan(temp).ExclusiveSum(padded,local,tilepad);
      long long offset=carry_pad+local;bool fits=offset>=0&&offset+padded<=nvs;
      if(e>=0){offsets[dest*E+e]=fits?int(offset):0;prefix[dest*E+e]=int(prior);totals[dest*E+e]=int(count);}
      if(dest==source&&g<Q) {
        cu[g]=fits?int(offset+padded):0;
        zeros[g*2]=fits&&padded>count?int(offset+count):0;zeros[g*2+1]=fits?int(padded-count):0;
      }
      __syncthreads();
      long long ignored=0,tilecount=0;Scan(temp).ExclusiveSum(count,ignored,tilecount);
      __syncthreads();
      if(tid==0){carry_pad+=tilepad;carry_tasks+=tilecount;}
      __syncthreads();
    }
    if(tid==0){if(carry_pad>nvs)atomicOr(status,64);if(carry_tasks!=N)atomicOr(status,32);}
    __syncthreads();
  }
}

__global__ void encode(const int* routes,const int* owner,const int* ids,const int* starts,
    const int* lut,const int* offsets,const int* prefix,const int* totals,const unsigned char* duplicates,
    int* dst,int* status,int N,int E,int nvs) {
  for(int i=blockIdx.x*THREADS+threadIdx.x;i<N;i+=gridDim.x*THREADS) {
    if(*status)continue;
    int p=ids[i],e=routes[p],d=owner[p];if(e<0||e>=E||d<0||d>=RANKS)continue;
    int key=d*E+e;if(lut[key]<0)continue;
    int ordinal=i-starts[key],within=prefix[key]+ordinal,row=offsets[key]+within;
    if(ordinal<0||within<0||within>=totals[key]||row<0||row>=nvs){atomicOr(status,64);continue;}
    int value=d*nvs+row;dst[p]=duplicates[p]?-value-1:value;
  }
}

__global__ void publish(const int* dst,int* meta,int N,int rank,int nvs,int stride,int src_info) {
  for(int p=blockIdx.x*THREADS+threadIdx.x;p<N;p+=gridDim.x*THREADS) {
    long long raw=dst[p],value=raw<0?-raw-1:raw;
    int dest=int(value/nvs),row=int(value%nvs);
    meta[int64_t(dest)*stride+src_info+row]=rank*nvs+p;
  }
  __threadfence_system();
}
} // namespace bounded_materialize

void fill_bounded(Tensor out,int64_t start,int64_t count,int value,int limit) {
  CHECK(out);TORCH_CHECK(start>=0&&count>=0&&start<=out.numel()&&count<=out.numel()-start);
  c10::cuda::CUDAGuard guard(out.device());auto stream=c10::cuda::getCurrentCUDAStream();
  bounded_materialize::fill<<<bounded_materialize::grid(count,limit),256,0,stream>>>(PTR(out)+start,count,value);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void decode_bounded(Tensor raw,Tensor owner,int nvs,int limit) {
  CHECK(raw);CHECK(owner);TORCH_CHECK(nvs>0&&raw.numel()==owner.numel());
  c10::cuda::CUDAGuard guard(raw.device());auto stream=c10::cuda::getCurrentCUDAStream();
  bounded_materialize::decode<<<bounded_materialize::grid(raw.numel(),limit),256,0,stream>>>(PTR(raw),PTR(owner),raw.numel(),nvs);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void support_bounded(Tensor input,Tensor lut,Tensor copies,Tensor status,int R,int E,int B,int limit) {
  TORCH_CHECK(input.is_cuda()&&input.is_contiguous()&&(input.scalar_type()==at::kBool||input.scalar_type()==at::kInt));
  CHECK(lut);CHECK(copies);CHECK(status);TORCH_CHECK(R==8&&limit>0);
  c10::cuda::CUDAGuard guard(input.device());auto stream=c10::cuda::getCurrentCUDAStream();
  bounded_materialize::fill<<<1,256,0,stream>>>(PTR(status),1,0);
  int blocks=std::min(R,limit);
  if(input.scalar_type()==at::kBool)bounded_materialize::support<<<blocks,512,0,stream>>>(input.data_ptr<bool>(),PTR(lut),PTR(copies),PTR(status),R,E,B);
  else bounded_materialize::support<<<blocks,512,0,stream>>>(PTR(input),PTR(lut),PTR(copies),PTR(status),R,E,B);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void prepare_bounded(Tensor routes,Tensor owner,Tensor seed,Tensor order,Tensor lut,Tensor kin,Tensor kout,
    Tensor vin,Tensor vout,Tensor starts,Tensor counts,Tensor hist,Tensor prefix,Tensor duplicates,Tensor status,
    int R,int E,int K,int limit,bool fuse) {
  CHECK(routes);CHECK(owner);CHECK(seed);CHECK(order);CHECK(lut);CHECK(kin);CHECK(kout);CHECK(vin);CHECK(vout);
  CHECK(starts);CHECK(counts);CHECK(hist);CHECK(prefix);CHECK(status);
  TORCH_CHECK(R==8&&K>=1&&K<=32&&limit>0&&routes.numel()>0&&routes.numel()%K==0);
  TORCH_CHECK(duplicates.is_cuda()&&duplicates.is_contiguous()&&duplicates.scalar_type()==at::kByte&&duplicates.numel()==routes.numel());
  int N=routes.numel(),tiles=(N+255)/256,width=1;while(width<K)width*=2;
  TORCH_CHECK(hist.numel()==int64_t(R)*tiles&&prefix.numel()==hist.numel());
  c10::cuda::CUDAGuard guard(routes.device());auto stream=c10::cuda::getCurrentCUDAStream();
  bounded_materialize::clear_prepare<<<bounded_materialize::grid(R*E,limit),256,0,stream>>>(PTR(counts),PTR(starts),R*E);
  if(fuse) {
    bounded_materialize::prepare_fused<<<bounded_materialize::grid(int64_t(N/K)*width,limit),256,0,stream>>>(PTR(routes),PTR(owner),PTR(seed),PTR(order),PTR(lut),PTR(kin),PTR(vin),PTR(counts),PTR(hist),duplicates.data_ptr<unsigned char>(),PTR(status),N,E,K,width,tiles);
  } else {
    bounded_materialize::metadata<<<bounded_materialize::grid(int64_t(N/K)*width,limit),256,0,stream>>>(PTR(routes),PTR(owner),duplicates.data_ptr<unsigned char>(),PTR(status),N/K,K,width);
    bounded_materialize::input<<<bounded_materialize::grid(N,limit),256,0,stream>>>(PTR(routes),PTR(owner),PTR(seed),PTR(order),PTR(lut),PTR(kin),PTR(vin),PTR(counts),PTR(status),N,E);
    bounded_materialize::histogram<<<bounded_materialize::grid(N,limit),256,0,stream>>>(PTR(kin),PTR(hist),N,tiles);
  }
  bounded_materialize::histogram_prefix<<<1,256,0,stream>>>(PTR(hist),PTR(prefix),tiles);
  bounded_materialize::scatter<<<bounded_materialize::grid(N,limit),256,0,stream>>>(PTR(kin),PTR(vin),PTR(prefix),PTR(kout),PTR(vout),N,tiles);
  bounded_materialize::starts<<<bounded_materialize::grid(N,limit),256,0,stream>>>(PTR(routes),PTR(owner),PTR(vout),PTR(starts),PTR(status),N,E);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void layout_encode_bounded(Tensor routes,Tensor owner,Tensor ids,Tensor starts,Tensor gathered,Tensor lut,
    Tensor copies,Tensor offsets,Tensor prefix,Tensor totals,Tensor duplicates,Tensor dst,Tensor cu,Tensor zeros,
    Tensor stats,Tensor status,int R,int E,int B,int K,int rank,int padding,int nvs,int limit) {
  CHECK(routes);CHECK(owner);TORCH_CHECK(R==8&&limit>0);
  c10::cuda::CUDAGuard guard(routes.device());auto stream=c10::cuda::getCurrentCUDAStream();int N=routes.numel();
  bounded_materialize::clear_layout<<<bounded_materialize::grid(std::max<int64_t>(N,R*E),limit),256,0,stream>>>(PTR(offsets),PTR(prefix),PTR(totals),PTR(dst),R*E,N);
  bounded_materialize::layout<<<std::min(R,limit),512,0,stream>>>(PTR(gathered),PTR(copies),PTR(offsets),PTR(prefix),PTR(totals),PTR(cu),PTR(zeros),PTR(status),R,E,B,rank,padding,nvs,N);
  bounded_materialize::encode<<<bounded_materialize::grid(N,limit),256,0,stream>>>(PTR(routes),PTR(owner),PTR(ids),PTR(starts),PTR(lut),PTR(offsets),PTR(prefix),PTR(totals),duplicates.data_ptr<unsigned char>(),PTR(dst),PTR(status),N,E,nvs);
  stats_kernel<<<1,256,0,stream>>>(PTR(totals),PTR(copies),PTR(stats),R,E,B,rank);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void publish_bounded(Tensor dst,Tensor meta,int rank,int nvs,int stride,int src_info,int limit) {
  CHECK(dst);CHECK(meta);c10::cuda::CUDAGuard guard(dst.device());auto stream=c10::cuda::getCurrentCUDAStream();
  bounded_materialize::publish<<<bounded_materialize::grid(dst.numel(),limit),256,0,stream>>>(PTR(dst),PTR(meta),dst.numel(),rank,nvs,stride,src_info);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
  m.def("workspace",&workspace);m.def("decode",&decode);m.def("support",&support);
  m.def("prepare",&prepare);m.def("layout_encode",&layout_encode);m.def("publish",&publish);
  m.def("fill_bounded",&fill_bounded);m.def("decode_bounded",&decode_bounded);
  m.def("support_bounded",&support_bounded);m.def("prepare_bounded",&prepare_bounded);
  m.def("layout_encode_bounded",&layout_encode_bounded);m.def("publish_bounded",&publish_bounded);
}
