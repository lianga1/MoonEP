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

PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
  m.def("workspace",&workspace);m.def("decode",&decode);m.def("support",&support);
  m.def("prepare",&prepare);m.def("layout_encode",&layout_encode);m.def("publish",&publish);
}
