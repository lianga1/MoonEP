// This parent's plan ABI: signed dst, copies, cu, zero-fill, remote stats.
// Never publish a candidate here: Python/caller makes the global decision first.
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cub/block/block_scan.cuh>
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
  int r=blockIdx.x,e=threadIdx.x,epn=E/R;
  int value=e<E?int(support[r*E+e]):0;
  bool home=e<E&&e/epn==r;
  if(e<E&&((value!=0&&value!=1)||(home&&!value))) atomicOr(status,1);
  int active=e<E&&!home&&value==1,slot=0,total=0;
  Scan(temp).ExclusiveSum(active,slot,total);
  if(e<B) copies[r*B+e]=-1;
  __syncthreads();
  if(e<E) {
    lut[r*E+e]=home&&value==1?e:(active&&slot<B?E+slot:-1);
    if(active&&slot<B) copies[r*B+slot]=e;
  }
  if(e==0&&total>B) atomicOr(status,2);
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
  if(p%K==0) for(int a=0;a<K;a++) for(int b=0;b<a;b++)
    if(routes[p+a]==routes[p+b]) atomicOr(status,8);
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
  using Scan=cub::BlockScan<int,512>;
  __shared__ typename Scan::TempStorage temp;
  int dest=blockIdx.x,g=threadIdx.x,Q=E+B,epn=E/R;
  int e=g<E?(g/epn==dest?g:-1):(g<Q?copies[dest*B+g-E]:-1);
  int count=0,prior=0;
  if(e>=0) for(int s=0;s<R;s++) {
    int n=gathered[(s*R+dest)*E+e];count+=n;if(s<source)prior+=n;
  }
  int padded=count?((count+padding-1)/padding)*padding:0,offset=0,totalpad=0;
  Scan(temp).ExclusiveSum(padded,offset,totalpad);
  if(e>=0) {offsets[dest*E+e]=offset;prefix[dest*E+e]=prior;totals[dest*E+e]=count;}
  if(dest==source&&g<Q) {
    cu[g]=offset+padded;
    zeros[g*2]=padded>count?offset+count:0;zeros[g*2+1]=padded-count;
  }
  if(g==0&&totalpad>nvs) atomicOr(status,64);
  __syncthreads();
  int ignored=0,totalcount=0;
  Scan(temp).ExclusiveSum(count,ignored,totalcount);
  if(g==0&&totalcount!=N) atomicOr(status,32);
}

__global__ void encode_kernel(const int* routes,const int* owner,const int* ids,const int* starts,
    const int* lut,const int* offsets,const int* prefix,const int* totals,int* dst,int* status,
    int N,int R,int E,int K,int nvs) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=N||*status)return;
  int p=ids[i],e=routes[p],d=owner[p];
  if(e<0||e>=E||d<0||d>=R) return;
  int key=d*E+e;
  if(lut[key]<0)return;
  int ordinal=i-starts[key],within=prefix[key]+ordinal,row=offsets[key]+within;
  if(ordinal<0||within<0||within>=totals[key]||row<0||row>=nvs) {atomicOr(status,64);return;}
  bool duplicate=false;
  for(int q=p-p%K;q<p;q++) duplicate|=owner[q]==d;
  int v=d*nvs+row;dst[p]=duplicate?-v-1:v;
}

__global__ void stats_kernel(const int* totals,const int* copies,int* stats,int R,int E,int B,int rank) {
  if(threadIdx.x)return;
  int incoming=0,outgoing=0,epn=E/R;
  for(int e=0;e<E;e++) if(e/epn!=rank&&totals[rank*E+e]>0)incoming++;
  // Native stats[1] counts planned prefetches from this home rank. A chosen
  // but presently idle copy still incurs a prefetch and therefore counts.
  for(int d=0;d<R;d++)for(int b=0;b<B;b++) {int e=copies[d*B+b];if(e>=0&&e/epn==rank)outgoing++;}
  stats[0]=incoming;stats[1]=outgoing;
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
    Tensor vin,Tensor vout,Tensor starts,Tensor counts,Tensor tmp,Tensor status,int R,int E,int K,bool reuse,int bits) {
  CHECK(routes);CHECK(owner);CHECK(seed);CHECK(order);CHECK(lut);
  c10::cuda::CUDAGuard guard(routes.device());auto stream=c10::cuda::getCurrentCUDAStream();int N=routes.numel();
  C10_CUDA_CHECK(cudaMemsetAsync(PTR(counts),0,R*E*sizeof(int),stream));
  C10_CUDA_CHECK(cudaMemsetAsync(PTR(starts),0,R*E*sizeof(int),stream));
  input_kernel<<<(N+255)/256,256,0,stream>>>(PTR(routes),PTR(owner),PTR(seed),PTR(order),PTR(lut),PTR(kin),PTR(vin),PTR(counts),PTR(status),N,R,E,K,reuse);
  size_t bytes=tmp.numel();
  C10_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(tmp.data_ptr(),bytes,PTR(kin),PTR(kout),PTR(vin),PTR(vout),N,0,bits,stream));
  starts_kernel<<<(N+255)/256,256,0,stream>>>(PTR(routes),PTR(owner),PTR(vout),PTR(starts),PTR(status),N,R,E);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void layout_encode(Tensor routes,Tensor owner,Tensor ids,Tensor starts,Tensor gathered,Tensor lut,
    Tensor copies,Tensor offsets,Tensor prefix,Tensor totals,Tensor dst,Tensor cu,Tensor zeros,
    Tensor stats,Tensor status,int R,int E,int B,int K,int rank,int padding,int nvs) {
  CHECK(routes);CHECK(owner);c10::cuda::CUDAGuard guard(routes.device());auto stream=c10::cuda::getCurrentCUDAStream();int N=routes.numel();
  C10_CUDA_CHECK(cudaMemsetAsync(PTR(offsets),0,R*E*sizeof(int),stream));
  C10_CUDA_CHECK(cudaMemsetAsync(PTR(prefix),0,R*E*sizeof(int),stream));
  C10_CUDA_CHECK(cudaMemsetAsync(PTR(totals),0,R*E*sizeof(int),stream));
  C10_CUDA_CHECK(cudaMemsetAsync(PTR(dst),0,N*sizeof(int),stream));
  layout_kernel<<<R,512,0,stream>>>(PTR(gathered),PTR(copies),PTR(offsets),PTR(prefix),PTR(totals),PTR(cu),PTR(zeros),PTR(status),R,E,B,rank,padding,nvs,N);
  encode_kernel<<<(N+255)/256,256,0,stream>>>(PTR(routes),PTR(owner),PTR(ids),PTR(starts),PTR(lut),PTR(offsets),PTR(prefix),PTR(totals),PTR(dst),PTR(status),N,R,E,K,nvs);
  stats_kernel<<<1,1,0,stream>>>(PTR(totals),PTR(copies),PTR(stats),R,E,B,rank);
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
