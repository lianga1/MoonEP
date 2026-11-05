// Single physical GPU semantic port of supplied MoonEP rt0, R=8,E=352,K=8.
// NOT the original distributed cooperative kernel. No NVLink barriers/offload.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cub/device/device_radix_sort.cuh>
#include <cuda_runtime.h>
constexpr int R=8,E=352,H=44,K=8;
#define CHECK(x) TORCH_CHECK((x).is_cuda()&&(x).is_contiguous()&&(x).scalar_type()==at::kInt,#x)
__global__ void alloc_kernel(const int* counts,int* alloc,int* z,int* starts,int cap){
 if(threadIdx.x||blockIdx.x)return;
 int bal[R]={0};int pref=0;
 for(int e=0;e<E;e++){starts[e]=pref;pref+=counts[e];for(int d=0;d<R;d++)alloc[e*R+d]=(e/H==d)?counts[e]:0;bal[e/H]+=counts[e];}
 for(int d=0;d<R;d++)bal[d]-=cap;
 for(int i=0;i<R*R;i++)z[i]=0;
 while(true){int hi=0,lo=0;for(int d=1;d<R;d++){if(bal[d]>bal[hi])hi=d;if(bal[d]<bal[lo])lo=d;}if(bal[hi]<=0||bal[lo]>=0)break;int amount=-bal[lo];z[hi*R+lo]=amount;bal[hi]-=amount;bal[lo]=0;}
 for(int h=0;h<R;h++){
  int quotas[R],remain[H];for(int d=0;d<R;d++)quotas[d]=z[h*R+d];for(int e=0;e<H;e++)remain[e]=counts[h*H+e];
  while(true){int d=0,e=0;for(int q=1;q<R;q++)if(quotas[q]>quotas[d])d=q;if(quotas[d]<=0)break;for(int q=1;q<H;q++)if(remain[q]>remain[e])e=q;if(remain[e]<=0)break;int n=min(quotas[d],remain[e]);quotas[d]-=n;remain[e]-=n;alloc[(h*H+e)*R+d]+=n;alloc[(h*H+e)*R+h]-=n;}
 }
}
__global__ void layout_kernel(const int* alloc,int* copies,int* offsets,int* physical_lut,int* cu,int* zfr,int* stats,int B,int padding){
 int d=blockIdx.x,lane=threadIdx.x;__shared__ int chosen[E];
 for(int e=lane;e<E;e+=32){chosen[e]=0;physical_lut[d*E+e]=(e/H==d)?e%H:-1;}__syncwarp();
 int rem=0;for(int e=lane;e<E;e+=32)rem+=(e/H!=d&&alloc[e*R+d]>0);for(int delta=16;delta;delta>>=1)rem+=__shfl_down_sync(0xffffffff,rem,delta);if(lane==0)stats[d*2]=rem;
 for(int s=0;s<B;s++){
  int best=0,ei=-1;for(int e=lane;e<E;e+=32){int n=(e/H!=d&&!chosen[e])?alloc[e*R+d]:0;if(n>best||(n==best&&e>ei)){best=n;ei=e;}}
  for(int delta=16;delta;delta>>=1){int n=__shfl_down_sync(0xffffffff,best,delta),e=__shfl_down_sync(0xffffffff,ei,delta);if(n>best||(n==best&&e>ei)){best=n;ei=e;}}
  best=__shfl_sync(0xffffffff,best,0);ei=__shfl_sync(0xffffffff,ei,0);
  if(lane==0){copies[d*B+s]=best>0?ei:-1;if(best>0){chosen[ei]=1;physical_lut[d*E+ei]=H+s;atomicAdd(stats+(ei/H)*2+1,1);}}
  __syncwarp();
 }
 if(lane==0){int off=0;for(int g=0;g<E+B;g++){
  int e=g<E?(chosen[g]?-1:g):copies[d*B+g-E];int n=e<0?0:alloc[e*R+d];int padded=n?((n+padding-1)/padding)*padding:0;
  if(n>0)offsets[d*E+e]=off;cu[d*(E+B)+g]=off+padded;zfr[(d*(E+B)+g)*2]=padded>n?off+n:0;zfr[(d*(E+B)+g)*2+1]=padded-n;off+=padded;
 }}
}
__global__ void mapping_kernel(const int* sorted_expert,const int* sorted_pair,const int* starts,const int* alloc,const int* offsets,const int* lut,int* owner,int* physical,int* dst,int* src_info,int N,int cap,int NvS,int B){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=N)return;int e=sorted_expert[i],pair=sorted_pair[i],j=i-starts[e],prev=0,d=0;
 for(;d<R-1;d++){int n=alloc[e*R+d];if(j<prev+n)break;prev+=n;}
 int local=offsets[d*E+e]+j-prev;int slot=lut[d*E+e];owner[pair]=d;physical[pair]=slot<0?-1:d*(H+B)+slot;dst[pair]=d*NvS+local;src_info[d*NvS+local]=(pair/cap)*NvS+pair%cap;
}
__global__ void dedup(int* dst,int T,int NvS){int t=blockIdx.x*blockDim.x+threadIdx.x;if(t>=T)return;unsigned seen=0;for(int k=0;k<K;k++){int v=dst[t*K+k],d=v/NvS;unsigned bit=1u<<d;if(seen&bit)dst[t*K+k]=-v-1;seen|=bit;}}
__global__ void owner_only_kernel(const int* sorted_expert,const int* sorted_pair,const int* starts,const int* alloc,int* owner,int N){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=N)return;int e=sorted_expert[i],j=i-starts[e],prefix=0,d=0;for(;d<R-1;d++){int n=alloc[e*R+d];if(j<prefix+n)break;prefix+=n;}owner[sorted_pair[i]]=d;}
size_t workspace(torch::Tensor experts,torch::Tensor outkeys,torch::Tensor ids,torch::Tensor outids){size_t n=0;cub::DeviceRadixSort::SortPairs(nullptr,n,experts.data_ptr<int>(),outkeys.data_ptr<int>(),ids.data_ptr<int>(),outids.data_ptr<int>(),experts.numel(),0,9);return n;}
void run(torch::Tensor experts,torch::Tensor counts,torch::Tensor ids,torch::Tensor outkeys,torch::Tensor outids,torch::Tensor tmp,torch::Tensor alloc,torch::Tensor z,torch::Tensor starts,torch::Tensor copies,torch::Tensor offsets,torch::Tensor lut,torch::Tensor cu,torch::Tensor zfr,torch::Tensor stats,torch::Tensor owner,torch::Tensor physical,torch::Tensor dst,torch::Tensor src_info,int B,int padding){
 CHECK(experts);CHECK(counts);CHECK(ids);CHECK(owner);c10::cuda::CUDAGuard guard(experts.device());auto stream=c10::cuda::getCurrentCUDAStream();int N=experts.numel(),cap=N/R,NvS=src_info.numel()/R;size_t nb=tmp.numel();
 cudaMemsetAsync(stats.data_ptr<int>(),0,stats.numel()*4,stream);cudaMemsetAsync(src_info.data_ptr<int>(),0xff,src_info.numel()*4,stream);
 alloc_kernel<<<1,1,0,stream>>>(counts.data_ptr<int>(),alloc.data_ptr<int>(),z.data_ptr<int>(),starts.data_ptr<int>(),cap);
 layout_kernel<<<R,32,0,stream>>>(alloc.data_ptr<int>(),copies.data_ptr<int>(),offsets.data_ptr<int>(),lut.data_ptr<int>(),cu.data_ptr<int>(),zfr.data_ptr<int>(),stats.data_ptr<int>(),B,padding);
 cub::DeviceRadixSort::SortPairs(tmp.data_ptr(),nb,experts.data_ptr<int>(),outkeys.data_ptr<int>(),ids.data_ptr<int>(),outids.data_ptr<int>(),N,0,9,stream);
 mapping_kernel<<<(N+255)/256,256,0,stream>>>(outkeys.data_ptr<int>(),outids.data_ptr<int>(),starts.data_ptr<int>(),alloc.data_ptr<int>(),offsets.data_ptr<int>(),lut.data_ptr<int>(),owner.data_ptr<int>(),physical.data_ptr<int>(),dst.data_ptr<int>(),src_info.data_ptr<int>(),N,cap,NvS,B);
 dedup<<<(N/K+255)/256,256,0,stream>>>(dst.data_ptr<int>(),N/K,NvS);C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void run_seed(torch::Tensor experts,torch::Tensor counts,torch::Tensor ids,torch::Tensor outkeys,torch::Tensor outids,torch::Tensor tmp,torch::Tensor alloc,torch::Tensor z,torch::Tensor starts,torch::Tensor owner){
 CHECK(experts);CHECK(counts);CHECK(ids);CHECK(owner);c10::cuda::CUDAGuard guard(experts.device());auto stream=c10::cuda::getCurrentCUDAStream();int N=experts.numel();size_t nb=tmp.numel();
 alloc_kernel<<<1,1,0,stream>>>(counts.data_ptr<int>(),alloc.data_ptr<int>(),z.data_ptr<int>(),starts.data_ptr<int>(),N/R);
 cub::DeviceRadixSort::SortPairs(tmp.data_ptr(),nb,experts.data_ptr<int>(),outkeys.data_ptr<int>(),ids.data_ptr<int>(),outids.data_ptr<int>(),N,0,9,stream);
 owner_only_kernel<<<(N+255)/256,256,0,stream>>>(outkeys.data_ptr<int>(),outids.data_ptr<int>(),starts.data_ptr<int>(),alloc.data_ptr<int>(),owner.data_ptr<int>(),N);C10_CUDA_KERNEL_LAUNCH_CHECK();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("workspace",&workspace);m.def("run",&run);m.def("run_seed",&run_seed);}
