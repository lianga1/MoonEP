#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cub/device/device_radix_sort.cuh>
#include <cuda_runtime.h>
__global__ void make_keys(const int* e,const int* o,const int* lut,int* keys,int N,int Q){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<N)keys[i]=o[i]*Q+lut[o[i]*352+e[i]];}
__global__ void boundaries(const int* key,int* starts,int* counts,int N){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=N)return;int k=key[i];if(i==0||key[i-1]!=k)starts[k]=i;if(i==N-1||key[i+1]!=k)counts[k]=i+1;}
__global__ void padded_layout(int* counts,const int* starts,int* offsets,int* cu,int* zfr,int Q,int padding){int rank=blockIdx.x;if(threadIdx.x)return;int off=0;for(int s=0;s<Q;s++){int q=rank*Q+s;int n=counts[q];n=n?n-starts[q]:0;counts[q]=n;int p=(n+padding-1)/padding*padding;offsets[q]=off;cu[q]=off+p;zfr[q*2]=p>n?off+n:0;zfr[q*2+1]=p-n;off+=p;}}
__global__ void rows(const int* keys,const int* ids,const int* starts,const int* offsets,int* raw,int* src,int N,int Q,int cap,int NvS){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=N)return;int k=keys[i],id=ids[i],rank=k/Q,local=i-starts[k]+offsets[k];int dest=rank*NvS+local;raw[id]=dest;src[dest]=id/cap*NvS+id%cap;}
__global__ void dedup_rows(const int* raw,const int* owner,int* dst,int N){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=N)return;int lane=threadIdx.x&31;unsigned same=__match_any_sync(__activemask(),owner[i]);unsigned previous=((1u<<lane)-1u)&(0xffu<<(lane&~7));int v=raw[i];dst[i]=(same&previous)?-v-1:v;}
size_t workspace(torch::Tensor key,torch::Tensor outkey,torch::Tensor ids,torch::Tensor outids){size_t n=0;cub::DeviceRadixSort::SortPairs(nullptr,n,key.data_ptr<int>(),outkey.data_ptr<int>(),ids.data_ptr<int>(),outids.data_ptr<int>(),key.numel(),0,9);return n;}
void run(torch::Tensor e,torch::Tensor owner,torch::Tensor lut,torch::Tensor ids,torch::Tensor key,torch::Tensor outkey,torch::Tensor outids,torch::Tensor tmp,torch::Tensor starts,torch::Tensor counts,torch::Tensor offsets,torch::Tensor cu,torch::Tensor zfr,torch::Tensor raw,torch::Tensor dst,torch::Tensor src,int Q,int padding){
 TORCH_CHECK(e.is_cuda()&&e.is_contiguous()&&owner.is_contiguous()&&owner.scalar_type()==at::kInt&&owner.numel()==e.numel());c10::cuda::CUDAGuard guard(e.device());auto s=c10::cuda::getCurrentCUDAStream();int N=e.numel(),cap=N/8,NvS=src.numel()/8;size_t nb=tmp.numel();
 cudaMemsetAsync(counts.data_ptr<int>(),0,counts.numel()*4,s);cudaMemsetAsync(src.data_ptr<int>(),0xff,src.numel()*4,s);
 make_keys<<<(N+255)/256,256,0,s>>>(e.data_ptr<int>(),owner.data_ptr<int>(),lut.data_ptr<int>(),key.data_ptr<int>(),N,Q);
 cub::DeviceRadixSort::SortPairs(tmp.data_ptr(),nb,key.data_ptr<int>(),outkey.data_ptr<int>(),ids.data_ptr<int>(),outids.data_ptr<int>(),N,0,9,s);
 boundaries<<<(N+255)/256,256,0,s>>>(outkey.data_ptr<int>(),starts.data_ptr<int>(),counts.data_ptr<int>(),N);
 padded_layout<<<8,1,0,s>>>(counts.data_ptr<int>(),starts.data_ptr<int>(),offsets.data_ptr<int>(),cu.data_ptr<int>(),zfr.data_ptr<int>(),Q,padding);
 rows<<<(N+255)/256,256,0,s>>>(outkey.data_ptr<int>(),outids.data_ptr<int>(),starts.data_ptr<int>(),offsets.data_ptr<int>(),raw.data_ptr<int>(),src.data_ptr<int>(),N,Q,cap,NvS);
 dedup_rows<<<(N+255)/256,256,0,s>>>(raw.data_ptr<int>(),owner.data_ptr<int>(),dst.data_ptr<int>(),N);C10_CUDA_KERNEL_LAUNCH_CHECK();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("workspace",&workspace);m.def("run",&run);}
