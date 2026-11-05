#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cub/device/device_radix_sort.cuh>
#include <cub/block/block_scan.cuh>
#include <cuda_runtime.h>

// Errors: invalid/overflowing expert support, invalid task, NvS overflow,
// unequal per-rank task load. Checked outside timed loops by the caller.
template <typename T> __global__ void support_lut(const T* support,int* lut,int* etc,int* error,int E,int B,int R){
 const int r=blockIdx.x,e=threadIdx.x,epn=E/R;
 using Scan=cub::BlockScan<int,512>;__shared__ typename Scan::TempStorage tmp;
 int flag=e<E && e/epn!=r && support[r*E+e]!=0,slot=0,total=0;
 Scan(tmp).ExclusiveSum(flag,slot,total);
 if(e<B)etc[r*B+e]=-1;__syncthreads();
 if(e==0&&total>B)atomicAdd(error,1);
 if(e<E){int g=-1;if(e/epn==r)g=e;else if(flag&&slot<B){g=E+slot;etc[r*B+slot]=e;}lut[r*E+e]=g;}
}
__global__ void make_keys(const int* route,const int* owner,const int* lut,int* key,int* phys,int* error,int M,int E,int B,int R){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=M)return;
 int e=route[i],r=owner[i],Q=E+B,epn=E/R;
 if(e<0||e>=E||r<0||r>=R){key[i]=R*Q;phys[i]=-1;atomicAdd(error+1,1);return;}
 int g=lut[r*E+e];if(g<0){key[i]=R*Q;phys[i]=-1;atomicAdd(error+1,1);return;}
 key[i]=r*Q+g;phys[i]=r*(epn+B)+(g<E?e%epn:epn+g-E);
}
__global__ void boundaries(const int* keys,int* starts,int* ends,int M,int Q){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=M)return;int k=keys[i];if(k>=Q)return;
 if(i==0||keys[i-1]!=k)starts[k]=i;
 if(i==M-1||keys[i+1]!=k)ends[k]=i+1;
}
__global__ void finish_counts(int* counts,const int* starts,int Q){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<Q&&counts[i])counts[i]-=starts[i];}
__global__ void layout(const int* gathered,int* offsets,int* prefix,int* cu,int* zfr,int* totals,int* error,int E,int B,int R,int rank,int padding,int NvS,int capacity){
 using Scan=cub::BlockScan<int,512>;__shared__ typename Scan::TempStorage tmp;
 int dest=blockIdx.x,g=threadIdx.x,Q=E+B,q=dest*Q+g,n=0,prior=0;
 if(g<Q){for(int s=0;s<R;s++){int v=gathered[s*R*Q+q];n+=v;if(s<rank)prior+=v;}}
 int padded=((n+padding-1)/padding)*padding,offset=0,totalpad=0;
 Scan(tmp).ExclusiveSum(padded,offset,totalpad);
 if(g==0&&totalpad>NvS)atomicAdd(error+2,1);
 if(g<Q){offsets[q]=offset;prefix[q]=prior;totals[q]=n;
  if(dest==rank){cu[g]=offset+padded;zfr[2*g]=padded>n?offset+n:0;zfr[2*g+1]=padded-n;}}
 // Independent total task-load audit, no token-content read.
 __syncthreads();int totaln=0,ignored=0;
 Scan(tmp).ExclusiveSum(n,ignored,totaln);
 if(g==0&&totaln!=capacity)atomicAdd(error+3,1);
}
__global__ void rows(const int* sorted_keys,const int* sorted_ids,const int* starts,const int* offsets,const int* prefix,int* raw,int* meta,int* error,int M,int E,int B,int R,int rank,int NvS,int stride,int src_info){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=M)return;int q=sorted_keys[i],id=sorted_ids[i],Q=E+B;
 if(q>=R*Q){raw[id]=-1;return;}
 int dest=q/Q,loff=offsets[q]+prefix[q]+i-starts[q];
 if(loff>=NvS){raw[id]=-1;atomicAdd(error+2,1);return;}
 raw[id]=dest*NvS+loff;
 meta[dest*stride+src_info+loff]=rank*NvS+id;
 // Kernel completion plus the following system-scope MoonEP rank barrier
 // publishes all peer writes before any fresh dispatch consumes src_info.
 __threadfence_system();
}
__global__ void dedup(const int* raw,const int* owner,int* dst,int S,int K){
 int t=blockIdx.x*blockDim.x+threadIdx.x;if(t>=S)return;
 unsigned seen=0;for(int k=0;k<K;k++){int i=t*K+k,r=owner[i],v=raw[i];if(r<0||r>=32||v<0){dst[i]=-1;continue;}unsigned bit=1u<<r;dst[i]=(seen&bit)?-v-1:v;seen|=bit;}
}
__global__ void stats_exact(const int* totals,const int* etc,int* stats,int E,int B,int R,int rank){
 if(threadIdx.x)return;int incoming=0,outgoing=0,Q=E+B,epn=E/R;
 for(int dest=0;dest<R;dest++)for(int b=0;b<B;b++)if(totals[dest*Q+E+b]>0){int e=etc[dest*B+b];if(dest==rank)incoming++;if(e>=0&&e/epn==rank)outgoing++;}
 stats[0]=incoming;stats[1]=outgoing;
}
size_t workspace(torch::Tensor key,torch::Tensor outkey,torch::Tensor ids,torch::Tensor outids,int bits){size_t n=0;cub::DeviceRadixSort::SortPairs(nullptr,n,key.data_ptr<int>(),outkey.data_ptr<int>(),ids.data_ptr<int>(),outids.data_ptr<int>(),key.numel(),0,bits);return n;}
void prepare_support(torch::Tensor support,torch::Tensor lut,torch::Tensor etc,torch::Tensor error,int E,int B,int R){
 TORCH_CHECK(support.is_cuda()&&(support.scalar_type()==at::kInt||support.scalar_type()==at::kBool)&&support.is_contiguous()&&support.numel()==R*E);TORCH_CHECK(E<=512&&E+B<=512&&B<=512&&E%R==0);
 c10::cuda::CUDAGuard guard(support.device());auto s=c10::cuda::getCurrentCUDAStream();cudaMemsetAsync(error.data_ptr<int>(),0,4*sizeof(int),s);
 if(support.scalar_type()==at::kBool) support_lut<bool><<<R,512,0,s>>>(support.data_ptr<bool>(),lut.data_ptr<int>(),etc.data_ptr<int>(),error.data_ptr<int>(),E,B,R);
 else support_lut<int><<<R,512,0,s>>>(support.data_ptr<int>(),lut.data_ptr<int>(),etc.data_ptr<int>(),error.data_ptr<int>(),E,B,R);C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void local_prepare(torch::Tensor route,torch::Tensor owner,torch::Tensor lut,torch::Tensor ids,torch::Tensor key,torch::Tensor phys,torch::Tensor outkey,torch::Tensor outids,torch::Tensor tmp,torch::Tensor starts,torch::Tensor counts,torch::Tensor meta,torch::Tensor error,int E,int B,int R,int rank,int NvS,int stride,int src_info,int bits){
 TORCH_CHECK(route.is_cuda()&&route.scalar_type()==at::kInt&&owner.scalar_type()==at::kInt&&route.is_contiguous()&&owner.is_contiguous()&&route.numel()==owner.numel());
 c10::cuda::CUDAGuard guard(route.device());auto s=c10::cuda::getCurrentCUDAStream();int M=route.numel();size_t nb=tmp.numel();
 cudaMemsetAsync(counts.data_ptr<int>(),0,R*(E+B)*sizeof(int),s);
 cudaMemsetAsync(meta.data_ptr<int>()+rank*stride+src_info,0xff,NvS*sizeof(int),s);
 make_keys<<<(M+255)/256,256,0,s>>>(route.data_ptr<int>(),owner.data_ptr<int>(),lut.data_ptr<int>(),key.data_ptr<int>(),phys.data_ptr<int>(),error.data_ptr<int>(),M,E,B,R);
 cub::DeviceRadixSort::SortPairs(tmp.data_ptr(),nb,key.data_ptr<int>(),outkey.data_ptr<int>(),ids.data_ptr<int>(),outids.data_ptr<int>(),M,0,bits,s);
 boundaries<<<(M+255)/256,256,0,s>>>(outkey.data_ptr<int>(),starts.data_ptr<int>(),counts.data_ptr<int>(),M,R*(E+B));
 finish_counts<<<(R*(E+B)+255)/256,256,0,s>>>(counts.data_ptr<int>(),starts.data_ptr<int>(),R*(E+B));C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void publish(torch::Tensor owner,torch::Tensor outkey,torch::Tensor outids,torch::Tensor starts,torch::Tensor gathered,torch::Tensor offsets,torch::Tensor prefix,torch::Tensor totals,torch::Tensor cu,torch::Tensor zfr,torch::Tensor raw,torch::Tensor dst,torch::Tensor meta,torch::Tensor etc,torch::Tensor stats,torch::Tensor error,int E,int B,int R,int rank,int K,int padding,int NvS,int stride,int src_info){
 c10::cuda::CUDAGuard guard(owner.device());auto s=c10::cuda::getCurrentCUDAStream();int M=owner.numel();
 layout<<<R,512,0,s>>>(gathered.data_ptr<int>(),offsets.data_ptr<int>(),prefix.data_ptr<int>(),cu.data_ptr<int>(),zfr.data_ptr<int>(),totals.data_ptr<int>(),error.data_ptr<int>(),E,B,R,rank,padding,NvS,M);
 rows<<<(M+255)/256,256,0,s>>>(outkey.data_ptr<int>(),outids.data_ptr<int>(),starts.data_ptr<int>(),offsets.data_ptr<int>(),prefix.data_ptr<int>(),raw.data_ptr<int>(),meta.data_ptr<int>(),error.data_ptr<int>(),M,E,B,R,rank,NvS,stride,src_info);
 dedup<<<(M/K+255)/256,256,0,s>>>(raw.data_ptr<int>(),owner.data_ptr<int>(),dst.data_ptr<int>(),M/K,K);
 stats_exact<<<1,1,0,s>>>(totals.data_ptr<int>(),etc.data_ptr<int>(),stats.data_ptr<int>(),E,B,R,rank);C10_CUDA_KERNEL_LAUNCH_CHECK();
}
__global__ void decode_owner_kernel(const int* dst,int* owner,int M,int NvS){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<M){int v=dst[i];int raw=v>=0?v:-v-1;owner[i]=raw/NvS;}}
void decode_owner(torch::Tensor dst,torch::Tensor owner,int NvS){c10::cuda::CUDAGuard guard(dst.device());auto s=c10::cuda::getCurrentCUDAStream();int M=dst.numel();TORCH_CHECK(dst.scalar_type()==at::kInt&&owner.scalar_type()==at::kInt&&owner.numel()==M);decode_owner_kernel<<<(M+255)/256,256,0,s>>>(dst.data_ptr<int>(),owner.data_ptr<int>(),M,NvS);C10_CUDA_KERNEL_LAUNCH_CHECK();}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("decode_owner",&decode_owner);m.def("workspace",&workspace);m.def("prepare_support",&prepare_support);m.def("local_prepare",&local_prepare);m.def("publish",&publish);}
