// Reuse native C1's stable expert order; no full-task radix sort.
// Segmented prefix is parallel over fixed 256-task tiles, avoiding one-CTA
// per-expert hot tails. Only the small tile-summary array uses DeviceScan.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cub/block/block_scan.cuh>
#include <cub/device/device_scan.cuh>
#include <cuda_runtime.h>
constexpr int BLOCK=256;
struct Seg {int ones;int head;};
struct SegOp {__host__ __device__ Seg operator()(const Seg& a,const Seg& b) const {return {b.head?b.ones:a.ones+b.ones,a.head|b.head};}};
__global__ void expert_prefix(const int* meta,int* starts,int* lengths,int* error,int E,int R,int rank,int stride,int plan_off,int M){
 using Scan=cub::BlockScan<int,512>;__shared__ typename Scan::TempStorage tmp;
 int e=threadIdx.x,n=0;
 if(e<E){int off=rank*stride+plan_off+E*R;n=meta[off+rank*E+e]-(rank?meta[off+(rank-1)*E+e]:0);}
 int start=0,total=0;Scan(tmp).ExclusiveSum(n,start,total);
 if(e<E){starts[e]=start;lengths[e]=n;}
 if(e==0&&total!=M)atomicAdd(error+3,1);
}
__global__ void block_prefix(const int* order,const int* routes,const int* owner,int* local_ones,Seg* summary,int* last_expert,int M){
 using Scan=cub::BlockScan<Seg,BLOCK>;__shared__ typename Scan::TempStorage tmp;
 int i=blockIdx.x*BLOCK+threadIdx.x;bool valid=i<M;
 int id=valid?order[i]:0,e=valid?routes[id]:-1,r=valid?owner[id]:0;
 int prev=__shfl_up_sync(0xffffffffu,e,1);
 if((threadIdx.x&31)==0)prev=i>0&&valid?routes[order[i-1]]:-1;
 Seg input{int(valid&&r==1),int(valid&&e!=prev)},out;
 Scan(tmp).InclusiveScan(input,out,SegOp());
 if(valid)local_ones[i]=out.ones;
 int last=min(BLOCK,M-blockIdx.x*BLOCK)-1;
 if(threadIdx.x==last){summary[blockIdx.x]=out;last_expert[blockIdx.x]=e;}
}
__device__ __forceinline__ int ones_to(int i,int e,const int* local_ones,const Seg* scanned,const int* last_expert){
 int b=i/BLOCK,value=local_ones[i];if(b&&last_expert[b-1]==e)value+=scanned[b-1].ones;return value;
}
__global__ void expert_counts(const int* starts,const int* lengths,const int* local_ones,const Seg* scanned,const int* last_expert,const int* lut,int* counts,int* error,int E,int B){
 int e=blockIdx.x*blockDim.x+threadIdx.x;if(e>=E)return;int n=lengths[e];if(!n)return;
 int count1=ones_to(starts[e]+n-1,e,local_ones,scanned,last_expert),Q=E+B;
 int ns[2]={n-count1,count1};for(int r=0;r<2;r++){int group=lut[r*E+e];if(ns[r]&&group<0){atomicAdd(error+1,1);continue;}if(group>=0)counts[r*Q+group]=ns[r];}
}
__global__ void publish_rows(const int* order,const int* route,const int* owner,const int* expert_starts,const int* local_ones,const Seg* scanned,const int* last_expert,const int* lut,const int* offsets,const int* prefix,int* physical,int* raw,int* meta,int* error,int M,int E,int B,int rank,int NvS,int stride,int src_info){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=M)return;
 int id=order[i],e=route[id],r=owner[id],Q=E+B,epn=E/2;
 int group=lut[r*E+e];if(group<0){raw[id]=-1;physical[id]=-1;atomicAdd(error+1,1);return;}
 int ones=ones_to(i,e,local_ones,scanned,last_expert);
 int ordinal=r?ones-1:i-expert_starts[e]-ones;
 int q=r*Q+group,loff=offsets[q]+prefix[q]+ordinal;
 if(loff<0||loff>=NvS){raw[id]=-1;physical[id]=-1;atomicAdd(error+2,1);return;}
 raw[id]=r*NvS+loff;physical[id]=r*(epn+B)+(group<E?e%epn:epn+group-E);
 meta[r*stride+src_info+loff]=rank*NvS+id;
 __threadfence_system();
}
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
__global__ void dedup(const int* raw,const int* owner,int* dst,int S,int K){
 int t=blockIdx.x*blockDim.x+threadIdx.x;if(t>=S)return;
 unsigned seen=0;for(int k=0;k<K;k++){int i=t*K+k,r=owner[i],v=raw[i];if(r<0||r>=32||v<0){dst[i]=-1;continue;}unsigned bit=1u<<r;dst[i]=(seen&bit)?-v-1:v;seen|=bit;}
}
__global__ void stats_exact(const int* totals,const int* etc,int* stats,int E,int B,int R,int rank){
 if(threadIdx.x)return;int incoming=0,outgoing=0,Q=E+B,epn=E/R;
 for(int dest=0;dest<R;dest++)for(int b=0;b<B;b++)if(totals[dest*Q+E+b]>0){int e=etc[dest*B+b];if(dest==rank)incoming++;if(e>=0&&e/epn==rank)outgoing++;}
 stats[0]=incoming;stats[1]=outgoing;
}

size_t workspace(torch::Tensor summary,torch::Tensor scanned){size_t bytes=0;cub::DeviceScan::InclusiveScan(nullptr,bytes,reinterpret_cast<Seg*>(summary.data_ptr<int>()),reinterpret_cast<Seg*>(scanned.data_ptr<int>()),SegOp(),summary.numel()/2);return bytes;}
void prepare(torch::Tensor order,torch::Tensor route,torch::Tensor owner,torch::Tensor lut,torch::Tensor expert_starts,torch::Tensor lengths,torch::Tensor local_ones,torch::Tensor summary,torch::Tensor scanned,torch::Tensor last_expert,torch::Tensor tmp,torch::Tensor counts,torch::Tensor meta,torch::Tensor error,int E,int B,int rank,int NvS,int stride,int plan_off,int src_info){
 TORCH_CHECK(route.is_cuda()&&route.scalar_type()==at::kInt&&owner.scalar_type()==at::kInt&&owner.numel()==route.numel()&&owner.is_contiguous());
 c10::cuda::CUDAGuard guard(route.device());auto s=c10::cuda::getCurrentCUDAStream();int M=route.numel(),blocks=(M+BLOCK-1)/BLOCK;size_t nbytes=tmp.numel();
 cudaMemsetAsync(counts.data_ptr<int>(),0,2*(E+B)*sizeof(int),s);
 cudaMemsetAsync(meta.data_ptr<int>()+rank*stride+src_info,0xff,NvS*sizeof(int),s);
 expert_prefix<<<1,512,0,s>>>(meta.data_ptr<int>(),expert_starts.data_ptr<int>(),lengths.data_ptr<int>(),error.data_ptr<int>(),E,2,rank,stride,plan_off,M);
 block_prefix<<<blocks,BLOCK,0,s>>>(order.data_ptr<int>(),route.data_ptr<int>(),owner.data_ptr<int>(),local_ones.data_ptr<int>(),reinterpret_cast<Seg*>(summary.data_ptr<int>()),last_expert.data_ptr<int>(),M);
 cub::DeviceScan::InclusiveScan(tmp.data_ptr(),nbytes,reinterpret_cast<Seg*>(summary.data_ptr<int>()),reinterpret_cast<Seg*>(scanned.data_ptr<int>()),SegOp(),blocks,s);
 expert_counts<<<(E+255)/256,256,0,s>>>(expert_starts.data_ptr<int>(),lengths.data_ptr<int>(),local_ones.data_ptr<int>(),reinterpret_cast<Seg*>(scanned.data_ptr<int>()),last_expert.data_ptr<int>(),lut.data_ptr<int>(),counts.data_ptr<int>(),error.data_ptr<int>(),E,B);
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void publish(torch::Tensor order,torch::Tensor route,torch::Tensor owner,torch::Tensor expert_starts,torch::Tensor local_ones,torch::Tensor scanned,torch::Tensor last_expert,torch::Tensor lut,torch::Tensor gathered,torch::Tensor offsets,torch::Tensor prefix,torch::Tensor totals,torch::Tensor cu,torch::Tensor zfr,torch::Tensor physical,torch::Tensor raw,torch::Tensor dst,torch::Tensor meta,torch::Tensor etc,torch::Tensor stats,torch::Tensor error,int E,int B,int rank,int K,int padding,int NvS,int stride,int src_info){
 c10::cuda::CUDAGuard guard(owner.device());auto s=c10::cuda::getCurrentCUDAStream();int M=owner.numel();
 layout<<<2,512,0,s>>>(gathered.data_ptr<int>(),offsets.data_ptr<int>(),prefix.data_ptr<int>(),cu.data_ptr<int>(),zfr.data_ptr<int>(),totals.data_ptr<int>(),error.data_ptr<int>(),E,B,2,rank,padding,NvS,M);
 publish_rows<<<(M+255)/256,256,0,s>>>(order.data_ptr<int>(),route.data_ptr<int>(),owner.data_ptr<int>(),expert_starts.data_ptr<int>(),local_ones.data_ptr<int>(),reinterpret_cast<Seg*>(scanned.data_ptr<int>()),last_expert.data_ptr<int>(),lut.data_ptr<int>(),offsets.data_ptr<int>(),prefix.data_ptr<int>(),physical.data_ptr<int>(),raw.data_ptr<int>(),meta.data_ptr<int>(),error.data_ptr<int>(),M,E,B,rank,NvS,stride,src_info);
 dedup<<<(M/K+255)/256,256,0,s>>>(raw.data_ptr<int>(),owner.data_ptr<int>(),dst.data_ptr<int>(),M/K,K);
 stats_exact<<<1,1,0,s>>>(totals.data_ptr<int>(),etc.data_ptr<int>(),stats.data_ptr<int>(),E,B,2,rank);C10_CUDA_KERNEL_LAUNCH_CHECK();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("workspace",&workspace);m.def("prepare",&prepare);m.def("publish",&publish);}
