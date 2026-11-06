// Bounded, same-stream planner utilities and same-node peer collectives.
// Every launch has <=limit CTAs; no hidden device library or NCCL launches.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <climits>

namespace {
constexpr int R=8, T=256, HEADER=4;
__device__ __forceinline__ int load_acquire(const int* p) {
  int v; asm volatile("ld.acquire.sys.global.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v;
}
__device__ __forceinline__ void signal_release(int* p,int v) {
  asm volatile("red.release.sys.global.add.s32 [%0], %1;" :: "l"(p),"r"(v) : "memory");
}
// Exactly one CTA participates per GPU. Two alternating signal slots and
// alternating signs avoid host resets, including across repeated CUDA Graphs.
__device__ void peer_barrier(int* shared,int stride,int rank) {
  __shared__ int phase,sign;
  __threadfence_system(); __syncthreads();
  if(threadIdx.x==0){int state=shared[rank*stride+2];phase=state&1;sign=(state>>1)&1;}
  __syncthreads();
  if(threadIdx.x<R)signal_release(shared+threadIdx.x*stride+phase,sign?-1:1);
  __syncthreads();
  if(threadIdx.x==0){
    // Keep only the four states: no signed counter overflow in long training.
    shared[rank*stride+2]=(shared[rank*stride+2]+1)&3;
    const int target=sign?0:R;
    const unsigned long long start=clock64();
    while(load_acquire(shared+rank*stride+phase)!=target){
      if(clock64()-start>100000000000ull){printf("sampled peer barrier timed out rank=%d phase=%d\n",rank,phase);asm volatile("trap;");}
    }
  }
  __syncthreads();
}
__global__ void barrier_kernel(int* shared,int stride,int rank){peer_barrier(shared,stride,rank);}
__global__ void copy_kernel(const int* src,int* dst,int64_t n,bool publish){
  for(int64_t i=int64_t(blockIdx.x)*T+threadIdx.x;i<n;i+=int64_t(gridDim.x)*T)dst[i]=src[i];
  if(publish)__threadfence_system();
}
__device__ void read_collective(const int* shared,int* dst,int stride,int n,int mode,int64_t start,int64_t step){
  const int64_t total=mode==0?int64_t(R)*n:n;
  for(int64_t i=start;i<total;i+=step){
    if(mode==0)dst[i]=shared[(i/n)*stride+HEADER+i%n];
    else {int value=mode==1?0:INT_MIN;
      for(int r=0;r<R;++r){int v=shared[r*stride+HEADER+i];value=mode==1?value+v:max(value,v);}dst[i]=value;}
  }
}
__global__ void read_kernel(const int* shared,int* dst,int stride,int n,int mode){
  read_collective(shared,dst,stride,n,mode,int64_t(blockIdx.x)*T+threadIdx.x,int64_t(gridDim.x)*T);
}
__global__ void fused_collective(const int* src,int* dst,int* shared,int stride,int rank,int n,int mode){
  for(int i=threadIdx.x;i<n;i+=T)shared[rank*stride+HEADER+i]=src[i];
  peer_barrier(shared,stride,rank);
  read_collective(shared,dst,stride,n,mode,threadIdx.x,T);
  // All readers complete before any rank may reuse its payload for the next collective.
  peer_barrier(shared,stride,rank);
}
// Larger tables use a cooperative grid, bounded by the same CTA budget as the
// local scans. Only CTA zero sends peer signals, while every CTA participates
// in each grid barrier. The system fence publishes ALL writers before CTA zero
// announces readiness; the second peer barrier prevents payload reuse until
// every local and remote reader has finished.
__global__ void fused_collective_parallel(const int* src,int* dst,int* shared,
    int stride,int rank,int n,int mode){
  const auto grid=cooperative_groups::this_grid();
  const int64_t start=int64_t(blockIdx.x)*T+threadIdx.x;
  const int64_t step=int64_t(gridDim.x)*T;
  for(int64_t i=start;i<n;i+=step)shared[rank*stride+HEADER+i]=src[i];
  __threadfence_system();
  grid.sync();
  if(blockIdx.x==0)peer_barrier(shared,stride,rank);
  grid.sync();
  read_collective(shared,dst,stride,n,mode,start,step);
  grid.sync();
  if(blockIdx.x==0)peer_barrier(shared,stride,rank);
  grid.sync();
}
__global__ void status_kernel(const int* a,const int* b,const int* reject,int* out){
  if(threadIdx.x==0)*out=max(max(*a,*b),*reject);
}
__global__ void select_kernel(const int* status,const int* candidate,const int* seed,int* out,int64_t n){
  const bool valid=*status==0;
  for(int64_t i=int64_t(blockIdx.x)*T+threadIdx.x;i<n;i+=int64_t(gridDim.x)*T)out[i]=valid?candidate[i]:seed[i];
}
struct Selection {const int* a;const int* b;int* out;int64_t n;};
__global__ void select_all_kernel(const int* status,bool* valid,Selection a,Selection b,Selection c,Selection d,Selection e){
  const bool ok=*status==0;
  if(blockIdx.x==0&&threadIdx.x==0)*valid=ok;
  Selection jobs[5]={a,b,c,d,e};
  const int64_t start=int64_t(blockIdx.x)*T+threadIdx.x,step=int64_t(gridDim.x)*T;
  for(int j=0;j<5;++j)for(int64_t i=start;i<jobs[j].n;i+=step)jobs[j].out[i]=ok?jobs[j].a[i]:jobs[j].b[i];
}
void check(const at::Tensor& t,const at::Tensor& ref){
  TORCH_CHECK(t.is_cuda()&&t.is_contiguous()&&t.scalar_type()==at::kInt&&t.device()==ref.device(),"expected same-device contiguous int32");
}
int blocks(int64_t n,int limit){TORCH_CHECK(limit>0&&limit<=1024,"invalid planner CTA limit");return int(std::max<int64_t>(1,std::min<int64_t>(limit,(n+T-1)/T)));}
}

void copy(at::Tensor src,at::Tensor dst,int limit){
  check(src,src);check(dst,src);TORCH_CHECK(src.numel()==dst.numel());
  c10::cuda::CUDAGuard guard(src.device());auto s=c10::cuda::getCurrentCUDAStream();
  copy_kernel<<<blocks(src.numel(),limit),T,0,s>>>(src.data_ptr<int>(),dst.data_ptr<int>(),src.numel(),false);C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void collective(at::Tensor src,at::Tensor dst,at::Tensor shared,int stride,int rank,int mode,int limit,bool fuse){
  check(src,src);check(dst,src);check(shared,src);
  TORCH_CHECK(src.numel()>0&&src.numel()<=INT_MAX&&stride>=HEADER+src.numel()&&shared.numel()==int64_t(R)*stride&&rank>=0&&rank<R&&mode>=0&&mode<=2);
  TORCH_CHECK(dst.numel()==src.numel()*(mode==0?R:1));
  TORCH_CHECK(limit>0&&limit<=1024,"invalid planner CTA limit");
  c10::cuda::CUDAGuard guard(src.device());auto s=c10::cuda::getCurrentCUDAStream();
  int n=int(src.numel());int* sh=shared.data_ptr<int>();
  if(fuse&&n>T){
    const int grid=blocks(dst.numel(),limit);
    const int* input=src.data_ptr<int>();int* output=dst.data_ptr<int>();
    void* args[]={&input,&output,&sh,&stride,&rank,&n,&mode};
    C10_CUDA_CHECK(cudaLaunchCooperativeKernel(
        reinterpret_cast<const void*>(fused_collective_parallel),
        dim3(grid),dim3(T),args,0,s));
  }
  else if(fuse)fused_collective<<<1,T,0,s>>>(src.data_ptr<int>(),dst.data_ptr<int>(),sh,stride,rank,n,mode);
  else{
    copy_kernel<<<blocks(n,limit),T,0,s>>>(src.data_ptr<int>(),sh+rank*stride+HEADER,n,true);
    barrier_kernel<<<1,T,0,s>>>(sh,stride,rank);
    read_kernel<<<blocks(dst.numel(),limit),T,0,s>>>(sh,dst.data_ptr<int>(),stride,n,mode);
    barrier_kernel<<<1,T,0,s>>>(sh,stride,rank);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void status(at::Tensor a,at::Tensor b,at::Tensor reject,at::Tensor out){
  for(auto t:{a,b,reject,out}){check(t,a);TORCH_CHECK(t.numel()==1);}
  c10::cuda::CUDAGuard guard(a.device());status_kernel<<<1,T,0,c10::cuda::getCurrentCUDAStream()>>>(a.data_ptr<int>(),b.data_ptr<int>(),reject.data_ptr<int>(),out.data_ptr<int>());C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void select_all(at::Tensor status,at::Tensor valid,std::vector<at::Tensor> candidates,std::vector<at::Tensor> seeds,std::vector<at::Tensor> outputs,int limit,bool fuse){
  check(status,status);TORCH_CHECK(status.numel()==1&&candidates.size()==5&&seeds.size()==5&&outputs.size()==5);
  TORCH_CHECK(valid.device()==status.device()&&valid.scalar_type()==at::kBool&&valid.numel()==1&&valid.is_contiguous());
  Selection jobs[5];int64_t maximum=1;
  for(int j=0;j<5;++j){check(candidates[j],status);check(seeds[j],status);check(outputs[j],status);
    TORCH_CHECK(candidates[j].numel()==seeds[j].numel()&&outputs[j].numel()==seeds[j].numel());
    jobs[j]={candidates[j].data_ptr<int>(),seeds[j].data_ptr<int>(),outputs[j].data_ptr<int>(),outputs[j].numel()};maximum=std::max(maximum,jobs[j].n);}
  c10::cuda::CUDAGuard guard(status.device());auto s=c10::cuda::getCurrentCUDAStream();
  // The tiny valid flag travels with the first launch in either mode.
  if(fuse)select_all_kernel<<<blocks(maximum,limit),T,0,s>>>(status.data_ptr<int>(),valid.data_ptr<bool>(),jobs[0],jobs[1],jobs[2],jobs[3],jobs[4]);
  else{
    Selection empty{nullptr,nullptr,nullptr,0};
    select_all_kernel<<<blocks(jobs[0].n,limit),T,0,s>>>(status.data_ptr<int>(),valid.data_ptr<bool>(),jobs[0],empty,empty,empty,empty);
    for(int j=1;j<5;++j)select_kernel<<<blocks(jobs[j].n,limit),T,0,s>>>(status.data_ptr<int>(),jobs[j].a,jobs[j].b,jobs[j].out,jobs[j].n);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("copy",&copy);m.def("collective",&collective);m.def("status",&status);m.def("select_all",&select_all);}
