// Exact two-rank specialization of analysis/vendor/residency.py reciprocal.
// d<=1024, E=352, K=8, top-2, B<=8. Keeps gain-0 fallback candidates and
// original first-category-occurrence order, including deterministic ties.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace {
constexpr int E=352,K=8,WORDS=6,CLASSES=36,RECORDS=72,MAX_PACKAGES=96;
using Word=unsigned long long;

__global__ void gather_owner_kernel(const int* owner,const int64_t* ids,int* sampled,int d,int S,int rank) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=d*K)return;
  int t=i/K,k=i%K;int64_t local=ids[t]-int64_t(rank)*S;
  sampled[i]=(local>=0 && local<S)?owner[local*K+k]:-1;
}

__global__ void gather_joint_kernel(const int* routes,const int* owner,const int64_t* ids,
 int* sampled,int d,int S,int rank) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=d*16)return;
  int t=i/16,k=i%16;int64_t local=ids[t]-int64_t(rank)*S;
  sampled[i]=(local>=0 && local<S)?(k<8?routes[local*8+k]:owner[local*8+k-8]):-1;
}

__global__ void unpack_joint_kernel(const int* joint,int* routes,int* owner,int d) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=d*8)return;
  int t=i/8,k=i%8;routes[i]=joint[t*16+k];owner[i]=joint[t*16+8+k];
}

__global__ void mandatory_kernel(const int* etc,bool* support,int* extra,int B) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<2*E) {
    int rank=i/E,e=i%E;bool yes=e/(E/2)==rank;
    for(int j=0;j<B;++j)yes|=etc[rank*B+j]==e;
    support[i]=yes;
  }
  if(i<2) {
    int n=0;
    for(int j=0;j<B;++j) {
      int e=etc[i*B+j];bool first=e>=0 && e<E && e/(E/2)!=i;
      for(int k=0;k<j;++k)if(etc[i*B+k]==e)first=false;
      n+=first;
    }
    extra[i]=n;
  }
}

__device__ __forceinline__ void sort_small(int* a,int n) {
  #pragma unroll
  for(int i=1;i<K;++i)if(i<n) {
    int v=a[i],j=i-1;
    while(j>=0 && a[j]>v) {a[j+1]=a[j];--j;}
    a[j+1]=v;
  }
}

__global__ void sample_groups(const int* routes,const int64_t* ids,const int* owner,
 const bool* support,const int* extra,int* classes,int* groups,Word* sig0,Word* sig1,int d,int S,int B) {
  int t=blockIdx.x*blockDim.x+threadIdx.x;if(t>=d)return;
  int src=int(ids[t]/S),sz[2]={0,0},g0[8],g1[8];
  #pragma unroll
  for(int k=0;k<K;++k) {
    int a=owner[t*K+k],e=routes[t*K+k];
    if(a==0)g0[sz[0]++]=e;
    else if(a==1)g1[sz[1]++]=e;
  }
  int gain[2]={-2,-2};
  #pragma unroll
  for(int a=0;a<2;++a) {
    if(sz[a]==0)continue;
    int b=1-a,v=int(a!=src)-int(b!=src && sz[b]==0),missing=0;
    if(v<0)continue;
    #pragma unroll
    for(int k=0;k<K;++k)if(k<sz[a])missing+=!support[b*E+(a?g1[k]:g0[k])];
    if(missing<=B-extra[b])gain[a]=v;
  }
  int a=-1;
  if(gain[0]>gain[1])a=0;
  else if(gain[1]>gain[0])a=1;
  else if(gain[0]>=0) {
    unsigned tie=unsigned(ids[t])+0x85ebca6bu;
    tie^=tie>>16;tie*=0x7feb352du;tie^=tie>>15;
    a=int(tie&1u);
  }
  if(a<0 || gain[a]<0) {classes[t]=-1;sig0[t]=sig1[t]=0;return;}
  int size=sz[a],g[8];
  #pragma unroll
  for(int k=0;k<K;++k)g[k]=k<size?(a?g1[k]:g0[k]):511;
  sort_small(g,size);
  Word lo=0;
  #pragma unroll
  for(int k=0;k<7;++k)lo=(lo<<9)|Word(g[k]);
  sig0[t]=lo;sig1[t]=Word(g[7]);classes[t]=(a*2+gain[a])*9+size;
  #pragma unroll
  for(int k=0;k<K;++k)groups[t*K+k]=k<size?g[k]:-1;
}

__global__ void frequency_kernel(const int* classes,const Word* sig0,const Word* sig1,int* frequencies,int d) {
  int t=blockIdx.x*blockDim.x+threadIdx.x;if(t>=d)return;
  int c=classes[t],freq=0,first=d;
  if(c>=0)for(int j=0;j<d;++j) {
    if(classes[j]==c && sig0[j]==sig0[t] && sig1[j]==sig1[t]) {++freq;first=min(first,j);}
  }
  frequencies[t]=first==t?freq:0;
}

__device__ __forceinline__ bool better_group(int candidate,int incumbent,const int* f,const Word* a,const Word* b) {
  return incumbent<0 || f[candidate]>f[incumbent] ||
    (f[candidate]==f[incumbent] && (a[candidate]<a[incumbent] ||
     (a[candidate]==a[incumbent] && b[candidate]<b[incumbent])));
}

__global__ void tops_kernel(const int* classes,const int* f,const Word* sig0,const Word* sig1,
 int* tops,int* first,int d) {
  int c=threadIdx.x;if(c>=CLASSES)return;
  int a=-1,b=-1,earliest=d;
  for(int t=0;t<d;++t)if(classes[t]==c) {
    earliest=min(earliest,t);
    if(f[t]>0) {
      if(better_group(t,a,f,sig0,sig1)) {b=a;a=t;}
      else if(better_group(t,b,f,sig0,sig1))b=t;
    }
  }
  tops[c*2]=a;tops[c*2+1]=b;first[c]=earliest;
}

__device__ __forceinline__ bool enabled(const Word* group,const Word* support) {
  bool yes=true;
  #pragma unroll
  for(int w=0;w<WORDS;++w)yes &= (group[w]&~support[w])==0;
  return yes;
}
__device__ __forceinline__ int match_gain(int x0,int x1,int y0,int y1) {
  int n=min(min(x0+x1,y0+y1),max(x1,y1));
  return min(n,x1)+min(n,y1);
}
__device__ __forceinline__ bool better_score(int ga,int ca,int ia,int gb,int cb,int ib) {
  if(ga<=0)return false;if(gb<=0)return true;
  int left=ga*cb,right=gb*ca;
  if(left!=right)return left>right;
  if(ga!=gb)return ga>gb;
  if(ca!=cb)return ca<cb;
  return ia<ib;
}

__global__ void greedy_kernel(const int* groups,const int* frequencies,const int* classes,
 const int* tops,const int* first,bool* support,int* extra,int* output_etc,
 int* history,int* history_count,int* diagnostics,int B,int d) {
  __shared__ Word resident[2][WORDS],masks[RECORDS][WORDS];
  __shared__ int rec[RECORDS],freq[RECORDS],value[RECORDS];
  __shared__ int pa[MAX_PACKAGES],pb[MAX_PACKAGES],psize[MAX_PACKAGES],packages;
  __shared__ int e0[18],e1[18],current_extra[2];
  __shared__ int score_gain[128],score_cost[128],score_idx[128];
  __shared__ int additions_a,additions_b,accepted;
  int tid=threadIdx.x;
  if(tid<12) {
    int r=tid/WORDS,w=tid%WORDS;Word bits=0;
    for(int j=0;j<64;++j) {int e=w*64+j;if(e<E && support[r*E+e])bits|=Word(1)<<j;}
    resident[r][w]=bits;
  }
  if(tid<2)current_extra[tid]=extra[tid];
  if(tid==0) {accepted=0;packages=0;}
  if(tid<18) {
    int dir=tid/9,size=tid%9,c0=dir*18+size,c1=c0+9;
    int first_gain=first[c0]<=first[c1]?0:1,pos=0;
    for(int v=0;v<2;++v) {
      int gain=v^first_gain,c=(dir*2+gain)*9+size;
      for(int j=0;j<2;++j)if(tops[c*2+j]>=0)rec[tid*4+pos++]=tops[c*2+j];
    }
    while(pos<4)rec[tid*4+pos++]=-1;
  }
  __syncthreads();
  if(tid<RECORDS) {
    int sample=rec[tid];freq[tid]=sample>=0?frequencies[sample]:0;
    value[tid]=sample>=0?(classes[sample]/9)%2:0;
    #pragma unroll
    for(int w=0;w<WORDS;++w)masks[tid][w]=0;
    if(sample>=0) {
      #pragma unroll
      for(int k=0;k<K;++k) {int e=groups[sample*K+k];if(e>=0)masks[tid][e/64]|=Word(1)<<(e%64);}
    }
  }
  __syncthreads();
  if(tid==0) {
    for(int size=1;size<=8;++size)for(int i=0;i<4;++i)for(int j=0;j<4;++j) {
      int ra=size*4+i,rb=(9+size)*4+j;
      if(rec[ra]<0 || rec[rb]<0 || value[ra]+value[rb]==0)continue;
      pa[packages]=ra;pb[packages]=rb;psize[packages]=size;++packages;
    }
  }
  __syncthreads();
  for(int iteration=0;iteration<=2*B;++iteration) {
    if(tid<18) {
      int dir=tid/9,x0=0,x1=0;
      for(int j=0;j<4;++j) {
        int r=tid*4+j;
        if(rec[r]>=0 && enabled(masks[r],resident[1-dir])) {
          if(value[r])x1+=freq[r];else x0+=freq[r];
        }
      }
      e0[tid]=x0;e1[tid]=x1;
    }
    __syncthreads();
    int gain=0,cost=0;
    if(tid<packages) {
      int ra=pa[tid],rb=pb[tid],size=psize[tid],ca=0,cb=0;
      Word new_a[WORDS],new_b[WORDS];
      #pragma unroll
      for(int w=0;w<WORDS;++w) {
        Word add_a=masks[rb][w]&~resident[0][w],add_b=masks[ra][w]&~resident[1][w];
        ca+=__popcll(add_a);cb+=__popcll(add_b);
        new_a[w]=resident[0][w]|add_a;new_b[w]=resident[1][w]|add_b;
      }
      cost=ca+cb;
      if(cost && current_extra[0]+ca<=B && current_extra[1]+cb<=B) {
        int x0=0,x1=0,y0=0,y1=0;
        for(int j=0;j<4;++j) {
          int r=size*4+j;
          if(rec[r]>=0 && enabled(masks[r],new_b)) {if(value[r])x1+=freq[r];else x0+=freq[r];}
          r=(9+size)*4+j;
          if(rec[r]>=0 && enabled(masks[r],new_a)) {if(value[r])y1+=freq[r];else y0+=freq[r];}
        }
        gain=match_gain(x0,x1,y0,y1)-match_gain(e0[size],e1[size],e0[9+size],e1[9+size]);
      }
    }
    score_gain[tid]=gain;score_cost[tid]=cost;score_idx[tid]=tid;
    __syncthreads();
    for(int delta=64;delta;delta/=2) {
      if(tid<delta && better_score(score_gain[tid+delta],score_cost[tid+delta],score_idx[tid+delta],score_gain[tid],score_cost[tid],score_idx[tid])) {
        score_gain[tid]=score_gain[tid+delta];score_cost[tid]=score_cost[tid+delta];score_idx[tid]=score_idx[tid+delta];
      }
      __syncthreads();
    }
    if(score_gain[0]<=0)break;
    int best=score_idx[0],ra=pa[best],rb=pb[best];
    if(tid==0) {
      additions_a=additions_b=0;
      #pragma unroll
      for(int w=0;w<WORDS;++w) {
        additions_a+=__popcll(masks[rb][w]&~resident[0][w]);
        additions_b+=__popcll(masks[ra][w]&~resident[1][w]);
      }
      current_extra[0]+=additions_a;current_extra[1]+=additions_b;
      if(accepted<2*B) {
        int* h=history+accepted*7;
        h[0]=0;h[1]=1;h[2]=additions_a;h[3]=additions_b;h[4]=psize[best];h[5]=score_gain[0];h[6]=score_cost[0];
      }
      ++accepted;
    }
    __syncthreads();
    if(tid<WORDS) {resident[0][tid]|=masks[rb][tid];resident[1][tid]|=masks[ra][tid];}
    __syncthreads();
  }
  for(int i=tid;i<2*E;i+=blockDim.x) {int r=i/E,e=i%E;support[i]=(resident[r][e/64]>>(e%64))&1;}
  if(tid<2) {
    extra[tid]=current_extra[tid];
    int n=0;
    for(int e=0;e<E;++e)if(e/(E/2)!=tid && ((resident[tid][e/64]>>(e%64))&1)) {
      if(n<B)output_etc[tid*B+n]=e;++n;
    }
    while(n<B)output_etc[tid*B+n++]=-1;
  }
  if(tid==0) {*history_count=accepted;diagnostics[0]=packages;diagnostics[1]=current_extra[0];diagnostics[2]=current_extra[1];}
}
} // namespace

void gather_owner(const at::Tensor& owner,const at::Tensor& ids,at::Tensor& sampled,int64_t S,int64_t rank) {
  c10::cuda::CUDAGuard guard(owner.device());int d=ids.numel();
  gather_owner_kernel<<<(d*8+255)/256,256,0,c10::cuda::getCurrentCUDAStream()>>>(owner.data_ptr<int>(),ids.data_ptr<int64_t>(),sampled.data_ptr<int>(),d,S,rank);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gather_joint(const at::Tensor& routes,const at::Tensor& owner,const at::Tensor& ids,
 at::Tensor& sampled,int64_t S,int64_t rank) {
  c10::cuda::CUDAGuard guard(owner.device());int d=ids.numel();
  gather_joint_kernel<<<(d*16+255)/256,256,0,c10::cuda::getCurrentCUDAStream()>>>(routes.data_ptr<int>(),owner.data_ptr<int>(),ids.data_ptr<int64_t>(),sampled.data_ptr<int>(),d,S,rank);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void unpack_joint(const at::Tensor& joint,at::Tensor& routes,at::Tensor& owner) {
  c10::cuda::CUDAGuard guard(joint.device());int d=routes.size(0);
  unpack_joint_kernel<<<(d*8+255)/256,256,0,c10::cuda::getCurrentCUDAStream()>>>(joint.data_ptr<int>(),routes.data_ptr<int>(),owner.data_ptr<int>(),d);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void select_residency(const at::Tensor& routes,const at::Tensor& ids,const at::Tensor& sample_owner,const at::Tensor& etc,
 at::Tensor& support,at::Tensor& extra,at::Tensor& classes,at::Tensor& groups,at::Tensor& sig0,at::Tensor& sig1,
 at::Tensor& frequencies,at::Tensor& tops,at::Tensor& first,at::Tensor& output_etc,at::Tensor& history,
 at::Tensor& history_count,at::Tensor& diagnostics,int64_t S,int64_t B) {
  TORCH_CHECK(routes.is_cuda() && routes.is_contiguous() && routes.scalar_type()==at::kInt && routes.dim()==2 && routes.size(1)==8,"routes require CUDA int32 [d,8]");
  TORCH_CHECK(ids.is_cuda() && ids.is_contiguous() && ids.scalar_type()==at::kLong && ids.numel()==routes.size(0),"ids require CUDA int64 [d]");
  TORCH_CHECK(etc.is_cuda() && etc.is_contiguous() && etc.scalar_type()==at::kInt && etc.numel()==2*B,"experts_to_copy requires CUDA int32 [2,B]");
  TORCH_CHECK(B>=1 && B<=8 && routes.size(0)>0 && routes.size(0)<=1024,"requires 1<=B<=8, 1<=d<=1024");
  c10::cuda::CUDAGuard guard(routes.device());auto stream=c10::cuda::getCurrentCUDAStream();int d=routes.size(0);
  mandatory_kernel<<<3,256,0,stream>>>(etc.data_ptr<int>(),support.data_ptr<bool>(),extra.data_ptr<int>(),B);
  sample_groups<<<(d+127)/128,128,0,stream>>>(routes.data_ptr<int>(),ids.data_ptr<int64_t>(),sample_owner.data_ptr<int>(),support.data_ptr<bool>(),extra.data_ptr<int>(),classes.data_ptr<int>(),groups.data_ptr<int>(),reinterpret_cast<Word*>(sig0.data_ptr<int64_t>()),reinterpret_cast<Word*>(sig1.data_ptr<int64_t>()),d,S,B);
  frequency_kernel<<<(d+127)/128,128,0,stream>>>(classes.data_ptr<int>(),reinterpret_cast<const Word*>(sig0.data_ptr<int64_t>()),reinterpret_cast<const Word*>(sig1.data_ptr<int64_t>()),frequencies.data_ptr<int>(),d);
  tops_kernel<<<1,64,0,stream>>>(classes.data_ptr<int>(),frequencies.data_ptr<int>(),reinterpret_cast<const Word*>(sig0.data_ptr<int64_t>()),reinterpret_cast<const Word*>(sig1.data_ptr<int64_t>()),tops.data_ptr<int>(),first.data_ptr<int>(),d);
  greedy_kernel<<<1,128,0,stream>>>(groups.data_ptr<int>(),frequencies.data_ptr<int>(),classes.data_ptr<int>(),tops.data_ptr<int>(),first.data_ptr<int>(),support.data_ptr<bool>(),extra.data_ptr<int>(),output_etc.data_ptr<int>(),history.data_ptr<int>(),history_count.data_ptr<int>(),diagnostics.data_ptr<int>(),B,d);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
  m.def("gather_owner",&gather_owner);m.def("gather_joint",&gather_joint);
  m.def("unpack_joint",&unpack_joint);m.def("select",&select_residency);
}
