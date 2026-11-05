#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cub/device/device_radix_sort.cuh>
#include <cub/block/block_scan.cuh>
#include <cuda_runtime.h>
#include <climits>
constexpr int R=8,K=8,E=352,BASE=E+E*E,MAXP=128,RESIDENTS=52,LBASE=RESIDENTS+RESIDENTS*RESIDENTS;
using U64=unsigned long long;
#define SUBSET_CHECK(x) TORCH_CHECK(x.is_cuda()&&x.is_contiguous(),#x " CUDA contiguous required")
__global__ void transpose(const int*x,int*y,int T){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<T*K)y[(i%K)*T+i/K]=x[i];}
__global__ void untranspose(const int*x,int*y,int T){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<T*K)y[i]=x[(i%K)*T+i/K];}
__global__ void resident_kernel(const bool*su,int*res){int e=threadIdx.x+blockIdx.x*blockDim.x;if(e<E){int v=0;for(int r=0;r<R;r++)v|=int(su[r*E+e])<<r;res[e]=v;}}
__global__ void slots_kernel(const bool*su,int*lut,int*slots){
 int d=blockIdx.x;if(threadIdx.x)return;int n=0;
 for(int e=0;e<E;e++){lut[d*E+e]=-1;if(su[d*E+e]){if(n<RESIDENTS){lut[d*E+e]=n;slots[d*RESIDENTS+n]=e;}n++;}}
}
__device__ void warpadd(int*ptr,int key){unsigned active=__activemask();unsigned eq=__match_any_sync(active,key);if((threadIdx.x&31)==__ffs(eq)-1)atomicAdd(ptr+key,__popc(eq));}
__global__ void mine_kernel(const int*x,const int*o,const int*res,const int*lut,int*counts,int T,int S,int h){
 int t=blockIdx.x*blockDim.x+threadIdx.x;if(t>=T)return;int ex[8],ow[8],occ[8]={0};
 #pragma unroll
 for(int k=0;k<8;k++){ex[k]=x[k*T+t];ow[k]=o[k*T+t];occ[ow[k]]++;}
 for(int i=0;i<8;i++)for(int j=-1;j<8;j++){
  if(j>=0&&(h<2||j<=i||ow[j]!=ow[i]))continue;
  int a=ow[i],size=j<0?1:2,pat=j<0?ex[i]:E+min(ex[i],ex[j])*E+max(ex[i],ex[j]);
  int possible=res[ex[i]]&~(1<<a);if(j>=0)possible&=res[ex[j]];
  for(int b=0;b<8;b++)if(possible&(1<<b)){
   int gain=int(a!=t/S&&size==occ[a])-int(b!=t/S&&occ[b]==0);if(gain<0)continue;
   int u=lut[b*E+ex[i]],v=j<0?-1:lut[b*E+ex[j]];if(u<0||(j>=0&&v<0))continue;int lp=j<0?u:RESIDENTS+min(u,v)*RESIDENTS+max(u,v);
   int key=((((a*8+b)*2+gain)*2+size-1)*LBASE)+lp;warpadd(counts,key);
  }
 }
}
// Keep g local maxima per thread while reading the dense count table once,
// then repeatedly reduce thread-local maxima. Tie: smallest pattern ID.
__global__ void select_kernel(const int*counts,const int*slots,int*top,int g){
 int cls=blockIdx.x,tid=threadIdx.x,edge=cls/4;if(edge/8==edge%8)return;
 int value[8],idx[8];for(int j=0;j<8;j++){value[j]=0;idx[j]=INT_MAX;}
 for(int lp=tid;lp<LBASE;lp+=256){int v=counts[cls*LBASE+lp];if(v<=0)continue;
  int d=edge%8;int p=lp<RESIDENTS?slots[d*RESIDENTS+lp]:E+slots[d*RESIDENTS+(lp-RESIDENTS)/RESIDENTS]*E+slots[d*RESIDENTS+(lp-RESIDENTS)%RESIDENTS];
  for(int j=0;j<g;j++)if(v>value[j]||(v==value[j]&&p<idx[j])){for(int q=g-1;q>j;q--){value[q]=value[q-1];idx[q]=idx[q-1];}value[j]=v;idx[j]=p;break;}
 }
 __shared__ int sv[256],si[256];
 for(int pick=0;pick<g;pick++){
  sv[tid]=value[0];si[tid]=idx[0];__syncthreads();
  for(int d=128;d;d/=2){if(tid<d&&(sv[tid+d]>sv[tid]||(sv[tid+d]==sv[tid]&&si[tid+d]<si[tid]))){sv[tid]=sv[tid+d];si[tid]=si[tid+d];}__syncthreads();}
  int winner=si[0];if(tid==0)top[cls*g+pick]=sv[0]>0?winner:-1;
  if(idx[0]==winner)for(int j=0;j<g;j++){value[j]=j+1<g?value[j+1]:0;idx[j]=j+1<g?idx[j+1]:INT_MAX;}
  __syncthreads();
 }
}
__global__ void assemble_kernel(const int*top,int*patterns,int*lengths,int g){
 int a=blockIdx.x/8,b=blockIdx.x%8;if(a>=b||threadIdx.x)return;
 int ps[MAXP],n=0;
 for(int dir=0;dir<2;dir++){int u=dir?b:a,v=dir?a:b;
  for(int gain=0;gain<2;gain++)for(int size=0;size<2;size++)for(int k=0;k<g;k++){
   int p=top[(((u*8+v)*2+gain)*2+size)*g+k];if(p<0)continue;int val=dir*BASE+p;
   bool seen=false;for(int j=0;j<n;j++)seen|=ps[j]==val;if(!seen)ps[n++]=val;
  }
 }
 for(int i=1;i<n;i++){int v=ps[i],j=i-1;while(j>=0&&ps[j]>v){ps[j+1]=ps[j];j--;}ps[j+1]=v;}
 for(int i=0;i<n;i++)patterns[(a*8+b)*MAXP+i]=ps[i];lengths[a*8+b]=n;
}
__global__ void propose(const int*x,const int*o,const unsigned char*used,const int*patterns,const int*lengths,U64*keys,int*masks,int*hist,int T,int S,int a,int b,int bits){
 __shared__ int ps[MAXP],np;if(threadIdx.x==0)np=lengths[a*8+b];if(threadIdx.x<MAXP)ps[threadIdx.x]=patterns[(a*8+b)*MAXP+threadIdx.x];__syncthreads();
 int t=blockIdx.x*256+threadIdx.x;if(t>=T)return;
 U64 key=~0ull;int mask=0;
 if(!used[t]){
  int ex[8],ow[8],na=0,nb=0;
  #pragma unroll
  for(int k=0;k<8;k++){ex[k]=x[k*T+t];ow[k]=o[k*T+t];na+=ow[k]==a;nb+=ow[k]==b;}
  int best=-1,bestcat=-1,first=MAXP;
  for(int p=0;p<np;p++){
   int dir=ps[p]/BASE,pat=ps[p]%BASE,old=dir?b:a,dest=dir?a:b;
   int size=pat<E?1:2,e1=pat<E?pat:(pat-E)/E,e2=pat<E?-1:(pat-E)%E,m=0;
   #pragma unroll
   for(int k=0;k<8;k++)if(ow[k]==old&&(ex[k]==e1||ex[k]==e2))m|=1<<k;
   if(__popc((unsigned)m)!=size)continue;
   int gain=int(old!=t/S&&size==(dir?nb:na))-int(dest!=t/S&&(dir?na:nb)==0);if(gain<0)continue;
   first=min(first,p);int score=gain*3+3-size;
   if(score>best){best=score;bestcat=(dir*2+gain)*3+size;mask=m;}
  }
  if(best>=0){key=(U64(bestcat*MAXP+first)<<bits)|t;atomicAdd(hist+bestcat,1);}
 }
 keys[t]=key;masks[t]=mask;
}
// Exact best count for target volume w with size1/2, positive/zero gains.
// Same tie as CPU bounded DP: minimize positive size2, then zero size2.
__device__ int volume(const int*c,int w,int*out){
 int A=c[4],B=c[5],C=c[1],D=c[2];int lo=max(0,w-2*(B+D)),hi=min(A+C,w);
 if((lo&1)!=(w&1))lo++;if((hi&1)!=(w&1))hi--;if(lo>hi)return -100000000;
 int piv[4]={lo,hi,A,w-2*B};int best=-100000000,bp=INT_MAX,bz=INT_MAX,bs=0;
 for(int j=0;j<4;j++)for(int d=-1;d<=1;d++){
  int s=max(lo,min(hi,piv[j]+d));if((s&1)!=(w&1))continue;int q=(w-s)/2;
  int p1=min(A,s),p2=min(B,q),z2=q-p2,v=p1+p2;
  if(v>best||(v==best&&(p2<bp||(p2==bp&&z2<bz)))){best=v;bp=p2;bz=z2;bs=s;}
 }
 if(out){out[1]=bs-min(A,bs);out[2]=bz;out[4]=min(A,bs);out[5]=bp;}
 return best;
}
__global__ void quota_blocks(const int*hist,U64*bestblocks,int T){
 __shared__ U64 scores[256];int cap=min(hist[1]+hist[4]+2*(hist[2]+hist[5]),hist[7]+hist[10]+2*(hist[8]+hist[11]));
 int w=blockIdx.x*256+threadIdx.x;U64 score=0;
 if(w<=cap){int gain=volume(hist,w,nullptr)+volume(hist+6,w,nullptr);if(gain>=0)score=(U64(gain)<<32)|(0xffffffffu-unsigned(w));}
 scores[threadIdx.x]=score;__syncthreads();for(int d=128;d;d/=2){if(threadIdx.x<d)scores[threadIdx.x]=max(scores[threadIdx.x],scores[threadIdx.x+d]);__syncthreads();}
 if(threadIdx.x==0)bestblocks[blockIdx.x]=scores[0];
}
__global__ void quota_finish(const int*hist,const U64*bestblocks,int*accept,int*offset,int blocks){
 __shared__ U64 scores[256];U64 v=0;for(int i=threadIdx.x;i<blocks;i+=256)v=max(v,bestblocks[i]);scores[threadIdx.x]=v;__syncthreads();
 for(int d=128;d;d/=2){if(threadIdx.x<d)scores[threadIdx.x]=max(scores[threadIdx.x],scores[threadIdx.x+d]);__syncthreads();}
 if(threadIdx.x==0){for(int k=0;k<12;k++)accept[k]=0;int w=int(0xffffffffu-unsigned(scores[0]));if((scores[0]>>32)>0){volume(hist,w,accept);volume(hist+6,w,accept+6);}int n=0;for(int k=0;k<12;k++){offset[k]=n;n+=hist[k];}}
}
__global__ void commit_kernel(const U64*keys,const int*masks,const int*accept,const int*offset,unsigned char*used,int*o,int T,int a,int b,int bits){
 int i=blockIdx.x*256+threadIdx.x;if(i>=T)return;U64 key=keys[i];if(key==~0ull)return;int cat=int(key>>bits)/MAXP;
 if(i-offset[cat]>=accept[cat])return;int t=int(key&((1ull<<bits)-1)),dest=cat/6?a:b,mask=masks[t];used[t]=1;
 #pragma unroll
 for(int k=0;k<8;k++)if(mask&(1<<k))o[k*T+t]=dest;
}
void prepare_gpu(torch::Tensor routes,torch::Tensor owner,torch::Tensor support,torch::Tensor ex,torch::Tensor ow,torch::Tensor res,torch::Tensor lut,torch::Tensor slots,torch::Tensor counts,torch::Tensor top,torch::Tensor patterns,torch::Tensor lengths,int h,int g){
 SUBSET_CHECK(routes);SUBSET_CHECK(owner);c10::cuda::CUDAGuard guard(routes.device());TORCH_CHECK(g>0&&g<=8&&h>=1&&h<=2,"h 1/2, g 1..8");int T=routes.numel()/8;auto stream=c10::cuda::getCurrentCUDAStream();
 transpose<<<(T*8+255)/256,256,0,stream>>>(routes.data_ptr<int>(),ex.data_ptr<int>(),T);transpose<<<(T*8+255)/256,256,0,stream>>>(owner.data_ptr<int>(),ow.data_ptr<int>(),T);
 resident_kernel<<<2,256,0,stream>>>(support.data_ptr<bool>(),res.data_ptr<int>());slots_kernel<<<8,1,0,stream>>>(support.data_ptr<bool>(),lut.data_ptr<int>(),slots.data_ptr<int>());cudaMemsetAsync(counts.data_ptr(),0,counts.nbytes(),stream);cudaMemsetAsync(top.data_ptr(),255,top.nbytes(),stream);cudaMemsetAsync(lengths.data_ptr(),0,lengths.nbytes(),stream);
 mine_kernel<<<(T+255)/256,256,0,stream>>>(ex.data_ptr<int>(),ow.data_ptr<int>(),res.data_ptr<int>(),lut.data_ptr<int>(),counts.data_ptr<int>(),T,T/8,h);
 select_kernel<<<256,256,0,stream>>>(counts.data_ptr<int>(),slots.data_ptr<int>(),top.data_ptr<int>(),g);assemble_kernel<<<64,1,0,stream>>>(top.data_ptr<int>(),patterns.data_ptr<int>(),lengths.data_ptr<int>(),g);C10_CUDA_KERNEL_LAUNCH_CHECK();
}
size_t temp_bytes(int T){size_t bytes=0;cub::DeviceRadixSort::SortKeys(nullptr,bytes,(U64*)nullptr,(U64*)nullptr,T);return bytes;}
void run_gpu(torch::Tensor owner,torch::Tensor ex,torch::Tensor ow,torch::Tensor patterns,torch::Tensor lengths,torch::Tensor used,torch::Tensor keys0,torch::Tensor keys1,torch::Tensor masks,torch::Tensor hist,torch::Tensor accept,torch::Tensor offsets,torch::Tensor scores,torch::Tensor temp,torch::Tensor output,int rounds){
 SUBSET_CHECK(owner);c10::cuda::CUDAGuard guard(owner.device());int T=owner.numel()/8,bits=0;while((1ull<<bits)<(unsigned)T)bits++;auto stream=c10::cuda::getCurrentCUDAStream();int threads=256,blocks=(T+255)/256,qblocks=(2*T+256)/256;size_t bytes=temp.nbytes();
 transpose<<<(T*8+255)/256,256,0,stream>>>(owner.data_ptr<int>(),ow.data_ptr<int>(),T);
 int as[28],bs[28],n=0;for(int a=0;a<8;a++)for(int b=a+1;b<8;b++){as[n]=a;bs[n++]=b;}
 for(int step=0;step<rounds;step++){
  cudaMemsetAsync(used.data_ptr(),0,used.nbytes(),stream);
  for(int p=0;p<28;p++){int pi=(p+step*7)%28,a=as[pi],b=bs[pi];cudaMemsetAsync(hist.data_ptr(),0,hist.nbytes(),stream);
   propose<<<blocks,threads,0,stream>>>(ex.data_ptr<int>(),ow.data_ptr<int>(),used.data_ptr<unsigned char>(),patterns.data_ptr<int>(),lengths.data_ptr<int>(),(U64*)keys0.data_ptr(),masks.data_ptr<int>(),hist.data_ptr<int>(),T,T/8,a,b,bits);
   cub::DeviceRadixSort::SortKeys(temp.data_ptr(),bytes,(U64*)keys0.data_ptr(),(U64*)keys1.data_ptr(),T,0,bits+11,stream);
   quota_blocks<<<qblocks,256,0,stream>>>(hist.data_ptr<int>(),(U64*)scores.data_ptr(),T);
   quota_finish<<<1,256,0,stream>>>(hist.data_ptr<int>(),(U64*)scores.data_ptr(),accept.data_ptr<int>(),offsets.data_ptr<int>(),qblocks);
   commit_kernel<<<blocks,256,0,stream>>>((U64*)keys1.data_ptr(),masks.data_ptr<int>(),accept.data_ptr<int>(),offsets.data_ptr<int>(),used.data_ptr<unsigned char>(),ow.data_ptr<int>(),T,a,b,bits);
  }
 }
 untranspose<<<(T*8+255)/256,256,0,stream>>>(ow.data_ptr<int>(),output.data_ptr<int>(),T);C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Fused round variant: all directions are considered from one token load.
// This deliberately changes sequential pair scheduling; audit quality separately.
__global__ void flatten_patterns(const int*patterns,const int*lengths,int*flat,int*nflat){
 if(threadIdx.x||blockIdx.x)return;int n=0;
 for(int a=0;a<8;a++)for(int b=a+1;b<8;b++)for(int j=0;j<lengths[a*8+b];j++){
  int val=patterns[(a*8+b)*MAXP+j],dir=val/BASE,pat=val%BASE;
  flat[n++]=((dir?b:a)*8+(dir?a:b))*BASE+pat;
 }*nflat=n;
}
__device__ int rotating_tie(int a,int b,int t,int step){
 int lo=min(a,b),hi=max(a,b),pair=lo*7-lo*(lo-1)/2+hi-lo-1;
 int rem=(pair-t-17*step)%28;if(rem<0)rem+=28;
 int priority=27-rem;bool asc=((t/28+step)&1)==0;
 return 2*priority+int((a<b)==asc);
}
__global__ void propose_parallel(const int*x,const int*o,const int*flat,const int*nflat,U64*keys,int*masks,int*hist,int T,int S,int bits,int step,int policy){
 __shared__ int ps[2048],np;if(threadIdx.x==0)np=*nflat;__syncthreads();
 for(int i=threadIdx.x;i<np;i+=256)ps[i]=flat[i];__syncthreads();
 int t=blockIdx.x*256+threadIdx.x;if(t>=T)return;
 int ex[8],ow[8],occ[8]={0};
 #pragma unroll
 for(int k=0;k<8;k++){ex[k]=x[k*T+t];ow[k]=o[k*T+t];occ[ow[k]]++;}
 int best=-1,bestcat=-1,mask=0;
 for(int p=0;p<np;p++){
  int edge=ps[p]/BASE,pat=ps[p]%BASE,a=edge/8,b=edge%8,size=pat<E?1:2;
  if(occ[a]<size)continue;
  int gain=int(a!=t/S&&size==occ[a])-int(b!=t/S&&occ[b]==0);
  int score=gain*3+3-size;
  if(policy)score=((gain*2+int((size-1)==((t/28+step)&1)))*9+occ[b])*56+rotating_tie(a,b,t,step);
  if(gain<0||score<=best)continue;
  int e1=pat<E?pat:(pat-E)/E,e2=pat<E?-1:(pat-E)%E,m=0;
  #pragma unroll
  for(int k=0;k<8;k++)if(ow[k]==a&&(ex[k]==e1||ex[k]==e2))m|=1<<k;
  if(__popc((unsigned)m)!=size)continue;
  int lo=min(a,b),hi=max(a,b),dir=a> b;
  best=score;bestcat=(lo*8+hi)*12+(dir*2+gain)*3+size;mask=m;
  if(!policy&&best==5)break;
 }
 keys[t]=best>=0?(U64(bestcat)<<bits)|t:~0ull;masks[t]=mask;
 if(best>=0)atomicAdd(hist+bestcat,1);
}
// Store one-based template order, preserving first-template tie behavior.
// The index contains only GPU-selected templates; no CPU-prepared lookup input.
__global__ void index_patterns(const int*flat,const int*nflat,const int*lut,int*member){
 int i=blockIdx.x*256+threadIdx.x;if(i>=*nflat)return;
 int edge=flat[i]/BASE,pat=flat[i]%BASE,b=edge%8;
 int e1=pat<E?pat:(pat-E)/E,e2=pat<E?-1:(pat-E)%E;
 int u=lut[b*E+e1],v=e2<0?-1:lut[b*E+e2];
 int lp=e2<0?u:RESIDENTS+min(u,v)*RESIDENTS+max(u,v);
 member[edge*LBASE+lp]=i+1;
}
__global__ void propose_indexed(const int*x,const int*o,const int*res,const int*lut,const int*member,U64*keys,int*masks,int*hist,int T,int S,int bits,int step,int policy){
 int t=blockIdx.x*256+threadIdx.x;if(t>=T)return;
 int ex[8],ow[8],available[8],occ[8]={0};
 #pragma unroll
 for(int k=0;k<8;k++){ex[k]=x[k*T+t];ow[k]=o[k*T+t];available[k]=res[ex[k]];occ[ow[k]]++;}
 int best=-1,bestorder=INT_MAX,bestcat=-1,mask=0;
 // Each token generates its own singletons/pairs exactly once. Unrelated
 // shortlisted templates never cause a token/expert scan.
 for(int i=0;i<8;i++)for(int j=i;j<8;j++){
  int a=ow[i],size=i==j?1:2;if(i!=j&&ow[j]!=a)continue;
  int possible=available[i]&available[j]&~(1<<a);
  for(int b=0;b<8;b++)if(possible&(1<<b)){
   int gain=int(a!=t/S&&size==occ[a])-int(b!=t/S&&occ[b]==0);if(gain<0)continue;
   int score=gain*3+3-size;
   if(policy)score=((gain*2+int((size-1)==((t/28+step)&1)))*9+occ[b])*56+rotating_tie(a,b,t,step);
   if(score<best)continue;
   int u=lut[b*E+ex[i]],v=lut[b*E+ex[j]];
   int lp=i==j?u:RESIDENTS+min(u,v)*RESIDENTS+max(u,v);
   int order=member[(a*8+b)*LBASE+lp];if(!order||(score==best&&order>=bestorder))continue;
   int lo=min(a,b),hi=max(a,b),dir=a>b;
   best=score;bestorder=order;bestcat=(lo*8+hi)*12+(dir*2+gain)*3+size;mask=(1<<i)|(1<<j);
  }
 }
 keys[t]=best>=0?(U64(bestcat)<<bits)|t:~0ull;masks[t]=mask;
 if(best>=0)warpadd(hist,bestcat);
}
__global__ void quota_parallel(const int*hist,int*accept){
 int edge=blockIdx.x,a=edge/8,b=edge%8;if(a>=b)return;const int*c=hist+edge*12;int*out=accept+edge*12;
 int cap=min(c[1]+c[4]+2*(c[2]+c[5]),c[7]+c[10]+2*(c[8]+c[11]));
 U64 score=0;for(int w=threadIdx.x;w<=cap;w+=256){int gain=volume(c,w,nullptr)+volume(c+6,w,nullptr);if(gain>=0)score=max(score,(U64(gain)<<32)|(0xffffffffu-unsigned(w)));}
 __shared__ U64 scores[256];scores[threadIdx.x]=score;__syncthreads();
 for(int d=128;d;d/=2){if(threadIdx.x<d)scores[threadIdx.x]=max(scores[threadIdx.x],scores[threadIdx.x+d]);__syncthreads();}
 if(threadIdx.x==0){int w=int(0xffffffffu-unsigned(scores[0]));if((scores[0]>>32)>0){volume(c,w,out);volume(c+6,w,out+6);}}
}
__global__ void histogram_offsets(const int*hist,int*offset){
 using Scan=cub::BlockScan<int,256>;
 __shared__ typename Scan::TempStorage storage;
 __shared__ int carry;
 if(threadIdx.x==0)carry=0;__syncthreads();
 // Three contiguous chunks keep both reads and writes coalesced. The carry
 // joins their exclusive scans into exactly the original 768-entry prefix.
 for(int base=0;base<64*12;base+=256){
  int prefix,total;Scan(storage).ExclusiveSum(hist[base+threadIdx.x],prefix,total);
  offset[base+threadIdx.x]=carry+prefix;
  __syncthreads();
  if(threadIdx.x==0)carry+=total;
  __syncthreads();
 }
}
__global__ void commit_parallel(const U64*keys,const int*masks,const int*accept,const int*offset,int*o,int T,int bits){
 int i=blockIdx.x*256+threadIdx.x;if(i>=T)return;U64 key=keys[i];if(key==~0ull)return;
 int cat=int(key>>bits);if(i-offset[cat]>=accept[cat])return;
 int edge=cat/12,dir=(cat%12)/6,dest=dir?edge/8:edge%8,t=int(key&((1ull<<bits)-1)),mask=masks[t];
 #pragma unroll
 for(int k=0;k<8;k++)if(mask&(1<<k))o[k*T+t]=dest;
}
void run_parallel_gpu(torch::Tensor owner,torch::Tensor ex,torch::Tensor ow,torch::Tensor flat,torch::Tensor nflat,torch::Tensor keys0,torch::Tensor keys1,torch::Tensor masks,torch::Tensor hist,torch::Tensor accept,torch::Tensor offsets,torch::Tensor temp,torch::Tensor output,int rounds,int policy){
 c10::cuda::CUDAGuard guard(owner.device());int T=owner.numel()/8,bits=0;while((1ull<<bits)<(unsigned)T)bits++;auto stream=c10::cuda::getCurrentCUDAStream();int blocks=(T+255)/256;size_t bytes=temp.nbytes();
 transpose<<<(T*8+255)/256,256,0,stream>>>(owner.data_ptr<int>(),ow.data_ptr<int>(),T);
 for(int step=0;step<rounds;step++){
  cudaMemsetAsync(hist.data_ptr(),0,hist.nbytes(),stream);cudaMemsetAsync(accept.data_ptr(),0,accept.nbytes(),stream);
  propose_parallel<<<blocks,256,0,stream>>>(ex.data_ptr<int>(),ow.data_ptr<int>(),flat.data_ptr<int>(),nflat.data_ptr<int>(),(U64*)keys0.data_ptr(),masks.data_ptr<int>(),hist.data_ptr<int>(),T,T/8,bits,step,policy);
  cub::DeviceRadixSort::SortKeys(temp.data_ptr(),bytes,(U64*)keys0.data_ptr(),(U64*)keys1.data_ptr(),T,0,bits+10,stream);
  quota_parallel<<<64,256,0,stream>>>(hist.data_ptr<int>(),accept.data_ptr<int>());histogram_offsets<<<1,256,0,stream>>>(hist.data_ptr<int>(),offsets.data_ptr<int>());
  commit_parallel<<<blocks,256,0,stream>>>((U64*)keys1.data_ptr(),masks.data_ptr<int>(),accept.data_ptr<int>(),offsets.data_ptr<int>(),ow.data_ptr<int>(),T,bits);
 }
 untranspose<<<(T*8+255)/256,256,0,stream>>>(ow.data_ptr<int>(),output.data_ptr<int>(),T);C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void flatten_gpu(torch::Tensor patterns,torch::Tensor lengths,torch::Tensor flat,torch::Tensor nflat){auto stream=c10::cuda::getCurrentCUDAStream();flatten_patterns<<<1,1,0,stream>>>(patterns.data_ptr<int>(),lengths.data_ptr<int>(),flat.data_ptr<int>(),nflat.data_ptr<int>());}
void index_gpu(torch::Tensor flat,torch::Tensor nflat,torch::Tensor lut,torch::Tensor member){
 c10::cuda::CUDAGuard guard(flat.device());auto stream=c10::cuda::getCurrentCUDAStream();
 cudaMemsetAsync(member.data_ptr(),0,member.nbytes(),stream);
 index_patterns<<<8,256,0,stream>>>(flat.data_ptr<int>(),nflat.data_ptr<int>(),lut.data_ptr<int>(),member.data_ptr<int>());C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void run_indexed_gpu(torch::Tensor owner,torch::Tensor ex,torch::Tensor ow,torch::Tensor res,torch::Tensor lut,torch::Tensor member,torch::Tensor keys0,torch::Tensor keys1,torch::Tensor masks,torch::Tensor hist,torch::Tensor accept,torch::Tensor offsets,torch::Tensor temp,torch::Tensor output,int rounds,int policy){
 c10::cuda::CUDAGuard guard(owner.device());int T=owner.numel()/8,bits=0;while((1ull<<bits)<(unsigned)T)bits++;auto stream=c10::cuda::getCurrentCUDAStream();int blocks=(T+255)/256;size_t bytes=temp.nbytes();
 transpose<<<(T*8+255)/256,256,0,stream>>>(owner.data_ptr<int>(),ow.data_ptr<int>(),T);
 for(int step=0;step<rounds;step++){
  cudaMemsetAsync(hist.data_ptr(),0,hist.nbytes(),stream);cudaMemsetAsync(accept.data_ptr(),0,accept.nbytes(),stream);
  propose_indexed<<<blocks,256,0,stream>>>(ex.data_ptr<int>(),ow.data_ptr<int>(),res.data_ptr<int>(),lut.data_ptr<int>(),member.data_ptr<int>(),(U64*)keys0.data_ptr(),masks.data_ptr<int>(),hist.data_ptr<int>(),T,T/8,bits,step,policy);
  cub::DeviceRadixSort::SortKeys(temp.data_ptr(),bytes,(U64*)keys0.data_ptr(),(U64*)keys1.data_ptr(),T,0,bits+10,stream);
  quota_parallel<<<64,256,0,stream>>>(hist.data_ptr<int>(),accept.data_ptr<int>());histogram_offsets<<<1,256,0,stream>>>(hist.data_ptr<int>(),offsets.data_ptr<int>());
  commit_parallel<<<blocks,256,0,stream>>>((U64*)keys1.data_ptr(),masks.data_ptr<int>(),accept.data_ptr<int>(),offsets.data_ptr<int>(),ow.data_ptr<int>(),T,bits);
 }
 untranspose<<<(T*8+255)/256,256,0,stream>>>(ow.data_ptr<int>(),output.data_ptr<int>(),T);C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("prepare",&prepare_gpu);m.def("run",&run_gpu);m.def("temp_bytes",&temp_bytes);m.def("flatten",&flatten_gpu);m.def("run_parallel",&run_parallel_gpu);m.def("index",&index_gpu);m.def("run_indexed",&run_indexed_gpu);}
