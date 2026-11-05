// Native eight-rank reciprocal sampled residency; only samples are collected.
// Unique top-8 expert IDs/token, E=352, d=256, top2/class, B<=44.
// Every rank runs the same global greedy. No independent per-pair budgets.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cub/block/block_reduce.cuh>
#include <cstdint>

namespace {
constexpr int R=8,E=352,K=8,D=256,PAIRS=28,WORDS=6,CLASSES=36,BUCKETS=576,MAXP=2688;
constexpr int PACKET_WIDTH=18,PACKET_SIZE=D*PACKET_WIDTH+1;
constexpr int BAD_COPIES=1,BAD_SAMPLE=2,BAD_OWNERSHIP=4,UNSUPPORTED_SEED=8,BUDGET_OVERFLOW=16;
using Word=unsigned long long;

__device__ __forceinline__ void pair_ids(int p,int& a,int& b) {
  a=0;while(p>=7-a){p-=7-a;++a;}b=a+1+p;
}
__device__ __forceinline__ int cls(int dir,int size,int gain){return (dir*9+size)*2+gain;}
__device__ __forceinline__ int bucket(int a,int b,int size){return (a*8+b)*9+size;}

// Non-owned rows are zero so one SUM collective recovers only d sampled rows.
// Last two row fields are presence and malformed-sample markers. The tail is
// a local copy-list validity flag; SUM makes any rank's error visible to all.
__global__ void gather_local(const int* routes,const int* owner,const int64_t* ids,
 const int* etc,int* packet,int* status,int S,int rank,int B) {
  int t=threadIdx.x;int64_t local=ids[t]-int64_t(rank)*S;bool owns=local>=0&&local<S;
  int bad=0;
  #pragma unroll
  for(int k=0;k<K;++k){int e=owns?routes[local*K+k]:0,o=owns?owner[local*K+k]:0;
    packet[t*PACKET_WIDTH+k]=e;packet[t*PACKET_WIDTH+K+k]=o;
    if(owns){bad|=e<0||e>=E||o<0||o>=R;
      for(int j=0;j<k;++j)bad|=routes[local*K+j]==e;}}
  packet[t*PACKET_WIDTH+16]=int(owns);packet[t*PACKET_WIDTH+17]=bad;
  if(t==0){int invalid=0;
    for(int r=0;r<R;++r)for(int j=0;j<B;++j){int e=etc[r*B+j];
      invalid|=e < -1||e>=E||(e>=0&&e/44==r);
      if(e>=0)for(int k=0;k<j;++k)invalid|=etc[r*B+k]==e;}
    packet[D*PACKET_WIDTH]=invalid;*status=0;
  }
}

__global__ void mandatory_validate(const int* packet,const int* etc,int* sampled,
 bool* support,int* extra,int* status,int B) {
  int t=threadIdx.x;
  for(int i=t;i<D*16;i+=256){int j=i/16,k=i%16;sampled[i]=packet[j*PACKET_WIDTH+k];}
  for(int i=t;i<R*E;i+=256){int r=i/E,e=i%E;bool yes=e/44==r;
    for(int j=0;j<B;++j)yes|=etc[r*B+j]==e;support[i]=yes;}
  if(t<R){int n=0;for(int j=0;j<B;++j){int e=etc[t*B+j];bool valid=e>=0&&e<E&&e/44!=t;
    for(int k=0;k<j;++k)if(etc[t*B+k]==e)valid=false;n+=valid;}extra[t]=n;
    if(n>B)atomicOr(status,BUDGET_OVERFLOW);}
  if(t==0&&packet[D*PACKET_WIDTH])atomicOr(status,BAD_COPIES);
  __syncthreads();
  int bad=0;
  if(packet[t*PACKET_WIDTH+16]!=1)bad|=BAD_OWNERSHIP;
  if(packet[t*PACKET_WIDTH+17])bad|=BAD_SAMPLE;
  for(int k=0;k<K;++k){int e=sampled[t*16+k],o=sampled[t*16+K+k];
    if(e<0||e>=E||o<0||o>=R)bad|=BAD_SAMPLE;
    else if(!support[o*E+e])bad|=UNSUPPORTED_SEED;
  }
  if(bad)atomicOr(status,bad);
}

__global__ void sample_records(const int* sampled,const int64_t* ids,const bool* support,const int* extra,
 int* classes,Word* sig0,Word* sig1,Word* masks,const int* status,int S,int B) {
  int p=blockIdx.x,t=threadIdx.x,a,b;pair_ids(p,a,b);int idx=p*D+t,src=int(ids[t]/S);
  if(*status){classes[idx]=-1;sig0[idx]=sig1[idx]=0;
    for(int w=0;w<WORDS;++w)masks[idx*WORDS+w]=0;return;}
  int ga[8],gb[8],na=0,nb=0;
  #pragma unroll
  for(int k=0;k<K;++k){int e=sampled[t*16+k],o=sampled[t*16+8+k];if(o==a)ga[na++]=e;if(o==b)gb[nb++]=e;}
  int va=-2,vb=-2;
  if(na){int v=int(a!=src)-int(b!=src&&nb==0),n=0;
    for(int k=0;k<na;++k)n+=!support[b*E+ga[k]];if(v>=0&&n<=B-extra[b])va=v;}
  if(nb){int v=int(b!=src)-int(a!=src&&na==0),n=0;
    for(int k=0;k<nb;++k)n+=!support[a*E+gb[k]];if(v>=0&&n<=B-extra[a])vb=v;}
  int dir=-1;
  if(va>vb)dir=0;else if(vb>va)dir=1;else if(va>=0){
    unsigned tie=unsigned(ids[t])+unsigned(a)*0x9e3779b9u+unsigned(b)*0x85ebca6bu;
    tie^=tie>>16;tie*=0x7feb352du;tie^=tie>>15;dir=tie&1u;}
  #pragma unroll
  for(int w=0;w<WORDS;++w)masks[idx*WORDS+w]=0;
  if(dir<0||(dir?vb:va)<0){classes[idx]=-1;sig0[idx]=sig1[idx]=0;return;}
  int n=dir?nb:na,g[8];
  #pragma unroll
  for(int k=0;k<K;++k)g[k]=k<n?(dir?gb[k]:ga[k]):511;
  for(int k=1;k<n;++k){int v=g[k],j=k-1;while(j>=0&&g[j]>v){g[j+1]=g[j];--j;}g[j+1]=v;}
  Word hi=0;for(int k=0;k<7;++k)hi=(hi<<9)|Word(g[k]);
  sig0[idx]=hi;sig1[idx]=Word(g[7]);classes[idx]=cls(dir,n,dir?vb:va);
  for(int k=0;k<n;++k)masks[idx*WORDS+g[k]/64]|=Word(1)<<(g[k]%64);
}

__global__ void record_frequencies(const int* classes,const Word* sig0,const Word* sig1,int* freq) {
  __shared__ int c[D];__shared__ Word h[D],l[D];
  int t=threadIdx.x,base=blockIdx.x*D;c[t]=classes[base+t];h[t]=sig0[base+t];l[t]=sig1[base+t];
  __syncthreads();int n=0,first=D;
  if(c[t]>=0)for(int j=0;j<D;++j)if(c[t]==c[j]&&h[t]==h[j]&&l[t]==l[j]){++n;first=min(first,j);}
  freq[base+t]=first==t?n:0;
}

struct Choice {Word hi,lo;int freq,id;};
struct TopTwo {Choice a,b;int first;};
__device__ __forceinline__ Choice empty_choice(){return {0,0,0,-1};}
__device__ __forceinline__ bool better(Choice a,Choice b){
  return a.id>=0&&(b.id<0||a.freq>b.freq||(a.freq==b.freq&&(a.hi<b.hi||(a.hi==b.hi&&a.lo<b.lo))));
}
struct MergeTop {
 __device__ __forceinline__ TopTwo operator()(TopTwo x,TopTwo y)const{
   Choice a=x.a,b=x.b;
   if(better(y.a,a)){b=a;a=y.a;}else if(better(y.a,b))b=y.a;
   if(better(y.b,a)){b=a;a=y.b;}else if(better(y.b,b))b=y.b;
   return {a,b,min(x.first,y.first)};
 }
};
__global__ void class_top2(const int* classes,const int* freq,const Word* sig0,const Word* sig1,
 int* tops,int* first) {
  using Reduce=cub::BlockReduce<TopTwo,D>;
  __shared__ typename Reduce::TempStorage tmp;
  int cc=blockIdx.x,c=cc%CLASSES,base=(cc/CLASSES)*D,t=threadIdx.x,i=base+t;
  bool match=classes[i]==c;Choice v=empty_choice();
  if(match&&freq[i]>0)v={sig0[i],sig1[i],freq[i],i};
  TopTwo out=Reduce(tmp).Reduce(TopTwo{v,empty_choice(),match?t:D},MergeTop());
  if(t==0){tops[cc*2]=out.a.id;tops[cc*2+1]=out.b.id;first[cc]=out.first;}
}

// Reference's dict insertion order matters: gain0/gain1 categories are appended
// in order of their first sample occurrence, then (-frequency, expert tuple).
__global__ void assemble(const int* tops,const int* first,const int* classes,int* records,
 int* sizes,int* packages,int* package_count) {
  if(threadIdx.x)return;
  for(int key=0;key<BUCKETS;++key){sizes[key]=0;for(int j=0;j<4;++j)records[key*4+j]=-1;}
  int np=0;
  for(int p=0;p<PAIRS;++p){int a,b;pair_ids(p,a,b);
    for(int size=1;size<=K;++size){
      for(int dir=0;dir<2;++dir){int key=bucket(dir?b:a,dir?a:b,size),base=p*CLASSES;
        int c0=base+cls(dir,size,0),c1=base+cls(dir,size,1),gfirst=first[c0]<=first[c1]?0:1;
        for(int pass=0;pass<2;++pass){int c=base+cls(dir,size,pass?1-gfirst:gfirst);
          for(int j=0;j<2;++j){int rec=tops[c*2+j];if(rec>=0)records[key*4+sizes[key]++]=rec;}}
      }
      int ka=bucket(a,b,size),kb=bucket(b,a,size);
      for(int ia=0;ia<sizes[ka];++ia)for(int ib=0;ib<sizes[kb];++ib){
        int ra=records[ka*4+ia],rb=records[kb*4+ib];
        if((classes[ra]&1)+(classes[rb]&1)==0)continue;
        int* q=packages+np*5;q[0]=a;q[1]=b;q[2]=size;q[3]=ra;q[4]=rb;++np;
      }
    }
  }
  *package_count=np;
}

__device__ __forceinline__ int match_gain(int x0,int x1,int y0,int y1){
  int n=min(min(x0+x1,y0+y1),max(x1,y1));return min(n,x1)+min(n,y1);
}
__device__ __forceinline__ bool enabled(const Word* mask,const Word* resident){
  bool ok=true;
  #pragma unroll
  for(int w=0;w<WORDS;++w)ok&=(mask[w]&~resident[w])==0;
  return ok;
}
struct Score{int gain,cost,idx;};
struct BestScore {
 __device__ __forceinline__ Score operator()(Score a,Score b)const{
   if(b.gain<=0)return a;if(a.gain<=0)return b;
   int lhs=b.gain*a.cost,rhs=a.gain*b.cost;
   if(lhs!=rhs)return lhs>rhs?b:a;
   if(a.gain!=b.gain)return b.gain>a.gain?b:a;
   if(a.cost!=b.cost)return b.cost<a.cost?b:a;
   return b.idx<a.idx?b:a;
 }
};
__global__ void global_greedy(const Word* masks,const int* freq,const int* classes,const int* records,
 const int* sizes,const int* packages,const int* package_count,bool* support,int* extra,
 int* output_etc,int64_t* history,int* history_count,const int* status,int B) {
  using Reduce=cub::BlockReduce<Score,256>;
  __shared__ typename Reduce::TempStorage tmp;
  __shared__ Word resident[R][WORDS];__shared__ int current_extra[R],e0[BUCKETS],e1[BUCKETS];
  __shared__ Score winner;__shared__ int accepted;
  int t=threadIdx.x;
  if(t<R*WORDS){int r=t/WORDS,w=t%WORDS;Word m=0;
    for(int bit=0;bit<64;++bit){int e=w*64+bit;if(e<E&&support[r*E+e])m|=Word(1)<<bit;}resident[r][w]=m;}
  if(t<R)current_extra[t]=extra[t];if(t==0)accepted=0;__syncthreads();
  for(int iteration=0;iteration<R*B&&*status==0;++iteration){
    for(int key=t;key<BUCKETS;key+=256){int x0=0,x1=0,b=(key/9)%R;
      for(int j=0;j<sizes[key];++j){int rec=records[key*4+j];if(enabled(masks+rec*WORDS,resident[b])){
        if(classes[rec]&1)x1+=freq[rec];else x0+=freq[rec];}}
      e0[key]=x0;e1[key]=x1;
    }
    __syncthreads();Score best{0,0,MAXP};
    for(int p=t;p<*package_count;p+=256){const int* q=packages+p*5;int a=q[0],b=q[1],size=q[2],ra=q[3],rb=q[4];
      Word aa[WORDS],ab[WORDS];int ca=0,cb=0;
      #pragma unroll
      for(int w=0;w<WORDS;++w){aa[w]=masks[rb*WORDS+w]&~resident[a][w];ab[w]=masks[ra*WORDS+w]&~resident[b][w];
        ca+=__popcll(aa[w]);cb+=__popcll(ab[w]);}
      int cost=ca+cb;if(cost==0||current_extra[a]+ca>B||current_extra[b]+cb>B)continue;
      int ka=bucket(a,b,size),kb=bucket(b,a,size),x0=0,x1=0,y0=0,y1=0;
      for(int j=0;j<sizes[ka];++j){int rec=records[ka*4+j];bool ok=true;
        #pragma unroll
        for(int w=0;w<WORDS;++w)ok&=(masks[rec*WORDS+w]&~(resident[b][w]|ab[w]))==0;
        if(ok){if(classes[rec]&1)x1+=freq[rec];else x0+=freq[rec];}}
      for(int j=0;j<sizes[kb];++j){int rec=records[kb*4+j];bool ok=true;
        #pragma unroll
        for(int w=0;w<WORDS;++w)ok&=(masks[rec*WORDS+w]&~(resident[a][w]|aa[w]))==0;
        if(ok){if(classes[rec]&1)y1+=freq[rec];else y0+=freq[rec];}}
      int gain=match_gain(x0,x1,y0,y1)-match_gain(e0[ka],e1[ka],e0[kb],e1[kb]);
      best=BestScore()(best,Score{gain,cost,p});
    }
    Score found=Reduce(tmp).Reduce(best,BestScore());if(t==0)winner=found;__syncthreads();
    if(winner.gain<=0)break;
    if(t==0){const int* q=packages+winner.idx*5;int a=q[0],b=q[1];int64_t* h=history+accepted*18;
      h[0]=winner.idx;h[1]=winner.gain;h[2]=a;h[3]=b;h[4]=q[2];h[5]=winner.cost;
      #pragma unroll
      for(int w=0;w<WORDS;++w){Word aa=masks[q[4]*WORDS+w]&~resident[a][w],ab=masks[q[3]*WORDS+w]&~resident[b][w];
        h[6+w]=int64_t(aa);h[12+w]=int64_t(ab);current_extra[a]+=__popcll(aa);current_extra[b]+=__popcll(ab);
        resident[a][w]|=aa;resident[b][w]|=ab;}
      ++accepted;
    }
    __syncthreads();
  }
  for(int i=t;i<R*E;i+=256){int r=i/E,e=i%E;support[i]=(resident[r][e/64]>>(e%64))&1;}
  if(t<R){extra[t]=current_extra[t];int n=0;
    for(int e=0;e<E;++e)if(e/44!=t&&((resident[t][e/64]>>(e%64))&1)){if(n<B)output_etc[t*B+n]=e;++n;}
    while(n<B)output_etc[t*B+n++]=-1;}
  if(t==0)*history_count=accepted;
}
} // namespace

void gather(const at::Tensor& routes,const at::Tensor& owner,const at::Tensor& ids,const at::Tensor& etc,
 at::Tensor& packet,at::Tensor& status,int64_t S,int64_t rank,int64_t B){
  TORCH_CHECK(routes.is_cuda()&&routes.scalar_type()==at::kInt&&routes.is_contiguous()&&routes.numel()==S*K,"routes: CUDA int32 [S,8]");
  TORCH_CHECK(owner.device()==routes.device()&&owner.scalar_type()==at::kInt&&owner.is_contiguous()&&owner.numel()==routes.numel(),"owner mismatch");
  TORCH_CHECK(ids.device()==routes.device()&&ids.scalar_type()==at::kLong&&ids.is_contiguous()&&ids.numel()==D,"ids: CUDA int64 [256]");
  TORCH_CHECK(etc.device()==routes.device()&&etc.scalar_type()==at::kInt&&etc.is_contiguous()&&etc.numel()==R*B,"etc: CUDA int32 [8,B]");
  TORCH_CHECK(packet.device()==routes.device()&&packet.scalar_type()==at::kInt&&packet.is_contiguous()&&packet.numel()==PACKET_SIZE,"packet mismatch");
  TORCH_CHECK(status.device()==routes.device()&&status.scalar_type()==at::kInt&&status.is_contiguous()&&status.numel()==1,"status mismatch");
  TORCH_CHECK(B>=1&&B<=44&&S>=32&&rank>=0&&rank<R,"requires B in 1..44, S>=32, rank in 0..7");
  c10::cuda::CUDAGuard guard(routes.device());auto stream=c10::cuda::getCurrentCUDAStream();
  gather_local<<<1,D,0,stream>>>(routes.data_ptr<int>(),owner.data_ptr<int>(),ids.data_ptr<int64_t>(),etc.data_ptr<int>(),packet.data_ptr<int>(),status.data_ptr<int>(),S,rank,B);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void select_residency(const at::Tensor& packet,const at::Tensor& ids,const at::Tensor& etc,
 at::Tensor& samples,at::Tensor& support,at::Tensor& extra,at::Tensor& classes,at::Tensor& sig0,
 at::Tensor& sig1,at::Tensor& masks,at::Tensor& freq,at::Tensor& tops,at::Tensor& first,
 at::Tensor& records,at::Tensor& sizes,at::Tensor& packages,at::Tensor& package_count,
 at::Tensor& output_etc,at::Tensor& history,at::Tensor& history_count,at::Tensor& status,int64_t S,int64_t B){
  TORCH_CHECK(packet.is_cuda()&&packet.scalar_type()==at::kInt&&packet.is_contiguous()&&packet.numel()==PACKET_SIZE,"packet mismatch");
  TORCH_CHECK(ids.device()==packet.device()&&ids.scalar_type()==at::kLong&&ids.is_contiguous()&&ids.numel()==D,"ids mismatch");
  TORCH_CHECK(etc.device()==packet.device()&&etc.scalar_type()==at::kInt&&etc.is_contiguous()&&etc.numel()==R*B,"etc mismatch");
  TORCH_CHECK(B>=1&&B<=44&&S>=32,"B must be 1..44 and S>=32");
  TORCH_CHECK(history.numel()>=R*B*18,"history must hold R*B selections");
  c10::cuda::CUDAGuard guard(packet.device());auto stream=c10::cuda::getCurrentCUDAStream();
  auto w=[](at::Tensor& x){return reinterpret_cast<Word*>(x.data_ptr<int64_t>());};
  mandatory_validate<<<1,D,0,stream>>>(packet.data_ptr<int>(),etc.data_ptr<int>(),samples.data_ptr<int>(),support.data_ptr<bool>(),extra.data_ptr<int>(),status.data_ptr<int>(),B);
  sample_records<<<PAIRS,D,0,stream>>>(samples.data_ptr<int>(),ids.data_ptr<int64_t>(),support.data_ptr<bool>(),extra.data_ptr<int>(),classes.data_ptr<int>(),w(sig0),w(sig1),w(masks),status.data_ptr<int>(),S,B);
  record_frequencies<<<PAIRS,D,0,stream>>>(classes.data_ptr<int>(),w(sig0),w(sig1),freq.data_ptr<int>());
  class_top2<<<PAIRS*CLASSES,D,0,stream>>>(classes.data_ptr<int>(),freq.data_ptr<int>(),w(sig0),w(sig1),tops.data_ptr<int>(),first.data_ptr<int>());
  assemble<<<1,32,0,stream>>>(tops.data_ptr<int>(),first.data_ptr<int>(),classes.data_ptr<int>(),records.data_ptr<int>(),sizes.data_ptr<int>(),packages.data_ptr<int>(),package_count.data_ptr<int>());
  global_greedy<<<1,256,0,stream>>>(w(masks),freq.data_ptr<int>(),classes.data_ptr<int>(),records.data_ptr<int>(),sizes.data_ptr<int>(),packages.data_ptr<int>(),package_count.data_ptr<int>(),support.data_ptr<bool>(),extra.data_ptr<int>(),output_etc.data_ptr<int>(),history.data_ptr<int64_t>(),history_count.data_ptr<int>(),status.data_ptr<int>(),B);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("gather",&gather);m.def("select",&select_residency);}
