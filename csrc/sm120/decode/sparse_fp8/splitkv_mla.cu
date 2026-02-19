// SM120 Sparse FP8 Decode MLA Kernel — single CTA, mma.sync, K-tiled
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cute/tensor.hpp>
#include "flashmla_utils.h"
#include "params.h"

using namespace cute;
using bf16 = cutlass::bfloat16_t;
using fp8 = cutlass::float_e4m3_t;

namespace sm120 {

static constexpr int SP_THREADS = 128;
static constexpr int SP_BLOCK_M = 64;
static constexpr int SP_TOPK_BS = 64;
static constexpr int SP_PAGE_BS = 64;
static constexpr int SP_QUANT_TILE = 128;
static constexpr int SP_HDK = 576;
static constexpr int SP_HDV = 512;
static constexpr int SP_NOPE = SP_HDV;
static constexpr int SP_ROPE = SP_HDK - SP_HDV;
static constexpr int SP_NSCALES = SP_NOPE / SP_QUANT_TILE;
static constexpr int SP_KTILE = 256;
static constexpr int SP_NKTILES = 3;
static constexpr int SP_LAST_KT = SP_HDK - 2 * SP_KTILE;
static constexpr int SP_HALFV = SP_HDV / 2;
static constexpr float SP_MAX_INIT = -1e30f;

using MmaAtom = MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>;
using MmaQK = decltype(make_tiled_mma(MmaAtom{}, Layout<Shape<_4,_1,_1>>{}));
using MmaPV = decltype(make_tiled_mma(MmaAtom{}, Layout<Shape<_4,_1,_1>>{}));
using SwzAtom = decltype(composition(Swizzle<3,3,3>{}, Layout<Shape<_8,_64>,Stride<_64,_1>>{}));
using SLQ = decltype(tile_to_shape(SwzAtom{}, Shape<Int<SP_BLOCK_M>,Int<SP_KTILE>>{}));
using SLK = decltype(tile_to_shape(SwzAtom{}, Shape<Int<SP_TOPK_BS>,Int<SP_KTILE>>{}));
using SLP = decltype(tile_to_shape(SwzAtom{}, Shape<Int<SP_BLOCK_M>,Int<SP_TOPK_BS>>{}));
// V half: (HALF_V, TOPK_BS) for TN B operand (N, K) — V loaded transposed
using SLV = decltype(tile_to_shape(SwzAtom{}, Shape<Int<SP_HALFV>,Int<SP_TOPK_BS>>{}));

struct SpFp8Plan {
    cute::array_aligned<bf16, cute::cosize_v<SLQ>> bm;
    cute::array_aligned<bf16, cute::cosize_v<SLK>> ba;
    float sM[SP_BLOCK_M], sL[SP_BLOCK_M];
    bool valid[SP_TOPK_BS];
};

__forceinline__ __device__ int sp_row(int lr, int t) {
    return (t/32)*16 + lr*8 + ((t%32)/4);
}

__global__ void __launch_bounds__(SP_THREADS, 1, 1)
sp_fp8_kernel(__grid_constant__ const DecodingParams p) {
#if IS_SM120
    const int hbi = blockIdx.x, sqi = blockIdx.y, pi = blockIdx.z, t = threadIdx.x;
    extern __shared__ char sb[];
    SpFp8Plan& pl = *reinterpret_cast<SpFp8Plan*>(sb);
    if (t < SP_BLOCK_M) { pl.sM[t] = SP_MAX_INIT; pl.sL[t] = 0.f; }
    __syncthreads();

    int* tsp = p.tile_scheduler_metadata_ptr + pi * TileSchedulerMetaDataSize;
    int4 ts = __ldg(reinterpret_cast<int4*>(tsp));
    if (ts.x >= p.b || ts.x < 0) return;
    int bnspi = __ldg(tsp + 4);

    MmaQK mqk; MmaPV mpv;

    #pragma unroll 1
    for (int bi = ts.x; bi <= ts.z; ++bi) {
        int nspi = bi == ts.x ? bnspi : 0;
        int sb_ = bi == ts.x ? ts.y : 0;
        int topk_blocks = (p.topk + SP_TOPK_BS - 1) / SP_TOPK_BS;
        int eb_ = bi == ts.z ? ts.w : topk_blocks;
        bool nosplit = sb_ == 0 && eb_ == topk_blocks;
        int* gIdx = p.indices_ptr + bi*p.indices_batch_stride + sqi*p.indices_row_stride;
        bf16* qp = (bf16*)p.q_ptr + bi*p.q_batch_stride + sqi*p.q_head_per_hk*p.q_row_stride + hbi*SP_BLOCK_M*p.q_row_stride;

        auto rOL = partition_fragment_C(mpv, Shape<Int<SP_BLOCK_M>,Int<SP_HALFV>>{});
        auto rOR = partition_fragment_C(mpv, Shape<Int<SP_BLOCK_M>,Int<SP_HALFV>>{});
        cute::fill(rOL, 0.f); cute::fill(rOR, 0.f);
        float rL[2]={0.f,0.f}, rM[2]={SP_MAX_INIT,SP_MAX_INIT};
        if (t < SP_BLOCK_M) { pl.sM[t] = SP_MAX_INIT; pl.sL[t] = 0.f; }
        __syncthreads();

        #pragma unroll 1
        for (int bk = sb_; bk < eb_; ++bk) {
            if (t < SP_TOPK_BS) { int idx = __ldg(gIdx + bk*SP_TOPK_BS + t); pl.valid[t] = (idx != -1); }
            __syncthreads();

            // QK^T K-tiled
            auto rP = partition_fragment_C(mqk, Shape<Int<SP_BLOCK_M>,Int<SP_TOPK_BS>>{});
            cute::fill(rP, 0.f);
            #pragma unroll 1
            for (int kt = 0; kt < SP_NKTILES; ++kt) {
                int ko = kt*SP_KTILE, ksz = (kt==SP_NKTILES-1) ? SP_LAST_KT : SP_KTILE;
                auto sQ = make_tensor(make_smem_ptr((bf16*)pl.bm.data()), SLQ{});
                auto sK = make_tensor(make_smem_ptr((bf16*)pl.ba.data()), SLK{});
                // Load Q tile
                for (int i = t; i < SP_BLOCK_M*ksz; i += SP_THREADS) {
                    int r=i/ksz, c=i%ksz; sQ(r,c) = qp[r*p.q_row_stride+ko+c];
                }
                if (ksz < SP_KTILE) for (int i=t; i<SP_BLOCK_M*(SP_KTILE-ksz); i+=SP_THREADS) {
                    int r=i/(SP_KTILE-ksz); sQ(r, ksz+i%(SP_KTILE-ksz)) = bf16(0.f);
                }
                // Load K tile with FP8 dequant
                for (int tok=t; tok<SP_TOPK_BS; tok+=SP_THREADS) {
                    int tidx = __ldg(gIdx + bk*SP_TOPK_BS + tok);
                    for (int c=0; c<SP_KTILE; ++c) {
                        int d = ko+c;
                        bf16 v = bf16(0.f);
                        if (tidx != -1 && d < SP_HDK) {
                            int blki = tidx/SP_PAGE_BS, reli = (tidx+SP_PAGE_BS)%SP_PAGE_BS;
                            fp8* gKb = (fp8*)p.k_ptr + blki*p.k_batch_stride + reli*p.k_row_stride;
                            if (d < SP_NOPE) {
                                float sc = __ldg((float*)(gKb+SP_NOPE) + d/SP_QUANT_TILE);
                                v = bf16(float(gKb[d])*sc);
                            } else {
                                bf16* rp = (bf16*)(gKb+SP_NOPE+SP_NSCALES*sizeof(float));
                                v = rp[d-SP_NOPE];
                            }
                        }
                        sK(tok, c) = v;
                    }
                    if (ksz<SP_KTILE) for (int c=ksz;c<SP_KTILE;++c) sK(tok,c)=bf16(0.f);
                }
                __syncthreads();
                auto th=mqk.get_slice(t); auto tsQ=th.partition_A(sQ); auto tsK=th.partition_B(sK);
                auto rA=th.partition_fragment_A(sQ); auto rB=th.partition_fragment_B(sK);
                CUTLASS_PRAGMA_UNROLL
                for (int k=0;k<size<2>(tsQ);++k) { cute::copy(tsQ(_,_,k),rA(_,_,k)); cute::copy(tsK(_,_,k),rB(_,_,k)); cute::gemm(mqk,rA(_,_,k),rB(_,_,k),rP); }
                __syncthreads();
            }

            // Softmax
            CUTLASS_PRAGMA_UNROLL
            for (int lr=0;lr<2;++lr) {
                int lane=t%32; float cm=-INFINITY;
                CUTLASS_PRAGMA_UNROLL
                for (int i=lr?2:0; i<size(rP); i+=4) {
                    int tk=(i/4)*8+(lane%4)*2;
                    if (!pl.valid[min(tk,SP_TOPK_BS-1)]) rP(i)=-INFINITY;
                    if (tk+1<SP_TOPK_BS && !pl.valid[tk+1]) rP(i+1)=-INFINITY;
                    cm=max(cm,max(rP(i),rP(i+1)));
                }
                cm=max(cm,__shfl_xor_sync(0xffffffff,cm,1));
                cm=max(cm,__shfl_xor_sync(0xffffffff,cm,2));
                cm*=p.scale_softmax_log2;
                float om=rM[lr]; rM[lr]=max(cm,om); float s4=exp2f(om-rM[lr]);
                CUTLASS_PRAGMA_UNROLL
                for (int i=lr?2:0;i<size(rOL);i+=4) { rOL(i)*=s4;rOL(i+1)*=s4;rOR(i)*=s4;rOR(i+1)*=s4; }
                float cs=0.f;
                CUTLASS_PRAGMA_UNROLL
                for (int i=lr?2:0;i<size(rP);i+=4) {
                    rP(i)=exp2f(rP(i)*p.scale_softmax_log2-rM[lr]);
                    rP(i+1)=exp2f(rP(i+1)*p.scale_softmax_log2-rM[lr]);
                    cs+=rP(i)+rP(i+1);
                }
                rL[lr]=rL[lr]*s4+cs;
            }

            // PV two halves
            auto dopv=[&](int voff, auto& rOh) {
                auto sP=make_tensor(make_smem_ptr((bf16*)pl.bm.data()),SLP{});
                auto sV=make_tensor(make_smem_ptr((bf16*)pl.ba.data()),SLV{});
                auto tq=mqk.get_slice(t); auto tsP=tq.partition_C(sP);
                CUTLASS_PRAGMA_UNROLL
                for (int i=0;i<size(rP);++i) tsP(i)=bf16(rP(i));
                // Load V transposed: sV is (HALF_V, TOPK_BS) for TN B operand (N, K)
                for (int tok=0;tok<SP_TOPK_BS;++tok) {
                    int tidx=__ldg(gIdx+bk*SP_TOPK_BS+tok);
                    for (int c=t;c<SP_HALFV;c+=SP_THREADS) {
                        bf16 v=bf16(0.f);
                        if (tidx!=-1) {
                            int blki=tidx/SP_PAGE_BS,reli=(tidx+SP_PAGE_BS)%SP_PAGE_BS;
                            fp8* gKb=(fp8*)p.k_ptr+blki*p.k_batch_stride+reli*p.k_row_stride;
                            int d=voff+c; float sc=__ldg((float*)(gKb+SP_NOPE)+d/SP_QUANT_TILE);
                            v=bf16(float(gKb[d])*sc);
                        }
                        sV(c,tok)=v;
                    }
                }
                __syncthreads();
                auto tp=mpv.get_slice(t); auto tsP2=tp.partition_A(sP); auto tsV=tp.partition_B(sV);
                auto rA2=tp.partition_fragment_A(sP); auto rB2=tp.partition_fragment_B(sV);
                CUTLASS_PRAGMA_UNROLL
                for (int k=0;k<size<2>(tsP2);++k) { cute::copy(tsP2(_,_,k),rA2(_,_,k)); cute::copy(tsV(_,_,k),rB2(_,_,k)); cute::gemm(mpv,rA2(_,_,k),rB2(_,_,k),rOh); }
                __syncthreads();
            };
            dopv(0,rOL); dopv(SP_HALFV,rOR);
        }

        for (int i=0;i<2;++i) { rL[i]+=__shfl_xor_sync(0xffffffff,rL[i],1); rL[i]+=__shfl_xor_sync(0xffffffff,rL[i],2); rL[i]=(rL[i]==0.f||rL[i]!=rL[i])?1.f:rL[i]; }
        if (t%32%4==0) for (int lr=0;lr<2;++lr) { int r=sp_row(lr,t); if(r<SP_BLOCK_M){pl.sL[r]=rL[lr];pl.sM[r]=rM[lr];} }
        __syncthreads();

        int nv=min(p.q_head_per_hk-hbi*SP_BLOCK_M,SP_BLOCK_M);
        int ssi=sqi*p.q_head_per_hk+hbi*SP_BLOCK_M;
        if (nosplit) {
            bf16* op=(bf16*)p.o_ptr+bi*p.o_batch_stride+ssi*p.o_row_stride;
            auto sOL=make_tensor(make_smem_ptr((bf16*)pl.bm.data()),make_layout(Shape<Int<SP_BLOCK_M>,Int<SP_HALFV>>{},make_stride(Int<SP_HALFV>{},_1{})));
            auto sOR=make_tensor(make_smem_ptr((bf16*)pl.ba.data()),make_layout(Shape<Int<SP_BLOCK_M>,Int<SP_HALFV>>{},make_stride(Int<SP_HALFV>{},_1{})));
            auto tp=mpv.get_slice(t);
            auto tOL=tp.partition_C(sOL); auto tOR=tp.partition_C(sOR);
            for (int i=0;i<size(rOL);++i){tOL(i)=bf16(rOL(i)/rL[i%4>=2]);tOR(i)=bf16(rOR(i)/rL[i%4>=2]);}
            __syncthreads();
            for (int i=t;i<nv*SP_HDV;i+=SP_THREADS){int r=i/SP_HDV,c=i%SP_HDV;op[r*p.o_row_stride+c]=(c<SP_HALFV)?sOL(r,c):sOR(r,c-SP_HALFV);}
            float* lp=(float*)p.softmax_lse_ptr+bi*p.q_seq_per_hk+ssi;
            if(t<nv){float cL=pl.sL[t],cM=pl.sM[t];lp[t]=(cL==0.f||cL!=cL)?INFINITY:logf(cL)+cM/(float)M_LOG2E;}
        } else {
            int si=__ldg(p.num_splits_ptr+bi)+nspi;
            float* oa=(float*)p.oaccum_ptr+(si*p.q_seq_per_hk+ssi)*SP_HDV;
            auto tp=mpv.get_slice(t);
            auto gOL=make_tensor(make_gmem_ptr(oa),make_layout(Shape<Int<SP_BLOCK_M>,Int<SP_HALFV>>{},make_stride(Int<SP_HDV>{},_1{})));
            auto gOR=make_tensor(make_gmem_ptr(oa+SP_HALFV),make_layout(Shape<Int<SP_BLOCK_M>,Int<SP_HALFV>>{},make_stride(Int<SP_HDV>{},_1{})));
            auto tOL=tp.partition_C(gOL); auto tOR=tp.partition_C(gOR);
            for(int i=0;i<size(rOL);++i){tOL(i)=rOL(i)/rL[i%4>=2];tOR(i)=rOR(i)/rL[i%4>=2];}
            float* la=(float*)p.softmax_lseaccum_ptr+si*p.q_seq_per_hk+ssi;
            if(t<nv){float cL=pl.sL[t],cM=pl.sM[t];la[t]=(cL==0.f||cL!=cL)?-INFINITY:log2f(cL)+cM;}
        }
        if(bi!=ts.z) __syncthreads();
    }
#else
    if(cute::thread0()){CUTE_INVALID_CONTROL_PATH("SM120 only");}
#endif
}

void run_flash_splitkv_mla_fp8_sparse_kernel(DecodingParams& params, cudaStream_t stream) {
    FLASH_ASSERT(params.h_k == 1);
    FLASH_ASSERT(params.topk % SP_TOPK_BS == 0);
    constexpr size_t smem = sizeof(SpFp8Plan);
    CHECK_CUDA(cudaFuncSetAttribute(sp_fp8_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    int nm = cute::ceil_div(params.q_head_per_hk, SP_BLOCK_M);
    sp_fp8_kernel<<<dim3(nm, params.s_q, params.num_sm_parts), dim3(SP_THREADS), smem, stream>>>(params);
    CHECK_CUDA_KERNEL_LAUNCH();
}

}  // namespace sm120
