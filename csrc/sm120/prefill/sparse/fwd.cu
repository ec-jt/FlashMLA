// SM120 Sparse Prefill Attention Kernel — mma.sync + K-tiling for 100KB smem
// Replaces SM90's WGMMA + 3 warpgroups + >200KB smem with:
//   - SM80 mma.sync.m16n8k16 (4 warps, 128 threads)
//   - K-dimension tiling (256 per tile, 3 tiles for D_QK=576)
//   - Direct global loads (no TMA for loads, simpler than cp.async pipeline)
//   - ~65KB peak shared memory

#include "fwd.h"

#include <math_constants.h>
#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>

#include "flashmla_utils.h"

namespace sm120 {

using namespace cute;
using bf16 = cutlass::bfloat16_t;

// ========== Constants ==========
constexpr int D_QK = 576;
constexpr int D_V = 512;
constexpr int B_H = 64;       // Q rows per CTA
constexpr int B_TOPK = 64;    // TopK block size (KV tokens per block)
constexpr int NUM_THREADS = 128;
static constexpr float MAX_INIT_VAL = -1e30f;

// K-dimension tiling: process 576 in chunks of 256
constexpr int K_TILE_SIZE = 256;
constexpr int NUM_K_TILES = 3;
constexpr int LAST_K_TILE_SIZE = D_QK - K_TILE_SIZE * (NUM_K_TILES - 1);  // 64

// ========== MMA Configuration ==========
using MMA_Atom_Arch = MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>;
using TiledMMA_QK = decltype(make_tiled_mma(
    MMA_Atom_Arch{},
    Layout<Shape<_4, _1, _1>>{}
));
using TiledMMA_PV = decltype(make_tiled_mma(
    MMA_Atom_Arch{},
    Layout<Shape<_4, _1, _1>>{}
));

// ========== Shared Memory Layouts ==========
using SmemLayoutAtom = decltype(
    composition(Swizzle<3, 3, 3>{},
                Layout<Shape<_8, _64>, Stride<_64, _1>>{}));

using SmemLayoutQTile = decltype(tile_to_shape(
    SmemLayoutAtom{}, Shape<Int<B_H>, Int<K_TILE_SIZE>>{}));

using SmemLayoutKTile = decltype(tile_to_shape(
    SmemLayoutAtom{}, Shape<Int<B_TOPK>, Int<K_TILE_SIZE>>{}));

using SmemLayoutS = decltype(tile_to_shape(
    SmemLayoutAtom{}, Shape<Int<B_H>, Int<B_TOPK>>{}));

// V half for PV gemm: B operand in TN layout needs (N, K) = (D_V/2, B_TOPK)
// V is loaded transposed from global (row-major K×N) to smem (N×K)
using SmemLayoutVHalf = decltype(tile_to_shape(
    SmemLayoutAtom{}, Shape<Int<D_V / 2>, Int<B_TOPK>>{}));

// ========== Shared Memory Plan ==========
// Peak: sQ(32KB) + sK(32KB) + persistent(~1.5KB) ≈ 65.5KB ✓
struct SharedMemoryPlan {
    cute::array_aligned<bf16, B_H * K_TILE_SIZE> buf_main;     // 32KB
    cute::array_aligned<bf16, B_TOPK * K_TILE_SIZE> buf_aux;   // 32KB
    float smem_sum[B_H];           // 256B — for cross-warp L reduction
    bool is_kv_valid[B_TOPK];      // 64B
};

// ========== Helpers ==========
__forceinline__ __device__ int get_row_idx(int local_row_idx, int tid) {
    return (tid / 32) * 16 + local_row_idx * 8 + ((tid % 32) / 4);
}

template<typename TiledMMA_T, typename SmemA, typename SmemB, typename FragC>
__forceinline__ __device__ void mma_smem_accum(
    TiledMMA_T& tiled_mma, int tid, SmemA& sA, SmemB& sB, FragC& rC
) {
    auto thr = tiled_mma.get_slice(tid);
    auto thr_sA = thr.partition_A(sA);
    auto thr_sB = thr.partition_B(sB);
    auto rA = thr.partition_fragment_A(sA);
    auto rB = thr.partition_fragment_B(sB);
    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < size<2>(thr_sA); ++k) {
        cute::copy(thr_sA(_, _, k), rA(_, _, k));
        cute::copy(thr_sB(_, _, k), rB(_, _, k));
        cute::gemm(tiled_mma, rA(_, _, k), rB(_, _, k), rC);
    }
}

// ========== Main Kernel ==========
__global__ void __launch_bounds__(NUM_THREADS, 1)
sparse_attn_fwd_kernel_sm120(const SparsePrefillParams params) {
#if IS_SM120
    const int q_h_idx = blockIdx.x % (params.h_q / B_H);
    const int s_q_idx = blockIdx.x / (params.h_q / B_H);
    const int tid = threadIdx.x;

    extern __shared__ char wksp_buf[];
    SharedMemoryPlan& plan = *reinterpret_cast<SharedMemoryPlan*>(wksp_buf);

    // Initialize persistent state
    if (tid < B_H) {
        plan.smem_sum[tid] = 0.0f;
    }
    __syncthreads();

    // ========== Pointers ==========
    const bf16* gQ = params.q + s_q_idx * params.stride_q_s_q
                     + q_h_idx * B_H * params.stride_q_h_q;
    const bf16* gKV = params.kv;
    const int* gIndices = params.indices + s_q_idx * params.stride_indices_s_q;
    bf16* gOut = params.out + s_q_idx * params.h_q * D_V + q_h_idx * B_H * D_V;

    const int num_topk_blocks = params.topk / B_TOPK;
    const float sm_scale = params.sm_scale_div_log2;

    // ========== Register accumulators ==========
    TiledMMA_PV tiled_mma_pv{};
    auto rO_lo = partition_fragment_C(tiled_mma_pv, Shape<Int<B_H>, Int<D_V / 2>>{});
    auto rO_hi = partition_fragment_C(tiled_mma_pv, Shape<Int<B_H>, Int<D_V / 2>>{});
    cute::fill(rO_lo, 0.0f);
    cute::fill(rO_hi, 0.0f);

    float rM[2] = {MAX_INIT_VAL, MAX_INIT_VAL};
    float rL[2] = {0.0f, 0.0f};

    // ========== Main loop over topk blocks ==========
    CUTE_NO_UNROLL
    for (int block_idx = 0; block_idx < num_topk_blocks; ++block_idx) {
        // Load validity mask
        if (tid < B_TOPK) {
            int t = __ldg(gIndices + block_idx * B_TOPK + tid);
            plan.is_kv_valid[tid] = (t >= 0 && t < params.s_kv);
        }
        __syncthreads();

        // ---- Phase 1: QK^T via K-tiled mma.sync ----
        TiledMMA_QK tiled_mma_qk{};
        auto rP = partition_fragment_C(tiled_mma_qk, Shape<Int<B_H>, Int<B_TOPK>>{});
        cute::fill(rP, 0.0f);

        CUTE_UNROLL
        for (int k_tile = 0; k_tile < NUM_K_TILES; ++k_tile) {
            int k_offset = k_tile * K_TILE_SIZE;
            int k_size = (k_tile == NUM_K_TILES - 1) ? LAST_K_TILE_SIZE : K_TILE_SIZE;

            auto sQ = make_tensor(make_smem_ptr(plan.buf_main.data()), SmemLayoutQTile{});
            auto sK = make_tensor(make_smem_ptr(plan.buf_aux.data()), SmemLayoutKTile{});

            // Load Q tile
            for (int idx = tid; idx < B_H * k_size; idx += NUM_THREADS) {
                int r = idx / k_size, c = idx % k_size;
                sQ(r, c) = gQ[r * params.stride_q_h_q + k_offset + c];
            }
            if (k_size < K_TILE_SIZE) {
                for (int idx = tid; idx < B_H * (K_TILE_SIZE - k_size); idx += NUM_THREADS) {
                    int r = idx / (K_TILE_SIZE - k_size);
                    int c = k_size + idx % (K_TILE_SIZE - k_size);
                    sQ(r, c) = bf16(0.0f);
                }
            }

            // Load K tile
            for (int idx = tid; idx < B_TOPK * k_size; idx += NUM_THREADS) {
                int r = idx / k_size, c = idx % k_size;
                int t = __ldg(gIndices + block_idx * B_TOPK + r);
                if (t >= 0 && t < params.s_kv) {
                    sK(r, c) = gKV[t * params.stride_kv_s_kv + k_offset + c];
                } else {
                    sK(r, c) = bf16(0.0f);
                }
            }
            if (k_size < K_TILE_SIZE) {
                for (int idx = tid; idx < B_TOPK * (K_TILE_SIZE - k_size); idx += NUM_THREADS) {
                    int r = idx / (K_TILE_SIZE - k_size);
                    int c = k_size + idx % (K_TILE_SIZE - k_size);
                    sK(r, c) = bf16(0.0f);
                }
            }
            __syncthreads();

            mma_smem_accum(tiled_mma_qk, tid, sQ, sK, rP);
            __syncthreads();
        }

        // ---- Mask invalid tokens ----
        CUTE_UNROLL
        for (int i = 0; i < size(rP); ++i) {
            int elem_in_tile = i % 4;
            int tile_n_idx = i / 4;
            int col = tile_n_idx * 8 + (tid % 4) * 2 + (elem_in_tile & 1);
            if (col < B_TOPK && !plan.is_kv_valid[col]) {
                rP(i) = -INFINITY;
            }
        }

        // ---- Online softmax with rescale ----
        CUTE_UNROLL
        for (int row_idx = 0; row_idx < 2; ++row_idx) {
            float cur_max = -INFINITY;
            CUTE_UNROLL
            for (int i = row_idx * 2; i < size(rP); i += 4) {
                cur_max = fmaxf(cur_max, fmaxf(rP(i), rP(i + 1)));
            }
            cur_max = fmaxf(cur_max, __shfl_xor_sync(0xffffffff, cur_max, 1));
            cur_max = fmaxf(cur_max, __shfl_xor_sync(0xffffffff, cur_max, 2));
            cur_max *= sm_scale;

            float new_max = fmaxf(rM[row_idx], cur_max);
            float scale_for_o = exp2f(rM[row_idx] - new_max);

            CUTE_UNROLL
            for (int i = row_idx * 2; i < size(rO_lo); i += 4) {
                rO_lo(i) *= scale_for_o;
                rO_lo(i + 1) *= scale_for_o;
            }
            CUTE_UNROLL
            for (int i = row_idx * 2; i < size(rO_hi); i += 4) {
                rO_hi(i) *= scale_for_o;
                rO_hi(i + 1) *= scale_for_o;
            }

            float cur_sum = 0.0f;
            CUTE_UNROLL
            for (int i = row_idx * 2; i < size(rP); i += 4) {
                rP(i) = exp2f(rP(i) * sm_scale - new_max);
                rP(i + 1) = exp2f(rP(i + 1) * sm_scale - new_max);
                cur_sum += rP(i) + rP(i + 1);
            }

            rL[row_idx] = rL[row_idx] * scale_for_o + cur_sum;
            rM[row_idx] = new_max;
        }

        // ---- Store P to smem as bf16 for PV gemm ----
        {
            auto sP = make_tensor(make_smem_ptr(plan.buf_main.data()), SmemLayoutS{});
            auto thr_qk = TiledMMA_QK{}.get_slice(tid);
            auto thr_sC = thr_qk.partition_C(sP);
            CUTE_UNROLL
            for (int i = 0; i < size(rP); ++i) {
                thr_sC(i) = bf16(rP(i));
            }
        }
        __syncthreads();

        // ---- Phase 2: PV gemm — O += P @ V (two halves) ----
        // P is (M=64, K=64), V is (K=64, N=256) in global memory
        // For TN MMA, B operand needs (N, K) layout in smem
        // So we load V transposed: global (row, col) → smem (col, row)

        // V lower half [0, D_V/2)
        {
            auto sP = make_tensor(make_smem_ptr(plan.buf_main.data()), SmemLayoutS{});
            auto sV = make_tensor(make_smem_ptr(plan.buf_aux.data()), SmemLayoutVHalf{});

            // Load V transposed: global V[k][n] → smem sV(n, k)
            for (int idx = tid; idx < B_TOPK * (D_V / 2); idx += NUM_THREADS) {
                int k = idx / (D_V / 2), n = idx % (D_V / 2);
                int t = __ldg(gIndices + block_idx * B_TOPK + k);
                if (t >= 0 && t < params.s_kv) {
                    sV(n, k) = gKV[t * params.stride_kv_s_kv + n];
                } else {
                    sV(n, k) = bf16(0.0f);
                }
            }
            __syncthreads();

            mma_smem_accum(tiled_mma_pv, tid, sP, sV, rO_lo);
            __syncthreads();
        }

        // V upper half [D_V/2, D_V)
        {
            auto sP = make_tensor(make_smem_ptr(plan.buf_main.data()), SmemLayoutS{});
            auto sV = make_tensor(make_smem_ptr(plan.buf_aux.data()), SmemLayoutVHalf{});

            // Load V transposed: global V[k][n+D_V/2] → smem sV(n, k)
            for (int idx = tid; idx < B_TOPK * (D_V / 2); idx += NUM_THREADS) {
                int k = idx / (D_V / 2), n = idx % (D_V / 2);
                int t = __ldg(gIndices + block_idx * B_TOPK + k);
                if (t >= 0 && t < params.s_kv) {
                    sV(n, k) = gKV[t * params.stride_kv_s_kv + (D_V / 2) + n];
                } else {
                    sV(n, k) = bf16(0.0f);
                }
            }
            __syncthreads();

            mma_smem_accum(tiled_mma_pv, tid, sP, sV, rO_hi);
            __syncthreads();
        }
    }  // end topk block loop

    // ========== Finalize: reduce L, normalize O, store ==========

    // Warp-level reduction for rL
    rL[0] += __shfl_xor_sync(0xffffffff, rL[0], 1);
    rL[0] += __shfl_xor_sync(0xffffffff, rL[0], 2);
    rL[1] += __shfl_xor_sync(0xffffffff, rL[1], 1);
    rL[1] += __shfl_xor_sync(0xffffffff, rL[1], 2);

    // Cross-warp reduction via smem atomics
    if (tid % 4 == 0) {
        for (int row_idx = 0; row_idx < 2; ++row_idx) {
            int real_row = get_row_idx(row_idx, tid);
            if (real_row < B_H) {
                atomicAdd(&plan.smem_sum[real_row], rL[row_idx]);
            }
        }
    }
    __syncthreads();

    // Read back reduced L and broadcast
    if (tid % 4 == 0) {
        for (int row_idx = 0; row_idx < 2; ++row_idx) {
            int real_row = get_row_idx(row_idx, tid);
            if (real_row < B_H) {
                rL[row_idx] = plan.smem_sum[real_row];
            }
        }
    }
    // Broadcast within groups of 4 threads sharing a row
    for (int row_idx = 0; row_idx < 2; ++row_idx) {
        rL[row_idx] = __shfl_sync(0xffffffff, rL[row_idx], (tid % 32) / 4 * 4, 32);
    }

    float inv_L[2];
    inv_L[0] = (rL[0] == 0.0f) ? 1.0f : 1.0f / rL[0];
    inv_L[1] = (rL[1] == 0.0f) ? 1.0f : 1.0f / rL[1];

    // Store output to global memory
    // mma.sync m16n8k16 fragment C layout for 4 warps along M:
    //   warp w handles rows [w*16, w*16+16)
    //   lane l holds: row0 = w*16 + l/4, row1 = w*16 + l/4 + 8
    //   For each n_tile: col = n_tile*8 + (l%4)*2 + {0,1}
    //   Fragment elements: [n_tile*4 + 0] = (row0, col), [n_tile*4 + 1] = (row0, col+1)
    //                      [n_tile*4 + 2] = (row1, col), [n_tile*4 + 3] = (row1, col+1)
    {
        int warp_id = tid / 32;
        int lane_id = tid % 32;
        int row0 = warp_id * 16 + lane_id / 4;
        int row1 = row0 + 8;
        int col_lane_base = (lane_id % 4) * 2;

        // O lower half: columns [0, D_V/2)
        int n_tiles_half = (D_V / 2) / 8;
        for (int nt = 0; nt < n_tiles_half; ++nt) {
            int col = nt * 8 + col_lane_base;
            int fi = nt * 4;

            bf16 v00 = bf16(rO_lo(fi + 0) * inv_L[0]);
            bf16 v01 = bf16(rO_lo(fi + 1) * inv_L[0]);
            bf16 v10 = bf16(rO_lo(fi + 2) * inv_L[1]);
            bf16 v11 = bf16(rO_lo(fi + 3) * inv_L[1]);

            gOut[row0 * D_V + col + 0] = v00;
            gOut[row0 * D_V + col + 1] = v01;
            gOut[row1 * D_V + col + 0] = v10;
            gOut[row1 * D_V + col + 1] = v11;
        }

        // O upper half: columns [D_V/2, D_V)
        for (int nt = 0; nt < n_tiles_half; ++nt) {
            int col = (D_V / 2) + nt * 8 + col_lane_base;
            int fi = nt * 4;

            bf16 v00 = bf16(rO_hi(fi + 0) * inv_L[0]);
            bf16 v01 = bf16(rO_hi(fi + 1) * inv_L[0]);
            bf16 v10 = bf16(rO_hi(fi + 2) * inv_L[1]);
            bf16 v11 = bf16(rO_hi(fi + 3) * inv_L[1]);

            gOut[row0 * D_V + col + 0] = v00;
            gOut[row0 * D_V + col + 1] = v01;
            gOut[row1 * D_V + col + 0] = v10;
            gOut[row1 * D_V + col + 1] = v11;
        }
    }

    // Store max_logits and lse
    {
        int warp_id = tid / 32;
        int lane_id = tid % 32;
        for (int row_idx = 0; row_idx < 2; ++row_idx) {
            int real_row = get_row_idx(row_idx, tid);
            if (real_row < B_H && lane_id % 4 == 0) {
                int g_offset = s_q_idx * params.h_q + q_h_idx * B_H + real_row;
                bool is_no_valid = (rL[row_idx] == 0.0f);
                params.max_logits[g_offset] = is_no_valid ? -INFINITY : rM[row_idx];
                params.lse[g_offset] = is_no_valid ? -INFINITY : log2f(rL[row_idx]) + rM[row_idx];
            }
        }
    }

#else
    if (cute::thread0()) {
        CUTE_INVALID_CONTROL_PATH("This kernel requires SM120");
    }
#endif
}


void run_fwd_kernel(const SparsePrefillParams& params) {
    FLASH_ASSERT(params.h_kv == 1);
    FLASH_ASSERT(params.topk % B_TOPK == 0);
    FLASH_ASSERT(params.topk > 0);
    FLASH_ASSERT(params.h_q % B_H == 0);

    constexpr size_t smem_size = sizeof(SharedMemoryPlan);
    static_assert(smem_size <= 100 * 1024, "SM120 shared memory exceeds 100KB limit");

    auto kernel = &sparse_attn_fwd_kernel_sm120;

    // SM120 has 100KB default smem, but we may still need to set attribute
    // if smem_size > 48KB (the default without attribute)
    if (smem_size > 48 * 1024) {
        CHECK_CUDA(cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    }

    dim3 grid((params.h_q / B_H) * params.s_q, 1, 1);
    dim3 block(NUM_THREADS, 1, 1);

    kernel<<<grid, block, smem_size, params.stream>>>(params);
    CHECK_CUDA_KERNEL_LAUNCH();
}

}  // namespace sm120
