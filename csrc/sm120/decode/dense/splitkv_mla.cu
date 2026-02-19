// SM120 Dense BF16 Decode MLA Kernel — mma.sync + K-tiling for 100KB smem
#include <cutlass/cutlass.h>
#include "flashmla_utils.h"
#include "params.h"
#include "config.h"
#include "traits.h"

using namespace cute;
namespace sm120 {

static constexpr float MAX_INIT_VAL_SM = -1e30f;
static constexpr float MAX_INIT_VAL = -1e33f;

__forceinline__ __device__ int get_row_idx(int local_row_idx, int tid) {
    return (tid / 32) * 16 + local_row_idx * 8 + ((tid % 32) / 4);
}

// Load a (rows × cols) tile from global to swizzled smem, zero-padding if cols < smem_cols
template<typename InputT, typename SmemTensor>
__forceinline__ __device__ void load_tile_to_smem(
    SmemTensor& sTile, const InputT* __restrict__ gptr,
    int rows, int cols, int g_row_stride, int smem_cols, int tid, int num_threads
) {
    for (int idx = tid; idx < rows * cols; idx += num_threads) {
        int r = idx / cols, c = idx % cols;
        sTile(r, c) = gptr[r * g_row_stride + c];
    }
    if (cols < smem_cols) {
        for (int idx = tid; idx < rows * (smem_cols - cols); idx += num_threads) {
            int r = idx / (smem_cols - cols);
            int c = cols + idx % (smem_cols - cols);
            sTile(r, c) = InputT(0.0f);
        }
    }
}

// MMA loop: copy smem partitions to registers, execute gemm
template<typename TiledMMA, typename SmemA, typename SmemB>
__forceinline__ __device__ void mma_smem_accum(
    TiledMMA& tiled_mma, int tid, SmemA& sA, SmemB& sB,
    decltype(partition_fragment_C(TiledMMA{}, Shape<_64, _64>{}))& rC
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

template<typename T>
__global__ void __launch_bounds__(T::NUM_THREADS, 1, 1)
flash_fwd_splitkv_mla_kernel_sm120(
    __grid_constant__ const DecodingParams params
) {
#if IS_SM120
    using InputT = typename T::InputT;
    using Plan = typename T::SharedMemoryPlan;
    constexpr int M = T::BLOCK_SIZE_M;
    constexpr int N = T::PAGE_BLOCK_SIZE;
    constexpr int K_TILE = T::K_TILE_SIZE;
    constexpr int HALF_V = T::HEAD_DIM_V / 2;

    const int m_block_idx = blockIdx.x;
    const int k_head_idx = blockIdx.y;
    const int partition_idx = blockIdx.z;
    const int tid = threadIdx.x;

    extern __shared__ char wksp_buf[];
    Plan& plan = *reinterpret_cast<Plan*>(wksp_buf);

    if (tid < M) plan.smem_sM[tid] = MAX_INIT_VAL_SM;
    __syncthreads();

    cudaGridDependencySynchronize();

    int* ts_ptr = params.tile_scheduler_metadata_ptr + partition_idx * TileSchedulerMetaDataSize;
    int4 ts = *(reinterpret_cast<int4*>(ts_ptr));
    if (ts.x >= params.b || ts.x < 0) return;
    int begin_n_split_idx = *(ts_ptr + 4);

    #pragma unroll 1
    for (int batch_idx = ts.x; batch_idx <= ts.z; ++batch_idx) {
        const int n_split_idx = batch_idx == ts.x ? begin_n_split_idx : 0;
        int seqlen_k = __ldg(params.seqlens_k_ptr + batch_idx);
        int start_blk = batch_idx == ts.x ? ts.y : 0;
        int end_blk = batch_idx == ts.z ? ts.w : cute::ceil_div(seqlen_k, N);
        bool is_no_split = __ldg(params.num_splits_ptr + batch_idx + 1) - __ldg(params.num_splits_ptr + batch_idx) == 1;

        // Causal mask
        int rRB[2];
        if (params.is_causal) {
            auto mask_len = [&](int lsq) {
                int gsq = m_block_idx * M + lsq;
                return gsq < params.q_seq_per_hk ? params.s_q - gsq / params.q_head_per_hk - 1 : 0;
            };
            int cml = mask_len(M - 1);
            int last_blk = cute::ceil_div(seqlen_k - cml, N);
            end_blk = batch_idx == ts.z ? min(ts.w, last_blk) : last_blk;
            for (int lr = 0; lr < 2; ++lr)
                rRB[lr] = min(seqlen_k - mask_len(get_row_idx(lr, tid)), end_blk * N);
        } else {
            rRB[0] = rRB[1] = seqlen_k;
        }

        InputT* q_ptr = (InputT*)params.q_ptr + batch_idx * params.q_batch_stride
                       + m_block_idx * M * params.q_row_stride + k_head_idx * params.q_head_stride;
        int* bt_ptr = params.block_table + batch_idx * params.block_table_batch_stride;
        InputT* o_ptr = (InputT*)params.o_ptr + batch_idx * params.o_batch_stride
                       + m_block_idx * M * params.o_row_stride + k_head_idx * params.o_head_stride;
        float* lse_ptr = (float*)params.softmax_lse_ptr
                        + (batch_idx * params.h_k + k_head_idx) * params.q_seq_per_hk + m_block_idx * M;

        // Accumulators
        typename T::TiledMMA_PV mma_pv;
        auto rO_L = partition_fragment_C(mma_pv, Shape<Int<M>, Int<HALF_V>>{});
        auto rO_R = partition_fragment_C(mma_pv, Shape<Int<M>, Int<HALF_V>>{});
        cute::fill(rO_L, 0.0f); cute::fill(rO_R, 0.0f);
        float rL[2] = {0.0f, 0.0f};

        typename T::TiledMMA_QK mma_qk;

        // Process each KV block
        #pragma unroll 1
        for (int bi = start_blk; bi < end_blk; ++bi) {
            int block_index = __ldg(bt_ptr + bi);
            int start_tok = bi * N;
            bool is_last = (bi == end_blk - 1);
            bool do_oob = is_last || (params.is_causal && bi >= end_blk - 2);
            int valid_win = is_last ? (seqlen_k - start_tok) : N;

            InputT* k_ptr = (InputT*)params.k_ptr + block_index * params.k_batch_stride
                           + k_head_idx * params.k_head_stride;

            // ===== QK^T with K-tiling =====
            auto rP = partition_fragment_C(mma_qk, Shape<Int<M>, Int<N>>{});
            cute::fill(rP, 0.0f);

            #pragma unroll 1
            for (int kt = 0; kt < T::NUM_K_TILES; ++kt) {
                int k_off = kt * K_TILE;
                int k_sz = (kt == T::NUM_K_TILES - 1) ? T::LAST_K_TILE_SIZE : K_TILE;

                auto sQ = make_tensor(make_smem_ptr((InputT*)plan.buf_main.data()), typename T::SmemLayoutQTile{});
                auto sK = make_tensor(make_smem_ptr((InputT*)plan.buf_aux.data()), typename T::SmemLayoutKTile{});

                load_tile_to_smem(sQ, q_ptr + k_off, M, k_sz, params.q_row_stride, K_TILE, tid, T::NUM_THREADS);
                load_tile_to_smem(sK, k_ptr + k_off, N, k_sz, params.k_row_stride, K_TILE, tid, T::NUM_THREADS);
                __syncthreads();

                mma_smem_accum(mma_qk, tid, sQ, sK, rP);
                __syncthreads();
            }

            // ===== Softmax + rescale O =====
            CUTLASS_PRAGMA_UNROLL
            for (int lr = 0; lr < 2; ++lr) {
                int row = get_row_idx(lr, tid);
                int lane = tid % 32;
                float cur_max = MAX_INIT_VAL;
                CUTLASS_PRAGMA_UNROLL
                for (int i = lr ? 2 : 0; i < size(rP); i += 4) {
                    if (do_oob) {
                        int tok = start_tok + (i / 4) * 8 + (lane % 4) * 2;
                        rP(i) = tok < rRB[lr] ? rP(i) : MAX_INIT_VAL;
                        rP(i+1) = tok+1 < rRB[lr] ? rP(i+1) : MAX_INIT_VAL;
                    }
                    cur_max = max(cur_max, max(rP(i), rP(i+1)));
                }
                cur_max = max(cur_max, __shfl_xor_sync(0xffffffff, cur_max, 1));
                cur_max = max(cur_max, __shfl_xor_sync(0xffffffff, cur_max, 2));
                cur_max *= params.scale_softmax_log2;

                float old_max = plan.smem_sM[row];
                float new_max = max(old_max, cur_max);
                float s4old = exp2f(old_max - new_max);
                __syncwarp();
                if (lane % 4 == 0) { plan.smem_sScale[row] = s4old; plan.smem_sM[row] = new_max; }

                // Rescale O
                CUTLASS_PRAGMA_UNROLL
                for (int i = lr ? 2 : 0; i < size(rO_L); i += 4) {
                    rO_L(i) *= s4old; rO_L(i+1) *= s4old;
                    rO_R(i) *= s4old; rO_R(i+1) *= s4old;
                }

                float cur_sum = 0.0f;
                CUTLASS_PRAGMA_UNROLL
                for (int i = lr ? 2 : 0; i < size(rP); i += 4) {
                    rP(i) = exp2f(rP(i) * params.scale_softmax_log2 - new_max);
                    rP(i+1) = exp2f(rP(i+1) * params.scale_softmax_log2 - new_max);
                    cur_sum += rP(i) + rP(i+1);
                }
                rL[lr] = rL[lr] * s4old + cur_sum;
            }

            // ===== PV GEMM (two halves) =====
            // Store P as BF16 in smem, load V half, compute O += P @ V_half
            auto do_pv = [&](InputT* v_ptr, auto& rO_half) {
                auto sP = make_tensor(make_smem_ptr((InputT*)plan.buf_main.data()), typename T::SmemLayoutP{});
                auto sV = make_tensor(make_smem_ptr((InputT*)plan.buf_aux.data()), typename T::SmemLayoutVHalf{});

                // Write P to smem
                auto thr_qk = mma_qk.get_slice(tid);
                auto thr_sP = thr_qk.partition_C(sP);
                CUTLASS_PRAGMA_UNROLL
                for (int i = 0; i < size(rP); ++i) thr_sP(i) = InputT(rP(i));

                // Load V half transposed: sV is (HALF_V, N) for TN B operand (N, K)
                // Global V is (token, feature), smem V is (feature, token)
                for (int idx = tid; idx < N * HALF_V; idx += T::NUM_THREADS) {
                    int tok = idx / HALF_V, feat = idx % HALF_V;
                    sV(feat, tok) = (tok < valid_win) ? v_ptr[tok * params.k_row_stride + feat] : InputT(0.0f);
                }
                __syncthreads();

                // PV gemm
                auto thr_pv = mma_pv.get_slice(tid);
                auto thr_sP2 = thr_pv.partition_A(sP);
                auto thr_sV2 = thr_pv.partition_B(sV);
                auto rA = thr_pv.partition_fragment_A(sP);
                auto rB = thr_pv.partition_fragment_B(sV);
                CUTLASS_PRAGMA_UNROLL
                for (int k = 0; k < size<2>(thr_sP2); ++k) {
                    cute::copy(thr_sP2(_, _, k), rA(_, _, k));
                    cute::copy(thr_sV2(_, _, k), rB(_, _, k));
                    cute::gemm(mma_pv, rA(_, _, k), rB(_, _, k), rO_half);
                }
                __syncthreads();
            };

            do_pv(k_ptr, rO_L);                    // Left half: V[:, 0:256]
            do_pv(k_ptr + HALF_V, rO_R);           // Right half: V[:, 256:512]
        }

        // Reduce rL within warp
        for (int i = 0; i < 2; ++i) {
            rL[i] += __shfl_xor_sync(0xffffffff, rL[i], 1);
            rL[i] += __shfl_xor_sync(0xffffffff, rL[i], 2);
            rL[i] = (rL[i] == 0.0f || rL[i] != rL[i]) ? 1.0f : rL[i];
        }

        if (batch_idx == ts.z) cudaTriggerProgrammaticLaunchCompletion();

        int num_valid = min(params.q_seq_per_hk - m_block_idx * M, M);

        // Store O
        if (is_no_split) {
            // BF16 output via smem
            auto sO_L = make_tensor(make_smem_ptr((InputT*)plan.buf_main.data()),
                make_layout(Shape<Int<M>, Int<HALF_V>>{}, make_stride(Int<HALF_V>{}, _1{})));
            auto sO_R = make_tensor(make_smem_ptr((InputT*)plan.buf_aux.data()),
                make_layout(Shape<Int<M>, Int<HALF_V>>{}, make_stride(Int<HALF_V>{}, _1{})));

            auto thr_pv = mma_pv.get_slice(tid);
            auto thr_sOL = thr_pv.partition_C(sO_L);
            auto thr_sOR = thr_pv.partition_C(sO_R);
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < size(rO_L); ++i) {
                thr_sOL(i) = InputT(rO_L(i) / rL[i % 4 >= 2]);
                thr_sOR(i) = InputT(rO_R(i) / rL[i % 4 >= 2]);
            }
            __syncthreads();

            for (int idx = tid; idx < num_valid * T::HEAD_DIM_V; idx += T::NUM_THREADS) {
                int r = idx / T::HEAD_DIM_V, c = idx % T::HEAD_DIM_V;
                o_ptr[r * params.o_row_stride + c] = (c < HALF_V) ? sO_L(r, c) : sO_R(r, c - HALF_V);
            }

            // Write LSE
            // Store rL to smem for cross-warp reduction
            int lane = tid % 32;
            if (lane % 4 == 0) {
                for (int lr = 0; lr < 2; ++lr) {
                    int row = get_row_idx(lr, tid);
                    if (row < M) plan.sL_reduction_wksp[row] = rL[lr];
                }
            }
            __syncthreads();
            if (tid < num_valid) {
                float cur_L = plan.sL_reduction_wksp[tid];
                float cur_M = plan.smem_sM[tid];
                lse_ptr[tid] = (cur_L == 0.0f || cur_L != cur_L) ? INFINITY : logf(cur_L) + cur_M / (float)M_LOG2E;
            }
        } else {
            int split_idx = params.num_splits_ptr[batch_idx] + n_split_idx;
            float* oacc = (float*)params.oaccum_ptr + ((split_idx * params.h_k + k_head_idx) * params.q_seq_per_hk + m_block_idx * M) * T::HEAD_DIM_V;
            float* lseacc = (float*)params.softmax_lseaccum_ptr + (split_idx * params.h_k + k_head_idx) * params.q_seq_per_hk + m_block_idx * M;

            // Write float O directly to global
            auto thr_pv = mma_pv.get_slice(tid);
            auto gOL = make_tensor(make_gmem_ptr(oacc),
                make_layout(Shape<Int<M>, Int<HALF_V>>{}, make_stride(Int<T::HEAD_DIM_V>{}, _1{})));
            auto gOR = make_tensor(make_gmem_ptr(oacc + HALF_V),
                make_layout(Shape<Int<M>, Int<HALF_V>>{}, make_stride(Int<T::HEAD_DIM_V>{}, _1{})));
            auto thr_gOL = thr_pv.partition_C(gOL);
            auto thr_gOR = thr_pv.partition_C(gOR);
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < size(rO_L); ++i) {
                thr_gOL(i) = rO_L(i) / rL[i % 4 >= 2];
                thr_gOR(i) = rO_R(i) / rL[i % 4 >= 2];
            }

            // Write LSE accum
            int lane = tid % 32;
            if (lane % 4 == 0) {
                for (int lr = 0; lr < 2; ++lr) {
                    int row = get_row_idx(lr, tid);
                    if (row < M) plan.sL_reduction_wksp[row] = rL[lr];
                }
            }
            __syncthreads();
            if (tid < num_valid) {
                float cur_L = plan.sL_reduction_wksp[tid];
                float cur_M = plan.smem_sM[tid];
                lseacc[tid] = (cur_L == 0.0f || cur_L != cur_L) ? -INFINITY : log2f(cur_L) + cur_M;
            }
        }

        if (batch_idx != ts.z) __syncthreads();

        // Reset sM for next batch
        if (tid < M) plan.smem_sM[tid] = MAX_INIT_VAL_SM;
        __syncthreads();
    }
#else
    if (cute::thread0()) {
        CUTE_INVALID_CONTROL_PATH("This kernel only supports sm120");
    }
#endif
}

// ========== Host launcher ==========
template<typename InputT>
void run_flash_splitkv_mla_kernel(DecodingParams& params, cudaStream_t stream) {
    using T = Traits<InputT>;

    auto kernel = &flash_fwd_splitkv_mla_kernel_sm120<T>;
    constexpr size_t smem_size = sizeof(typename T::SharedMemoryPlan);
    CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    const int num_m_block = cute::ceil_div(params.q_seq_per_hk, T::BLOCK_SIZE_M);
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t cfg = {
        dim3(num_m_block, params.h_k, params.num_sm_parts),
        dim3(T::NUM_THREADS, 1, 1),
        smem_size, stream, attrs, 1
    };
    cudaLaunchKernelEx(&cfg, kernel, params);
    CHECK_CUDA_KERNEL_LAUNCH();
}

template void run_flash_splitkv_mla_kernel<cutlass::bfloat16_t>(DecodingParams& params, cudaStream_t stream);

#ifndef FLASH_MLA_DISABLE_FP16
template void run_flash_splitkv_mla_kernel<cutlass::half_t>(DecodingParams& params, cudaStream_t stream);
#endif

}  // namespace sm120
