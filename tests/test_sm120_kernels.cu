// test_sm120_kernels.cu — Correctness validation for all SM120 FlashMLA kernels
//
// Tests:
//   1. SM120 Decode Dense BF16 kernel
//   2. SM120 Decode Sparse FP8 kernel
//   3. SM120 Prefill Sparse kernel
//
// Each test creates synthetic data, runs the kernel, and compares against
// a CPU reference implementation of scaled dot-product attention.
//
// Build:
//   nvcc -std=c++17 -arch=sm_120a \
//     -I ../csrc -I ../csrc/cutlass/include -I ../csrc/sm120 \
//     --expt-relaxed-constexpr --use_fast_math \
//     test_sm120_kernels.cu \
//     ../csrc/sm120/decode/dense/splitkv_mla.cu \
//     ../csrc/sm120/prefill/sparse/fwd.cu \
//     -o test_sm120_kernels
//
// Run:
//   ./test_sm120_kernels

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <cutlass/numeric_types.h>
#include "params.h"

// Forward declarations of SM120 kernel launchers
namespace sm120 {
    template<typename InputT>
    void run_flash_splitkv_mla_kernel(DecodingParams& params, cudaStream_t stream);

    void run_flash_splitkv_mla_fp8_sparse_kernel(DecodingParams& params, cudaStream_t stream);

    void run_fwd_kernel(const SparsePrefillParams& params);
}

using fp8 = cutlass::float_e4m3_t;

using bf16 = cutlass::bfloat16_t;

#define CHECK_CUDA(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while (0)

// ============================================================
// CPU Reference: Scaled Dot-Product Attention
// Q: [M, D_K], K: [N, D_K], V: [N, D_V] -> O: [M, D_V]
// ============================================================
void cpu_attention_ref(
    const float* Q, const float* K, const float* V,
    float* O, float* lse,
    int M, int N, int D_K, int D_V, float scale,
    bool is_causal = false, int causal_offset = 0
) {
    std::vector<float> S(M * N);
    for (int m = 0; m < M; ++m) {
        // QK^T
        float row_max = -1e30f;
        for (int n = 0; n < N; ++n) {
            float dot = 0.0f;
            for (int d = 0; d < D_K; ++d) {
                dot += Q[m * D_K + d] * K[n * D_K + d];
            }
            dot *= scale;
            if (is_causal && (causal_offset + m) < n) {
                dot = -1e30f;
            }
            S[m * N + n] = dot;
            row_max = std::max(row_max, dot);
        }
        // Softmax
        float row_sum = 0.0f;
        for (int n = 0; n < N; ++n) {
            S[m * N + n] = expf(S[m * N + n] - row_max);
            row_sum += S[m * N + n];
        }
        if (row_sum == 0.0f) row_sum = 1.0f;
        for (int n = 0; n < N; ++n) {
            S[m * N + n] /= row_sum;
        }
        // PV
        for (int d = 0; d < D_V; ++d) {
            float acc = 0.0f;
            for (int n = 0; n < N; ++n) {
                acc += S[m * N + n] * V[n * D_V + d];
            }
            O[m * D_V + d] = acc;
        }
        if (lse) {
            lse[m] = logf(row_sum) + row_max;
        }
    }
}

// ============================================================
// Test 1: SM120 Decode Dense BF16
// ============================================================
bool test_decode_dense() {
    printf("=== Test 1: SM120 Decode Dense BF16 ===\n");

    // Small test dimensions
    constexpr int B = 1;          // batch
    constexpr int S_Q = 1;        // query seq len (decode = 1)
    constexpr int H_Q = 1;        // query heads
    constexpr int H_K = 1;        // kv heads
    constexpr int D_K = 576;      // key dim
    constexpr int D_V = 512;      // value dim
    constexpr int PAGE_BS = 64;   // page block size
    constexpr int SEQ_K = 64;     // kv seq len (1 page)
    constexpr int NUM_BLOCKS = 1; // total pages
    constexpr int Q_PER_HK = H_Q / H_K * S_Q;

    float scale = 1.0f / sqrtf((float)D_K);

    // Allocate host data
    std::vector<float> h_Q_f(Q_PER_HK * D_K);
    std::vector<float> h_K_f(PAGE_BS * D_K);
    std::vector<float> h_V_f(PAGE_BS * D_V);
    std::vector<bf16> h_Q(Q_PER_HK * D_K);
    std::vector<bf16> h_KV(PAGE_BS * D_K);  // K and V interleaved in same buffer (K dim = 576, V = first 512)

    srand(123);
    for (int i = 0; i < Q_PER_HK * D_K; ++i) {
        float v = (float)(rand() % 200 - 100) / 200.0f;
        h_Q[i] = bf16(v);
        h_Q_f[i] = float(h_Q[i]);
    }
    for (int i = 0; i < PAGE_BS * D_K; ++i) {
        float v = (float)(rand() % 200 - 100) / 200.0f;
        h_KV[i] = bf16(v);
        float fv = float(h_KV[i]);
        h_K_f[i] = fv;
        if (i % D_K < D_V) {
            int row = i / D_K;
            int col = i % D_K;
            h_V_f[row * D_V + col] = fv;
        }
    }

    // CPU reference
    std::vector<float> h_O_ref(Q_PER_HK * D_V, 0.0f);
    std::vector<float> h_lse_ref(Q_PER_HK, 0.0f);
    cpu_attention_ref(h_Q_f.data(), h_K_f.data(), h_V_f.data(),
                      h_O_ref.data(), h_lse_ref.data(),
                      Q_PER_HK, SEQ_K, D_K, D_V, scale);

    // Allocate device memory
    bf16 *d_Q, *d_KV, *d_O;
    float *d_lse, *d_oaccum, *d_lseaccum;
    int *d_block_table, *d_seqlens_k, *d_tile_sched, *d_num_splits;

    CHECK_CUDA(cudaMalloc(&d_Q, Q_PER_HK * D_K * sizeof(bf16)));
    CHECK_CUDA(cudaMalloc(&d_KV, NUM_BLOCKS * PAGE_BS * D_K * sizeof(bf16)));
    CHECK_CUDA(cudaMalloc(&d_O, Q_PER_HK * D_V * sizeof(bf16)));
    CHECK_CUDA(cudaMalloc(&d_lse, B * H_K * Q_PER_HK * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_oaccum, 2 * H_K * Q_PER_HK * D_V * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_lseaccum, 2 * H_K * Q_PER_HK * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_block_table, B * 1 * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_seqlens_k, B * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_tile_sched, 1 * TileSchedulerMetaDataSize * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_num_splits, (B + 1) * sizeof(int)));

    CHECK_CUDA(cudaMemcpy(d_Q, h_Q.data(), Q_PER_HK * D_K * sizeof(bf16), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_KV, h_KV.data(), PAGE_BS * D_K * sizeof(bf16), cudaMemcpyHostToDevice));

    // Block table: batch 0 -> block 0
    int h_bt = 0;
    CHECK_CUDA(cudaMemcpy(d_block_table, &h_bt, sizeof(int), cudaMemcpyHostToDevice));

    // Seqlens
    int h_seqlen = SEQ_K;
    CHECK_CUDA(cudaMemcpy(d_seqlens_k, &h_seqlen, sizeof(int), cudaMemcpyHostToDevice));

    // Tile scheduler: 1 partition covering batch 0, blocks [0, 1)
    // [begin_idx, begin_block, end_idx, end_block, begin_n_split, _, _, _]
    int h_ts[TileSchedulerMetaDataSize] = {0, 0, 0, 1, 0, 0, 0, 0};
    CHECK_CUDA(cudaMemcpy(d_tile_sched, h_ts, TileSchedulerMetaDataSize * sizeof(int), cudaMemcpyHostToDevice));

    // num_splits: [0, 1] (batch 0 has 1 split starting at index 0)
    int h_ns[2] = {0, 1};
    CHECK_CUDA(cudaMemcpy(d_num_splits, h_ns, 2 * sizeof(int), cudaMemcpyHostToDevice));

    // Set up params
    DecodingParams params = {};
    params.b = B;
    params.s_q = S_Q;
    params.q_seq_per_hk = Q_PER_HK;
    params.d = D_K;
    params.d_v = D_V;
    params.h_q = H_Q;
    params.h_k = H_K;
    params.num_blocks = NUM_BLOCKS;
    params.q_head_per_hk = H_Q / H_K;
    params.is_causal = false;
    params.scale_softmax = scale;
    params.scale_softmax_log2 = scale * (float)M_LOG2E;
    params.topk = 0;

    params.q_ptr = d_Q;
    params.k_ptr = d_KV;
    params.o_ptr = d_O;
    params.softmax_lse_ptr = d_lse;
    params.indices_ptr = nullptr;

    params.q_batch_stride = Q_PER_HK * H_K * D_K;
    params.k_batch_stride = PAGE_BS * D_K;
    params.o_batch_stride = Q_PER_HK * H_K * D_V;
    params.q_row_stride = D_K;
    params.k_row_stride = D_K;
    params.o_row_stride = D_V;
    params.q_head_stride = Q_PER_HK * D_K;
    params.k_head_stride = 0;  // single head
    params.o_head_stride = Q_PER_HK * D_V;

    params.block_table = d_block_table;
    params.block_table_batch_stride = 1;
    params.page_block_size = PAGE_BS;
    params.seqlens_k_ptr = d_seqlens_k;

    params.tile_scheduler_metadata_ptr = d_tile_sched;
    params.num_sm_parts = 1;
    params.num_splits_ptr = d_num_splits;
    params.total_num_splits = 1;
    params.softmax_lseaccum_ptr = d_lseaccum;
    params.oaccum_ptr = d_oaccum;

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    sm120::run_flash_splitkv_mla_kernel<cutlass::bfloat16_t>(params, stream);
    CHECK_CUDA(cudaStreamSynchronize(stream));

    // Read back results
    std::vector<bf16> h_O_gpu_bf16(Q_PER_HK * D_V);
    CHECK_CUDA(cudaMemcpy(h_O_gpu_bf16.data(), d_O, Q_PER_HK * D_V * sizeof(bf16), cudaMemcpyDeviceToHost));

    // Compare
    float max_abs_err = 0.0f;
    float max_rel_err = 0.0f;
    int mismatch_count = 0;
    for (int i = 0; i < Q_PER_HK * D_V; ++i) {
        float gpu_val = float(h_O_gpu_bf16[i]);
        float ref_val = h_O_ref[i];
        float abs_err = fabsf(gpu_val - ref_val);
        float rel_err = (fabsf(ref_val) > 1e-6f) ? abs_err / fabsf(ref_val) : abs_err;
        max_abs_err = std::max(max_abs_err, abs_err);
        max_rel_err = std::max(max_rel_err, rel_err);
        // BF16 has ~0.8% relative error, allow 5% for accumulated errors
        if (rel_err > 0.10f && abs_err > 0.05f) {
            if (mismatch_count < 5) {
                printf("  Mismatch [%d]: GPU=%.6f CPU=%.6f abs_err=%.6f rel_err=%.4f\n",
                       i, gpu_val, ref_val, abs_err, rel_err);
            }
            mismatch_count++;
        }
    }

    printf("  Max abs error: %.6f, Max rel error: %.4f\n", max_abs_err, max_rel_err);
    printf("  Mismatches: %d / %d\n", mismatch_count, Q_PER_HK * D_V);

    bool pass = (mismatch_count == 0);
    printf("  %s\n\n", pass ? "PASS" : "FAIL");

    CHECK_CUDA(cudaFree(d_Q)); CHECK_CUDA(cudaFree(d_KV)); CHECK_CUDA(cudaFree(d_O));
    CHECK_CUDA(cudaFree(d_lse)); CHECK_CUDA(cudaFree(d_oaccum)); CHECK_CUDA(cudaFree(d_lseaccum));
    CHECK_CUDA(cudaFree(d_block_table)); CHECK_CUDA(cudaFree(d_seqlens_k));
    CHECK_CUDA(cudaFree(d_tile_sched)); CHECK_CUDA(cudaFree(d_num_splits));
    CHECK_CUDA(cudaStreamDestroy(stream));

    return pass;
}

// ============================================================
// Test 2: SM120 Prefill Sparse
// ============================================================
bool test_prefill_sparse() {
    printf("=== Test 2: SM120 Prefill Sparse ===\n");

    // Test dimensions — h_q must be multiple of B_H=64
    constexpr int S_Q = 1;       // query seq len
    constexpr int S_KV = 64;     // kv seq len (1 block of topk tokens)
    constexpr int H_Q = 64;      // query heads (must be multiple of B_H=64)
    constexpr int H_KV = 1;      // kv heads
    constexpr int D_QK = 576;    // qk dim
    constexpr int D_V = 512;     // value dim
    constexpr int TOPK = 64;     // topk

    float scale = 1.0f / sqrtf((float)D_QK);

    // Allocate host data
    std::vector<float> h_Q_f(S_Q * H_Q * D_QK);
    std::vector<float> h_KV_f(S_KV * H_KV * D_QK);
    std::vector<float> h_K_f(TOPK * D_QK);
    std::vector<float> h_V_f(TOPK * D_V);
    std::vector<bf16> h_Q(S_Q * H_Q * D_QK);
    std::vector<bf16> h_KV(S_KV * H_KV * D_QK);
    std::vector<int> h_indices(S_Q * H_KV * TOPK);

    srand(456);
    for (int i = 0; i < S_Q * H_Q * D_QK; ++i) {
        float v = (float)(rand() % 200 - 100) / 200.0f;
        h_Q[i] = bf16(v);
        h_Q_f[i] = float(h_Q[i]);
    }
    for (int i = 0; i < S_KV * H_KV * D_QK; ++i) {
        float v = (float)(rand() % 200 - 100) / 200.0f;
        h_KV[i] = bf16(v);
        h_KV_f[i] = float(h_KV[i]);
    }
    // Indices: select first TOPK tokens from KV
    for (int i = 0; i < S_Q * H_KV * TOPK; ++i) {
        h_indices[i] = i % TOPK;  // tokens 0..63
    }

    // Build K and V for reference from selected indices
    for (int t = 0; t < TOPK; ++t) {
        int kv_idx = h_indices[t];
        for (int d = 0; d < D_QK; ++d) {
            h_K_f[t * D_QK + d] = h_KV_f[kv_idx * D_QK + d];
        }
        for (int d = 0; d < D_V; ++d) {
            h_V_f[t * D_V + d] = h_KV_f[kv_idx * D_QK + d];
        }
    }

    // CPU reference
    std::vector<float> h_O_ref(S_Q * H_Q * D_V, 0.0f);
    std::vector<float> h_lse_ref(S_Q * H_Q, 0.0f);
    cpu_attention_ref(h_Q_f.data(), h_K_f.data(), h_V_f.data(),
                      h_O_ref.data(), h_lse_ref.data(),
                      S_Q * H_Q, TOPK, D_QK, D_V, scale);

    // Allocate device memory
    bf16 *d_Q, *d_KV, *d_O;
    float *d_max_logits, *d_lse;
    int *d_indices;

    CHECK_CUDA(cudaMalloc(&d_Q, S_Q * H_Q * D_QK * sizeof(bf16)));
    CHECK_CUDA(cudaMalloc(&d_KV, S_KV * H_KV * D_QK * sizeof(bf16)));
    CHECK_CUDA(cudaMalloc(&d_O, S_Q * H_Q * D_V * sizeof(bf16)));
    CHECK_CUDA(cudaMalloc(&d_max_logits, S_Q * H_Q * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_lse, S_Q * H_Q * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_indices, S_Q * H_KV * TOPK * sizeof(int)));

    CHECK_CUDA(cudaMemcpy(d_Q, h_Q.data(), S_Q * H_Q * D_QK * sizeof(bf16), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_KV, h_KV.data(), S_KV * H_KV * D_QK * sizeof(bf16), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_indices, h_indices.data(), S_Q * H_KV * TOPK * sizeof(int), cudaMemcpyHostToDevice));

    SparsePrefillParams params = {};
    params.s_q = S_Q;
    params.s_kv = S_KV;
    params.h_q = H_Q;
    params.h_kv = H_KV;
    params.d_qk = D_QK;
    params.d_v = D_V;
    params.topk = TOPK;
    params.sm_scale = scale;
    params.sm_scale_div_log2 = scale * (float)M_LOG2E;

    params.q = (bf16*)d_Q;
    params.kv = (bf16*)d_KV;
    params.indices = d_indices;

    params.stride_q_s_q = H_Q * D_QK;
    params.stride_q_h_q = D_QK;
    params.stride_kv_s_kv = H_KV * D_QK;
    params.stride_kv_h_kv = D_QK;
    params.stride_indices_s_q = H_KV * TOPK;
    params.stride_indices_h_kv = TOPK;

    params.out = (bf16*)d_O;
    params.max_logits = d_max_logits;
    params.lse = d_lse;
    params.stream = 0;

    sm120::run_fwd_kernel(params);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Read back
    std::vector<bf16> h_O_gpu_bf16(S_Q * H_Q * D_V);
    CHECK_CUDA(cudaMemcpy(h_O_gpu_bf16.data(), d_O, S_Q * H_Q * D_V * sizeof(bf16), cudaMemcpyDeviceToHost));

    // Compare
    float max_abs_err = 0.0f;
    float max_rel_err = 0.0f;
    int mismatch_count = 0;
    for (int i = 0; i < S_Q * H_Q * D_V; ++i) {
        float gpu_val = float(h_O_gpu_bf16[i]);
        float ref_val = h_O_ref[i];
        float abs_err = fabsf(gpu_val - ref_val);
        float rel_err = (fabsf(ref_val) > 1e-6f) ? abs_err / fabsf(ref_val) : abs_err;
        max_abs_err = std::max(max_abs_err, abs_err);
        max_rel_err = std::max(max_rel_err, rel_err);
        if (rel_err > 0.10f && abs_err > 0.05f) {
            if (mismatch_count < 5) {
                printf("  Mismatch [%d]: GPU=%.6f CPU=%.6f abs_err=%.6f rel_err=%.4f\n",
                       i, gpu_val, ref_val, abs_err, rel_err);
            }
            mismatch_count++;
        }
    }

    printf("  Max abs error: %.6f, Max rel error: %.4f\n", max_abs_err, max_rel_err);
    printf("  Mismatches: %d / %d\n", mismatch_count, S_Q * H_Q * D_V);

    bool pass = (mismatch_count == 0);
    printf("  %s\n\n", pass ? "PASS" : "FAIL");

    CHECK_CUDA(cudaFree(d_Q)); CHECK_CUDA(cudaFree(d_KV)); CHECK_CUDA(cudaFree(d_O));
    CHECK_CUDA(cudaFree(d_max_logits)); CHECK_CUDA(cudaFree(d_lse));
    CHECK_CUDA(cudaFree(d_indices));

    return pass;
}

// ============================================================
// Test 3: SM120 Decode Sparse FP8
// ============================================================
bool test_decode_sparse_fp8() {
    printf("=== Test 3: SM120 Decode Sparse FP8 ===\n");

    // FP8 KV cache layout per token row:
    //   [FP8 NOPE values: 512 bytes] [float scales: 4 * 4 = 16 bytes] [BF16 ROPE: 64 * 2 = 128 bytes]
    // Total per row = 512 + 16 + 128 = 656 bytes
    //
    // K access: dims 0..511 = FP8 dequant (scale per 128-dim tile), dims 512..575 = BF16 rope
    // V access: dims 0..511 = FP8 dequant (same scales)

    constexpr int B = 1;
    constexpr int S_Q = 1;
    constexpr int H_Q = 64;       // q heads
    constexpr int H_K = 1;        // kv heads
    constexpr int D_K = 576;
    constexpr int D_V = 512;
    constexpr int NOPE = 512;
    constexpr int ROPE = 64;
    constexpr int QUANT_TILE = 128;
    constexpr int NSCALES = NOPE / QUANT_TILE;  // 4
    constexpr int PAGE_BS = 64;
    constexpr int TOPK = 64;      // must be multiple of 64
    constexpr int Q_PER_HK = H_Q / H_K * S_Q;

    // Per-token row size in bytes
    constexpr int ROW_BYTES = NOPE + NSCALES * sizeof(float) + ROPE * sizeof(bf16);
    // k_row_stride is in units of fp8 (1 byte each)
    constexpr int K_ROW_STRIDE = ROW_BYTES;
    constexpr int K_BATCH_STRIDE = PAGE_BS * K_ROW_STRIDE;

    float scale = 1.0f / sqrtf((float)D_K);

    // Generate random Q (BF16)
    std::vector<float> h_Q_f(Q_PER_HK * D_K);
    std::vector<bf16> h_Q_bf16(Q_PER_HK * D_K);
    srand(789);
    for (int i = 0; i < Q_PER_HK * D_K; ++i) {
        float v = (float)(rand() % 200 - 100) / 200.0f;
        h_Q_bf16[i] = bf16(v);
        h_Q_f[i] = float(h_Q_bf16[i]);
    }

    // Generate random FP8 KV cache (1 page, 64 tokens)
    // Layout: [NOPE fp8 values][NSCALES floats][ROPE bf16 values]
    std::vector<uint8_t> h_kv_raw(PAGE_BS * ROW_BYTES, 0);
    std::vector<float> h_K_f(TOPK * D_K, 0.0f);  // dequantized K for reference
    std::vector<float> h_V_f(TOPK * D_V, 0.0f);  // dequantized V for reference

    for (int tok = 0; tok < PAGE_BS; ++tok) {
        uint8_t* row = h_kv_raw.data() + tok * ROW_BYTES;
        fp8* nope_vals = reinterpret_cast<fp8*>(row);
        float* scales = reinterpret_cast<float*>(row + NOPE);
        bf16* rope_vals = reinterpret_cast<bf16*>(row + NOPE + NSCALES * sizeof(float));

        // Set scales (use small values to keep FP8 in range)
        for (int s = 0; s < NSCALES; ++s) {
            scales[s] = 0.1f;  // uniform scale for simplicity
        }

        // Set FP8 NOPE values
        for (int d = 0; d < NOPE; ++d) {
            float v = (float)(rand() % 100 - 50) / 100.0f;
            nope_vals[d] = fp8(v);
            float dequant = float(nope_vals[d]) * scales[d / QUANT_TILE];
            h_K_f[tok * D_K + d] = dequant;
            h_V_f[tok * D_V + d] = dequant;
        }

        // Set BF16 ROPE values
        for (int d = 0; d < ROPE; ++d) {
            float v = (float)(rand() % 100 - 50) / 100.0f;
            rope_vals[d] = bf16(v);
            h_K_f[tok * D_K + NOPE + d] = float(rope_vals[d]);
        }
    }

    // CPU reference attention
    std::vector<float> h_O_ref(Q_PER_HK * D_V, 0.0f);
    std::vector<float> h_lse_ref(Q_PER_HK, 0.0f);
    cpu_attention_ref(h_Q_f.data(), h_K_f.data(), h_V_f.data(),
                      h_O_ref.data(), h_lse_ref.data(),
                      Q_PER_HK, TOPK, D_K, D_V, scale);

    // Allocate device memory
    bf16 *d_Q;
    uint8_t *d_KV;
    bf16 *d_O;
    float *d_lse, *d_oaccum, *d_lseaccum;
    int *d_indices, *d_tile_sched, *d_num_splits;

    CHECK_CUDA(cudaMalloc(&d_Q, Q_PER_HK * D_K * sizeof(bf16)));
    CHECK_CUDA(cudaMalloc(&d_KV, PAGE_BS * ROW_BYTES));
    CHECK_CUDA(cudaMalloc(&d_O, Q_PER_HK * D_V * sizeof(bf16)));
    CHECK_CUDA(cudaMalloc(&d_lse, B * Q_PER_HK * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_oaccum, 2 * Q_PER_HK * D_V * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_lseaccum, 2 * Q_PER_HK * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_indices, B * S_Q * TOPK * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_tile_sched, 1 * TileSchedulerMetaDataSize * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_num_splits, (B + 1) * sizeof(int)));

    CHECK_CUDA(cudaMemcpy(d_Q, h_Q_bf16.data(), Q_PER_HK * D_K * sizeof(bf16), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_KV, h_kv_raw.data(), PAGE_BS * ROW_BYTES, cudaMemcpyHostToDevice));

    // Indices: tokens 0..63 (all in page 0)
    std::vector<int> h_indices(B * S_Q * TOPK);
    for (int i = 0; i < TOPK; ++i) h_indices[i] = i;
    CHECK_CUDA(cudaMemcpy(d_indices, h_indices.data(), TOPK * sizeof(int), cudaMemcpyHostToDevice));

    // Tile scheduler: 1 partition covering batch 0, topk blocks [0, 1)
    int h_ts[TileSchedulerMetaDataSize] = {0, 0, 0, 1, 0, 0, 0, 0};
    CHECK_CUDA(cudaMemcpy(d_tile_sched, h_ts, TileSchedulerMetaDataSize * sizeof(int), cudaMemcpyHostToDevice));

    // num_splits: [0, 1]
    int h_ns[2] = {0, 1};
    CHECK_CUDA(cudaMemcpy(d_num_splits, h_ns, 2 * sizeof(int), cudaMemcpyHostToDevice));

    // Set up params
    DecodingParams params = {};
    params.b = B;
    params.s_q = S_Q;
    params.q_seq_per_hk = Q_PER_HK;
    params.d = D_K;
    params.d_v = D_V;
    params.h_q = H_Q;
    params.h_k = H_K;
    params.num_blocks = 1;
    params.q_head_per_hk = H_Q / H_K;
    params.is_causal = false;
    params.scale_softmax = scale;
    params.scale_softmax_log2 = scale * (float)M_LOG2E;
    params.topk = TOPK;

    params.q_ptr = d_Q;
    params.k_ptr = d_KV;
    params.o_ptr = d_O;
    params.softmax_lse_ptr = d_lse;
    params.indices_ptr = d_indices;

    params.q_batch_stride = Q_PER_HK * D_K;
    params.k_batch_stride = K_BATCH_STRIDE;
    params.o_batch_stride = Q_PER_HK * D_V;
    params.q_row_stride = D_K;
    params.k_row_stride = K_ROW_STRIDE;
    params.o_row_stride = D_V;
    params.q_head_stride = 0;
    params.k_head_stride = 0;
    params.o_head_stride = 0;
    params.indices_batch_stride = S_Q * TOPK;
    params.indices_row_stride = TOPK;

    params.block_table = nullptr;
    params.block_table_batch_stride = 0;
    params.page_block_size = PAGE_BS;
    params.seqlens_k_ptr = nullptr;

    params.tile_scheduler_metadata_ptr = d_tile_sched;
    params.num_sm_parts = 1;
    params.num_splits_ptr = d_num_splits;
    params.total_num_splits = 1;
    params.softmax_lseaccum_ptr = d_lseaccum;
    params.oaccum_ptr = d_oaccum;

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    sm120::run_flash_splitkv_mla_fp8_sparse_kernel(params, stream);
    CHECK_CUDA(cudaStreamSynchronize(stream));

    // Read back
    std::vector<bf16> h_O_gpu_bf16(Q_PER_HK * D_V);
    CHECK_CUDA(cudaMemcpy(h_O_gpu_bf16.data(), d_O, Q_PER_HK * D_V * sizeof(bf16), cudaMemcpyDeviceToHost));

    // Compare
    float max_abs_err = 0.0f;
    float max_rel_err = 0.0f;
    int mismatch_count = 0;
    for (int i = 0; i < Q_PER_HK * D_V; ++i) {
        float gpu_val = float(h_O_gpu_bf16[i]);
        float ref_val = h_O_ref[i];
        float abs_err = fabsf(gpu_val - ref_val);
        float rel_err = (fabsf(ref_val) > 1e-6f) ? abs_err / fabsf(ref_val) : abs_err;
        max_abs_err = std::max(max_abs_err, abs_err);
        max_rel_err = std::max(max_rel_err, rel_err);
        // FP8 has lower precision, allow wider tolerance
        if (rel_err > 0.15f && abs_err > 0.1f) {
            if (mismatch_count < 5) {
                printf("  Mismatch [%d]: GPU=%.6f CPU=%.6f abs_err=%.6f rel_err=%.4f\n",
                       i, gpu_val, ref_val, abs_err, rel_err);
            }
            mismatch_count++;
        }
    }

    printf("  Max abs error: %.6f, Max rel error: %.4f\n", max_abs_err, max_rel_err);
    printf("  Mismatches: %d / %d\n", mismatch_count, Q_PER_HK * D_V);

    bool pass = (mismatch_count == 0);
    printf("  %s\n\n", pass ? "PASS" : "FAIL");

    CHECK_CUDA(cudaFree(d_Q)); CHECK_CUDA(cudaFree(d_KV)); CHECK_CUDA(cudaFree(d_O));
    CHECK_CUDA(cudaFree(d_lse)); CHECK_CUDA(cudaFree(d_oaccum)); CHECK_CUDA(cudaFree(d_lseaccum));
    CHECK_CUDA(cudaFree(d_indices)); CHECK_CUDA(cudaFree(d_tile_sched)); CHECK_CUDA(cudaFree(d_num_splits));
    CHECK_CUDA(cudaStreamDestroy(stream));

    return pass;
}

// ============================================================
// Main
// ============================================================
int main() {
    printf("========================================\n");
    printf("SM120 FlashMLA Kernel Validation Tests\n");
    printf("========================================\n\n");

    int device;
    CHECK_CUDA(cudaGetDevice(&device));
    cudaDeviceProp props;
    CHECK_CUDA(cudaGetDeviceProperties(&props, device));
    printf("GPU: %s (SM %d.%d)\n", props.name, props.major, props.minor);
    printf("Shared Memory per SM: %zu bytes (%.1fKB)\n\n",
           props.sharedMemPerMultiprocessor,
           props.sharedMemPerMultiprocessor / 1024.0);

    int pass_count = 0, total = 0;

    // Test 1: Decode Dense
    total++;
    if (test_decode_dense()) pass_count++;

    // Test 2: Prefill Sparse
    total++;
    if (test_prefill_sparse()) pass_count++;

    // Test 3: Decode Sparse FP8
    total++;
    if (test_decode_sparse_fp8()) pass_count++;

    printf("========================================\n");
    printf("Results: %d / %d tests passed\n", pass_count, total);
    printf("========================================\n");

    return (pass_count == total) ? 0 : 1;
}
