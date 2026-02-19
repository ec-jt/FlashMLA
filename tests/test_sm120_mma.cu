// test_sm120_mma.cu — SM120 mma.sync validation test
//
// Validates that SM120 (RTX 5090 / GB200) correctly executes:
//   1. SM80_16x8x16_F32BF16BF16F32_TN MMA atom via CuTe
//   2. Swizzled shared memory layout
//   3. 128-thread / 4-warp configuration with per-warp MMA
//   4. Shared memory within 100KB budget
//
// Build:
//   nvcc -arch=sm_120a \
//     -I ../../csrc/cutlass/include \
//     -I ../../csrc/cutlass/tools/util/include \
//     --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math \
//     test_sm120_mma.cu -o test_sm120_mma

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <cute/tensor.hpp>
#include <cutlass/numeric_types.h>

using namespace cute;
using bf16 = cutlass::bfloat16_t;

constexpr int M = 64;
constexpr int N = 64;
constexpr int K = 256;
constexpr int NUM_THREADS = 128;

// SM80 mma.sync m16n8k16 with 4 warps along M
using MMA_Atom_SM120 = MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>;
using TiledMMA_SM120 = decltype(make_tiled_mma(
    MMA_Atom_SM120{},
    Layout<Shape<_4, _1, _1>>{}
));

// Swizzled smem layout compatible with SM80 mma.sync
// Swizzle<3,3,3> = 8-byte swizzle with 8 rows, good for 128-bit bank-conflict-free access
using SmemLayoutAtom = decltype(
    composition(Swizzle<3, 3, 3>{},
                Layout<Shape<_8, _64>, Stride<_64, _1>>{}));

using SmemLayoutA = decltype(tile_to_shape(SmemLayoutAtom{}, Shape<Int<M>, Int<K>>{}));
using SmemLayoutB = decltype(tile_to_shape(SmemLayoutAtom{}, Shape<Int<N>, Int<K>>{}));

struct SharedMemory {
    cute::array_aligned<bf16, cute::cosize_v<SmemLayoutA>> a;
    cute::array_aligned<bf16, cute::cosize_v<SmemLayoutB>> b;
};

#define CHECK_CUDA(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while (0)

__global__ void __launch_bounds__(NUM_THREADS)
test_gemm_kernel(const bf16* __restrict__ gA_ptr,
                 const bf16* __restrict__ gB_ptr,
                 float* __restrict__ gC_ptr)
{
    extern __shared__ char smem_buf[];
    SharedMemory& smem = *reinterpret_cast<SharedMemory*>(smem_buf);

    int tid = threadIdx.x;

    // ---- Load A and B into swizzled smem ----
    Tensor sA = make_tensor(make_smem_ptr(smem.a.data()), SmemLayoutA{});
    Tensor sB = make_tensor(make_smem_ptr(smem.b.data()), SmemLayoutB{});

    // Simple element-wise copy from global to swizzled smem
    for (int idx = tid; idx < M * K; idx += NUM_THREADS) {
        int row = idx / K;
        int col = idx % K;
        sA(row, col) = gA_ptr[row * K + col];
    }
    for (int idx = tid; idx < N * K; idx += NUM_THREADS) {
        int row = idx / K;
        int col = idx % K;
        sB(row, col) = gB_ptr[row * K + col];
    }

    __syncthreads();

    // ---- Compute C = A @ B^T using mma.sync ----
    TiledMMA_SM120 tiled_mma;
    auto thr_mma = tiled_mma.get_slice(tid);

    // Partition shared memory tensors for this thread's MMA
    // partition_A/B give us the smem view partitioned for this thread
    Tensor thr_sA = thr_mma.partition_A(sA);  // (MMA, MMA_M, MMA_K)
    Tensor thr_sB = thr_mma.partition_B(sB);  // (MMA, MMA_N, MMA_K)

    // Create register fragments (empty, need to be filled)
    Tensor rA = thr_mma.partition_fragment_A(sA);  // (MMA, MMA_M, MMA_K)
    Tensor rB = thr_mma.partition_fragment_B(sB);  // (MMA, MMA_N, MMA_K)

    // Accumulator
    Tensor rC = partition_fragment_C(tiled_mma, Shape<Int<M>, Int<N>>{});
    cute::fill(rC, 0.0f);

    // MMA loop over K dimension
    // For each K tile, copy from smem to registers, then do MMA
    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < size<2>(thr_sA); ++k) {
        // Copy smem -> registers for this K tile
        cute::copy(thr_sA(_, _, k), rA(_, _, k));
        cute::copy(thr_sB(_, _, k), rB(_, _, k));
        // Execute MMA
        cute::gemm(tiled_mma, rA(_, _, k), rB(_, _, k), rC);
    }

    // ---- Store C to global memory ----
    // Use partition_C to map accumulator elements to output coordinates
    Tensor gC = make_tensor(make_gmem_ptr(gC_ptr), make_layout(Shape<Int<M>, Int<N>>{}, make_stride(Int<N>{}, _1{})));
    Tensor thr_gC = thr_mma.partition_C(gC);

    // Copy accumulator to global
    cute::copy(rC, thr_gC);
}

void cpu_gemm_ref(const float* A, const float* B, float* C, int m, int n, int k) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int l = 0; l < k; ++l) {
                sum += A[i * k + l] * B[j * k + l];
            }
            C[i * n + j] = sum;
        }
    }
}

int main() {
    printf("=== SM120 mma.sync + swizzled smem validation test ===\n\n");

    int device;
    CHECK_CUDA(cudaGetDevice(&device));
    cudaDeviceProp props;
    CHECK_CUDA(cudaGetDeviceProperties(&props, device));
    printf("GPU: %s (SM %d.%d)\n", props.name, props.major, props.minor);

    printf("SharedMemory size: %zu bytes (%.1fKB) - %s 100KB limit\n",
           sizeof(SharedMemory), sizeof(SharedMemory) / 1024.0,
           sizeof(SharedMemory) <= 102400 ? "within" : "EXCEEDS");
    printf("Test GEMM: C[%d×%d] = A[%d×%d] @ B[%d×%d]^T\n", M, N, M, K, N, K);
    printf("Threads: %d (4 warps)\n\n", NUM_THREADS);

    int sizeA = M * K, sizeB = N * K, sizeC = M * N;
    float *h_A = new float[sizeA], *h_B = new float[sizeB];
    float *h_C_ref = new float[sizeC], *h_C_gpu = new float[sizeC];
    bf16 *h_A_bf16 = new bf16[sizeA], *h_B_bf16 = new bf16[sizeB];

    srand(42);
    for (int i = 0; i < sizeA; ++i) {
        h_A[i] = (float)(rand() % 100 - 50) / 50.0f;
        h_A_bf16[i] = bf16(h_A[i]);
        h_A[i] = float(h_A_bf16[i]);
    }
    for (int i = 0; i < sizeB; ++i) {
        h_B[i] = (float)(rand() % 100 - 50) / 50.0f;
        h_B_bf16[i] = bf16(h_B[i]);
        h_B[i] = float(h_B_bf16[i]);
    }
    cpu_gemm_ref(h_A, h_B, h_C_ref, M, N, K);

    bf16 *d_A, *d_B; float *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, sizeA * sizeof(bf16)));
    CHECK_CUDA(cudaMalloc(&d_B, sizeB * sizeof(bf16)));
    CHECK_CUDA(cudaMalloc(&d_C, sizeC * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_A, h_A_bf16, sizeA * sizeof(bf16), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B_bf16, sizeB * sizeof(bf16), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_C, 0, sizeC * sizeof(float)));

    constexpr size_t smem_size = sizeof(SharedMemory);
    printf("Requesting %zu bytes dynamic shared memory...\n", smem_size);
    cudaError_t attr_err = cudaFuncSetAttribute(test_gemm_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    if (attr_err != cudaSuccess) {
        printf("FAIL: cudaFuncSetAttribute failed: %s\n", cudaGetErrorString(attr_err));
        return 1;
    }
    printf("cudaFuncSetAttribute succeeded\n");

    test_gemm_kernel<<<1, NUM_THREADS, smem_size>>>(d_A, d_B, d_C);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    printf("Kernel completed successfully\n\n");

    CHECK_CUDA(cudaMemcpy(h_C_gpu, d_C, sizeC * sizeof(float), cudaMemcpyDeviceToHost));

    float max_err = 0.0f;
    int err_count = 0;
    for (int i = 0; i < sizeC; ++i) {
        float err = fabsf(h_C_gpu[i] - h_C_ref[i]);
        float ref_abs = fabsf(h_C_ref[i]);
        float rel_err = (ref_abs > 1e-6f) ? err / ref_abs : err;
        if (err > max_err) max_err = err;
        if (rel_err > 0.05f && err > 2.0f) {
            if (err_count < 5)
                printf("  Mismatch [%d,%d]: GPU=%.4f CPU=%.4f err=%.4f\n",
                       i/N, i%N, h_C_gpu[i], h_C_ref[i], err);
            err_count++;
        }
    }

    printf("Max absolute error: %.6f\n", max_err);
    printf("Error count: %d / %d\n", err_count, sizeC);
    printf("Sample: C[0,0]=%.4f (ref=%.4f), C[0,1]=%.4f (ref=%.4f)\n",
           h_C_gpu[0], h_C_ref[0], h_C_gpu[1], h_C_ref[1]);

    bool pass = (err_count == 0);
    printf("\n%s: SM120 mma.sync %s\n", pass ? "PASS" : "FAIL",
           pass ? "works correctly" : "has errors");

    delete[] h_A; delete[] h_B; delete[] h_C_ref; delete[] h_C_gpu;
    delete[] h_A_bf16; delete[] h_B_bf16;
    CHECK_CUDA(cudaFree(d_A)); CHECK_CUDA(cudaFree(d_B)); CHECK_CUDA(cudaFree(d_C));
    return pass ? 0 : 1;
}
