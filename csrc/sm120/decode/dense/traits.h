#pragma once

#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>

#include "config.h"

using namespace cute;

namespace sm120 {

template<typename InputT_>
struct Traits {
    using InputT = InputT_;

    static constexpr int BLOCK_SIZE_M = Config::BLOCK_SIZE_M;       // 64
    static constexpr int PAGE_BLOCK_SIZE = Config::PAGE_BLOCK_SIZE; // 64
    static constexpr int HEAD_DIM_K = Config::HEAD_DIM_K;           // 576
    static constexpr int HEAD_DIM_V = Config::HEAD_DIM_V;           // 512
    static constexpr int K_TILE_SIZE = Config::K_TILE_SIZE;         // 256
    static constexpr int NUM_K_TILES = Config::NUM_K_TILES;         // 3
    static constexpr int LAST_K_TILE_SIZE = Config::LAST_K_TILE_SIZE; // 64

    // SM120: 128 threads = 4 warps (no warpgroups)
    static constexpr int NUM_THREADS = 128;

    static_assert(std::is_same_v<InputT, cutlass::bfloat16_t> || std::is_same_v<InputT, cutlass::half_t>);

    // ========== MMA Atoms ==========
    // SM80 mma.sync m16n8k16 for BF16/FP16
    // 4 warps tiled along M dimension: each warp handles 16 rows, total 64 rows
    using MMA_Atom_Arch = MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>;
    using TiledMMA_QK = decltype(make_tiled_mma(
        MMA_Atom_Arch{},
        Layout<Shape<_4, _1, _1>>{}  // 4 warps along M
    ));

    // PV uses same MMA atom configuration
    using TiledMMA_PV = decltype(make_tiled_mma(
        MMA_Atom_Arch{},
        Layout<Shape<_4, _1, _1>>{}  // 4 warps along M
    ));

    // ========== Shared Memory Layouts ==========
    // Swizzle<3,3,3> provides bank-conflict-free access for SM80 mma.sync
    using SmemLayoutAtom = decltype(
        composition(Swizzle<3, 3, 3>{},
                    Layout<Shape<_8, _64>, Stride<_64, _1>>{}));

    // Q tile: (BLOCK_SIZE_M × K_TILE_SIZE) = (64 × 256) = 32KB
    using SmemLayoutQTile = decltype(tile_to_shape(
        SmemLayoutAtom{},
        Shape<Int<BLOCK_SIZE_M>, Int<K_TILE_SIZE>>{}
    ));

    // K tile: (PAGE_BLOCK_SIZE × K_TILE_SIZE) = (64 × 256) = 32KB
    using SmemLayoutKTile = decltype(tile_to_shape(
        SmemLayoutAtom{},
        Shape<Int<PAGE_BLOCK_SIZE>, Int<K_TILE_SIZE>>{}
    ));

    // P: (BLOCK_SIZE_M × PAGE_BLOCK_SIZE) = (64 × 64) = 8KB
    using SmemLayoutP = decltype(tile_to_shape(
        SmemLayoutAtom{},
        Shape<Int<BLOCK_SIZE_M>, Int<PAGE_BLOCK_SIZE>>{}
    ));

    // V half: (HEAD_DIM_V/2 × PAGE_BLOCK_SIZE) = (256 × 64) = 32KB
    // For PV gemm with TN MMA: B operand needs (N, K) layout
    // P(M=64, K=64) × V(N=256, K=64) = O(M=64, N=256)
    // V is loaded transposed from global memory
    using SmemLayoutVHalf = decltype(tile_to_shape(
        SmemLayoutAtom{},
        Shape<Int<HEAD_DIM_V / 2>, Int<PAGE_BLOCK_SIZE>>{}
    ));

    // Output buffer for no-split TMA store: (BLOCK_SIZE_M × HEAD_DIM_V) = (64 × 512) = 64KB
    using SmemLayoutO = decltype(tile_to_shape(
        SmemLayoutAtom{},
        Shape<Int<BLOCK_SIZE_M>, Int<HEAD_DIM_V>>{}
    ));

    // ========== Shared Memory Plan ==========
    // SM120 has 100KB shared memory limit.
    //
    // Memory layout uses phases with overlapping buffers:
    //
    // Phase 1 (QK^T): For each K-tile iteration:
    //   sQTile(32KB) + sKTile(32KB) = 64KB
    //   + persistent: sM(256B) + sScale(256B) + sL(512B) + barriers(~64B) ≈ 1KB
    //   Total: ~65KB ✓
    //
    // Phase 2 (PV): For each KV block, after QK^T is done:
    //   sP(8KB) + sVHalf(32KB) = 40KB (reuses QK phase memory)
    //   + persistent ≈ 1KB
    //   Total: ~41KB ✓
    //
    // Phase 3 (store O): After all blocks processed:
    //   For no-split: sO(64KB) via TMA store (reuses all phase memory)
    //   For split: write float accumulators directly from registers to gmem
    //
    // Peak: 65KB ✓ (well within 100KB)

    // Main shared memory buffer — large enough for the biggest phase
    static constexpr int MAIN_BUF_ELEMS = BLOCK_SIZE_M * K_TILE_SIZE;  // 64×256 = 16384 bf16 = 32KB
    static constexpr int AUX_BUF_ELEMS = PAGE_BLOCK_SIZE * K_TILE_SIZE; // 64×256 = 16384 bf16 = 32KB

    struct SharedMemoryPlan {
        // Main buffer: used for sQTile in QK phase, sP in PV phase, sO in store phase
        cute::array_aligned<InputT, MAIN_BUF_ELEMS> buf_main;    // 32KB
        // Aux buffer: used for sKTile in QK phase, sVHalf in PV phase
        cute::array_aligned<InputT, AUX_BUF_ELEMS> buf_aux;      // 32KB

        // Persistent state (always valid)
        cute::array_aligned<float, BLOCK_SIZE_M> smem_sM;                // 256B
        cute::array_aligned<float, BLOCK_SIZE_M> smem_sScale;            // 256B
        cute::array_aligned<float, 2 * BLOCK_SIZE_M> sL_reduction_wksp;  // 512B
    };
};

}  // namespace sm120
