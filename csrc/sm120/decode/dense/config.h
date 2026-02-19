#pragma once

namespace sm120 {
namespace Config {

static constexpr int BLOCK_SIZE_M = 64;
static constexpr int PAGE_BLOCK_SIZE = 64;

static constexpr int HEAD_DIM_K = 576;
static constexpr int HEAD_DIM_V = 512;

// K-dimension tiling for SM120's 100KB shared memory limit
// Process QK^T in chunks of K_TILE_SIZE along the K dimension
// ceil(576/256) = 3 iterations
static constexpr int K_TILE_SIZE = 256;
static constexpr int NUM_K_TILES = (HEAD_DIM_K + K_TILE_SIZE - 1) / K_TILE_SIZE;  // 3

// For the last K tile, only 576-512=64 elements are valid
static constexpr int LAST_K_TILE_SIZE = HEAD_DIM_K - (NUM_K_TILES - 1) * K_TILE_SIZE;  // 64

}  // namespace Config
}  // namespace sm120
