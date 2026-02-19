#!/bin/bash
# SM120 FlashMLA Kernel NCU Profiling Script
#
# Usage (inside Docker container on RTX 5090):
#   # Full profiling of all kernels
#   bash /app/FlashMLA/benchmark/profile_sm120.sh
#
#   # Profile specific kernel
#   bash /app/FlashMLA/benchmark/profile_sm120.sh decode_dense
#   bash /app/FlashMLA/benchmark/profile_sm120.sh prefill_sparse
#   bash /app/FlashMLA/benchmark/profile_sm120.sh decode_sparse_fp8
#
#   # Quick timing only (no NCU)
#   bash /app/FlashMLA/benchmark/profile_sm120.sh timing
#
# Output: .ncu-rep files in /tmp/sm120_profiles/
# View with: ncu-ui /tmp/sm120_profiles/*.ncu-rep

set -euo pipefail

OUTDIR="/tmp/sm120_profiles"
BENCH="/app/FlashMLA/benchmark/bench_sm120.py"
KERNEL="${1:-all}"

mkdir -p "$OUTDIR"

echo "============================================"
echo "SM120 FlashMLA Kernel Profiling"
echo "============================================"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "CUDA: $(nvcc --version | grep release | awk '{print $6}')"
echo "Output: $OUTDIR"
echo "Kernel: $KERNEL"
echo ""

# ============================================================================
# Quick timing benchmark (no NCU overhead)
# ============================================================================
if [[ "$KERNEL" == "timing" || "$KERNEL" == "all" ]]; then
    echo "=== Quick Timing Benchmark ==="
    python3 "$BENCH" --kernel all --iters 200
    echo ""
fi

if [[ "$KERNEL" == "timing" ]]; then
    exit 0
fi

# ============================================================================
# Decode Dense BF16 — flash_fwd_splitkv_mla_kernel_sm120
# ============================================================================
if [[ "$KERNEL" == "decode_dense" || "$KERNEL" == "all" ]]; then
    echo "=== NCU: Decode Dense BF16 ==="

    # 1. Full metrics collection
    echo "  [1/4] Full metrics..."
    ncu --set full \
        --kernel-name-base demangled \
        --kernel-name regex:"flash_fwd_splitkv_mla_kernel_sm120" \
        --launch-count 5 \
        --launch-skip 2 \
        -f -o "$OUTDIR/decode_dense_full" \
        python3 "$BENCH" --kernel decode_dense --ncu 2>&1 | tail -5

    # 2. Roofline analysis
    echo "  [2/4] Roofline..."
    ncu --set roofline \
        --kernel-name-base demangled \
        --kernel-name regex:"flash_fwd_splitkv_mla_kernel_sm120" \
        --launch-count 3 \
        --launch-skip 2 \
        -f -o "$OUTDIR/decode_dense_roofline" \
        python3 "$BENCH" --kernel decode_dense --ncu 2>&1 | tail -5

    # 3. Key metrics summary (text output)
    echo "  [3/4] Key metrics..."
    ncu --kernel-name-base demangled \
        --kernel-name regex:"flash_fwd_splitkv_mla_kernel_sm120" \
        --launch-count 3 \
        --launch-skip 2 \
        --metrics \
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
sm__sass_inst_executed_op_shared_ld.sum,\
sm__sass_inst_executed_op_shared_st.sum,\
sm__sass_inst_executed_op_global_ld.sum,\
sm__sass_inst_executed_op_global_st.sum,\
sm__inst_executed_pipe_tensor.sum,\
smsp__warps_issue_stalled_long_scoreboard_per_issue_active.ratio,\
smsp__warps_issue_stalled_short_scoreboard_per_issue_active.ratio,\
smsp__warps_issue_stalled_wait_per_issue_active.ratio,\
smsp__warps_issue_stalled_mio_throttle_per_issue_active.ratio,\
smsp__warps_issue_stalled_barrier_per_issue_active.ratio,\
smsp__warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio \
        python3 "$BENCH" --kernel decode_dense --ncu 2>&1 | tee "$OUTDIR/decode_dense_metrics.txt"

    # 4. Source-correlated (for SASS analysis)
    echo "  [4/4] Source correlation..."
    ncu --set source \
        --kernel-name-base demangled \
        --kernel-name regex:"flash_fwd_splitkv_mla_kernel_sm120" \
        --launch-count 1 \
        --launch-skip 2 \
        -f -o "$OUTDIR/decode_dense_source" \
        python3 "$BENCH" --kernel decode_dense --ncu 2>&1 | tail -5

    echo "  Done: $OUTDIR/decode_dense_*.ncu-rep"
    echo ""
fi

# ============================================================================
# Prefill Sparse BF16 — sparse_attn_fwd_kernel_sm120
# ============================================================================
if [[ "$KERNEL" == "prefill_sparse" || "$KERNEL" == "all" ]]; then
    echo "=== NCU: Prefill Sparse BF16 ==="

    # 1. Full metrics
    echo "  [1/4] Full metrics..."
    ncu --set full \
        --kernel-name-base demangled \
        --kernel-name regex:"sparse_attn_fwd_kernel_sm120" \
        --launch-count 5 \
        --launch-skip 2 \
        -f -o "$OUTDIR/prefill_sparse_full" \
        python3 "$BENCH" --kernel prefill_sparse --ncu 2>&1 | tail -5

    # 2. Roofline
    echo "  [2/4] Roofline..."
    ncu --set roofline \
        --kernel-name-base demangled \
        --kernel-name regex:"sparse_attn_fwd_kernel_sm120" \
        --launch-count 3 \
        --launch-skip 2 \
        -f -o "$OUTDIR/prefill_sparse_roofline" \
        python3 "$BENCH" --kernel prefill_sparse --ncu 2>&1 | tail -5

    # 3. Key metrics
    echo "  [3/4] Key metrics..."
    ncu --kernel-name-base demangled \
        --kernel-name regex:"sparse_attn_fwd_kernel_sm120" \
        --launch-count 3 \
        --launch-skip 2 \
        --metrics \
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
sm__sass_inst_executed_op_shared_ld.sum,\
sm__sass_inst_executed_op_shared_st.sum,\
sm__sass_inst_executed_op_global_ld.sum,\
sm__sass_inst_executed_op_global_st.sum,\
sm__inst_executed_pipe_tensor.sum,\
smsp__warps_issue_stalled_long_scoreboard_per_issue_active.ratio,\
smsp__warps_issue_stalled_short_scoreboard_per_issue_active.ratio,\
smsp__warps_issue_stalled_wait_per_issue_active.ratio,\
smsp__warps_issue_stalled_mio_throttle_per_issue_active.ratio,\
smsp__warps_issue_stalled_barrier_per_issue_active.ratio,\
smsp__warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio \
        python3 "$BENCH" --kernel prefill_sparse --ncu 2>&1 | tee "$OUTDIR/prefill_sparse_metrics.txt"

    # 4. Source correlation
    echo "  [4/4] Source correlation..."
    ncu --set source \
        --kernel-name-base demangled \
        --kernel-name regex:"sparse_attn_fwd_kernel_sm120" \
        --launch-count 1 \
        --launch-skip 2 \
        -f -o "$OUTDIR/prefill_sparse_source" \
        python3 "$BENCH" --kernel prefill_sparse --ncu 2>&1 | tail -5

    echo "  Done: $OUTDIR/prefill_sparse_*.ncu-rep"
    echo ""
fi

# ============================================================================
# Decode Sparse FP8 — sp_fp8_kernel
# ============================================================================
if [[ "$KERNEL" == "decode_sparse_fp8" || "$KERNEL" == "all" ]]; then
    echo "=== NCU: Decode Sparse FP8 ==="
    echo "  NOTE: FP8 sparse decode requires full serving stack with quantized KV cache."
    echo "  Skipping standalone NCU profiling."
    echo "  To profile in-server, use:"
    echo "    ncu --target-processes all \\"
    echo "        --kernel-name regex:'sp_fp8_kernel' \\"
    echo "        --launch-count 10 --launch-skip 100 \\"
    echo "        -f -o $OUTDIR/decode_sparse_fp8_full \\"
    echo "        python -m sglang.launch_server ..."
    echo ""
fi

# ============================================================================
# Summary
# ============================================================================
echo "============================================"
echo "Profiling Complete"
echo "============================================"
echo ""
echo "Output files:"
ls -la "$OUTDIR"/*.ncu-rep 2>/dev/null || echo "  (no .ncu-rep files — NCU may not be installed)"
ls -la "$OUTDIR"/*.txt 2>/dev/null || echo "  (no .txt files)"
echo ""
echo "View in NCU GUI:"
echo "  ncu-ui $OUTDIR/decode_dense_full.ncu-rep"
echo "  ncu-ui $OUTDIR/prefill_sparse_full.ncu-rep"
echo ""
echo "Key things to look for:"
echo "  1. sm__throughput vs gpu__compute_memory_throughput → memory or compute bound?"
echo "  2. sm__inst_executed_pipe_tensor → MMA utilization (expect <10% currently)"
echo "  3. sm__sass_inst_executed_op_shared_ld → scalar smem loads (expect very high)"
echo "  4. l1tex__data_bank_conflicts → bank conflicts from swizzle"
echo "  5. long_scoreboard stalls → waiting for global memory (no pipeline)"
echo "  6. barrier stalls → __syncthreads() overhead"
