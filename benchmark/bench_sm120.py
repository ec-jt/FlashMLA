#!/usr/bin/env python3
"""Standalone benchmark for SM120 FlashMLA kernels.

Usage (inside Docker container):
    # Quick timing benchmark
    python3 /app/FlashMLA/benchmark/bench_sm120.py

    # NCU profiling (decode dense)
    ncu --set full --kernel-name regex:"sm120" --launch-count 5 --launch-skip 2 \
        -o /tmp/sm120_decode_dense \
        python3 /app/FlashMLA/benchmark/bench_sm120.py --kernel decode_dense --ncu

    # NCU profiling (prefill sparse)
    ncu --set full --kernel-name regex:"sm120" --launch-count 5 --launch-skip 2 \
        -o /tmp/sm120_prefill_sparse \
        python3 /app/FlashMLA/benchmark/bench_sm120.py --kernel prefill_sparse --ncu

    # NCU profiling (decode sparse fp8)
    ncu --set full --kernel-name regex:"sm120|sp_fp8" --launch-count 5 --launch-skip 2 \
        -o /tmp/sm120_decode_sparse_fp8 \
        python3 /app/FlashMLA/benchmark/bench_sm120.py --kernel decode_sparse_fp8 --ncu
"""

import argparse
import time
import torch
import sys

def get_flashmla_ops():
    """Import flashmla_ops from sgl_kernel."""
    try:
        from sgl_kernel import flashmla_ops
        return flashmla_ops
    except ImportError:
        print("ERROR: sgl_kernel.flashmla_ops not available. Run inside Docker container.")
        sys.exit(1)


def bench_decode_dense(num_iters=100, warmup=10, batch=128, seqlen=4096, h_q=128, h_kv=1, d=576, dv=512, ncu_mode=False):
    """Benchmark the decode dense BF16 kernel (flash_mla_with_kvcache)."""
    print(f"=== Decode Dense BF16 ===")
    print(f"  batch={batch}, seqlen={seqlen}, h_q={h_q}, h_kv={h_kv}, d={d}, dv={dv}")

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Setup tensors matching flash_mla_with_kvcache signature
    block_size = 64
    s_q = 1  # decode: 1 query token
    max_seqlen_pad = ((seqlen + block_size - 1) // block_size) * block_size
    num_blocks_per_seq = max_seqlen_pad // block_size

    q = torch.randn(batch, s_q, h_q, d, dtype=torch.bfloat16, device=device)
    block_table = torch.arange(batch * num_blocks_per_seq, dtype=torch.int32, device=device).view(batch, num_blocks_per_seq)
    blocked_k = torch.randn(block_table.numel(), block_size, h_kv, d, dtype=torch.bfloat16, device=device)
    cache_seqlens = torch.full((batch,), seqlen, dtype=torch.int32, device=device)

    # Get metadata
    from sgl_kernel.flashmla_ops import get_mla_metadata
    tile_scheduler_metadata, num_splits = get_mla_metadata(cache_seqlens, s_q * h_q // h_kv, h_kv)

    from sgl_kernel.flashmla_ops import fwd_kvcache_mla

    def run():
        return fwd_kvcache_mla(
            q, blocked_k, block_table, cache_seqlens, dv,
            tile_scheduler_metadata, num_splits, causal=True,
        )

    if ncu_mode:
        # Just run a few iterations for NCU to capture
        for _ in range(3):
            run()
        torch.cuda.synchronize()
        return

    # Warmup
    for _ in range(warmup):
        run()
    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(num_iters):
        run()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    avg_ms = elapsed / num_iters * 1000
    total_seqlens = batch * seqlen
    FLOPS = s_q * total_seqlens * h_q * (d + dv) * 2
    bytes_moved = (total_seqlens * h_kv * d + batch * s_q * h_q * d + batch * s_q * h_q * dv) * 2  # bf16 = 2 bytes

    print(f"  avg time: {avg_ms:.3f} ms")
    print(f"  TFLOPS: {FLOPS / 1e12 / (avg_ms / 1000):.2f}")
    print(f"  bandwidth: {bytes_moved / 1e9 / (avg_ms / 1000):.1f} GB/s")
    print()


def bench_prefill_sparse(num_iters=50, warmup=5, s_q=64, h_q=128, h_kv=1, d=576, dv=512, s_kv=4096, topk=256, ncu_mode=False):
    """Benchmark the prefill sparse BF16 kernel (sparse_prefill_fwd)."""
    print(f"=== Prefill Sparse BF16 ===")
    print(f"  s_q={s_q}, h_q={h_q}, h_kv={h_kv}, d={d}, dv={dv}, s_kv={s_kv}, topk={topk}")

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    from sgl_kernel.flashmla_ops import sparse_prefill_fwd

    q = torch.randn(s_q, h_q, d, dtype=torch.bfloat16, device=device)
    kv = torch.randn(s_kv, h_kv, d, dtype=torch.bfloat16, device=device)
    # Indices: (s_q, topk) — which KV tokens to attend to
    indices = torch.randint(0, s_kv, (s_q, topk), dtype=torch.int32, device=device)

    def run():
        return sparse_prefill_fwd(q, kv, indices, 1.0 / (d ** 0.5), s_kv)

    if ncu_mode:
        for _ in range(3):
            run()
        torch.cuda.synchronize()
        return

    # Warmup
    for _ in range(warmup):
        run()
    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(num_iters):
        run()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    avg_ms = elapsed / num_iters * 1000
    FLOPS = s_q * topk * h_q * (d + dv) * 2
    bytes_moved = (s_q * h_q * d + topk * s_q * h_kv * d + s_q * h_q * dv) * 2

    print(f"  avg time: {avg_ms:.3f} ms")
    print(f"  TFLOPS: {FLOPS / 1e12 / (avg_ms / 1000):.2f}")
    print(f"  bandwidth: {bytes_moved / 1e9 / (avg_ms / 1000):.1f} GB/s")
    print()


def bench_decode_sparse_fp8(num_iters=100, warmup=10, batch=128, h_q=128, h_kv=1, d=576, dv=512, topk=256, ncu_mode=False):
    """Benchmark the decode sparse FP8 kernel."""
    print(f"=== Decode Sparse FP8 ===")
    print(f"  batch={batch}, h_q={h_q}, h_kv={h_kv}, d={d}, dv={dv}, topk={topk}")
    print(f"  NOTE: This kernel requires FP8 KV cache format — may need special setup")

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # The FP8 sparse decode kernel has a complex setup with quantized KV cache
    # For now, just report that it needs the full serving stack
    print(f"  Skipping standalone benchmark — FP8 sparse decode requires full serving stack")
    print(f"  Use NCU on the running server instead:")
    print(f"    ncu --target-processes all --kernel-name regex:'sp_fp8' ...")
    print()


def main():
    parser = argparse.ArgumentParser(description="SM120 FlashMLA Kernel Benchmark")
    parser.add_argument("--kernel", type=str, default="all",
                       choices=["all", "decode_dense", "prefill_sparse", "decode_sparse_fp8"],
                       help="Which kernel to benchmark")
    parser.add_argument("--ncu", action="store_true",
                       help="NCU mode: run fewer iterations, no timing")
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--seqlen", type=int, default=4096)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")
    print()

    if args.kernel in ("all", "decode_dense"):
        bench_decode_dense(num_iters=args.iters, batch=args.batch, seqlen=args.seqlen, ncu_mode=args.ncu)

    if args.kernel in ("all", "prefill_sparse"):
        bench_prefill_sparse(num_iters=args.iters, ncu_mode=args.ncu)

    if args.kernel in ("all", "decode_sparse_fp8"):
        bench_decode_sparse_fp8(ncu_mode=args.ncu)


if __name__ == "__main__":
    main()
