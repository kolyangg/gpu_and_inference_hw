import torch
import statistics

# ============================================================================
# Part 1: Implement PyTorch Functions
# ============================================================================
#
# TASK 1a: Implement an operation with the lowest arithmetic intensity.
# Use an op that performs essentially memory traffic with ~0 useful FLOPs
# per element.


def lowest_ai_fn(x: torch.Tensor) -> torch.Tensor:
    """Lowest arithmetic intensity baseline (0 FLOP/Byte)."""
    # DONE (1 line): implement a lowest-AI op
    return x.clone()


# TASK 1b: Implement a function with configurable arithmetic intensity.
# Build an element-wise compute operation where work increases with `num_ops`.
# Design it so fused arithmetic intensity grows roughly linearly with `num_ops`,
# while each element is still read/written once at the kernel boundary.
# Return either the eager function or a compiled version depending on the
# `compiled` flag so we can compare both on the roofline plot.
#
# Use an accumulator variable and implement fused multiply-add (FMA) style work
# explicitly, e.g. `acc = acc * x + x`, so each loop iteration contributes
# about 2 FLOPs per element in a realistic GPU-friendly pattern. We prefer this
# pattern here mainly because it gives clean FLOP accounting and resembles the
# kind of floating-point work GPUs are designed to do; Avoid patterns like repeated
# doubling (`x = x + x`), since long self-dependent pointwise chains can trigger
# very poor Inductor compile-time behavior and are also less useful for this
# roofline exercise.


def make_compute_fn(num_ops: int, compiled: bool = True):
    """Return an eager or compiled function whose work scales with num_ops."""

    def fn(x: torch.Tensor) -> torch.Tensor:
        acc = x
        for _ in range(num_ops):
            acc = acc * x + x
        return acc

    # DONE (1 line): return either `fn` or `torch.compile(fn)` based on `compiled`
    return torch.compile(fn) if compiled else fn


# ============================================================================
# Part 2: Benchmarking
# ============================================================================
#
# TASK 2: Complete the benchmark function using CUDA events.
# CUDA events measure GPU time precisely (not CPU wall time), which avoids
# including kernel launch overhead or CPU-GPU synchronization delays.


def benchmark_fn(fn, *args, warmup=25, rep=100) -> float:
    """Benchmark a GPU function using CUDA events.

    Returns median execution time in milliseconds.
    """
    # Warmup (triggers torch.compile on first call, then warms caches)
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    # DONE: time `rep` runs using CUDA events and return median latency (ms)
    times = []
    for _ in range(rep):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        fn(*args)
        end.record()

        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    return float(statistics.median(times))


# TASK 3: Compute element-wise operation metrics from measured runtime.
# Count every arithmetic operation performed inside the loop (careful: each
# `acc = acc * x + x` iteration does more than one FLOP per element).
#
# Use different byte-traffic models for the two variants:
#   - compiled: assume the operation is fused, so each element is read once and
#     written once at the kernel boundary
#   - eager: estimate the traffic from the separate multiply and add operations
#     launched by PyTorch in each loop iteration, including intermediate tensors
#
# Return a tuple with:
#   - total_flops
#   - arithmetic_intensity  (FLOP / Byte)
#   - achieved_flops        (FLOP / s)


def compute_elementwise_metrics(num_elements, num_ops, bytes_per_element, ms, variant):
    # DONE: compute total FLOPs, arithmetic intensity, and achieved FLOP/s
    total_flops = num_elements * num_ops * 2

    if variant == "compiled":
        # Fused kernel: one read + one write at kernel boundary.
        total_bytes = num_elements * 2 * bytes_per_element
    elif variant == "eager":
        # Each loop iteration: multiply reads 2 tensors+writes 1,
        # add reads 2 tensors+writes 1 => 6 elements of traffic per op.
        total_bytes = num_elements * num_ops * 6 * bytes_per_element
    else:
        raise ValueError(f"Unknown variant: {variant}")

    ai = total_flops / total_bytes
    achieved_flops = total_flops / (ms * 1e-3)
    return total_flops, ai, achieved_flops


# ============================================================================
# Part 3: Short Writeup
# ============================================================================
# Answer these after you generate `results/roofline.png` and inspect the points.
#
# Q1. Look at the compiled element-wise operations from `1 ops` through `64 ops`.
# Why does performance rise as arithmetic intensity increases even though the
# measured runtime changes only a little?

# The compiled element-wise operations are fused by torch.compile, so the tensor
# is read from memory and written back roughly once at the kernel boundary, while
# the extra arithmetic work is done mostly in registers. Therefore, increasing
# num_ops increases the number of FLOPs but does not increase external memory
# traffic by the same amount.
#
# In my run, compiled runtime stayed almost flat from 1 ops to 64 ops:
#   1 ops:  0.870 ms, AI 0.25 FLOP/B, 0.15 TFLOP/s
#   64 ops: 0.875 ms, AI 16 FLOP/B, 9.82 TFLOP/s
#
# So performance rises because achieved FLOP/s = total FLOPs / runtime. The
# runtime is almost unchanged, but the amount of arithmetic grows linearly with
# num_ops. This is the expected roofline behavior for a fused operation moving
# along the memory-bandwidth-limited part of the roofline.
#

# Q2. In one sample run, `matmul 1024x1024` achieved lower FLOP/s than the
# `128 ops` compiled element-wise operation. Give one or two reasons why that can
# happen on a large GPU like an H100.
 
# In my L40S run this specific inversion did not happen: matmul 1024x1024 reached
# 23.04 TFLOP/s, while the 128 ops compiled element-wise kernel reached
# 19.62 TFLOP/s. However, it can happen on a very large GPU like H100 because
# 1024x1024 matmul is relatively small and may not fully occupy all SMs or hide
# overheads well. Kernel launch overhead, cuBLAS kernel selection, tiling
# inefficiencies, and insufficient parallel work can all reduce achieved FLOP/s
# for a small matmul.
#
# By contrast, the compiled element-wise benchmark runs over a very large vector
# of 64M elements, giving the GPU a huge amount of independent parallel work.
# Even though it is a simple operation, the fused kernel can keep many threads
# active and produce a high measured FLOP/s when num_ops is large.
 

# Q3. Between `64 ops` and `128 ops`, runtime increases more noticeably than it
# did for smaller operations. What does that suggest about what resource is
# becoming the bottleneck?
 
# In my run the increase was still tiny: 64 ops took 0.875 ms and 128 ops took
# 0.876 ms. But conceptually, if the 64 -> 128 jump starts increasing runtime
# more noticeably, it suggests the kernel is moving away from being purely
# memory-bandwidth limited and is starting to run into compute-side limits.
#
# At low num_ops, the bottleneck is mostly reading/writing memory, so adding more
# arithmetic can be hidden inside roughly the same memory-transfer time. At high
# num_ops, there is enough arithmetic per element that instruction throughput,
# register pressure, occupancy, and FP32/FMA execution capacity begin to matter.
# In roofline terms, the point is moving closer to the ridge point, where the
# bottleneck transitions from memory bandwidth toward compute throughput.

# Q4. Why do the eager `ops-K` points look so different from the compiled ones?
 
# The eager points look different because eager PyTorch does not fuse the loop
# into one kernel. Each iteration of:
#
#     acc = acc * x + x
#
# is effectively executed as separate multiply and add kernels:
#
#     tmp = acc * x
#     acc = tmp + x
#
# This materializes intermediate tensors and causes much more memory traffic.
# Per iteration, the multiply reads acc and x and writes tmp: 3 element transfers.
# The add then reads tmp and x and writes acc: another 3 element transfers. So
# eager mode has about 6 element transfers per iteration, while doing only
# 2 FLOPs per element.
#
# Therefore, as num_ops increases in eager mode, both FLOPs and memory traffic
# increase together, so arithmetic intensity stays almost constant. In my run,
# all eager points had AI ~= 0.0833 FLOP/B and stayed around 0.05-0.06 TFLOP/s.
#
# In compiled mode, torch.compile fuses the operations, so memory traffic stays
# closer to one read plus one write, while FLOPs increase with num_ops. That is
# why the compiled points move up and to the right along the roofline, while the
# eager points remain clustered at low arithmetic intensity.