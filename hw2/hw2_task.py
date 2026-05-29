import torch
from utils import (
    build_model,
    get_input_ids,
    slow_loop,
    time_generation,
    MODEL_NAME,
    PROFILE_STEPS,
    RESULTS_DIR,
)


def optimized_loop(model, input_ids, n_steps):
    # DONE: fix the performance issues you found — changes may include
    # both `optimized_loop` and `generate_optimized`
    # generated_ids = input_ids.clone()
    # generated_tokens = []
    # for _ in range(n_steps):
    #     outputs = model(input_ids=generated_ids)
    #     next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1)
    #     token_value = next_token_id.item()
    #     generated_tokens.append(token_value)
    #     generated_ids = torch.cat([generated_ids, next_token_id.unsqueeze(0)], dim=1)
    # return generated_tokens

    # Fast autoregressive generation loop.
    #
    # Main fixes vs slow_loop:
    # 1. Use KV cache: prefill full prompt once, then decode one token at a time.
    # 2. Avoid recomputing the whole growing sequence every step.
    # 3. Avoid `.item()` inside the loop, because it synchronizes GPU -> CPU.
    # 4. Avoid repeated `torch.cat` of generated_ids inside the loop.
    # 5. Ask the model to return logits only for the last token when supported.

    if n_steps <= 0:
        return []

    generated_tokens = []

    with torch.inference_mode():
        # Prefill: process the whole prompt once and build the KV cache.
        outputs = model(
            input_ids=input_ids,
            use_cache=True,
            logits_to_keep=1,
        )
        past_key_values = outputs.past_key_values

        # First generated token comes from the last prompt position.
        next_token_id = torch.argmax(
            outputs.logits[:, -1, :],
            dim=-1,
            keepdim=True,
        )
        generated_tokens.append(next_token_id)

        # Decode: from now on, feed only the latest token and reuse KV cache.
        for _ in range(n_steps - 1):
            outputs = model(
                input_ids=next_token_id,
                past_key_values=past_key_values,
                use_cache=True,
                logits_to_keep=1,
            )
            past_key_values = outputs.past_key_values

            next_token_id = torch.argmax(
                outputs.logits[:, -1, :],
                dim=-1,
                keepdim=True,
            )
            generated_tokens.append(next_token_id)

    # Convert to a Python list once at the end.
    # This creates only one GPU -> CPU synchronization, not one per token.
    return torch.cat(generated_tokens, dim=1).squeeze(0).detach().cpu().tolist()




def profile(loop_fn, model, input_ids, trace_name: str):
    # DONE: wrap loop_fn(model, input_ids, PROFILE_STEPS) with torch.profiler,
    # print the summary table, and export a Chrome trace to RESULTS_DIR / trace_name
    # Profile one short generation run and export a Chrome trace.
    # PROFILE_STEPS is intentionally small, so the trace remains navigable.

    torch.cuda.synchronize()

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        loop_fn(model, input_ids, PROFILE_STEPS)

    torch.cuda.synchronize()

    print(
        prof.key_averages().table(
            sort_by="self_cuda_time_total",
            row_limit=25,
        )
    )

    prof.export_chrome_trace(str(RESULTS_DIR / trace_name))


def generate_optimized(optimized_trace_name: str) -> float:
    # DONE: load the model (consider dtype and other loading options),
    # then call profile() and time_generation() on optimized_loop.
    # Return the elapsed time from time_generation so main() can print a speedup.

    # Enable TF32 for matmul-like operations. This is usually faster on NVIDIA GPUs
    # and is acceptable for this synthetic benchmark.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    # Use BF16 to reduce memory bandwidth and speed up matmuls on L40S/H100.
    # If this causes issues on your GPU, change to torch.float16 or torch.float32.
    model = build_model(torch.bfloat16)
    input_ids = get_input_ids()

    profile(optimized_loop, model, input_ids, optimized_trace_name)
    optimized_elapsed = time_generation(optimized_loop, model, input_ids, "Optimized")

    del model
    torch.cuda.empty_cache()

    return optimized_elapsed


def main():
    print("=" * 60)
    print("HW2: LLM Inference Optimization")
    print(f"Model: {MODEL_NAME}")
    print("=" * 60)

    print("\n--- Part 1: Slow baseline ---")
    model = build_model(torch.float32)
    input_ids = get_input_ids()
    profile(slow_loop, model, input_ids, "v0_slow_trace.json")
    slow_elapsed = time_generation(slow_loop, model, input_ids, "Slow")
    del model
    torch.cuda.empty_cache()

    print("\n--- Part 2: Optimized ---")
    optimized_elapsed = generate_optimized(optimized_trace_name="v1_optimized_trace.json")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if optimized_elapsed is None or optimized_elapsed <= 0:
        print("generate_optimized() did not return a positive elapsed time; "
              "cannot compute speedup.")
    else:
        speedup = slow_elapsed / optimized_elapsed
        print(f"  Slow:      {slow_elapsed:6.2f}s")
        print(f"  Optimized: {optimized_elapsed:6.2f}s")
        print(f"  Speedup:   {speedup:6.2f}x  (vs V0 slow baseline)")


if __name__ == "__main__":
    main()


# ============================================================================
# Writeup
# ============================================================================
#
# Changes made and speedup per fix:
# Final result:
#   Slow baseline: 128 tokens in 1.52s  (84.0 tok/s)
#   Optimized:     128 tokens in 0.20s  (636.7 tok/s)
#   Speedup:       7.58x
#
# The optimized loop produced the same token preview as the baseline:
#   [775, 1973, 97, 2453, 295, 695, 775, 866]
#
# Main changes:
#
# 1. KV cache:
#    The baseline recomputed the full growing sequence at every decode step:
#    length 1024, then 1025, then 1026, etc. The optimized version does one
#    prompt prefill, saves past_key_values, and then decodes one token at a time.
#    This was the biggest optimization.
#
# 2. BF16 model:
#    The optimized path loads the model in bfloat16, which reduces memory traffic
#    and uses faster BF16 kernels on L40S/H100-class GPUs.
#
# 3. Removed per-step .item():
#    The baseline calls .item() every step, causing GPU -> CPU synchronization.
#    The optimized loop keeps tokens on GPU and converts to a Python list only
#    once at the end.
#
# 4. Removed repeated full-sequence torch.cat:
#    With KV cache, the next forward pass only needs the latest token plus
#    past_key_values, so we avoid repeatedly rebuilding generated_ids.
#
# 5. Used logits_to_keep=1:
#    During generation only the final-position logits are needed, so this avoids
#    unnecessary logits computation/storage for earlier positions.
#
# Profiler evidence:
#   Total self CUDA time: 119.667ms -> 7.495ms
#   aten::mm self CUDA:   103.384ms -> 5.040ms
#   Baseline aten::cat allocated ~792.94 MB CUDA memory, showing the overhead of
#   repeatedly materializing the growing sequence.
#
# Biggest impact and why:
#
# KV caching had the biggest impact because it removes redundant full-context
# transformer computation. Instead of recomputing the 1024-token prompt and all
# previous generated tokens on every step, the optimized loop computes the prompt
# once and reuses cached keys/values for cheap one-token decode steps.
