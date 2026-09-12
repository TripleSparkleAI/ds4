# PR: CUDA support for DeepSeek V4.1 Flash

Branch `cuda-deepseek-v4.1-flash`, one commit, forked from `bd66c40`
("DeepSeek v4.1 Flash support for Metal"). Engine files only - no plans,
no notes, no working material.

```
 Makefile             4 +-
 ds4.c              128 ++++----
 ds4_cuda.cu         14 +-
 ds4_cuda_dsv41.cuh 593 +++++++++++  (new)
 ds4_gpu.h            7 +-
```

## What it does

Runs the V4.1 Flash graph on CUDA alongside the existing Metal support.
Verified on an NVIDIA DGX Spark (GB10, sm_121, CUDA 13.0) against the Q2
GGUF with SSD streaming. Metal builds clean and is behaviourally
untouched.

## Why it is mostly gate-opening

The V4.1 graph was already shared GPU code - `metal_graph_*` compiles for
both backends and `ds41_gpu_graph` is backend-neutral. But the graph
region and all thirty dispatch sites sat behind `#ifdef __APPLE__`, so a
V4.1 model on CUDA silently ran the **V4** graph and died at the first FFN
encode. Most of the diff opens those gates and supplies the CUDA halves of
what they call.

## The eight things the port actually needed

1. **33 guards opened** from `__APPLE__` to `!DS4_NO_GPU` - three in the
   V4.1 graph region, thirty at dispatch sites.
2. **`ds4_cuda_dsv41.cuh`** - ten kernels ported 1:1 from
   `metal/dsv41.metal`, host wrappers mirroring `ds4_metal.m` validation.
   Reductions are all-lane xor butterflies, not `ds4_cuda.cu`'s
   `shfl_down` `warp_sum_f32`: Metal's `simd_sum` broadcasts to every
   lane and the ported kernels depend on that.
3. **General router** - the CUDA router was hard-locked to 256 experts /
   6 used / scale 1.5. V4.1 routes 6-of-384. The V4 shape stays as the
   fast path; a general kernel handles the rest.
4. **`ds4_engram.o` into the CUDA link** - Engram symbols become
   reachable once the V4.1 path is live.
5. **Shared type block** (`DS4_V41_TYPES_DEFINED` in `ds4_gpu.h`) - the
   CUDA translation unit does not include `ds4_gpu.h`, so both files
   define the same enums without clashing.
6. **Linux host-memory branch** in `glm_graph_host_memory_bytes()` - it
   returned 0 off-Apple, so the V4.1 memory admission refused to plan.
7. **Model cache tolerates unmapped tensors** - V4.1 keeps its Engram
   tables outside the mapped region by design (rows are read from disk),
   and every load failed on them.
8. **`ds41_cuda_stream_selected_load()`** - primes CUDA's per-layer
   selected-expert cache before the routed MoE, mirroring what
   `metal_graph_decode_cuda_selected_load` does for V4.

`DS4_V41_TRACE=1` names the failing graph step. Without it a V4.1 fault
surfaces as "gpu layer 0 ffn batch encode failed", which points nowhere.

## Performance - honest, and not yet good

**1.63 prefill / 1.69 generation t/s** on the Spark.

The cost is not the port's control flow. Timing split, per MoE layer:

```
  GPU drain (sync + id readback)   1.1 ms   ████
  synchronous expert fetch        12.4 ms   ████████████████████████████████████████████████
```

V4.1 is 20 causal encoder + 20 decoder layers, every one of them MoE.
Forty layers pulling six 9.5 MiB experts each is ~2.3 GiB per token, and
the fetch does not overlap with compute - so **~87% of every token is
spent waiting on SSD**.

An 87 GiB expert cache was tested and moved generation 1.61 -> 1.79 t/s,
so the cache is not the lever either. Prefetching each layer's experts
against the previous layer's compute is the obvious next step. It is not
attempted here; this PR is about correctness.

## Submitting

`origin` is `antirez/ds4` for fetch **and push**, and this project's
standing rule is never to push there. Submitting needs a fork under our
own account first. That is the navigator's call, not a lane's.
