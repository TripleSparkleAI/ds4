# PR: CUDA support for DeepSeek V4.1 Flash

Branch `v4.1-flash-cuda`, one commit, forked from `bd66c40`
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

## Performance - the bottleneck was found and it was not the port

Initial: **1.63 prefill / 1.69 generation t/s** on the Spark. Timing split, per MoE layer:

```
  GPU drain (sync + id readback)   1.1 ms   ████
  synchronous expert fetch        12.4 ms   ████████████████████████████████████████████████
```

V4.1 is 20 causal encoder + 20 decoder layers, every one MoE. Forty layers pulling
six 9.5 MiB experts each is ~2.3 GiB per token, and the fetch did not overlap with
compute - **~87% of every token was SSD reads.**

The cause is upstream's, at the fork point, not this port's: the five
`ds4_gpu_stream_expert_cache_*` entry points are stubs in `ds4_cuda.cu`
(`configured_count()` returns 0, the budget setters are `(void)` no-ops). Metal has
a full resident expert cache; CUDA had none, and `--ssd-streaming-cache-experts`
printed an 87 GiB plan nothing ever allocated. That is fixed on its own branch,
**`cuda-resident-expert-cache`**, because it helps any CUDA streaming model:

```
  generation  1.69 -> 5.28 t/s   (3.1x)   steady state, hit rate 0.868, driver 580.159.03
  prefill     1.63 -> 2.84 t/s
```

Verified byte-identical: greedy `--temp 0 --dump-logprobs`, 32 steps x 20
alternatives, cache on vs `DS4_CUDA_EXPERT_CACHE=0`, same sha256.

Still open, in flight on four lanes: parallel preads on the miss path, a persistent
hot list so the cache starts warm, page-cache eviction, and the 44 ms/token of
per-layer drain (now ~23% of a token). No published antirez V4.1 number exists to
compare against; his README says V4.1 Q2 streams from SSD on a 128 GB Mac too.

## Submitting

`origin` is `antirez/ds4` for fetch **and push**, and this project's
standing rule is never to push there. Submitting needs a fork under our
own account first. That is the navigator's call, not a lane's.
