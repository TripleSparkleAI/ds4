# triple-cuda-backend-archive

The TripleSparkle CUDA backend for DeepSeek V4.1 Flash on the NVIDIA DGX Spark
(GB10, sm_121, CUDA 13.0, 121 GB unified memory), against the Q2 GGUF with SSD
streaming. Forked from upstream `bd66c40` ("DeepSeek v4.1 Flash support for
Metal"), before upstream's own V4.1 CUDA commit `a04f46fa4`, so this branch
and upstream `main` are two independent CUDA paths for the same model.

Every commit on this branch was gated byte-identical on greedy
`--temp 0 --dump-logprobs` runs before it was kept: a short V4.1 prompt, a
~588-token V4.1 prompt run twice, and V4 Flash with the expert cache on and
off. Generation on the Spark went from 1.69 to 9.5-10.1 t/s.

## What it contains

- **The port.** Thirty-three Metal-only guards opened, ten V4.1 kernels ported
  from `metal/dsv41.metal` into `ds4_cuda_dsv41.cuh`, a general 6-of-384
  router (block argmax rounds, same tie rule as the one-thread sort it
  replaced), Engram tables tolerated outside the mapped region, a Linux
  host-memory branch, per-layer selected-expert priming, Engram rows read in
  parallel on Linux, four host-view segfaults on ~600-token prompts fixed, and
  the batched prefill loading its experts through the cache instead of
  resolving whole 3.6 GiB layers into device memory.
- **The resident expert arena.** Upstream's five `ds4_gpu_stream_expert_cache_*`
  entry points were stubs on CUDA. This is a resident LRU arena of expert
  triples, sized from `/proc/meminfo` MemAvailable because `cudaMemGetInfo`
  reports MemFree on the GB10 and `cudaMalloc` overcommits. 1.69 to 4.56 t/s
  on its own.
- **Zero-copy views.** The arena is planar and the MoE kernels index it
  directly, for decode and prefill, so the 2.2 GiB per token of
  arena-to-scratch copy is gone. 5.91 to 7.30 t/s.
- **DRAINCUT.** An event-gated expert-id readback instead of a device drain
  per layer.
- **The pread pool.** Eight worker threads, pinned staging, one upload stream
  each, O_DIRECT. Misses are NVMe-bound at any thread count above that.
- **The persistent hot list.** `~/.cache/ds4/cuda_expert_hotlist.txt` warms
  the arena at startup from the previous run's hit counts.
- **HITS-FIRST and STAGED HITS-FIRST.** The layer runs gate/up and the per-slot
  down projection for the experts already resident while the misses stream,
  then the miss slots, then the same slot-ordered sum. +5% (9.96/10.06/9.99
  against 9.53/9.51/9.45 t/s). Staged reads a miss's gate/up tensors before
  its down tensor so the miss gate/up kernel runs while the down bytes land.
- **The IQ2_XXS LUT gate/up kernel** covers 32 input blocks; V4.1's 5120-wide
  input is 20 blocks, one past the old cap of 16. Routed MoE 34 to 23 ms per
  token.
- **Opt-in, measured null, kept for the record.** CUDA-graph islands
  (`DS4_CUDA_V41_DECODE_GRAPH=<mask>`), router-ahead prefetch
  (`DS4_CUDA_ROUTER_AHEAD=1`), the 16-lane LUT kernel (not bit-identical),
  the LUT kernel at 16 rows per block.

## Knobs

`DS4_CUDA_EXPERT_CACHE=0`, `DS4_CUDA_EXPERT_CACHE_STATS=1`,
`DS4_CUDA_EXPERT_CACHE_MARGIN_GB` (default 12; raise it on a shared box),
`DS4_CUDA_EXPERT_ZERO_COPY=0`, `DS4_CUDA_HITS_FIRST=0`,
`DS4_CUDA_HITS_FIRST_STAGED=0`, `DS4_CUDA_MOE_EXPERT_TILES=1`,
`DS4_CUDA_MOE_UNSTABLE_SCATTER=1`, `DS4_V41_PROFILE=1`,
`DS4_V41_PREFILL_CHECKSUM=1`, `DS4_V41_TRACE=1`.

## Against upstream a04f46fa4, same box, same day, interleaved

Upstream 5.45-6.23 t/s, this branch 6.60-9.38 t/s, with the arena squeezed to
~50 GB by other tenants. Hit rate 0.90-0.91 here against 0.84-0.87 there at
equal slot counts. Upstream wins short exact prefill (4.2-4.5 vs 2.1-2.5
t/s) and has session batching and dual-Spark RoCE, which this branch does not.

The full measurement log and the gate script are on `triple-notes`.
