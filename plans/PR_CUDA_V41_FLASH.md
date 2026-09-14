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
**`cuda-resident-expert-cache`**, because it helps any CUDA streaming model (the
branch stands on `bd66c40` alone and is gated on V4 Flash by itself, cache on
equal to cache off at sha `de8682e8546bc2df`, and on V4.1 through this port):

```
  generation  1.69 -> 5.28 t/s   (3.1x)   steady state, hit rate 0.868, driver 580.159.03
  prefill     1.63 -> 2.84 t/s
```

Verified byte-identical: greedy `--temp 0 --dump-logprobs`, 32 steps x 20
alternatives, cache on vs `DS4_CUDA_EXPERT_CACHE=0`, same sha256.

The four follow-ups landed the same day, each gated byte-identical to this
baseline (sha `e73f7635...`) and split across the two branches by where their
`ds4.c` half lives:

```
  cuda-resident-expert-cache (general)     v4.1-flash-cuda (this port)
    HOTLIST   persistent hot list             DRAINCUT  CUDA event replaces the
    PREADPOOL 8-thread pinned pread pool                 per-layer cudaDeviceSynchronize
    PAGECACHE fadvise ordering fix + readahead
    MemAvailable arena sizing
    the resident cache
```

A third branch, **`cuda-iq2-lut-gate-32`** (one commit off `bd66c40`, 15 lines), widens
the IQ2_XXS LUT gate/up decode kernel from 16 to 32 quantised input blocks. V4.1's
expert input is 5120 wide = 20 blocks, one past the old cap, so it fell to the generic
kernel. Measured with antirez's own `DS4_CUDA_MOE_PROFILE=1`: gate/up 0.660 -> 0.398
ms per layer, down unchanged, routed MoE 34 -> 23.3 ms per token. Byte-identical to
the same baseline sha.

Two more on `cuda-resident-expert-cache`, measured the same night with the same gate:

- **PRIMESPLIT** - the cache stats line now splits the prime phase into miss reads
  and copy-out. Measured: 33 ms of miss reads (180 MB per token off NVMe) and
  30 ms of arena-to-scratch device copy per token, hit rate 0.92.
- **ZEROCOPY** - the arena becomes three planes with the scratch's own per-tensor
  stride, and for decode the MoE kernels read the arena directly through a view;
  the copy-out is gone. **5.91 -> 7.30 t/s generation** (load 1.2, gpu 1%, driver
  580.159.03, ctx 32768), byte-identical. `DS4_CUDA_EXPERT_ZERO_COPY=0` restores
  the copy.


Measured payoff: the hot list lifts the cold-start hit rate from **0.501 to
0.733** at the first 4,000 lookups (load-independent). A matched A/B on a quiet
box (load ~3, same prompt, hot list warmed on it, driver 580.159.03, ctx 32768)
reads **5.37 -> 5.78 t/s, +7.6%**, morning tree against all four lanes - one
clean pair, so treat it as the size of the effect, not a precise figure. The
mechanism agrees: hit rate was 0.868 before these, so they trim slivers; the
resident cache was the lever. No published antirez V4.1 number exists to compare against;
his README says V4.1 Q2 streams from SSD on a 128 GB Mac too.

And two on `v4.1-flash-cuda`, from an nsys map of the decode token (dense Q8_0
matmuls already at ~240 GB/s, the byte floor; the general router at 197 us per
layer because one thread ran a 384x6 insertion sort):

- **ROUTERARGMAX** - the general router selects by block-wide argmax rounds with
  the insertion sort's tie rule (equal score to the lower index) and the same
  weight summation order: bit-identical by construction. **7.76 -> 8.12 t/s.**
- **ENGRAMTHREADS** - a decode token reads its 2x24 Engram rows (random reads into
  ~189 GiB of tables outside the map) through the batch reader, which gains a
  pthread path on Linux (it was Apple-dispatch only). 5.2 -> 2.2 ms per token.

A crash class fixed, found by a ~600-token prompt: the V4.1 prefill touched device
tensors through `ds4_gpu_tensor_contents`, which is shared memory on Metal and the
raw device pointer on CUDA - a host memset of `selected_comp`, a host read of the
selected ids in the prefill seed, host writes of the Engram rows (batch, per-row,
and the pipelined prefetch). All four are explicit device fills, reads and uploads
now, with a new `ds4_gpu_tensor_memset` on both backends. The short-prompt gate
passes on the fixed build. Two more gaps sat under it and are fixed on the same
branches: the batched prefill loaded no selected-expert cache on CUDA, so the launch
resolved whole expert tensors into device memory (an OOM kill around layer 13), and
the MMQ IQ2 prefill tier was not gated on streaming mode. A ~588-token prompt now
runs (batched prefill 8-9 t/s). It was then not bit-reproducible run to run: per-stage
checksums put the divergence in layer 0's routed MoE, and a switch sweep named the
expert-tile tier (reproducible with tiles off, not slower). The sorted-pairs scatter is
stable now (one thread, pair order) and on a streaming model the tile tier is opt-in
(`DS4_CUDA_MOE_EXPERT_TILES=1`). Two runs of the ~588-token prompt and a zero-copy-off
run all give `5566a268d8baab75`; the gate script carries it as `--long`.

One rejected: a 16-lane LUT gate/up kernel was 0.375 -> 0.307 ms per layer and
failed the gate - both arms agreed with each other and not with the baseline sha,
because the two kernels contract differently under `--use_fast_math`. It is kept
opt-in (`DS4_CUDA_MOE_LUT16=1`) and off. The gate script now asserts the baseline
sha as well as on-vs-off equality; that case passes the second check alone.

End of night, same box, same prompt, load 1-2, gpu 1%, driver 580.159.03, ctx
32768: **generation 1.69 -> 8.2 t/s** at a 48 GiB arena (five stamped runs 8.21-8.47, the minimum claimed); with the box freed to a 63 GiB arena, 8.48-9.25 over three runs, hit rate 0.95, every rung byte-identical to the baseline.

The CUDA-graph island lane was finished and measured: all four position-free parts
of the decode layer capture and replay byte-identically (160 captures per run), and
the speed is a null (three interleaved pairs, on 8.7-9.8 against off 9.6-9.9 t/s).
Launch issue was never on the critical path. Opt-in, off by default.

HITS-FIRST, on `cuda-resident-expert-cache`: the selected load starts the miss reads
and returns, and the routed MoE runs gate/up and a per-slot down projection for the
resident experts while the misses stream, then the miss slots, then the slot-ordered
sum. Bit-identical by construction and by the gates; **+5% on three interleaved
pairs** (9.96 / 10.06 / 9.99 vs 9.53 / 9.51 / 9.45 t/s).

## Submitting

`origin` is `antirez/ds4` for fetch **and push**, and this project's
standing rule is never to push there. Submitting needs a fork under our
own account first. That is the navigator's call, not a lane's.

STAGED HITS-FIRST (`lane-integrate`, d958c2a0e): the pool takes a stage boundary;
a layer's miss reads go gate/up first, down after; the miss gate/up kernel launches
after the first stage lands. Both gates pass; the speed A/B is owed on a quiet box.

ROUTER-AHEAD PREFETCH (`lane-integrate`, e89954d1b, opt-in, default off): correct
(both gates) and measured slower - the drive cannot land the guesses inside one
attention window (11-21 GB/s needed against ~10 available). Recorded as a null
with its mechanism; a faster drive changes the arithmetic.

Closed nulls this round: O_DIRECT off (-40%), pool at 24 threads, pread into the
arena (copy already 0.0 s), mapped-flag id readback (DRAINCUT already), sequential
short-prompt prefill (already the behaviour under 256 tokens), chunked miss reads
(a 3 MiB direct read is 416 us, bandwidth-bound).

Build recipe note for reviewers: the baseline sha e73f763588d4f253 belongs to the
no-arch build (`make -B ds4 ds4-server ds4-bench ds4-eval ds4-agent`, no CUDA_ARCH,
JIT from PTX); `make cuda-spark` (sm_121a native, MXF4) gives 187cf54a55a03b34 on
both arms from identical source. A gate must name its recipe.

## Upstream moved (2026-09-14)

antirez a04f46fa4 lands his own V4.1 CUDA port: bounded SSD streaming, exact short
prefills, accelerated large prefills, session batching, two-Spark RoCE tensor
parallelism. This branch is therefore no longer "the port"; it is a set of measured
levers on the streaming path: O_DIRECT expert reads, the read pool, MemAvailable
arena sizing, zero-copy hits, the Engram reader threads, router argmax, hits-first
and its staged form. Head-to-head on one box under one set of stamps (both arenas
squeezed to ~50 GB): upstream 5.45 / 5.62 t/s, ours 6.60 / 6.69 / 6.62 (+20%).
His greedy sha (bb06e711bc498bb9) differs from ours (e73f763588d4f253) in the last
ulp; each is self-consistent across its cache arms. The PR shape is now a decision:
rebase the levers onto a04f46fa4 as small PRs against his streaming path, or keep
the fork. Navigator's call.
