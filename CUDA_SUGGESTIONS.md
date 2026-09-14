# CUDA suggestions for DeepSeek V4.1 Flash

This branch is upstream `a04f46fa4` ("DeepSeek v4.1 Flash support for CUDA")
plus the parallel pread pool (`triple-pr-parallel-ssd-reads`) plus one commit per
lever below. Each lever was ported by hand onto upstream's slot cache and
gated on a DGX Spark (GB10, 128 GB unified memory) against the same model.

Gate recipe, for anyone who wants to repeat it:

```
make -B ds4 ds4-server ds4-bench ds4-eval ds4-agent
./ds4 -m DeepSeek-V4.1-Flash-Q2.gguf --cuda --ssd-streaming \
      --ssd-streaming-cache-experts 90GB --ctx 32768 --temp 0 -n 32 \
      --dump-logprobs short.json -p 'Explain in two sentences why the sky is blue.'
sha256sum short.json | cut -c1-16     # bb06e711bc498bb9 on the pre-lever build
```

The long arm is the same with `-n 16` and a 588-token prompt (twelve copies of
one MoE-routing question); it dumps `d3355c94c70a4bb1` on the pre-lever build.
Byte-identical means both shas are unchanged after the lever. A greedy dump
that moves means a kernel changed the logits, which none of these should.

Speed numbers in this file are marked one of three ways. "Measured" is a
quiet box. "Under load, indicative only" carries the load average and
MemAvailable it was taken under and proves nothing. "Unmeasured" is honest.

## 1. Expert cache reserve knob

Commit: `cuda: expert cache reserve knob (DS4_CUDA_EXPERT_CACHE_MARGIN_GB) and a sizing stats line`

What it changes: the slot cache is sized from the nonmovable host memory
(MemAvailable minus CmaFree) with a fixed 8 GiB reserve. The reserve becomes a
knob. The default stays 8 GiB, so the slot count is unchanged unless the knob
is set.

Why: on a box shared with other work, 8 GiB was enough for the slots but not
for the KV cache, the staging pool and the page cache that arrive afterwards,
and the process was the OOM killer's first pick. Raising the reserve to 12
costs about 400 slots on a 128 GB part and survives a shared box. An idle box
can lower it.

What was not ported: the fork's own sizing source (MemAvailable instead of
cudaMemGetInfo) is already what upstream does here, with the CmaFree
subtraction on top. Nothing to gain.

Gate: byte-identical (short bb06e711bc498bb9, long d3355c94c70a4bb1).
Speed: no data path change; slot count unchanged at the default.
Knobs: `DS4_CUDA_EXPERT_CACHE_MARGIN_GB` (GiB, default 8, max 120).
`DS4_CUDA_EXPERT_CACHE_STATS=1` prints the free, reserve and budget the
sizing was made from.

## 2. Persistent learned hot list

Commit: `cuda: persistent learned hot list seeds the V4.1 SSD expert cache at startup`

What it changes: the CUDA slot cache counts every unique (layer, expert) it is
asked for and writes the counts at exit to
`~/.cache/ds4/cuda_expert_hotlist.txt`, hits descending, in the streaming
hotlist text format the loader in `ds4.c` already reads. On the next run the
V4.1 session seeds the cache from that file before the first prefill, the way
the GLM path seeds from its built-in list. Seeds are not counted as demand.
History is halved on load so hotness decays run over run. A `# model_size`
header keeps one model's list from seeding another. Seeds fill empty slots
only: when the cache is smaller than the list, each layer seeds its hottest
experts up to the free slots and skips the rest, so a small cache never
reads gigabytes it cannot hold.

Why: the cache starts empty every run and the hit rate is still climbing tens
of thousands of lookups later. Expert routing is stable across prompts, so
the last run's demand is a good first guess.

Where it hooks in: `ds4_session` creation for a V4.1 engine calls
`metal_graph_seed_streaming_expert_cache_from_hotlist` once per engine, with
the same zeroed seed graph the GLM path uses. That function now falls back to
the learned file for V4.1 Flash on CUDA when `DS4_METAL_STREAMING_EXPERT_HOTLIST`
is unset. A missing or unusable list never fails a session.

Gate: byte-identical with the seed active (short bb06e711bc498bb9 with 1278 and
1606 seeded slots on two runs, long d3355c94c70a4bb1 with 1450 seeded slots),
measured on the full branch tree with the learned file present. Under a
contested box (MemAvailable under 30 GiB) the long arm was OOM-killed three
times before the seed-fills-empty-slots rule landed and the box freed up; the
kills were NVRM out-of-memory, not a logit change.
Speed: unmeasured. Every run during this port shared the box with a 40 to 72 GB
neighbour and load 8 to 140, so the cache was sized at 355 to 1,931 slots instead
of the ~8,600 the budget allows, and a warm-versus-cold comparison at that size
says nothing. The mechanism is observable: DS4_CUDA_EXPERT_CACHE_STATS=1 prints
the seeded slot count at startup and the file is rewritten at exit (5,177 experts
after the first short run, 8,679 after the long one).
Knobs: `DS4_METAL_STREAMING_EXPERT_HOTLIST=<file>` names a list to read
(upstream's knob, unchanged). `DS4_CUDA_EXPERT_HOTLIST_WRITE=<file>` names the
file to write; `0` disables the writer. `DS4_METAL_DISABLE_STREAMING_EXPERT_HOTLIST=1`
disables seeding (upstream's knob). `DS4_METAL_STREAMING_EXPERT_AUTO_PRELOAD_CAP`
caps how many experts are seeded (upstream's knob, default 4096).

## 3. Event-gated expert-id readback (DRAINCUT)

Commit: `cuda: event-gated expert-id readback for the V4.1 streaming decode (DRAINCUT)`

What it changes: the streaming MoE path learns the router's expert ids with a
blocking `cudaMemcpy`. On the V4.1 decode path the shared-expert gate, up and
down matmuls are queued between the router and that read, once per MoE
layer, so the read waited for them too. Now `ds41_moe_partial` queues an
async copy of the ids into a pinned buffer plus an event straight after the
router select, before the shared expert. The streaming load waits on that
event only. Kernel order in the stream is unchanged; only what the host
blocks on changed.

Why: every layer drained the GPU to idle for work the ids never depended on,
then kept it idle while the host primed the expert cache.

Scope: single GPU only. The TP=2 path already runs its shared expert on its
own stream. Refused during CUDA graph capture; the blocking read then runs
as before.

Gate: byte-identical (short bb06e711bc498bb9, long d3355c94c70a4bb1).
Speed: unmeasured as an A/B. An interleaved on/off pair was attempted at load 6
and 28 to 30 GiB MemAvailable; three of four arms could not stage their experts
at all. The one complete pair, at load 6 to 8 and 41 to 42 GiB MemAvailable with the
GPU shared (9% and 38% utilisation), read generation 4.91 t/s with the event wait
and 5.10 t/s with the blocking read; under load, indicative only, and not a
verdict either way. The path is live: with
DS4_CUDA_EXPERT_CACHE_STATS=1 the stats line carries `armed_readbacks`, which
read 0 during prefill and 881 after 12,004 decode lookups on a 32-token run.
Knobs: `DS4_CUDA_SELECTED_DRAIN_SYNC=1` restores the blocking read for A/B.

## 4. Router argmax

Skipped. Upstream's `ds4_gpu_router_select_tensor` already dispatches a
warp-wide top-k kernel (`router_select_warp_topk_kernel`) for 256 and 384
experts by default; the one-thread insertion sort only runs when
`DS4_CUDA_NO_WARP_ROUTER_SELECT` is set. The fork's block-wide argmax rounds
were written against its own router and have nothing to add here.

## 5. Zero-copy planar arena

Skipped for decode, noted for prefill.

Decode: upstream's slot cache is already planar (`gate_ptr`, `up_ptr`,
`down_ptr`, one region per tensor, indexed by slot) and the routed MoE
kernels read it directly through a slot-remapped selected-id tensor
(`slot_selected_tensor`, `compact_count`). There is no per-token device
copy-out to remove.

Prefill: `cuda_stream_compact_prefill` still compacts the layer's selected
slots into a contiguous prefill buffer with device-to-device copies (or an
aligned repack on the IQ2 MMQ path) before the batched kernels run. The
fork's prefill-goes-zero-copy commit sized the batched tables from the slot
table it indexes and skipped that copy. It is a device-to-device copy of a
few hundred MiB per layer, not an SSD read, and porting it means teaching
the MMQ prefill kernels to index a sparse slot table. Left for a separate
branch.

## 6. Page cache

Commit: `cuda: drop staged expert pages after upload in the right order, readahead misses on the buffered path`

What it changes: the staged upload already asked the kernel to drop the file
pages it had just copied, but ran `posix_fadvise(DONTNEED)` before
`madvise(DONTNEED)`. Linux skips `FADV_DONTNEED` for pages still mapped by a
process, and the model file is mmap'd by the loader, so mapped pages were
never dropped. The release helper zaps the mapping first, then drops the
cache, at all three staged-copy sites. Misses on the buffered read path get a
`POSIX_FADV_WILLNEED` for every tensor of the layer before the pool
dispatches.

Why: on a unified-memory part every cached page of the model is a page the
expert slots cannot have.

Note: upstream opens the model `O_DIRECT` by default on Linux, so the
readahead half is a no-op by design there (the read bypasses the cache it
would fill) and the drop half only matters for pages touched through the
mmap. With `DS4_CUDA_NO_DIRECT_IO=1` both halves are live.

Gate: byte-identical (short bb06e711bc498bb9, long d3355c94c70a4bb1).
Speed: unmeasured.
Knobs: `DS4_CUDA_KEEP_MODEL_PAGES=1` keeps the pages; `DS4_CUDA_NO_EXPERT_READAHEAD=1`
skips the readahead.

## 7. Hits-first (opt-in)

Commit: `cuda: HITS-FIRST for the SSD expert cache (opt-in, DS4_CUDA_HITS_FIRST=1; measured a null on this tree)`

What it changes: with `DS4_CUDA_HITS_FIRST=1` the layer runs gate/up for the
resident experts and a per-slot down partial while the miss reads stream,
then the miss slots, then a slot-ordered sum from 0.0f. Off by default.

Why it is here: on the TripleSparkle backend the same idea is +5%. On
upstream's tree it measured a null (8.21 / 8.39 / 8.31 t/s with, 8.37 / 8.27 /
8.61 without, load 3-7) because upstream already overlaps the shared expert
with the miss reads, so the window hits-first fills is already busy. Kept
opt-in for the record.

Porting note: the base branch dropped the staged pool entry points the commit
relies on (`cuda_expert_pread_pool_dispatch_start_staged`,
`cuda_expert_pread_pool_wait_stage`, the `stage_boundary` fields), so the commit
carries them itself, and its state and entry points are forward-declared
before `routed_moe_launch`, which consumes them.

Gate: byte-identical with the flag off (the full branch tree: short bb06e711bc498bb9, long d3355c94c70a4bb1).
With the flag on, the short arm also matches (bb06e711bc498bb9, 1,122 slots); the long
arm with the flag on was not run on this tree.
Speed: measured a null on this tree (numbers above).
Knobs: `DS4_CUDA_HITS_FIRST=1` enables it.

## Cumulative speed

Unmeasured. The DGX Spark was shared with another tenant's 40 to 72 GB process
and load 8 to 140 for the whole port; the sparkport speed gate refuses a box
above load 8, and a t/s taken there measures the box. One arm of one pair read
`prefill 5.21 t/s, generation 4.33 t/s` at load 6.6 and 30 GiB MemAvailable
with the cache starved to a few hundred slots; under load, indicative only, and
not an A/B. The correctness gates are load-independent and were run on every
lever; the speed pass is owed on a quiet box, with the pool commit's own
+12% as the reference point.
