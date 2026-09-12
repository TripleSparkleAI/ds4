# SPARKPORT log — 2026-09-13

## ★ IT RUNS: DeepSeek V4.1 Flash generating on the DGX Spark (CUDA), coherent output
`./ds4 -m gguf/DeepSeek-V4.1-Flash-Q2.gguf --cuda --ssd-streaming --ctx 32768 -n 40 -p "The capital of France is"`
-> "The user asks a simple question ... likely wants the answer "Paris". Need to respond concisely. Paris."
prefill 1.38 t/s, generation 1.50-1.90 t/s (untuned; DS4_CUDA_DIRECT_MODEL=1 still required).

## What the port needed, in the order the failures appeared
1. ds4_cuda_dsv41.cuh - 10 kernels 1:1 from metal/dsv41.metal + wrappers (rope, quantize, engram_add,
   pool2, candidate blocks/filter, carry_copy, gather_kv, bf16_linear); unpacked indexer path.
2. ds4_cuda.cu - attention-output split into an impl with a round_low_bf16 hook.
3. ds4_gpu.h - V4.1 types moved into a DS4_V41_TYPES_DEFINED block (ds4_cuda.cu does not include it).
4. ds4.c - 33 guards flipped from __APPLE__ to !DS4_NO_GPU (3 in the graph, 30 dispatch sites);
   without the dispatch flips a V4.1 model silently ran the V4 graph and died in its FFN.
5. Makefile - ds4_engram.o added to the CUDA CORE_OBJS.
6. glm_graph_host_memory_bytes - Linux branch (sysconf), else the V4.1 memory admission sees host=0.
7. General CUDA router - the V4 router is hard-locked to 256 experts/top-6/scale 1.5; V4.1 routes 6 of
   384. Tuned path kept for the V4 shape, everything else takes dsv41_router_select_general.
8. ds41_cuda_stream_selected_load - CUDA's routed MoE requires the per-layer selected-expert streaming
   cache primed with this token's experts (the V4 decode path does this via
   metal_graph_decode_cuda_selected_load). Without it: "CUDA streaming selected experts are unavailable".

## Next
- Remove the DS4_CUDA_DIRECT_MODEL=1 requirement: accelerator_{prepare_model_tensor_spans,cache_q8_tensors}
  return false on tensors outside the map; V4.1's Engram tables are deliberately outside it -> skip, don't fail.
- Speed: ds41_cuda_stream_selected_load ends/begins the command buffer every layer (40 syncs/token).
  Batch the seeding or prime from the previous layer's selection.
- Gate: quality batch (gguf-tools/quality-testing/deepseek-v4.1-flash-20260910-batch) + M5 Metal golden
  (rsync at 41/366 GB) for greedy top-1 agreement.

## Later that day: the speed question, answered and then moved
- DS4_CUDA_DIRECT_MODEL=1 requirement removed (model cache skips unmapped tensors). Committed.
- Debug scaffolding stripped; 6 layer-level DS4_V41_TRACE calls kept. Metal still builds clean.
- Two PR branches cut off bd66c40, engine files only, named to upstream house style
  (glm-tp-rocm / glm-5.3-flash pattern): `v4.1-flash-cuda` (the port) and
  `cuda-resident-expert-cache` (the perf fix, general to any CUDA streaming model).
- MEASURED with a split timer: per MoE layer the GPU drain (sync + id readback) is 1.1 ms and the
  expert fetch is 12.4 ms. 40 layers -> 496 ms of a 575 ms token was SSD reads. The per-layer sync
  I suspected was NOT the cost.
- ROOT CAUSE: the five ds4_gpu_stream_expert_cache_* entry points are STUBS in upstream ds4_cuda.cu
  at bd66c40 (configured_count returns 0, setters are (void) no-ops). Metal has a full resident
  cache; CUDA had none. --ssd-streaming-cache-experts printed an 87 GiB plan nothing honoured.
  Verified at the fork point: not our error. Also not better than his - Metal's cache is richer.
- BUILT the cache: device arena of expert triples keyed (layer, expert), LRU. Byte-identical greedy
  logprob A/B (32 steps x 20 alts, same sha256) between cache on and DS4_CUDA_EXPERT_CACHE=0.
  generation 1.69 -> 4.56 t/s.
- OOM lesson: on GB10 unified memory cudaMalloc overcommits and the OOM killer fires on first
  touch, so a halving loop never triggers. Cap against real free memory.
- SECOND lesson: cudaMemGetInfo reports MemFree; MemAvailable was 59.68 vs MemFree 31.62 because
  28.76 GiB was reclaimable page cache. Sizing from MemAvailable: 5512 experts / 51 GiB, stable.
  generation -> 5.28 t/s steady state, hit rate 0.868 and still climbing at 72k lookups.
- NEGATIVE: hotness-with-decay eviction measured worse than LRU (0.851 vs 0.868; 4.95 vs 5.28).
  Reverted. Confounded by non-greedy text; not re-run because it is the smallest lever.
- V4.1 Q2 file is 340.6 GiB, NOT ~142 GiB as earlier notes said. That is why it streams on 121 GB.
- antirez publishes NO V4.1 t/s. The 39.35 tok/s M5 number is V4 Flash (81 GB, resident). His
  README: V4.1 Q2 "runs with SSD streaming on one 128 GB Mac" - same regime as us.
- NAVIGATOR RULE: never run models on the M5; Spark only. M5 partial V4.1 copy deleted (+115 GiB).

## Fleet (2026-09-13 evening): port the rest of Metal's streaming subsystem to CUDA
Census of ds4_metal.m vs ds4_cuda.cu found the gaps cluster in one subsystem. Four lanes, own
worktree + branch each, off sparkport-v41-cuda at dc09a1969. Spark contested: BUILD ONLY, no runs.
  PREADPOOL  lane-preadpool  parallel pread worker pool (CUDA reads serially)
  HOTLIST    lane-hotlist    persistent hot-expert list, warm start (hit rate starts at 0 every run)
  PAGECACHE  lane-pagecache  posix_fadvise DONTNEED + readahead (page cache fights the arena)
  DRAINCUT   lane-draincut   the 44 ms/token of per-layer sync, ~23% of a token now; exploratory
All four died on a rate limit seconds in, resumed on the new login with context intact.
Verification: sparkport-verify.sh (committed 3d50648a5) - correctness gate is load-independent
and runs now; speed gate refuses a busy box. Driver stamp 580.159.03 captured; earlier numbers
were driver-unstamped.

## Fleet results (2026-09-13, later): four lanes, four passes, one integrated tree
Every lane's greedy logprob A/B (cache on vs DS4_CUDA_EXPERT_CACHE=0) is byte-identical AND matches the
untouched baseline's sha256 e73f763588d4f253 - so PREADPOOL's threaded reads and DRAINCUT's event path
emit the identical logit distribution to the original serial, device-synced code.
  PAGECACHE  1bb1a7041  PASS  (an earlier FAIL was an OOM from ~60 GiB of someone else's work on the
                              box, not the code; re-run passed). Finding: fadvise(DONTNEED) already
                              existed but ran BEFORE the madvise that unmapped the pages, so Linux
                              skipped every page and it silently no-op'd. O_DIRECT is on by default,
                              so the WILLNEED readahead half is inert on this path.
  PREADPOOL  865ffbd01  PASS  8 pthreads, pinned staging + non-blocking streams, serial fallback.
  DRAINCUT   4ff4c2a27  PASS  CUDA event + pinned async D2H replaces cudaDeviceSynchronize per layer.
                              Also PASS on its own toggle: DS4_CUDA_SELECTED_DRAIN_SYNC=1 vs unset,
                              byte-identical. Corrected my brief: end_commands is cudaDeviceSynchronize,
                              not a stream sync, unless DS4_CUDA_END_STREAM_SYNC is set.
  HOTLIST    c98c0acf2  PASS  persistent hot list at ~/.cache/ds4/cuda_expert_hotlist.txt, Metal v1
                              format, fills empty slots only. Run 1 wrote 4801 experts; run 2 loaded
                              and seeded per layer (166/302/421/537 resident during token 1).
Integrated: lane-integrate at b52898fab. Two textual conflicts, both the four lanes inserting at the
same anchor in cuda_stream_selected_cache_begin_load; resolved as hotlist record+prewarm FIRST, then
PAGECACHE readahead, then the PREADPOOL pool block. Builds on the Spark rc=0, 8 pre-existing warnings.
Its own correctness gate + (if quiet) speed gate are in flight. Box memory contested all evening: the
arena sized 2184-3657 experts where an empty box gives ~5500, so no speed number today is comparable
to this morning's 5.28 t/s.

## 2026-09-14 ~03:50 THE FIRST V4.1 PER-PHASE PROFILE (DS4_V41_PROFILE=1, commit 4efacc59f)
Integrated tree, load 4.60 / gpu 6% / driver 580.159.03 / ctx 32768, arena 3450 (box-capped, 64 GiB
avail), hit 0.769, wall 4.87 t/s = 205 ms/token WITH profiling syncs. Marginal over tokens 88->96,
un-nesting moe.prime/moe.routed from moe:
   moe.prime  72 ms 32%   misses at hit 0.77 - contention-capped arena, hotlist+full arena is the fix
   before_moe 48 ms 21%   UNEXPECTED - candidate: Engram rows read from DISK synchronously mid-token
   moe.routed 35 ms 15%
   attention  34 ms 15%
   other      30 ms 13%   (router+shared expert inside moe, loop driver, sampling)
   before/after_attention, after_moe ~12 ms 5%
   sum ~228 ms vs wall 205: the loop driver's own syncs are NOT a big hidden term.
Byte census: 9.1 GiB/token at 5.4 t/s = 49 GB/s = 21% of the 230.5 GB/s ceiling -> the wall is
latency/serialisation, not bytes. Roofline ceiling ~25 t/s at achieved bandwidth.
LEVER CANDIDATE (paper-backed, our own kgraph EngramContestedNotSettled notes it): Engram's hash
depends only on token IDs, so its rows can be PREFETCHED before layer 1 - we read them synchronously.
- CORRECTION to the profile reading above (same hour): `before_moe` is NOT a separate 48 ms - the
  function ds41_graph_before_moe IS the nested attention triple (before_attention + attention +
  after_attention = 6+34+5), the same nesting as moe containing moe.prime. And Sinkhorn is ONE
  kernel launch with the iterations inside - not a launch storm. Two hypotheses (Engram disk reads,
  Sinkhorn storm) died on reading: Engram rows are already prefetched on a thread by antirez.
  UN-NESTED marginal per token (wall 205): prime 72 · routed 35 · attention 34 · router+shared 26 ·
  untraced 26 · small phases 12. After the prime, the kernels run at 27-42% of the 230.5 GB/s
  ceiling (routed: 2.2 GiB in 35 ms = 63 GB/s). The lever after residency is BATCH-1 KERNEL
  EFFICIENCY - the same finding as V4 on this box, worse on V4.1. Antirez's Metal fuses gate+up+swiglu
  (mul_mv_addr_iq2_pair_swiglu); whether CUDA does is being read.
- MEMORY TRACE (each tree alone, 5 s samples): morning tree MemAvailable 69 -> 7 -> 7 GiB, min 3,
  survived; integrated 69 -> 1 GiB, min 1, killed rc=137 with a foreign job at load 12 on top. Cache
  build saw ~44 GiB avail both times and took ~36 GiB of arena. ⇒ BOTH trees run the box to 1-7 GiB
  free; the 8 GiB margin is too thin once KV, staging and the streaming reads' page cache land. Any
  foreign allocation then kills whichever tree is running; integrate is a few hundred MiB heavier
  (pinned pool staging, hot-list tables) so it dies first. Not a lane bug - a margin. Fix: margin ->
  a knob, DS4_CUDA_EXPERT_CACHE_MARGIN_GB, default 12; costs ~430 experts of arena on this box.
