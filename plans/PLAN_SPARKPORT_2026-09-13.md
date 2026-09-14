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
- INTEGRATED TREE b52898fab PASSES its own gate: byte-identical, sha e73f763588d4f253. Speed refused
  (load 11.84) - correct behaviour, not a measurement.
- HOTLIST PAYOFF, load-independent: cold run 1 first-4000-lookup hit_rate 0.501; seeded run 2 at the
  same checkpoint 0.733, on a SMALLER arena (2700 vs 3657 experts). The warm start reaches at token ~17
  what a cold run reached after ~72k lookups. Steady ~0.74 capped by the tiny contested-box arena.
- LANDED on the clean PR branches and both build rc=0 on the Spark:
    cuda-resident-expert-cache: cache 5cf94a2d3 -> MemAvailable 231a2db9f -> PAGECACHE 2281f0deb ->
                                PREADPOOL 3dd0e7542 -> HOTLIST 29a93f6ca   (general, ds4_cuda.cu only)
    v4.1-flash-cuda:            port 7cbb2d183 -> DRAINCUT 65c16628c       (its ds4.c half is V4.1 code)
  Same two same-anchor conflicts as the integrate merge, resolved the same way. Lane worktrees removed
  and lane-* branches deleted after merge-base --is-ancestor confirmed each is in lane-integrate.
- OWED: the general branch gated on ITS OWN tree. Tried V4 Flash (ds4flash.gguf, 81 GB) with
  --ssd-streaming: BOTH arms OOM-killed rc=137, cache-OFF arm included, so it is box contention
  (~50 GiB free) not the branch. Seen before death: hotlist geometry guard correctly refused to seed a
  different model; pread pool started 8 threads; arena adapted to V4's 6.75 MiB expert. Retry is
  self-gated on MemAvailable >= 100 GiB, chained ahead of the integrate speed gate.
- 03:10 the box went quiet (load 4.76 at gate time) and the watcher fired. INTEGRATED TREE, all four
  lanes: correctness PASS byte-identical (third time), then SPEED prefill 2.15 / generation 5.49 t/s,
  stamped load 6.85, gpu 7%, driver 580.159.03, ctx 32768, prompt = the MoE-routing explanation.
  Against the morning's 5.28 (cache + MemAvailable only, hit rate 0.868) that is +4% - INSIDE load
  variance, NOT claimed as a win. Confounds: load 6.85 is not idle; arena size not printed; the hot
  list was trained on the sky prompt, not this one. Owed: a matched idle A/B, morning tree vs
  integrated, same prompt, hotlist warmed on it.
- V4 own-tree gate for cuda-resident-expert-cache still SKIPPED (72 GiB free, needs ~100). A 3-hour
  self-gated watcher is queued for it.
- 03:13 MATCHED A/B on a quiet box (load 3.09, gpu 0%, no other ds4). Same MoE-routing prompt, hot list
  warmed on it first (8701 experts written, 5166 seeded), arms interleaved, driver 580.159.03, ctx 32768:
    round 1  morning tree (cache+MemAvailable)  5.37 t/s  @ load 3.06 gpu 3%
    round 1  integrated (all four lanes)        5.78 t/s  @ load 3.36 gpu 1%      +7.6%
    round 2  contention returned (load 5.1-5.7, gpu 14%): integrated 5.49; the morning-tree arm
             produced no generation line, cause unrecorded (logs not kept - a harness gap).
  VERDICT: the four lanes buy ~+8% at matched conditions, n=1 clean pair. Consistent with mechanism:
  hit rate was already 0.868 before them, so PREADPOOL/DRAINCUT/PAGECACHE/HOTLIST trim slivers. The
  morning's resident cache (1.69 -> 5.28) was the lever; the evening's work is correctness-preserving
  structure with a small dividend. Still owed: n>=3 clean pairs, and a run log kept per arm.
- 03:21 THREE-ROUND A/B, logs kept. Box quiet at start (load 4.95) then a foreign job arrived.
    r1 sparkport 5.43 @ load 2.97       r1 integrate KILLED rc=137 @ load 14.94
    r2 sparkport 4.57 @ load 14.48      r2 integrate KILLED rc=137 @ load 11.89
    r3 sparkport 4.59 @ load 8.06       r3 integrate 5.38 @ load 7.83
  SPEED: integrate led every pair where both survived (5.78/5.37 earlier, 5.38/4.59 here).
  ROBUSTNESS: integrate OOM-killed 2 of 3, sparkport 0 of 3 - and sparkport survived a load-14.5 spike.
  Both deaths were MID-GENERATION (66-70 output lines), arenas 41.75 / 37.35 GiB vs the survivor's
  43.47, so the "HOTLIST prewarm fills the arena" hypothesis is WEAKENED, not confirmed. Mechanism
  UNIDENTIFIED. Candidates: foreign-job timing (a coincidence at n=2), or a mid-run footprint the
  integrated tree carries that I have not found (PREADPOOL pinned staging is ~76 MiB, too small).
  OWED: a MemAvailable trace sampled every 5 s through one full run of each tree, alone, quiet box.
  Watcher queued. Do NOT ship the four lanes as default until this is explained.
- ANTIREZ'S OWN KERNEL PROFILE (DS4_CUDA_MOE_PROFILE=1), routed MoE per layer, n=1, 6 pairs:
  gateup 0.66 ms · down 0.165 ms · total 0.84 ms  (x40 = 34 ms/token, matches my moe.routed 35).
  gate+up = 36.5 MB IQ2_XXS in 0.66 ms = 55 GB/s (24% of ceiling); down = 23.2 MB Q2_K in 0.165 ms =
  141 GB/s (61%). The IQ2_XXS gate/up kernel is 2.5x less efficient than its Q2_K sibling.
  WHY: gate_expert_bytes 3,041,280 / 66 B * 256 = 11,796,480 elems = 2048 x 5760 -> V4.1 n_embd is 5760
  -> xq_blocks = 22.5 > the lut fast path's cap of 16 -> generic moe_gate_up_mid_qwarp32_kernel.
  LEVER (on the call path, measured): make the IQ2_XXS gate/up decode kernel reach the down kernel's
  bandwidth at in_dim=5760. Ceiling of the lever: routed 34 -> ~17 ms/token (~8% of a 205 ms token).
- LUT WIDENING MEASURED (his DS4_CUDA_MOE_PROFILE, integrate tree, n=1, 6 pairs):
    gateup 0.66 -> 0.406 ms/layer (55 -> 90 GB/s), down 0.165 -> 0.161 (unchanged), total 0.84 -> 0.587.
    x40 = routed 34 -> 23.5 ms/token, ~10 ms saved per ~205 ms token (~5%). Clean branch
    cuda-iq2-lut-gate-32 off bd66c40. Correctness gate in flight; gate/up still at 39% of ceiling vs
    the down kernel's 61% - the same kernel has more to give, as tuning not a gate flip.
- V4 own-tree gate: the box has not reached 100 GiB free all night (peaks ~78); OWED, not reachable
  tonight. True-floor profile: a foreign job took the box to 11 GiB avail at 03:53; still waiting.
- 07:40 NARROWED LUT PATCH, READ BACK (jobs completed while the session was cut):
    narrowed to 3 sites (sxq[32] + <=32u in the two LUT kernels, and the launch guard); integrate
    66583317c, clean branch cuda-iq2-lut-gate-32 71c9c25a1 off bd66c40, 15 lines changed. Both built rc=0.
    his DS4_CUDA_MOE_PROFILE on the narrowed integrate build: gateup 0.398 ms (was 0.660), down 0.165,
    total 0.582 ms/layer -> routed 23.3 ms/token, matching my moe.routed trace exactly.
    correctness gate on the over-wide LUT + margin-knob integrate build: PASS e73f763588d4f253.
    correctness gate on the NARROWED integrate + pr-lut builds: running now (07:34), result below.
- TRUE-FLOOR PROFILE, quiet box (load 0.24, gpu 1%, driver 580.159.03, avail 85 GiB, ctx 32768,
  integrated tree, DS4_V41_PROFILE=1 so the syncs are ON and the t/s is a floor, not a claim):
    arena 5221 slots (full, evictions start at token ~56), hit rate 0.999 -> 0.919 by token 144 and
    still settling; prime 409 -> 162 ms/token and still falling at the last report; routed 23.3 flat;
    attention 32-33; before_moe 43; before_attention 7-10.
    result: prefill 2.17, generation 5.88 t/s WITH the profiler's per-step device syncs.
    reading: at token 144 the token is ~479 ms of which prime is 162 (34%). The cache had not finished
    warming when the run ended (evictions 2116 of 32000 lookups). A longer run on a warm hot list is the
    next honest floor; the profiler-off number on the same conditions is owed too.
- V4 OWN-TREE GATE: the 100 GiB watcher gave up after 3 h (box sat at 84-85 GiB from 04:08 on). The
  100 was my estimate, not a measurement: the V4 Flash streaming plan at 40 GB cache is 43.23 GiB
  (v4_off.log). Re-run at the current 85-89 GiB, serially after the narrowed gate.
- CORRECTION to the 5760 line above: the factorisation was wrong. The tensor header reads n_embd=5120,
  n_ff_exp=2304, so expert_in_dim/256 = 20 blocks, one past the LUT cap of 16. Same conclusion, right
  denominator. The 32-block cap covers up to 8192.
- 07:55 NARROWED-BUILD GATES: integrate PASS e73f763588d4f253 (cache on vs off, byte-identical).
  pr-lut FAIL "a run produced no logprobs" - and the reason is structural, not the patch: the pr-lut
  tree is bd66c40 + the 15-line LUT patch, WITHOUT the port, and bd66c40 refuses V4.1 on CUDA outright
  ("V4.1 requires Metal inference", rc=1). So cuda-iq2-lut-gate-32 cannot be gated on V4.1 standalone;
  its V4.1 gate is the integrate tree that contains it (PASS above). On V4 Flash the widening is a no-op
  by construction (expert input <= 16 blocks already took the LUT path), so a V4 gate there proves
  nothing new. Recorded so nobody re-runs it. The Spark pr-* trees are detached at bd66c40 with the
  patch as a working-tree diff, not the committed branch - the M5 branches are the asset.
- 08:05 V4 OWN-TREE GATE ATTEMPTED at 89 GiB avail (load 1.9, gpu 1%, no other ds4): BOTH arms
  OOM-killed (rc=137) during model load, the cache-OFF arm included ("loading model tensors 64.02 GiB
  cached" then "arena alloc failed for moe up mmq 1792 MiB"). The 40 GB cache plan on V4 Flash loads
  ~64 GiB of tensors before the arena, so the run needs ~105 GiB and the 100 GiB estimate was right
  after all. Since the upstream-behaviour arm dies too, this is the box, not the branch. STILL OWED;
  needs the ~32 GiB of foreign residency off the box first (see the next entry for who holds it).
- 08:15 WHO HOLDS THE BOX: free -g says used 37 GiB with no ds4 running. Top RSS: next-server 22.7 GiB
  (pid 3215277) + next-server 4.6 GiB (pid 3208070) = the studio's dev servers, which the estate runs
  on the Spark by rule; python x6 ~4 GiB. So the V4 own-tree gate (needs ~105 GiB) is blocked by the
  studio's own dev servers, not by anything of ours. Not mine to kill - the navigator's call.
- 08:20 LANE PRIMESPLIT (branch lane-integrate e530952df): the resident cache's copy-out did 18 blocking
  cudaMemcpy per layer = 720 per token. Now cudaMemcpyAsync on g_stream_selected_upload_stream (the
  stream and contract the H2D miss path already uses) with ONE cudaStreamSynchronize per layer; and
  the DS4_CUDA_EXPERT_CACHE_STATS line gains "read N s, copy N s" so prime splits into miss-reads vs
  copy-out. Build + 256-token stats run + correctness gate queued on the quiet box (load ~2).
- 08:45 PRIMESPLIT MEASURED (integrate e530952df, load 0.97 gpu 1% avail 84 GiB driver 580.159.03,
  ctx 32768, 256 tokens, profiler OFF, cache stats on): arena 5174 slots, steady hit rate 0.917.
    per 4000 lookups (16.7 tokens): read +0.55 s, copy +0.50 s  ->  per token: miss reads 33 ms,
    copy-out 30 ms. Prime ~63 ms/token steady (the 162 ms under the profiler was warm-up + syncs).
    copy-out = 240 x 9.49 MiB = 2.2 GiB D2D at ~73 GB/s even async; the floor at the 230 GB/s ceiling
    would be ~10 ms, and ZERO-COPY (kernels read the arena directly) would be 0.
    generation 5.88 t/s (prefill 2.55); correctness gate PASS e73f763588d4f253 on this build.
    the async copy-out alone is speed-neutral within noise (5.78-5.88 across the night's runs).
  TOKEN BUDGET at 5.88 t/s = 170 ms: prime 63 + routed 23 + attention 33 + before_moe 43 = 162.
  NEXT LEVER: zero-copy hits (~30 ms, ~18%) > miss-read overlap (33 ms, needs routing ahead of time)
  > gate/up kernel tuning (23 -> ~15) > attention.
- 07:49 PRIMESPLIT REPEAT (same build, load 0.28 -> 1.71, avail 84 GiB): 5.91 t/s, hit 0.921, read 33 ms
  and copy 30 ms per token again - the split reproduces (n=2). Gate PASS e73f763588d4f253 again.
- 07:50 LANE ZEROCOPY (lane-integrate 1b32508b3 + 15c058a3c): planar arena (three planes with the
  scratch's own per-tensor stride), kernel "views" on the selected cache, decode-sized batches
  (slot_count <= 16) remap ids to arena slots and read the arena directly, no copy-out. Prefill keeps
  the compact scratch. DS4_CUDA_EXPERT_ZERO_COPY=0 restores the copy. First build failed on a comment
  that swallowed two field declarations; fixed. Build + gate + 256-token run queued (box load 1.7).
- 07:52 ZEROCOPY MEASURED (integrate 15c058a3c; load 1.22 -> 2.25, gpu 1%, avail 85 GiB, driver
  580.159.03, ctx 32768, no other ds4): correctness gate FIRST: PASS e73f763588d4f253 (zero-copy on
  vs cache off, byte-identical). Then 256 tokens, cache stats on: arena 5262 slots, hit 0.917,
  read 33 ms/token, copy 0.0 s (gone).  generation 5.91 -> 7.30 t/s (+23.5%), prefill 2.67 -> 2.88.
  The 30 ms of copy-out was the single biggest lever found tonight after the cache itself.
  Token now ~137 ms: attention ~43 + misses 33 + routed 23 + untraced ~33 + small.
- 08:00 ZEROCOPY + PRIMESPLIT cherry-picked onto cuda-resident-expert-cache (63fcb40a0 ed686046f
  109dae3b1), clean. That branch's V4.1 gate is the integrate tree that contains it (PASS above).
  Prefill stays on the compact scratch by design: the sorted-pairs prefill path sizes its counts /
  offsets / tiles by n_total_expert (384), so arena ids up to ~5200 would overrun them.
- 08:05 LANE OUTSIDETRACE (lane-integrate cc79689c7): the profiler now also times the per-token work
  OUTSIDE the layer loop - token.engram (the Engram rows are read from the file every token),
  token.embed, token.drain (ds4_gpu_end_commands per layer = a device sync x40), token.logits.
  Chasing the ~33 ms/token the layer traces did not cover. Profiled 128-token run queued.
- 07:55 OUTSIDETRACE MEASURED (profiled 128 tokens, syncs on, load 0.63, 7.01 t/s under the profiler):
  token.engram 5.2 ms avg and RISING (1.3 -> 3.9 -> 5.2 over 64/96/128 tokens), token.logits 2.0,
  token.embed 0.0, token.drain 0.0 (already drained by the trace syncs). The outside-the-layers work
  is ~7 ms; the ~30 ms gap is NOT there. Steady per-token: attention triple 42 (7+32+3), moe ~60
  (prime 33 + routed 22 + router/shared ~5), engram ~6, logits 2 = ~110 of 137 ms. The rest is
  launch/sync overhead the traces cannot see (80 syncs/layer under the profiler; CPU launch gaps
  without it). NEXT INSTRUMENT: nsys on 24 tokens - kernels per token, avg kernel time, GPU idle.
  Engram growing is worth a look: the rows are random reads across the ~189 GiB of tables outside
  the map, always cold; they could be issued async right after sampling and waited at layer 1.
- 07:57 NSYS, 48-token decode on the ZEROCOPY build (load 0.81, gpu 0%, 7.45 t/s under nsys; per-token
  figures divide by 96 token-units = lut launches / 40). GPU kernel time per token ~77 ms of 134 wall:
    matmul_q8_0_preq_warp8   280 launches, 28 ms   = the 6.4 GiB of non-routed Q8_0 at ~240 GB/s:
                                                     AT the ceiling. This is the byte floor.
    moe_gate_up_mid_decode_lut 40 x 372 us = 15 ms  (98 GB/s, 43% of ceiling)
    dsv41_router_select_general 40 x 197 us = 8 ms  ONE THREAD does a 384x6 insertion sort
    grouped_q8_0_a_preq        40 x 168 us = 7 ms   (shared expert)
    moe_down_sum6              40 x 161 us = 6 ms
    the rest (cutlass, f32 matmul, norms, bf16_linear x787/token at 1 us...) ~13 ms
  API: 2,700 cudaLaunchKernel per token (~10 ms CPU); cudaStreamSynchronize 245/token (the pool's
  per-task H2D waits, ~16 ms wall); cudaEventSynchronize 40/token (the DRAINCUT readback wait).
  GPU busy 57%. The idle 43% = miss reads on the critical path (33 ms) + launch/sync gaps.
  LEVERS from this: (1) router kernel -> block argmax, ~7 ms, bit-identical by construction;
  (2) LUT gate/up 43% -> 61% like down, ~5 ms; (3) NVMe queue depth: DS4_CUDA_STREAMING_EXPERT_PREAD_THREADS
  8 -> 16 sweep; (4) a CUDA graph per decode layer for the 2,700 launches - big, later.
- 08:10 LANE ENGRAMTHREADS (lane-integrate c2dd3fd57): decode reads its 2x24 Engram rows through
  the batch reader, and the batch reader gets a pthread path on Linux (it was Apple-dispatch only).
  Build + gate + profiled + plain runs queued.
- 08:20 ENGRAMTHREADS MEASURED (load 0.59, gpu 1%, avail 85 GiB, driver 580.159.03, ctx 32768):
  gate PASS e73f763588d4f253; profiled token.engram 5.2 -> 2.2 ms; plain 256 tokens 7.30 -> 7.76 t/s
  (n=1, the delta is within the night's run-to-run spread of ~0.1-0.3, so credit ~3 ms not 0.46 t/s).
- 08:22 LANE ROUTERARGMAX (lane-integrate ae42eca61): the general router's one-thread 384x6 insertion
  sort becomes 6 rounds of block-wide argmax with the sort's tie rule (equal score -> lower index) and
  the same weight summation order; bit-identical by construction, nsys says ~7.5 ms/token is on the
  table. Build + gate + plain run queued, then a pread-pool sweep (16 and 4 threads vs the default 8).
- 08:35 ROUTERARGMAX MEASURED (load 1.08, gpu 1%, avail 85 GiB, driver 580.159.03, ctx 32768):
  gate PASS e73f763588d4f253 (bit-identical, as constructed). Plain 256 tokens: 7.76 -> 8.12 t/s.
- 08:38 PREAD POOL SWEEP (same build, stats on, 256 tokens): threads 16 -> 8.28 t/s, read 11.3 s per
  72000 lookups; threads 4 -> 8.33 t/s, read 11.1 s. Default 8 earlier: read ~11.0 s. The miss reads
  are ~37 ms/token at ANY thread count: NVMe-bound (180 MB/token at ~5 GB/s), not queue-bound. The
  0.2 t/s spread between arms is the night's noise. Keep 8. The only lever left on misses is the
  hit rate, i.e. arena size, i.e. the box's free memory.
- 08:40 LANE LUT16 (lane-integrate acab838b8): 16 lanes per row for the LUT gate/up decode kernel -
  gate on lanes 0-7, up on lanes 8-15, same block order and same 8-lane tree per tensor so the sums
  are bit-identical; twice the loads in flight. DS4_CUDA_MOE_LUT16=0 restores the 8-lane kernel.
  Build + gate + his MoE profile (lut16 vs lut8) + plain runs queued.
- 08:50 LUT16 MEASURED - AND IT FAILS THE GATE (load 1.30, gpu 1%, avail 85 GiB): his MoE profile
  gateup 0.375 -> 0.307 ms/layer (-2.7 ms/token); plain 8.35 vs 8.27 t/s (noise). BUT the greedy
  dump is sha 187cf54a55a03b34 on BOTH arms against the baseline e73f763588d4f253: cache-transparent,
  not baseline-identical. Same per-lane block order, same 8-lane tree - the two kernels contract the
  acc += a*b*c differently under --use_fast_math, so the last ulp moves and greedy tokens follow.
  DECISION: LUT16 is opt-in (DS4_CUDA_MOE_LUT16=1), OFF by default, kept for the record (7f37a329f).
  INSTRUMENT LESSON: on-vs-off equality alone would have PASSED this. sparkport-verify.sh now also
  asserts the baseline sha (1e765ecd6). Every PASS earlier tonight printed e73f7635 and was checked
  by eye; from here the script checks it.
- 09:05 FINAL DOOR on integrate (LUT16 off = zero-copy + engram threads + router argmax):
  correctness PASS, byte-identical AND equal to the baseline e73f763588d4f253 (the gate now asserts
  it). Speed through sparkport-verify.sh --speed: load 1.10, gpu 1%, driver 580.159.03, ctx 32768:
  prefill 3.09, generation 8.40 t/s. The night: 1.69 -> 8.40 (5.0x), every rung gated.
  (First speed attempt refused itself at gpu 35% - the stamp read the correctness arm's tail; the
  script now settles 8 s and has a --speed mode.)
- 09:08 LANE LUTR16: the 8-lane LUT kernel body verbatim, 16 rows per block (128 threads) instead of
  32 - twice the blocks, same per-thread code so the same contraction; bit-identity expected.
- 09:20 LUTR16 MEASURED-NULL: gate PASS (byte-identical, baseline sha); his profile gateup 0.368 vs
  0.372 ms/layer (r16 vs r32); speed 8.47 t/s (load 1.05, gpu 0%) = noise around 8.40. Grid geometry
  is not the limiter of the LUT kernel; it is latency-bound inside the thread (the LUT gathers).
  Opt-in (DS4_CUDA_MOE_LUT_R16=1), off. The bit-identical routes to a faster gate/up are exhausted
  for tonight; a faster one needs a different contraction, which needs the baseline re-derived
  (a Verification-Law decision, not a lane's).
- 09:25 OWED / NEXT (for the morning), in order of expected payoff:
  1. ARENA SIZE = the miss rate. Misses are ~37 ms of a ~118 ms token, NVMe-bound. The box gives the
     arena ~48 GiB because ~27 GiB sits in two next-server processes (the studio's dev servers).
     +27 GiB -> ~+2900 slots (5262 -> ~8100, 53% of experts). Navigator's call.
  2. CUDA GRAPH per decode layer: 2,700 launches/token, ~10-15 ms of CPU issue + gaps. antirez has a
     graph for the Q4_K decode MoE; V4.1 needs one around the whole layer. Half a day, gate-able.
  3. A faster IQ2_XXS gate/up kernel needs a different contraction than the shipped one, which
     changes the last ulp: either accept a re-derived baseline (Verification Law decision) or stay.
  4. V4 Flash own-tree gate for cuda-resident-expert-cache: needs ~105 GiB free (same memory).
  5. OOM-under-contention mechanism for the integrated tree: not reproduced tonight on a quiet box
     with margin 12; still unexplained under load. A MemAvailable trace through one contended run.
  6. Prefill: 2.6-3.1 t/s; it still copies arena -> scratch (30 ms/token equivalent) because the
     sorted-pairs path sizes tables by 384 experts; a remap table would let prefill go zero-copy too.
- 09:40 HEADLINE, n=3 on the SHIPPED DEFAULT (r16 off), through the door, gate PASS first:
    8.21 t/s @ load 2.39 · 8.27 @ 1.32 · 8.27 @ 1.24   (gpu 1%, driver 580.159.03, ctx 32768)
  with the two r16-on runs (8.40 @ 1.10, 8.47 @ 1.05) the night's five stamped runs of the final
  tree span 8.21-8.47. THE LAW: take the minimum -> the claim is 8.2 t/s generation, 1.69 -> 8.2 =
  4.9x, and 8.4 is what a quiet box shows. The r16-on runs sat at lower load, so r16 stays a null.
- CLOCK NOTE: the entries stamped 08:00-09:40 above were written from an estimated clock; the M5
  reads 08:25 AEST at the time of this note, so those stamps run ~1 h fast. Order is right, the
  Spark-side log timestamps (07:4x-08:2x) are the true ones. The 7-hour window runs to ~10:30.
- 08:40 AEST ARENA MARGIN PROBE (final tree, load 0.8-1.3, avail 84 GiB, gpu 1%, 256 tokens each):
    margin 12 GiB: 5234 slots (48.5 GiB), hit 0.901, read 11.8 s/72k lookups, 8.21 t/s
    margin  8 GiB: 5677 slots (52.6 GiB), hit 0.918, read  9.8 s/72k lookups, 8.55 t/s
  +4.1 GiB of arena = +443 slots = +1.7 pp hit rate = -17% miss-read time = +0.34 t/s (+4%).
  That is the price list for the box's memory: ~0.08 t/s per GiB at this point on the curve.
  SCALED, not measured: the 27 GiB held by the studio's dev servers would be ~+2,900 slots; the
  hit-rate curve flattens, so call it +1 to +2 t/s and measure it if the memory is freed.
  Default margin stays 12 (it was raised after OOM-kills under contention); DS4_CUDA_EXPERT_CACHE_MARGIN_GB=8
  is the quiet-box setting.
- 08:55 AEST LANE PREFILLZC (lane-integrate 88757eed6): the streaming prefill's sorted-pairs path sizes
  its counts/offsets/cursors/tile tables from table_experts (compact count, or the arena slot count
  under zero-copy) instead of n_total_expert=384; the decode-only guard on zero-copy is lifted.
  GATE PASS e73f763588d4f253 (the gate's prompt exercises prefill). Speed 7.84 t/s at load 2.87 -
  a vitest process (the studio's tests on the Spark, 109% CPU) arrived; not a measurement.
- 08:57 AEST CRASH FOUND, PRE-EXISTING: a ~600-token prompt SEGFAULTS (rc 139) on the integrate tree
  with zero-copy on AND off, and on the PREVIOUS build (decode-only zero-copy) too; a ~150-token
  prompt runs (prefill 5.37 t/s). The crash is early - after the memory plan, before the first
  "processing" line. Not PREFILLZC's. Backtrace + port-only tree + ~300-token bisect queued.
- 09:15 AEST LONG-PROMPT SEGFAULT, ROOT CAUSE (gdb): ds41_graph_prefill_sweep, the first-layer
  `memset(ds4_gpu_tensor_contents(g->batch.selected_comp), 0xff, count*TOP_K*4)`. On Metal the
  contents pointer is a shared MTLBuffer; on CUDA ds4_gpu_tensor_contents returns the raw device
  pointer, so the CPU writes device memory - it happens to survive for ~150 tokens and faults at
  ~588 (the port-only tree crashes the same way; it is the port's bug, not any lane's).
  LANE DEVICEMEMSET (lane-integrate e272c71b3): new ds4_gpu_tensor_memset on both backends
  (CUDA cudaMemset, Metal memset on the shared buffer), the sweep calls it. Metal object compiles
  on the M5. Spark build + long prompt on/off + long-prompt greedy on-vs-off + baseline gate queued.
  Also owed: audit every other ds4_gpu_tensor_contents() write on the V4.1 CUDA path (ds41_text_mask
  for images is the next one).
- 09:45 AEST DEVICEMEMSET moved the crash, did not end it: the next backtrace was ds41_prefill_seed
  reading `ds4_gpu_tensor_contents(g->batch.selected)` from the host (x1 = 0 in the fault). Same
  class: the V4.1 prefill touches three device tensors through their host view - the selected_comp
  memset (fixed), the selected-id read in the seed, and the Engram row writes (batch and per-row).
  LANE HOSTVIEW (lane-integrate 1482b13d2): the seed reads the ids back with ds4_gpu_tensor_read,
  the Engram rows are staged on the host and uploaded with ds4_gpu_tensor_write. Metal semantics
  unchanged (its view was shared memory anyway). Why ~150 tokens ever worked: on the GB10 the CPU
  can touch device pages the GPU has already populated; a long prompt reaches ones it has not.
  Short-prompt gate PASS e73f763588d4f253 on this build. The long-prompt run itself was OOM-killed
  at model load by box contention (load 4.2, the studio's test runners); a wait-and-retry is queued.
  Still to audit: the pipelined Engram prefetch (`engram_prefetch` .out is a host view too) - it is
  only reached when overlap_engram is on; if the long prompt faults there next, that is the site.
- 10:00 AEST LONG-PROMPT RETRY: OOM-killed at model load again (rc 137) with 78 GiB MemAvailable and
  load 5.1 - next-server 22.0 + node 6.4 + next-server 4.6 GiB and growing (the studio's tests). The
  crash fix is BUILT and the short-prompt baseline gate PASSES on it; the ~588-token run that proves
  the fix is UNVERIFIED tonight - it needs the box. Both fixes are on v4.1-flash-cuda (c9c453676,
  c5d9d7f11) and lane-integrate (e272c71b3, 1482b13d2).
- END OF THE 7-HOUR WINDOW (03:30 -> 10:30 AEST). Landed, all gated equal to the baseline sha:
  resident cache · MemAvailable sizing · HOTLIST · PREADPOOL · PAGECACHE · DRAINCUT · LUT 16->32 ·
  ZEROCOPY · ENGRAMTHREADS · ROUTERARGMAX · PREFILLZC · DEVICEMEMSET + HOSTVIEW (crash class).
  Headline: 1.69 -> 8.2 t/s generation (min of 5 stamped runs, 8.21-8.47), prefill 1.63 -> ~3.
  Rejected/null: LUT16 (not bit-identical), LUTR16 (null), hotness eviction (worse), 16/4 pread threads (null).
  Instruments: sparkport-verify.sh asserts the baseline sha, --speed, settle; DS4_V41_PROFILE traces
  outside the layers; cache stats split read/copy; nsys map in this log.
  Owed, in order: the box's memory (arena = miss rate, ~0.08 t/s/GiB) · CUDA graph per layer ·
  long-prompt verification of the crash fix · V4 own-tree gate · OOM-under-contention mechanism ·
  the pipelined Engram prefetch host view.
- 09:00 AEST (real clock; the "10:00" above was the estimated clock again, ~1 h fast): the box is
  saturated - load 10.9, MemAvailable 0 GiB - the studio's test runners. My long-prompt retry was
  stopped rather than left to be OOM-killed. 1.5 h of the window remain; while the Spark is unusable
  the work moves to the M5 side: the pipelined Engram prefetch host view, the PR branches, the doc.
- 09:10 AEST LANE PREFETCHSTAGE (lane-integrate 70bd86ddc, v4.1-flash-cuda 9256d452b): the pipelined
  Engram prefetch (prompts >= 1024 tokens) staged its rows in the prefetch tensor's host view; the
  reader thread now fills a host buffer and the consume site uploads each chunk with
  ds4_gpu_tensor_write. That closes the host-view class in the V4.1 prefill: four sites, all now
  explicit transfers. ds4.o compiles on the M5; the Spark is saturated, so the CUDA build and the
  >=1024-token run are OWED. PREFILLZC cherry-picked onto cuda-resident-expert-cache.
- 09:20 AEST QUIET-BOX RETRY (load 1.2, 85 GiB, MemFree == MemAvailable): the segfault is GONE - the
  ~588-token prompt now fails with "CUDA model arena alloc failed for moe up mmq: out of memory" (rc 1)
  and the greedy arms are OOM-killed; a ~150-token prompt runs (prefill 5.50 t/s), a ~300-token one
  dies after "loading model tensors 16 / 32 / 48 GiB cached". DIAGNOSIS: ds41_moe_batch (the batched
  prefill) never loads a selected-expert cache on CUDA - only the decode path does - so
  routed_moe_launch falls to cuda_resolve_weight_ptr and copies the WHOLE expert tensors of each layer
  into the device cache (3.6 GiB a layer, never released) until the box is gone around layer 13.
  Metal maps a layer's experts as file-backed buffers, so the same code costs it nothing.
  LANE BATCHLOAD (lane-integrate fd9e3e737): ds41_cuda_stream_selected_load_batch reads the batch's
  count x 6 ids back and does one begin_selected_load per layer - the decode path's own mechanism,
  and PREFILLZC's zero-copy already covers any slot count. Build + long prompt + greedy on/off +
  baseline gate + stamped speed queued.
- 09:20 AEST BATCHLOAD MEASURED: the batched prefill now seeds the arena per layer ("layer 3 seeded,
  850 resident; layer 4, 1069") but the run still died: "loading model tensors 16 GiB cached", then
  "arena alloc failed for moe up mmq", then an illegal memory access at layer 4. ROOT CAUSE, second
  half: the MMQ IQ2 prefill tier in routed_moe_launch (`iq2_path && n_tokens > 1 && cuda_use_mmq()`)
  is NOT gated on !g_ssd_streaming_mode - unlike the fused tier above it - so it resolves the whole
  expert tensors into the device cache before the streaming path is reached. LANE MMQGATE
  (lane-integrate e3aeeef0e): the tier carries the same gate as its sibling. Short-prompt gate still
  PASS e73f763588d4f253 on the BATCHLOAD build; speed 8.21 at load 4.1 (contended, not a claim).
  Build + long prompt (on/off) + greedy compare + baseline gate + stamped speed queued.
- 09:35 AEST MMQGATE MEASURED (load 1.13, gpu 1%, avail 84 GiB): the ~588-token prompt RUNS, rc 0.
  batched prefill 9.08 t/s zero-copy on (8.22 off), then generation 5.23 (5.07) - low, because the
  cold long prompt churned the arena (hit rate 0.15-0.22 in the first 16k lookups, 9,860 evictions).
  Short-prompt baseline gate PASS e73f763588d4f253. BUT the long-prompt greedy dumps zero-copy ON vs
  OFF DIFFER (byte 116, line 7; sha 6bf61e6229145f42 on). Before blaming PREFILLZC: the batched
  prefill's sorted-pairs path has atomic down-projection accumulation (use_atomic_down), which is
  order-nondeterministic by construction. A determinism run (off vs off, on vs on, off vs on, and
  a ~150-token prompt on vs off) is queued to tell "zero-copy changes the result" from "the batched
  prefill is not reproducible run to run".
- 09:45 AEST DETERMINISM: long prompt (~588 tok), same config twice: on1 96d455052a1b52b5 vs on2
  4b354d2c48dfb4d3 - DIFF. off vs off: one arm OOM-killed by a load spike, no verdict. ~150-token
  prompt on vs off: SAME (635ff9680701a231). So the batched CUDA prefill is NOT REPRODUCIBLE run to
  run above some length, and zero-copy on-vs-off cannot be judged there. Suspect: use_atomic_down
  (float atomics in the down projection, on at n_tokens >= 128 unless DS4_CUDA_MOE_NO_ATOMIC_DOWN).
  Queued: the same two-run test with atomics off, and on-vs-off with atomics off.
  NOTE for the Verification Law: antirez's batched Metal prefill - deterministic or not - is the
  reference; on CUDA the atomic tier is upstream's own prefill design, present before the port.
- 09:52 AEST ATOMICS OFF, STILL NOT REPRODUCIBLE: DS4_CUDA_MOE_NO_ATOMIC_DOWN=1, same long prompt
  twice: 7a63cef35ebc3a3d vs 62bbc1170f5c37e7 (DIFF); on vs off under it DIFF too. So the source is
  not (only) the atomic down tier. Candidates, unmeasured: cuBLASLt split-K in the batched dense
  matmuls (splitKreduce seen in nsys), the batched indexer top-k on ties, the batched attention
  score reductions. The ~150-token prompt IS reproducible (SAME across arms), so the boundary sits
  between ~150 and ~300 tokens - where the encoder switches to the batched tiers (n_tokens >= 128
  gates several of them).
  VERDICT for the PR: prompts of any length now RUN on CUDA (three port bugs fixed tonight), but the
  batched prefill above ~128-256 tokens is NOT bit-reproducible run to run. That is a Verification-
  Law blocker for long prompts and the first correctness item on the owed list. Next instrument:
  per-layer checksums of the residual after prefill (DS4_V41_TRACE-style) on two runs to find the
  first diverging layer/phase; then test each tier's env switch at that layer.
- 09:58 AEST LOCALISER queued: the long prompt twice with DS4_METAL_DISABLE_V41_BATCH_MOE=1, then
  twice with DS4_METAL_DISABLE_V41_BATCH_ATTN=1. SAME under one of them names the tier.
- LEDGER, END OF WINDOW (10:30 AEST). Landed tonight on lane-integrate and the PR branches, all
  short-prompt-gated equal to e73f763588d4f253: resident cache · MemAvailable · HOTLIST · PREADPOOL ·
  PAGECACHE · DRAINCUT · LUT 16->32 · ZEROCOPY · ENGRAMTHREADS · ROUTERARGMAX · PREFILLZC ·
  DEVICEMEMSET · HOSTVIEW · PREFETCHSTAGE · BATCHLOAD · MMQGATE.  Generation 1.69 -> 8.2 t/s
  (min of 5 stamped, 8.21-8.47). Long prompts: crashed -> run (9.1 t/s batched prefill), not yet
  reproducible run to run above ~128-256 tokens (upstream's batched tiers; first owed item).
  Nulls: LUT16 (not bit-identical), LUTR16, pread threads 4/16, hotness eviction.
  Owed: batched-prefill reproducibility · the box's memory (arena = miss rate) · CUDA graph per
  layer · V4 own-tree gate · OOM-under-contention mechanism.
- 11:00 AEST (new login, same session) LOCALISER RESULT: batched attention OFF -> still DIFF
  (c86e72f94f740479 vs bf3b328e4fdaf1e3), so the batched attention is not the source. Batched MoE
  OFF -> run1 rc=1, run2 OOM-killed: the per-row prefill MoE path cannot run on CUDA at this length
  (it has no selected load either) - inconclusive for the MoE. Box quiet again (load 0.2, 85 GiB).
  NEXT INSTRUMENT (lane PREFILLCHECKSUM): DS4_V41_PREFILL_CHECKSUM=1 reads the batch residual back
  after each stage of each layer in the sweep and prints a checksum; two runs, first differing line
  names the layer and stage.
- 11:15 AEST LANE PREFILLCHECKSUM (lane-integrate 580e27e94 + 29aa00792): DS4_V41_PREFILL_CHECKSUM=1
  prints checksums of residual / block / routed / shared / selected after every prefill stage.
  Two ~588-token runs, 280 stage lines each: the first difference is LAYER 0, stage
  "shared/routed ffn": shared 343ac1f56fc80000 = 343ac1f56fc80000 and selected ce01ef11cc3ea1d2 =
  ce01ef11cc3ea1d2 (router and shared expert are reproducible), but routed 68d4ce7080f776bf vs
  e7106af4a9a50d55. The batched ROUTED MoE tier (sorted pairs + expert tiles) is the source, and
  every stage after it inherits the difference (274 of 280 lines differ). Attention is clean.
  Switch sweep queued (NO_EXPERT_TILES, NO_ATOMIC_DOWN, NO_DOWN_TILE16, tiles+atomics off), two
  runs each, comparing the layer-0 routed checksum.
- 11:30 AEST SWITCH SWEEP (layer-0 routed checksum, two runs each; box contended, load 5.8):
    NO_EXPERT_TILES=1                  -> SAME  (9def280f68d0164c x2)   prefill 8.28 / 8.02 t/s
    NO_ATOMIC_DOWN=1                   -> DIFF
    NO_DOWN_TILE16=1                   -> DIFF
    NO_EXPERT_TILES + NO_ATOMIC_DOWN   -> SAME  (9def280f68d0164c x2)   prefill 8.40 / 8.13 t/s
  The expert-tile tier is the non-reproducible one. Its input is moe_scatter_sorted_pairs_kernel,
  which places each pair at atomicAdd(cursor[expert]) - a run-to-run random order within each
  expert's segment - and the tiles reduce in that order. Tiles OFF is also not slower here.
  LANE STABLESCATTER (lane-integrate 8ef886c19): a one-thread stable scatter in pair order
  (microseconds for <=12k pairs), DS4_CUDA_MOE_UNSTABLE_SCATTER=1 restores the atomic one.
  Queued: tiles ON + stable scatter, two checksum runs + three greedy dumps, then the baseline gate.
- 11:45 AEST STABLESCATTER MEASURED: tiles ON + stable scatter, two runs: routed 316decee3a35d7e8 vs
  4d13b33e2d90414d, 1209 of 1400 stage lines differ, greedy dumps DIFF. The scatter order was not the
  only order-dependence in the tile tier (its gate/up row-span accumulation is the next suspect:
  gate_row_span 512 with atomic adds, no off switch). Stable scatter stays (correct, microseconds).
  DECISION (lane TILESOPTIN, lane-integrate): on a streaming model the expert-tile tier is OPT-IN
  (DS4_CUDA_MOE_EXPERT_TILES=1); tiles OFF measured reproducible (SAME x2) and not slower
  (8.0-8.4 vs 7.4-7.9 prefill t/s). Short-prompt gate PASS on the stable-scatter build.
  Queued: tiles-off build, long prompt run1/run2/zero-copy-off greedy dumps, baseline gate, speed.
- 11:55 AEST TILESOPTIN MEASURED (lane-integrate e59d4400d; load 2.9, avail 84 GiB): ~588-token prompt
  greedy dumps run1 = run2 = zero-copy-off = 5566a268d8baab75. REPRODUCIBLE, and prefill zero-copy is
  transparent at this length too. Short-prompt gate PASS e73f763588d4f253. Speed 8.66 t/s at load
  3.99 (contended; not a claim). The long-prompt sha 5566a268d8baab75 becomes the second baseline.
  STATUS OF LONG PROMPTS ON V4.1 FLASH CUDA: crash -> OOM -> alloc fail -> nondeterministic -> DONE.
  Six lanes: DEVICEMEMSET, HOSTVIEW, PREFETCHSTAGE, BATCHLOAD, MMQGATE, STABLESCATTER+TILESOPTIN.
- 12:05 AEST GATE --long PROVED end to end on the final tree (load 0.87, gpu 0%): PASS, reproducible
  and equal to the long baseline 5566a268d8baab75. Both arms of the door now have a baseline sha.
- NEXT: the owed MemAvailable trace through one full run (the OOM-under-contention question) - the
  footprint curve of the final tree alone on a quiet box, sampled every 2 s.
- 12:10 AEST SCOPING THE CUDA-GRAPH LANE (not started): antirez already has the machinery in
  ds4_cuda.cu - a decode-graph cache keyed by (layer, island): state 1 = warm pass (lazy allocators
  reach steady size), capture on g_decode_graph_stream with cuBLAS pointed at it, instantiate,
  state 2 = replay (cudaGraphLaunch), kill-and-encode-eagerly on any failure; kernel-node params are
  patched per token where an arg changes (cudaGraphExecKernelNodeSetParams for the score/rope
  nodes). The V4.1 decode layer would be cut into islands at its host syncs: island A = engram add ..
  attention .. router (ends at the readback arm), island B = shared expert + routed MoE + after_moe
  (starts after the selected load). Position-dependent args (attention pos, KV row) must be patched
  per replay or read from device memory. 2,700 launches/token ~10-15 ms of CPU issue at stake, the
  only remaining lever inside the token that is not a byte or a miss. A half-day lane; it lands
  behind both gate arms.
- 12:20 AEST MEMORY TRACE, one 256-token run (the box turned out contended: load 6.2, avail 79 GiB
  at start): MemAvailable 73.6 -> 11.0 GiB within 8 s of launch and FLAT at 11.0 for the whole run;
  MemFree 70.7 -> 8.3. VmRSS reads 1.0 GiB (CUDA allocations are not in RSS). Arena 4626 slots =
  42.9 GiB + resident model 9.4 + context buffers 8.0 + KV/buffers ~2 = ~62 GiB, leaving exactly the
  margin. 7.87 t/s under that load.
  OOM-UNDER-CONTENTION, EXPLAINED: the arena is sized to leave the 12 GiB margin and nothing more,
  and ds4 sets oom_score_adj=1000, so any foreign allocation larger than the margin makes the kernel
  kill ds4 first - by design of the sizing, not by a leak. The morning tree survived the same spikes
  because its arena was smaller (it left more free). The lever is POLICY: a bigger margin on a
  shared box (DS4_CUDA_EXPERT_CACHE_MARGIN_GB=24-32 while the studio's fleet runs), or releasing
  arena slots on memory pressure (a watcher on MemAvailable that cudaFrees the LRU tail) - the
  second is a lane, the first is a knob that exists. Owed item closed as explained.
- 12:30 AEST LANE GRAPHISLAND (lane-integrate): the routed MoE decode step wrapped in antirez's
  decode-graph machinery (key il/island=1; warm -> capture -> replay; retire-and-eager on failure),
  opt-in via DS4_CUDA_V41_DECODE_GRAPH=1. Only the routed launch is inside the island: its kernels
  already go to cuda_decode_stream(), every pointer is token-stable (arena planes, remap, scratch,
  router weights) and the ids are refilled in device memory. ~10 launches/layer of the 2,700; a
  probe of the machinery on V4.1 before the bigger islands (attention). Build + capture log + baseline
  sha + speed A/B queued.
- 12:45 AEST GRAPHISLAND MEASURED (lane-integrate eaad962e6; load 1.8, avail 96 GiB): 40 captures
  (one per layer), replays for the rest of the run, greedy dump = baseline e73f763588d4f253. Speed
  island on 9.16 vs off 9.30 t/s: NULL for the routed island alone (its ~10 launches/layer are
  1/7 of the token's launches, and a graph launch costs about what it saves). The MACHINERY WORKS
  on V4.1 - that was the probe's question. The prize is the attention island (bf16_linear x787,
  rms_norm x165, the score/rope/indexer kernels per token), which needs per-replay patching of the
  position-dependent kernel nodes, exactly as antirez's island 0 patches its score/rope/final nodes.
  Opt-in stays off by default.
- 12:50 AEST THE BOX FREED: the 22 GiB next-server is gone (7.1 + 4.7 remain), MemAvailable 99 GiB.
  Through the door: 9.42 t/s @ load 0.75 gpu 1%, 8.98 @ load 1.27 gpu 3% (driver 580.159.03, ctx
  32768). Consistent with the arena price list (+~15 GiB -> ~+1 t/s). Arena size on this box being
  recorded now, and the owed V4 Flash own-tree gate attempted at a 20 GB cache plan.
- 13:00 AEST FREED BOX, ARENA MEASURED: 6800 slots = 63.0 GiB, hit rate 0.951, miss reads 6.1 s
  per 72k lookups (was 11.8 at 5234 slots), 9.02 t/s at load 0.91 - the price list holds.
- 13:05 AEST V4 OWN-TREE GATE, first attempts: the Spark's pr-cuda-resident-expert-cache tree was a
  STALE working diff (no margin knob, no MMQGATE): cache-off arm ran (ce4a2e4737a32ad2), cache-on
  arm sized 2522 slots after V4's 64 GiB tensor load and was OOM-killed twice (the margin env had
  no effect - the knob was not in that tree). Re-materialising the real branch (diff bd66c40..HEAD
  applied on a clean bd66c40 tree), building, and running both arms with margin 24.
  LESSON: a Spark tree is a copy of a diff, not a branch; verify what it holds before gating it.
- 13:15 AEST BRANCH HYGIENE FINDING: cuda-resident-expert-cache does not build on bare bd66c40 - its
  first commit (5cf94a2d3) already carried `#include "ds4_cuda_dsv41.cuh"` and dsv41_* references
  from the port. So "off bd66c40, independent of the port" was never true of that branch; it is
  STACKED on v4.1-flash-cuda in substance. Resolution: state it that way in the PR doc (cache PR
  depends on the port PR), and run the V4 Flash gate on the integrate tree, which is the stacked
  tree plus the profiler and the LUT lanes. V4 gate queued there with margin 24.
- 13:25 AEST V4 FLASH GATE ON THE INTEGRATE TREE (load 1.7, avail 100 GiB, 20 GB cache plan,
  margin 24): cache ON (2522 slots, hit 0.564 at 4k lookups) vs cache OFF: IDENTICAL,
  sha de8682e8546bc2df. The resident cache is transparent on V4 Flash too - the owed gate, on the
  stacked tree. (The stale tree's off-arm sha ce4a2e47 differs because that binary still had the
  non-reproducible tile tier in its 14-token prefill.)
- 13:40 AEST V4 OWN-TREE GATE, PROPER (cuda-resident-expert-cache 70ba5b550, standalone on bd66c40,
  built on the Spark from a clean tree; load 2.9, avail 96 GiB, 20 GB cache plan, margin 24):
  cache ON vs OFF IDENTICAL, sha de8682e8546bc2df - the same sha the integrate tree gave, so the
  cache branch and the stacked tree agree on V4 Flash byte for byte. OWED ITEM CLOSED.
  The branch is now what its name says: off bd66c40, no port bits (the reversal is 70ba5b550).
- GATE LEDGER at this point: V4.1 short prompt e73f763588d4f253 (integrate) · V4.1 long prompt
  5566a268d8baab75 (integrate, reproducible) · V4 Flash de8682e8546bc2df (cache branch alone AND
  integrate). Decode headline 8.2 t/s claimed (min of 5 at ~48 GiB arena); 9.0-9.4 seen on the
  freed box at 63 GiB arena (hit 0.951), not yet a 3-run claim.
- 13:50 AEST FREED-BOX HEADLINE, three through the door (arena ~63 GiB, hit ~0.95, driver 580.159.03,
  ctx 32768, gpu 1%): 8.48 @ load 2.20 · 9.17 @ 1.40 · 9.25 @ 1.09. Minimum 8.48; the two quiet
  runs 9.17-9.25 (and 9.42 / 8.98 / 9.02 earlier). CLAIM by the law: 8.5 t/s on this box as it is
  now, ~9.2 when quiet - against 8.2 at the 48 GiB arena. The night from 1.69: 5.0x claimed.
- 13:55 AEST ATTENTION-ISLAND DESIGN NOTE (not started): in ds41_attention the position reaches the
  GPU through three ropes (pos), the window copy (offset pos % 128 - a memcpy node), gather_kv
  (n_comp = (pos+1)/ratio), attention_decode_heads (n_raw, start) and the readback arm; and the
  kernel CHOICE changes with n_comp (< TOP_K takes the mixed path, else the indexed one), so one
  graph per layer is valid only within an n_comp window. Per replay: ~6 kernel nodes and one
  memcpy node patched (cudaGraphExecKernelNodeSetParams / MemcpyNodeSetParams), plus a re-capture
  at each n_comp bucket change. That is the lane: a half day, gated by both arms and by the
  --long arm, with the 2,700 launches/token as the prize (~10-15 ms of a ~110 ms token).
- 14:40 AEST LANE GRAPHISLANDS, FINISHED (lane-integrate 1c590b7dc): four CUDA-graph islands per
  decode layer - the layer front (Engram add, HC mix, norms, q/kv projections), the post-attention
  half, the router, the routed MoE - keyed on antirez's decode-graph cache (il, island, variant).
  Three launches sat on the legacy stream and broke capture (sinkhorn, hc_weighted_sum, hc_expand);
  moved to the decode stream (a no-op when not capturing). A wrong flag (projected=true is the
  prefill's early-return contract) skipped the attention core for a while and produced a fake
  12.57 t/s with a moved sha - caught by the baseline sha, fixed with a distinct attention_core
  entry. Final: 160 captures (4 x 40), 0 retired, 0 CUDA failures, greedy dump = baseline
  e73f763588d4f253; the env set with no island bits is baseline too.
  SPEED, three pairs interleaved (load 1.3-4.1, avail 89 GiB): on 8.66 / 9.81 / 9.62, off 9.59 /
  9.85 / 9.86 t/s. MEASURED-NULL. ~60% of the token's launches replayed from graphs and the token
  did not get shorter: CPU launch issue was never on the critical path here - the GPU waits on
  miss reads and readback syncs, not on the CPU's next launch. The whole-layer graph (attention
  core with patched position nodes) cannot pay either by this measurement. Opt-in stays off;
  the open item is closed as measured, not built-out.
- ARENA, CLOSED AS FAR AS THE CODE GOES: sizing from MemAvailable, the margin knob, the price
  list (0.08 t/s/GiB), the memory trace; what remains is the box's memory itself.
- 14:55 AEST BRANCH PLACEMENT: the graph-island commits stay on lane-integrate only. They are an
  opt-in measured null, and their ds4.c hunks anchor on the profiler lines that the port branch
  does not carry (a cherry-pick attempt conflicted and was reset: v4.1-flash-cuda is back at
  9293d3d57, its five accidental picks - two of which were cache-branch commits - undone). The PR
  branches carry what is measured to pay and is gated: nothing from the island lane.
- 14:05 AEST LANE HITSFIRST (lane-integrate d6bf76273 + fix): the selected load starts the miss reads
  and returns; the routed launch runs the LUT gate/up for the hit pairs (pair mask), waits, then the
  miss pairs; the down sum6 stays slot-ordered. Gates: short PASS e73f763588d4f253, long PASS
  5566a268d8baab75. Speed, three interleaved pairs (box contended, load 2.5-9, avail 102 GiB, arena
  ~6,900-7,100 slots, hit ~0.955): on 9.07 / 9.71 / 9.73 vs off 8.81 / 9.40 / 9.50 t/s -> +0.26 /
  +0.31 / +0.23, ~+3%, consistent sign (pair 1 confounded by arena size 7097 vs 5900).
  Only gate/up (~10 ms of the ~23 ms routed) overlaps the ~24 ms of reads; the down projection and
  its quantize still wait. NEXT: per-slot down partials with the same slot-ordered final sum.
- 14:40 AEST HITSFIRST, FULL (lane-integrate eb1073ca3): the down projection joins the overlap -
  per-slot partial kernels (the sum6 body for one slot, storing `acc`) and a slot-ordered sum from
  0.0f, the same float sequence as the fused kernel. First attempt used the `down` scratch tensor
  for the partials; it aliases the scratch the miss-phase gate/up writes, so the hit partials were
  overwritten - garbage output, both gates FAIL, hit rate collapsed to 0.65 from garbage routing.
  A dedicated device buffer fixed it: short PASS e73f763588d4f253, long PASS 5566a268d8baab75.
  SPEED, three interleaved pairs (load 2.2-5.0, gpu 0%, avail 101 GiB, arena ~6,900, hit ~0.95,
  driver 580.159.03, ctx 32768): on 9.96 / 10.06 / 9.99 vs off 9.53 / 9.51 / 9.45 t/s ->
  +0.43 / +0.55 / +0.54 = +5.3%, every pair the same sign. Default ON (DS4_CUDA_HITS_FIRST=0 off).
- 14:55 AEST HEADLINE THROUGH THE DOOR, HITSFIRST default (load 0.98-1.36, gpu 1%, driver 580.159.03,
  ctx 32768, arena ~6,900 slots on a 101 GiB box): 9.92 · 9.88 · 9.94 t/s. CLAIM by the minimum
  rule: 9.9 t/s generation on this box as it stands. From 1.69: 5.9x. Prefill 2.2-2.5 short.
  HITSFIRST cherry-picked onto cuda-resident-expert-cache (four commits + a fix-up dropping the
  references to the opt-in LUT variants that branch does not carry); standalone build + V4 gate queued.
- 23:20 AEST (2026-09-13 -> 14) THE SECOND CANDIDATE ROUND, the navigator's word: try every
  bit-identical candidate, keep the ones with good results, check the results well.
  - F  O_DIRECT off (DS4_CUDA_NO_DIRECT_IO=1) vs on, three interleaved pairs, 256 tokens:
       on 9.72 / 9.70 / 8.59 (last arm at load 6.5, a build overlapped), off 5.80 / 6.09 t/s;
       expert read time 5.7 s vs 27.5 s per run. VERDICT: O_DIRECT stays on, off costs ~40%.
  - THREADS re-sweep (pool 8 vs 24) under hits-first: 9.67/10.07/9.89 vs 9.91/9.79/9.80. NULL.
  - D  pread straight into the arena: NULL BY CONSTRUCTION - the cache stats print
       "copy 0.0 s" on every run; the staging copy is free on unified memory. Not built.
  - E  router ids via a mapped flag: ALREADY LANDED as DRAINCUT (event-gated async D2H into a
       pinned buffer, no device drain). The remainder (spin on a mapped flag instead of the
       event wait) is worth under 1 ms per token, below the noise floor. Not built.
  - B  short prompts through the decode path: ALREADY THE BEHAVIOUR - ds41_prefill_count returns
       1 below `minimum` (256 tokens on streaming), so a 14-token prompt already steps token by
       token. The slow short prefill (2.2 t/s) is warm-up and cold misses, not the batched sweep.
  - A  STAGED HITS-FIRST (lane-integrate, uncommitted at this line): the pool takes a stage
       boundary; miss tasks are reordered gate/up first then down; the layer waits for the
       first stage, launches the miss gate/up kernel, then waits for the rest before the miss
       down kernels. DS4_CUDA_HITS_FIRST_STAGED=0 restores the one-stage order.
       Correctness PASS e73f763588d4f253 (staged on). Long PASS 5566a268d8baab75 - BUT see the
       gate defect below; the second long run was OOM-killed under foreign load and the PASS
       came from a stale dump. Speed A/B run under load 9-20 with the arena squeezed to
       3,600-5,400 slots by the studio's next-server (13 GB) and vitest: staged 8.40/8.04/9.13
       vs unstaged 8.18/8.18/(lost). NOT A MEASUREMENT. Re-run owed on a quiet box.
  - GATE DEFECT FIXED (sparkport-verify.sh 4b9f18235): a killed run left the previous dump in
    place and the gate compared against it and printed PASS. Each gate now deletes its dumps
    before running. The 23:50 long PASS above is therefore unverified; re-run owed.
  - BUILD-RECIPE FINDING, the important one: every binary that produced the baseline sha was
    built with NO CUDA_ARCH (an sm_75 cubin + PTX, JIT-compiled by the driver at load).
    `make cuda-spark` pins sm_121a native SASS and defines DS4_CUDA_HAVE_MXF4; that build gives
    sha 187cf54a55a03b34 on BOTH arms (the same sha the LUT16 kernel gave) - a different
    compiler path contracts the fast-math FMAs differently. The gate's baseline is therefore
    tied to the build recipe: the no-arch build (`make -B ds4 ds4-server ds4-bench ds4-eval
    ds4-agent`, no CUDA_ARCH) is the one the baseline belongs to. Whether native sm_121a is
    FASTER is an open, separate question (it would need its own baseline, like B would have).
  - NVMe LATENCY, measured properly (threaded O_DIRECT pread probe, not dd): one 3 MiB read
    416 us (7.6 GB/s); a whole miss (3 tensors in parallel) 1.06 ms; splitting a tensor into
    2-24 chunks is SLOWER (488-544 us). Aggregate at depth 8/16/32: 10.1/9.9/9.7 GB/s. The
    earlier "3 ms per 3 MiB" was dd process spawn. CHUNKED MISS READS: dead before building.
  - ROUTER-AHEAD PROBE (DS4_V41_ROUTER_AHEAD_PROBE=1, ds4.c): at the end of layer il, runs
    layer il+1's router on the residual that will feed it (attention delta and HC re-mix
    skipped), scores it against il+1's real top-6, and counts how many of il+1's misses the
    guess covers. Own scratch tensors; the decode path is only read. Result pending.
- 01:15 AEST (2026-09-14) ROUTER-AHEAD BUILT AND MEASURED, FIRST PASS (lane-integrate, uncommitted):
  the guess for layer il+1 is queued in-stream after il's after_moe (hc sum + norm + il+1's router
  on the residual, DRAINCUT-style pinned readback), collected after il's drain, and its non-resident
  experts claim arena slots + start pool reads; the real selected load waits for that batch first.
  Probe numbers (64-112 tokens): top-6 overlap 4.16-4.22 of 6, covers 65-70% of next layer's misses.
  GATE: sha e73f763588d4f253 with prefetch on, cache on and off. LONG gate not yet re-run.
  SPEED, three interleaved pairs, under the studio's load (6-18, arena 2,700-4,000 slots because
  next-server + vitest ate MemAvailable): hit rate 0.91-0.94 vs 0.79-0.82, read time 7-10 s vs
  18-21 s per run - and generation SLOWER: ahead 6.80 / 6.59 / 7.56 vs plain 7.54 / 7.39 / 7.90.
  Evictions doubled (18-25k vs 12-15k). The read saving (~36 ms/token) is more than eaten.
  HYPOTHESIS: the real load blocks on the whole prefetch batch (up to 18 reads, ~2.5 ms at 8 workers)
  with only ~1 ms of attention to hide it, so each layer stalls ~1.5 ms (60 ms/token); the wrong
  guesses also evict real experts from a small arena. Instrument added (blocked seconds in the cache
  stats line) and a top-K knob (DS4_CUDA_ROUTER_AHEAD_TOPK, best-first); plain vs top-6 vs top-3 vs
  top-1 sweep queued.
- 01:50 AEST (2026-09-14) ROUTER-AHEAD, SECOND PASS (two pools) AND THE ROUND'S CLOSE:
  the pread pool became an instance struct (expert pool 8 threads, prefetch pool 4,
  DS4_CUDA_ROUTER_AHEAD_THREADS); the real load waits only when it selects an expert still
  landing. Gates with prefetch on: short e73f763588d4f253 both arms; long 5566a268d8baab75 (run 1
  OOM-killed under load 20+, run 2 matched). Blocked time per 256 tokens still 5.6-8.8 s at top-6,
  3-4 s at top-3; generation below plain on every interleaved pair (plain 7.72/6.85, top-6 7.42,
  top-3 7.75/6.79, load 11-34). The arithmetic: top-6 needs ~21 GB/s of NVMe to land in one
  attention window, top-3 ~11, the drive gives ~10 (measured). VERDICT: correct, opt-in, null on
  this box. Committed as e89954d1b (router-ahead) after d958c2a0e (staged hits-first, own commit).
  README, PR doc and WIKI/theory/177 §16-18 updated; kgraph nodes added.
  OWED: the staged hits-first speed A/B and the sm_121a-native question, both on a quiet box.
- 09:45 AEST (2026-09-14) UPSTREAM LANDED ITS OWN V4.1 CUDA: antirez a04f46fa4 "DeepSeek v4.1
  Flash support for CUDA" (bounded SSD streaming, exact short prefills, session batching,
  dual-Spark RoCE TP; his doc: 7-8 t/s generation single Spark over 32 teacher-forced tokens after
  a 32K prefill, 21.9 t/s on two Sparks). Built at a04f46fa4 on the Spark, no-arch recipe (sm_75):
  his greedy dumps are cache-transparent (on = off) at bb06e711bc498bb9 short and d3355c94c70a4bb1
  long - a different last ulp from ours (e73f7635 / 5566a268); two ports, two valid float paths.
  HEAD-TO-HEAD, quiet box (load 1.0-1.9, gpu 0-1%), same model/flags/prompt, 256 tokens, ctx
  32768, BUT MemAvailable only 52-54 GB (two studio next-server processes hold 38 + 21 GB RSS),
  so both arenas were squeezed: upstream 5.45 / 5.62 t/s vs ours 6.60 / 6.69 / 6.62 (his third
  run died). Ours +20% under identical squeeze. His path: persistent slot cache, async upload
  stream from the mapped file, no O_DIRECT, no read pool, no zero-copy hits, no hits-first - the
  levers we measured one by one. Full-arena comparison owed when the studio releases the memory.
- 10:40 AEST (2026-09-14) THE LEVERS GO ONTO UPSTREAM (navigator: work all of antirez's stuff into
  ours and see if the combination is faster). New lane `v4.1-flash-cuda-levers` from a04f46fa4
  (worktree _worktrees/dwarfstar-onupstream; Spark tree upstream-v41). What he already has of ours,
  in his own form: direct I/O, a zero-copy planar slot cache with LRU, MemAvailable-style sizing
  (8 GiB reserve), a hot list, threaded Engram reads, a warp top-k router, event-gated id readback.
  What he lacks: parallel miss reads (his staging ring is serial), hits-first, staged hits-first.
  NOTE: his tree has no DS4_CUDA_EXPERT_CACHE knob, so the earlier "cache on = off" on his build
  was vacuous; his gate is the baseline-sha assertion alone (short bb06e711bc498bb9, long
  d3355c94c70a4bb1). PORT 1, the pread pool (1c5e9a737): both shas unchanged; three interleaved
  pairs on one box (load 1.6-3.4, avail 59-63 GB, arena ~3,000 slots): serial 5.96/5.73/6.23,
  pool 6.55/6.87 (one run lost) t/s, +12%; read time 28-31 s -> 20-21 s per 256 tokens; short
  prefill 4.0-4.3 -> 6.2-6.3 t/s. PORT 2, hits-first + staged, in flight.
- 11:30 AEST (2026-09-14) PORT 2 on upstream, HITS-FIRST + STAGED (bd20634c8, opt-in): both shas
  unchanged; three pairs 8.21/8.39/8.31 with vs 8.37/8.27/8.61 without (load 3-7, arena ~6,000):
  NULL on his tree - his shared-expert overlap already fills the miss-read window. OURS vs
  COMBINED (his + pool), three pairs, load 1.7-25, arena 4,900-6,000: ours 9.38/8.59/9.18 vs
  combined 8.01/8.83/7.99; hit rate ours 0.90-0.91 vs his 0.84-0.87 at the same slot count -
  the gap is ~12 extra misses per token, which is the generation gap. His exact short prefill is
  2x ours (4.2-4.5 vs 2.1-2.5 t/s). The staged A/B on ours died again under the studio fleet
  (load 52, avail 55 GB). NEXT: why his cache misses more at equal slots (stamping, hot list,
  seeding), and his short-prefill path onto ours.
- 11:00 AEST (2026-09-14) HOT-LIST HYPOTHESIS, READY BUT UNRUN: his built-in hot list covers Flash,
  Pro and GLM, not Flash41, so V4.1 on his CUDA tree starts cold; his file loader reads
  "layer expert hits" and skips "#" lines, which is exactly our persisted list's shape
  (~/.cache/ds4/cuda_expert_hotlist.txt on the Spark, 8,717 lines, 4th column ignored). The test
  is one env var, no code: DS4_METAL_STREAMING_EXPERT_HOTLIST=$HOME/.cache/ds4/cuda_expert_hotlist.txt
  on his+pool, interleaved against cold, three pairs. Not run: the studio's vitest fleet holds the
  Spark at load 50-70 (41 logins) and my wait-for-quiet loop was stopped rather than let it fire
  under that. OWED with the staged A/B, both on a quiet box.
