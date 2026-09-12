# PLAN · SPARKPORT — DeepSeek V4.1 Flash CUDA port for the DGX Spark (GB10)

Branch `sparkport-v41-cuda`, forked from antirez/ds4 main `bd66c40` ("DeepSeek v4.1 Flash support
for Metal", 2026-09-12). Goal: the V4.1 path runs on the Spark's CUDA backend with SSD streaming,
same 128 GB-class design as the M5 (main weights 152 GiB stream; Engram 189 GiB read row-wise from
disk, never resident). Verification is paramount: every kernel golden-checked, then a greedy
token-identity gate against the Metal reference on the M5 with the same GGUF.

## Ground truth (read off the commit)
- V4.1 graph in ds4.c (ds41_gpu_graph, lines ~39054-67410) calls 173 ds4_gpu_* primitives.
  CUDA already has 140. Missing 33; ~10 are TP/M5/GLM-vision/gfx1151 = NOT needed on one Spark.
- Core port = 24 primitives (~1,072 Metal host lines) + 10 shader kernels (metal/dsv41.metal,
  293 lines) + moe.metal diff (39 lines):
  dsv41: attention_output_batch(9) attention_output_tp_batch(12, TP - skip) candidate_blocks(6)
  candidate_filter(7) carry_copy(29) engram_add(35) gather_kv(37) indexer_pack(35)
  indexer_packed_bytes(0) indexer_scores_batch(10) indexer_scores_packed(44) indexer_topk_batch(7)
  pool2(45) projection_rows(9) quantize(35) rope(4) rope_stride(50) tensor_ops_available(3)
  fused: hc_expand_add_rms_norm_mix_split_norm_f16_tensor(237) hc_rms_norm_mix_f16_tensor(78)
  hc_rms_norm_mix_split_norm_f16_tensor(168) hc_rms_norm_mix_f16_available(8)
  router_project_select_fused_tensor(106) router_shared_gate_up_q8_0_tensor(98)
  stragglers: hc_rms_scale_project_f16_tensor(129) stream_expert_cache_seed_experts_gpu_copy(15)
  stream_expert_cache_seed_from_layer_selected(0) stream_expert_cache_release_layer_cache(0)
- Engram path is CPU (ds4_engram.c: read/read_batch + prefetch thread) - backend-agnostic.
- SSD expert streaming already exists on CUDA (9 stream_expert_cache_* impls).
- Gate today: ds4.c refuses with "V4.1 requires Metal inference" (find + relax for CUDA).

## Build steps (each gated before the next)
1. [ ] Inventory every missing primitive's exact signature (ds4_gpu.h) + Metal semantics.
2. [ ] Port the 10 shader kernels to ds4_cuda.cu (dsv41 section), 1:1 with the Metal math.
3. [ ] Port the 24 host wrappers (CUDA launches) matching ds4_gpu.h contracts.
4. [ ] Relax the Metal-only gate for CUDA; build on the Spark; run the V4.1 Q2 GGUF with
       --ssd-streaming; first goal = it decodes tokens at all.
5. [ ] GATE 1 (quality batch): gguf-tools/quality-testing/deepseek-v4.1-flash-20260910-batch
       prompts vs official continuations (antirez's own release bar).
6. [ ] GATE 2 (golden): Metal build on the M5, same GGUF, greedy, token-identical vs Spark CUDA
       (G1-style). Needs the GGUF on the M5 (rsync started 2026-09-12).
7. [ ] Measure t/s on the Spark; compare to antirez's M5 numbers; log honestly.
8. [ ] Upstreamable PR shape: keep changes to ds4_cuda.cu + minimal ds4.c gate changes.

## Working rules
- Lane: SPARKPORT. Worktree `_worktrees/dwarfstar-sparkport` (M5 edit), Spark build tree
  `~/dwarfstar/_worktrees/sparkport` (rsync source files, proven flow). Commit per step.
- No stash. Small prefixed commits "sparkport: ...".
- Status file: this PLAN, updated at every beat.

## Log
- 2026-09-12: branch + plan created. GGUF Q2 on Spark (byte-exact). Vision GGUF on Spark.
  Official safetensors pulling (~13 h). M5 golden copy started.
- 2026-09-13: PORT COMPILES ON BOTH BACKENDS. New ds4_cuda_dsv41.cuh (10 kernels 1:1 with metal/dsv41.metal,
  all-lane warp reductions; wrappers mirror ds4_metal.m validation; packed Metal-tensor-op indexer path reported
  unavailable so ds4.c takes the unpacked path; TP off). ds4_cuda.cu: attention-output split into an impl with a
  round_low_bf16 hook (V4.1 BF16-rounds `low` between the two Q8 projections). ds4.c: 3 guards flipped to
  !DS4_NO_GPU, backend gate accepts CUDA. ds4_gpu.h: V4.1 types in a DS4_V41_TYPES_DEFINED block (ds4_cuda.cu
  does not include the header). Spark CUDA build 48.9 MB clean; Metal build unchanged. Steps 1-3 done, 4 = first run.
