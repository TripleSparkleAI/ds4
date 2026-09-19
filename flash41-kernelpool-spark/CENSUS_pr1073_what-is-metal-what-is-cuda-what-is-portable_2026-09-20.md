# CENSUS - antirez/ds4 PR #1073 (kernelpool): what is Metal, what is CUDA, what is portable

Lane KERNELPOOLSPARK, 2026-09-20. Read at PR head `af7c02b59` (36 commits on `8db1d1d15`, which is
antirez `origin/main` at the time of reading; `git merge-base pr1073 origin/main` = `8db1d1d15`).
Every count below was taken with `git show --stat` and `git diff` over the fetched ref `pr1073`,
never from the PR description. Re-derive rather than quote: `git log --oneline origin/main..pr1073`.

## The headline

```
   36 commits
   ───────────────────────────────────────────────────────
   METAL-ONLY        11   the speed wins live here
   CUDA-TOUCHING      4   429 lines in ds4_deepseek41_cuda.cuh, of which
                          9 entry points return 0 ("the caller runs the old path"),
                          8 are compositions of ops main already ran,
                          4 are real kernels (2 of them the DSpark Markov draft chain)
   ENGINE-GENERIC    11   scheduling, converter, DSpark control, server, engram
   TP-ONLY            8   two Mac Studios over a wire; no GB10 equivalent
   DOCS / TESTS       2
```

The PR's 1.8x (19.2 -> 34.7 t/s at 2048 on MXFP4, its own table) is a Metal number. On the CUDA
path the graph runs the SAME operator sequence as main: every fused Metal entry has a CUDA twin
that either composes the pre-existing ops (op count unchanged) or returns 0 so `ds4.c` falls
through to the unfused chain it ran before. Two engine-generic commits DO change what the CUDA
decode loop does per token (rows 1 and 5 below); those are the only places a CUDA decode delta
can come from without a port, and the KP1 seal measures exactly that.

⚠ **THE AS-IS PR DOES NOT COMPILE ON CUDA.** `make cuda-spark` at `af7c02b59` on the GB10 exits 2:
six `identifier is undefined` errors in `ds4_deepseek41_cuda.cuh` (`ds4_gpu_matmul_q8_0_bf16_tensor`
x3, `ds4_gpu_rms_norm_weight_bf16_tensor` x2, `ds4_gpu_hc_weighted_sum_bf16_tensor` x1). The
fused entries added by `d8d1523b3` call the bf16 wrappers that `d95f8b610` defined at the END of
the same header, and `ds4_cuda.cu` never includes `ds4_gpu.h`, so no prototype exists. Three
forward declarations fix it (our ref `pr1073-cudafix` = `577814f0d`, one commit on `af7c02b59`,
no behaviour change). Metal never compiles this header, which is why the branch's own gates did
not see it. Details: `flash41-kernelpool-spark/BUILD_as-is-cuda_2026-09-20.md`.

## The table

Class key: M = METAL-ONLY · C = CUDA-TOUCHING · G = ENGINE-GENERIC · TP = TP-ONLY · D = DOCS/TESTS.
CUDA status is read from `ds4_cuda.cu` and `ds4_deepseek41_cuda.cuh` at `8db1d1d15`, by function
name. Port cost S/M/L is a judgement; "share of the 1.8x" is UNKNOWN wherever the PR's own
numbers do not isolate the commit (they never do; the table is per-branch, not per-commit).

| # | commit | class | mechanism | CUDA status at main | port cost | share of 1.8x |
|---|---|---|---|---|---|---|
| 1 | `2a281b080` Queue decode layers on a single node | G | `queue_layers` was `tp_world == 2` only; now every non-resident single-node decode encodes all layers in one command stream with a flush every `DS4_METAL_V41_DECODE_FLUSH_LAYERS` (default 2) layers. Main drained after EVERY layer. | **CHANGES CUDA BEHAVIOUR.** `ds4_gpu_begin_commands` is a no-op on CUDA and `ds4_gpu_end_commands` / `ds4_gpu_flush_commands` are `cudaDeviceSynchronize`. Main: ~60 device syncs per token. PR default: ~31 (a flush every 2 layers is still a blocking sync on CUDA, unlike a Metal commit). `FLUSH_LAYERS=0`: 2 per token. | none (already applies) | UNKNOWN; on CUDA the P1 seal prices it |
| 2 | `d95f8b610` Round activations inside the producing kernels | C + M | Metal folds the bf16 rounding into rms_norm, q8 matmul, hc weighted sum/expand, rope. | Shim: each `*_bf16_tensor` = the unchanged op + the standalone `ds4_gpu_dsv41_quantize` pass main already ran as `ds41_bf16`. Op count on CUDA unchanged (main: 19 `DS4_V41_BF16` sites; PR: 25, the extra 6 are inside the wrappers). | M (an epilogue in ~6 kernels) | UNKNOWN |
| 3 | `edceb7ab7` Select experts in one dispatch | M | sigmoid + bias + top-k + normalise in one Metal dispatch (`ds4_gpu_dsv41_router_one`, `#ifdef __APPLE__` in `ds4.c`). | CUDA already one launch: `router_select_warp_topk_kernel` under `ds4_gpu_router_select_tensor`. Metal catch-up. | none | UNKNOWN (small) |
| 4 | `10118742a` Publish ratio-1 keys in batches on Metal | M | drops an `#ifdef __APPLE__ ratio == 2u` guard so Apple batches every ratio. Non-Apple never had the guard. | unchanged | none | 0 on CUDA |
| 5 | `e76839617` Skip candidate selection while every block is kept | G | when `(n_comp + 7) / 8 <= 2048` every block is a candidate: skip `ds4_gpu_indexer_topk_tensor` + `ds4_gpu_dsv4_topk_mask_tensor` on index-source layers. | **CHANGES CUDA BEHAVIOUR** at every context up to 16,384 compressed rows: two launches per index layer per token removed (they selected all blocks anyway). Lossless by construction. | none | UNKNOWN (small; at 4096/6144 it applies) |
| 6 | `ccbf2c083` Decoder suffix from 2541 tokens | G | wide-sweep prefill threshold `8192 -> 1 + (n_layer - 20) * 127`. | applies; prefill path only | none | prefill, not decode |
| 7 | `ce5a812fd` Select candidate blocks in batches | C | batched candidate top-k/mask for verify rows (DSpark) and prefill rows. | CUDA gets `ds4_gpu_dsv41_candidate_topk_batch` / `_mask_batch`: a per-row loop over the existing `ds4_gpu_indexer_topk_tensor` and `ds4_gpu_dsv4_topk_mask_tensor`. Real code, no new kernel. | S if a fused batch kernel is wanted | UNKNOWN |
| 8 | `d744ebb9e` Q4_K experts on compact resident tiles | M | Metal resident Q4_K expert tiles. | n/a to our file (IQ2_XXS/Q2_K experts); the Q4 question is the glm53-q4 lane's | none | UNKNOWN (Q4 only) |
| 9 | `3b7f8f224` Select every key for short index rows | C | when `visible <= 512` keys, write ids 0..visible-1 instead of scoring + top-k. | new tiny kernel `dsv41_indexer_all_kernel` on CUDA. Real, applies to the first 512*ratio tokens only. | none | ~0 at our frontiers |
| 10 | `c016ea773` Keep released MXFP4 experts in the converter | G | converter | n/a | none | 0 |
| 11 | `a60edc61f` DSpark stages to a separate support GGUF | G | converter | n/a | none | 0 |
| 12 | `d8d1523b3` Speed up Metal decode + DSpark verify rows on decode kernels | M + C + G | THE big one: 1,193 lines `ds4_metal.m`, 1,594 lines of `.metal` (fused rope+quantize store, hc block input chain, norm pairs, attention decode rows, matmul+expand, project pairs, shared swiglu), 1,220 lines `ds4.c` (verify-rows plumbing, engram second table). | 286 lines in the `.cuh`: **9 entries `return 0`** (`hc_block_input`, `hc_block_input_rows`, `norm_pair_rows`, `attention_decode_rows`, `matmul_expand`, `matmul_expand_rows`, `hc_project`, `router_rows`, `matmul_f32_rows`), 7 compositions of existing ops (`project_pair_q8`, `shared_swiglu`, `rope_quantize` = rope + quantize + copy, `project_f16_bf16`, `attention_decode`, `project_q`, `attention_low`, `norm_pair`, `hc_input`), 3 real kernels (`dsv41_hc_mean_kernel`; `dsv41_markov_part_kernel` + `dsv41_markov_final_kernel` = the DSpark draft chain). | L (this is the port) | UNKNOWN; Metal's share is most of the 1.8x |
| 13 | `4fbbc4a45` Encode the vocab head before the last drain; wide sweep at 3072 | G | `queued_head`: the output projection is encoded inside the layer stream before the final drain (gated on `queue_layers`, so now live single-node); short prefill sweeps keep a partial tile. | applies on CUDA once row 1 makes `queue_layers` true: one fewer begin/end pair per token. Prefill half is prefill. | none | UNKNOWN (small) |
| 14 | `9f3892d2f` TP partials into the slab slot, batch kslice rows | TP | | n/a | none | 0 |
| 15 | `dc81436da` Bind owned expert range once on one-token MoE fallback | TP | `ds4_metal.m` only | n/a | none | 0 |
| 16 | `1923131d4` Match the rope contraction of the fused rope-quantize store | M + D | Metal correctness fix + `tests/test_dsv41_rope_quantize.c` | CUDA `rope_quantize` composes rope then quantize; test is Metal-shaped | none | 0 |
| 17 | `a3f6f313a` Port the decode-control regression test | D | | | none | 0 |
| 18 | `8d73a0eff` Balance TP decode experts by list position | TP | | n/a | none | 0 |
| 19 | `a13b513e6` Split the shared expert across TP ranks | TP | | n/a | none | 0 |
| 20 | `867465cc6` Fold TP gate sums into the hc expansion | TP | | n/a | none | 0 |
| 21 | `e49643fff` Shared expert beside the routed experts on Metal | M | concurrent shared-expert encode (`#ifdef __APPLE__`, `shared_queued`). | CUDA already does this: `ds4_gpu_dsv41_shared_start` on a second stream, `ds4_gpu_dsv41_shared_join` via `cudaStreamWaitEvent`. Metal catch-up. | none | UNKNOWN |
| 22 | `262b4a66e` Verify rows through simdgroup matrices | M | Metal simdgroup matmul for DSpark verify rows (n>1). | CUDA: `n_tok > 1` q8 matmuls already route to cuBLAS (`g_cublas_ready && n_tok > 1` in the dispatch). | none | DSpark only |
| 23 | `a6b50ff6c` Shared expert + attention output rows through simdgroup matrices | M | simdgroup matmul for the shared expert and attn-out (n>1 rows). | as row 22: cuBLAS for batched rows; the n=1 matvec kernels are unchanged on both. | none | DSpark/batch only |
| 24 | `c674b198d` Verify DSpark blocks across two TP nodes | TP | | n/a | none | 0 |
| 25 | `1e3bcf8d8` Release verify-block gates through the spin kernel | TP | `ds4_metal.m` spin kernel | n/a | none | 0 |
| 26 | `bd85ba6ac` Keep the row kernels on the exact rows path | M | | n/a | none | 0 |
| 27 | `7da9535f2` Sync idle server slots in the prefill chunk | G | `ds4_server.c` | applies | none | server, not bench |
| 28 | `b7bcccdea` Stop drafts before EOS | G | DSpark control | applies with `--dspark` | none | DSpark only |
| 29 | `29ce27171` Second Engram table for prefill rows | G | one line | applies | none | prefill |
| 30 | `5c2dac7e3` End the TP bind line | TP | | n/a | none | 0 |
| 31 | `aea1a0bae` Docs | D | | | none | 0 |
| 32 | `0b2d6e28c` Converter writes only text tensors | G | converter | n/a | none | 0 |
| 33 | `1c6920666` Stop proposing drafts while they lose | G | DSpark control (the "diminishing past 32k" fix) | applies with `--dspark` | none | DSpark only |
| 34 | `fd7091916` Score verify rows with the tiled indexer kernel | M | | n/a | none | DSpark only |
| 35 | `4b1b1519e` One thread per index key on Metal | M | Metal indexer scoring layout. | CUDA indexer scoring already a per-key kernel family (`indexer_top1_kernel`, `indexer_topk_pow2_kernel`, the 4096-chunk + merge path, CUB at 8192). | none | UNKNOWN |
| 36 | `af7c02b59` Radix select for index candidates on Metal | M | `metal/argsort.metal` radix select replaces a sort for the 2048-of-N candidate pick. | CUDA `ds4_gpu_indexer_topk_tensor` already dispatches a pow2 bitonic at `n_comp <= 4096` and chunk+merge above, with CUB for 8192. Metal catch-up. | none | UNKNOWN |

## What the CUDA path already has, in one list (Metal catching up, not the other way)

- one-dispatch expert select (row 3) - `router_select_warp_topk_kernel`
- shared expert concurrent with routed experts (row 21) - second stream + event join
- batched-row matmuls for verify/shared/attn-out (rows 22, 23) - cuBLAS at `n_tok > 1`
- top-k candidate selection without a sort (row 36) - bitonic pow2 + chunk merge + CUB

## What CUDA lacks that Metal now has (the port candidates, ranked in the DESIGN doc)

- in-kernel bf16 rounding epilogues (row 2): CUDA runs a separate `ds4_gpu_dsv41_quantize` pass
  after norm, matmul, hc sum/expand and before rope. ~6 extra launches per layer, each a read +
  write of an activation row (5,120 to 8 x 5,120 floats). Bytes are small; launches are not free
  at 60 layers.
- fused rope + quantize + store (row 12): CUDA = 3 launches (`ds4_gpu_dsv41_rope`,
  `ds4_gpu_dsv41_quantize`, `ds4_gpu_tensor_copy`).
- the hc block-input chain and norm pairs as one kernel (row 12): CUDA = `return 0`, the caller
  runs sinkhorn + weighted sum + quantize + norm + quantize as separate launches.
- matmul + hc expand fused (row 12): CUDA = `return 0`.
- Q4_K compact resident tiles (row 8): not our file.

## What is NOT portable

- Metal `simdgroup_matrix` paths (rows 22, 23): CUDA's equivalent is cuBLAS / mma, already used.
- Everything TP (rows 14, 15, 18, 19, 20, 24, 25, 30): two Macs over a wire; the GB10 is one box.
- Command-buffer semantics (row 1): a Metal "flush" commits work and returns; a CUDA
  `ds4_gpu_flush_commands` is `cudaDeviceSynchronize`. The PR's default of a flush every 2
  layers is a Metal pipelining choice that on CUDA is a blocking bubble every 2 layers. The
  right CUDA reading of this commit is `DS4_METAL_V41_DECODE_FLUSH_LAYERS=0` (two syncs per
  token), which is a follow-up seal (KP2), not a port.

## Overlap with our branches, by function name

`git diff origin/main pr1073` touches `ds4.c` (1,725 lines), `ds4_deepseek41_cuda.cuh` (429),
`ds4_deepseek41_gpu.h` (143), `ds4_gpu.h` (71), `ds4_engram.c` (61), `ds4_server.c` (7),
`ds4_tp.c` (6), `ds4_gpu_tp.h` (2), `ds4_metal.m` (1,988) and `metal/*`. **It does not touch
`ds4_cuda.cu` at all.**

- **hits-first** (`6803d6dba`, PR #1083 on our side): `ds4_cuda.cu` only (753 lines) in
  `moe_down_f32_kernel`, `moe_down_sum_qwarp32_kernel`, `moe_gate_up_mid_decode_lut_qwarp32_kernel`,
  `ds4_gpu_stream_expert_cache_reset_route_hotness`, `cuda_model_copy_to_device_streamed`,
  `cuda_stream_compact_prefill`, `cuda_stream_selected_cache_begin_load`,
  `cuda_stream_selected_ranges_valid`, `routed_moe_launch`. **Zero functions in common with
  #1073.** The two stack cleanly (different files).
- **glm53 / glm53-q4** (`fdb63f4a3`, `a378e1535`): `ds4.c` in `glm_graph_dense_tensor_layout`
  only (31 lines). #1073's 46 touched `ds4.c` functions are all `ds41_*` or the V4.1 graph;
  **zero in common.** `glm53-hitsfirst` = glm53 + hits-first, same conclusion.
- `triple-hitsfirst-passthrough` (`d788449f0`): `ds4_cuda.cu` only. Zero in common.

Derive: `git diff origin/main pr1073 -- ds4.c | grep '^@@' | sed -E 's/^@@[^@]*@@ //;s/\(.*//' | sort -u`
against the same over `origin/main..6803d6dba -- ds4_cuda.cu` and `origin/main..glm53 -- ds4.c`.

## Two things the seal must not get wrong

1. **The model file.** The PR is the DeepSeek **V4.1** Flash engine (`ds41_*`). `ds4flash.gguf` on
   the spark resolves to `DeepSeek-V4-Flash-IQ2XXS-...-0731.gguf`, a V4 file that never enters the
   `ds41_*` graph. The standing flash41 rounds (R28, R29, GLM-R1/R2) use
   `gguf/DeepSeek-V4.1-Flash-Q2.gguf` and their G1 references (`5ed3e6df` / `c5cb8256`) are minted
   on it. KP1 runs on that file; a run on the V4 file would measure nothing the PR changed.
2. **The arm that can run.** `pr1073` as fetched does not build. The bench arm is `pr1073-cudafix`
   (`577814f0d`), which is `af7c02b59` + three forward declarations. The seal names both shas and
   the diff between them is the whole content of that commit.
