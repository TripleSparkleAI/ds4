# DESIGN - porting PR #1073's Metal mechanisms to the CUDA path, ranked

Lane KERNELPOOLSPARK, 2026-09-20. Companion to `CENSUS_pr1073_*.md` (what each commit is) and
`2026-09-20-KP1-PREREGISTERED-RULE.txt` (what the branch does on CUDA before any port). This
is a PLAN. No port is written here; each port lands as its own commit on
`flash41-kernelpool-spark` with its own seal, its own G1 gate and a before/after CSV, in
antirez's style: one branch, one claim per commit.

## The frame that decides the ranking

The PR's 1.8x is a Metal result on a launch-bound decode: 60 layers of small row ops, each a
Metal dispatch, and the Mac's fix was to fuse them. On the GB10 at the standing regime
(`--ssd-streaming --ssd-streaming-cache-experts 90GB`, this Q2 file) the tip decodes at
9.5-9.8 t/s (R28/R29 TIP floors), i.e. ~100 ms per token, and our own profile of the V4 file
(`WIKI/theory` MoEIsNotTheDecode: routed ~23 %, non-routed ~61 % of decode TIME) says the time
is in weight reads, not in dispatch. So every launch-fusion port below is bounded by the
launch budget it removes, and that budget is small against 100 ms. The ranking is honest about
that: the biggest CUDA-side number in this whole PR is not a port at all, it is a sync knob.

Score = (expected share on CUDA x confidence) / port cost. Shares are DERIVED from launch
counts and per-launch overhead, never measured; every one is a prediction to be sealed.

## The ranking

| rank | mechanism | Metal commit | CUDA function(s) it would change | expected share on CUDA | conf | cost | G1 gate | seal |
|---|---|---|---|---|---|---|---|---|
| 0 | **Decode queueing without mid-token syncs** (`DS4_METAL_V41_DECODE_FLUSH_LAYERS=0`) | `2a281b080` | none: a knob already in the PR. On CUDA `ds4_gpu_flush_commands` is `cudaDeviceSynchronize`, so the Metal default (flush every 2 layers) is half a fix; 0 removes ~29 more syncs/token | +1 .. +4 % (29 x 0.05-0.10 ms bubble on ~100 ms) | 0.5 | none | same-process token ids + logprob sha vs tip | **KP2** |
| 1 | **bf16 rounding as a kernel epilogue** | `d95f8b610` | `ds4_gpu_rms_norm_weight_tensor` (add a round-on-store flag), `ds4_gpu_hc_weighted_sum_tensor`, `ds4_gpu_hc_weighted_sum_split_tensor`, `ds4_gpu_hc_expand_split_tensor` (same), then make the `*_bf16_tensor` shims call the flagged kernel instead of op + `ds4_gpu_dsv41_quantize`. The q8 matmul stays two ops (n=1 goes to custom kernels that can take the epilogue later; n>1 goes to cuBLAS and cannot) | ~4 launches per layer removed x 60 = ~240 launches/token; at 3-6 us each and pipelined, +0.5 .. +1.5 % | 0.5 | M | **bit-exact by construction**: round-to-nearest-even of the same fp32 in the epilogue or in a later pass yields the same bf16. G1 must be sha-identical, and a divergence means the epilogue rounded a different value (a fused accumulate) and is a bug | **KP3** |
| 2 | **fused rope + quantize + store** for the KV publish | `d8d1523b3` (`ds4_gpu_dsv41_rope_quantize`) | `ds4_gpu_dsv41_rope` gains a quantize-and-copy epilogue; the CUDA shim (rope, quantize, `ds4_gpu_tensor_copy`) becomes one launch | 2 launches per KV-source layer removed; +0.2 .. +0.6 % | 0.5 | S | rope then RNE-to-bf16 in one kernel is the same arithmetic in the same order; sha-identical or bug | KP4 |
| 3 | **hc block-input chain as one kernel** (sinkhorn + weighted sum + round + norm + round) | `d8d1523b3` (`ds4_gpu_dsv41_hc_block_input`, `_rows`) | new kernel replacing `ds4_gpu_hc_split_sinkhorn_tensor` + `ds4_gpu_hc_weighted_sum_bf16_tensor` + `ds4_gpu_rms_norm_weight_bf16_tensor` for the 5,120-wide row; the `return 0` stub becomes real | ~4 launches x 2 (attn, ffn) x 60 = ~480 launches/token; +0.5 .. +2 % | 0.4 | M-L (the sinkhorn iterations need a block-wide reduction; the norm needs another) | the reduction ORDER changes (a block reduce vs the existing kernels' reduce). G1 may legitimately move by fp32 noise: the gate is sha-identical first, and a mismatch is adjudicated by the `--dspark-verify-depth` style logit-gap reading, never waved through | KP5 |
| 4 | **matmul + hc expand fused** | `d8d1523b3` (`ds4_gpu_dsv41_matmul_expand`) | the n=1 q8 matvec kernel writes straight into the `n_hc x n_embd` expansion with the split weights; the `return 0` stub becomes real | 1 launch + one 5,120-float round trip per layer x 2; +0.2 .. +0.5 % | 0.4 | M | the matvec's own reduction is unchanged; the expand is a per-element scale-add; sha-identical expected | KP6 |
| 5 | verify rows on decode kernels (DSpark verify batch) | `d8d1523b3`, `262b4a66e`, `fd7091916` | the `*_rows` stubs; CUDA already batches n>1 through cuBLAS | 0 without `--dspark`; UNKNOWN with it | 0.3 | L | separate DSpark G1 (the same-process rule, this repo's own `--dspark-g1`) | after KP3-6 |
| - | one-dispatch router | `edceb7ab7` | already `router_select_warp_topk_kernel` | 0 | - | none | - | - |
| - | shared expert beside routed | `e49643fff` | already a second stream + `cudaStreamWaitEvent` | 0 | - | none | - | - |
| - | radix select / one thread per key | `af7c02b59`, `4b1b1519e` | already pow2 bitonic + chunk merge + CUB | 0 | - | none | - | - |
| - | simdgroup matmuls | `262b4a66e`, `a6b50ff6c` | cuBLAS at n>1 | 0 | - | not portable | - | - |
| - | Q4_K compact resident tiles | `d744ebb9e` | not our file (IQ2_XXS/Q2_K); the Q4 question belongs to glm53-q4 | n/a | - | - | - | - |
| - | everything TP | 8 commits | two Macs over a wire | n/a | - | not portable | - | - |

## The first port, named: KP3, the bf16 epilogue on the norm and hc kernels

Why this one and not rank 0: rank 0 is a knob, not a port, and it runs first as KP2 because it
is free. Among the PORTS, the epilogue wins on three counts at once:

1. **It cannot change a bit.** Rounding the same fp32 to bf16 inside the producing kernel or in
   the pass after it is the same function of the same value. So its G1 gate is strict
   sha-identity with no adjudication step, which is the cheapest gate this repo has, and a red
   there is unambiguously a bug in the port rather than fp32 noise to argue about.
2. **It removes the most launches per line changed.** Four kernels gain a `round` flag and an
   `if (round) out = bf16(out)` on the store; the shims in the `.cuh` flip from
   `op && quantize` to `op_round`. ~240 launches per token go away for ~40 lines.
3. **It touches nothing hits-first or glm53 touch.** `ds4_gpu_rms_norm_weight_tensor` and the
   hc kernels are in `ds4_cuda.cu` but in none of the nine functions hits-first changes
   (`routed_moe_launch`, the `moe_*` kernels, the stream cache), so the branches stack.

What it will NOT do: move the number much. The prediction (to be sealed in KP3's own file) is
+0.5 .. +1.5 % at min-across-frontiers, and a TIE by the R-rule is the likely verdict. That is
the honest expectation for a launch-fusion port on a bandwidth-bound decode, and it is the
reason the design carries KP2 ahead of it: if KP2 (sync removal) reads +3 % and KP3 reads +1 %,
the estate has learned that on the GB10 the PR's mechanism class is worth ~4 % total and the
remaining ports (KP4-6, ~+1-3 % combined by the same arithmetic) are priced before anyone
writes them.

## What is NOT portable, stated once

- Metal `simdgroup_matrix` code: the CUDA equivalent is cuBLAS / mma, already on the path.
- Metal command-buffer pipelining semantics: a Metal flush commits and returns; the CUDA
  `flush` is a blocking sync. The PR's flush cadence is a Metal tuning; on CUDA the right
  value is 0 (KP2) and the Makefile-level port is to make `ds4_gpu_flush_commands` a no-op on
  CUDA when queueing (the stream already orders the work), which is a one-line follow-up if
  KP2 reads positive.
- Tensor parallelism across two Macs (`ds4_gpu_tp_flag_fold_request`,
  `ds4_gpu_add_tensor_tp_flag`, the spin-kernel gates, the slab slot): the GB10 is one box.
- Q4_K compact resident tiles: a Metal resident-weight layout; our file is IQ2_XXS/Q2_K.

## The rule every port lands under

Each port is one commit on `flash41-kernelpool-spark`, cut from `8db1d1d15`, carrying: the
kernel change, its seal file `2026-MM-DD-KPn-PREREGISTERED-RULE.txt` committed BEFORE the run,
the G1 gate result as AMENDMENT A1 in that seal, and a `RESULT_KPn_*.md` with the before/after
CSV (`ds4-bench --csv`, same machine, same file, same sweep, load and driver 580.159.03
stamped). A port whose G1 is not sha-identical does not land, whatever its t/s reads.
