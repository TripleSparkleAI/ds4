# DRAFT comment for antirez/ds4 PR #1073 - NOT POSTED

> ⛔ THIS FILE IS THE DELIVERABLE. Nothing below has been submitted anywhere: no
> `gh pr comment`, no `gh pr review`, no issue, nothing on antirez's repository. The navigator
> posts it by hand if he chooses. The measurement slots are now FILLED from the sealed KP1 round
> (fired 2026-09-19, `2026-09-20-KP1-RESULT-ties-at-plus1.74pct-...md`). Every number present was measured by lane KERNELPOOLSPARK on 2026-09-19/20 and is
> traceable to `BUILD_as-is-cuda_2026-09-20.md` or to the KP1 RESULT doc. No slot is a guess.

---

We built this branch with CUDA on a DGX Spark GB10 (driver 580.159.03, CUDA 13.0, `make cuda-spark`,
sm_121) and found:

**It does not compile as-is on CUDA.** At `af7c02b59`, six `identifier is undefined` errors in
`ds4_deepseek41_cuda.cuh`: the fused entries from d8d1523b3 (`ds4_gpu_dsv41_project_pair_q8`,
`_shared_swiglu`, `_project_q`, `_norm_pair`, `_hc_input`) call `ds4_gpu_matmul_q8_0_bf16_tensor`,
`ds4_gpu_rms_norm_weight_bf16_tensor` and `ds4_gpu_hc_weighted_sum_bf16_tensor`, which d95f8b610
defines at the end of the same header, and `ds4_cuda.cu` does not include `ds4_gpu.h`. Three
forward declarations at the top of the `.cuh` fix the compile. The link then fails on
`ds4_gpu_tp_flag_fold_request` and `ds4_gpu_add_tensor_tp_flag` (five references each), which only
`ds4_metal.m` defines: the two V4.1 call sites from 9f3892d2f and 867465cc6 lack the
`#if defined(__APPLE__)` guard that main puts on the V4 site. With those two guards it builds:
`ds4-server` .text 24,862,747 (main 8db1d1d1) -> 24,945,559 (+82,812 B). Both hunks are in the
PR's own style and I can send them as a diff if useful.

**Which of the 36 commits touch CUDA:** d95f8b610, ce5a812fd, 3b7f8f224, d8d1523b3 (429 lines in
`ds4_deepseek41_cuda.cuh` between them). Read by function: 9 entry points `return 0` so `ds4.c`
runs the unfused chain, 8 are compositions of ops main already ran (the bf16 wrappers are the old
op plus the same standalone `ds4_gpu_dsv41_quantize` pass main called through `ds41_bf16`), and 4
are new kernels (`dsv41_indexer_all_kernel`, `dsv41_hc_mean_kernel`, the two Markov draft
kernels). So on CUDA the decode graph is the same operator sequence as main. The two commits that
DO change what a CUDA decode does per token are not Metal commits: 2a281b080 (single-node layer
queueing; on CUDA `ds4_gpu_end_commands` and `ds4_gpu_flush_commands` are both
`cudaDeviceSynchronize`, so main's drain-every-layer is ~60 syncs/token and your default flush of 2
is ~31) and e76839617 (skipping candidate selection under 2048 blocks).

**Same machine, same model, same sweep, main vs this branch (+ the two build fixes):**
`ds4-bench -m DeepSeek-V4.1-Flash-Q2.gguf --cuda --ssd-streaming --ssd-streaming-cache-experts 90GB
--prompt-file speed-bench/promessi_sposi.txt --ctx-start 2048 --ctx-max 6144 --step-incr 2048
--gen-tokens 128`, 3 interleaved cycles, load stamped, gen t/s at the worse of 4096/6144:

| | main 8db1d1d1 | this branch (0bbca98ee) | delta |
|---|---:|---:|---:|
| gen t/s, min across 4096/6144 | 9.55 | 9.67 | **+1.74 %** (paired vs bracketing main runs, sign 3/3, SE 1.05, range +0.26 .. +3.88) |
| prefill t/s (2048-token increment) | 82.7 | 81.7 | +1.74 % paired, sign 2/3, SE 3.48, range -6.78 .. +4.85 (not separable) |
| 1-min load before / after | 1.30-1.61 / 1.34-3.07 | 1.30-1.87 / 1.69-2.97 | expert cache 8941 slots / 82.88 GiB on every run, both arms |
| greedy identity vs main, `--temp 0 --dump-logprobs`, 2 prompts | reference | **NOT IDENTICAL on either prompt** | short: first differing token at index 6 of 32; long: same 16 tokens, different logprobs |

Sealed prediction, written before the run so it can be wrong in public: the header contributes
0.0 % on CUDA; the branch as a whole reads +1 .. +6 % from the removed syncs and the skipped
selection, and is greedy-identical to main on both prompts.

**Outcome: the speed half landed, the losslessness half did not, and the header half is not yet
separable.** The "header contributes 0.0 %" clause is UNSCOREABLE from this round: the arm carries
the whole branch, so the header's own share cannot be read off it. A per-mechanism matrix (each of
your commits behind its own switch, run against tip one at a time) is what isolates it, and it has
not been run yet. +1.74 % is inside the predicted
band, but it is also inside our own tip-versus-tip floor for this round (3.10 % max over three
adjacent main pairs), so by our pre-registered rule the branch **TIES** main on CUDA decode here and
we are not claiming a speed win. The prediction we got wrong is the one worth your attention: the
branch is **not greedy-identical to main on CUDA**. Same binary pair, same file, same 90 GB cache,
`--temp 0`: the short prompt's 7th generated token flips (main takes `.` over ` why` by 0.904 logits,
the branch reverses it), and the long prompt keeps all 16 tokens while its logprobs differ, with a
peak top-20 logit delta of 0.136. Our control is main against itself, twice, which is identical to
0.000000 in every top-20 logit over 32 steps, so this is the branch and not run-to-run noise.
One corroborating build fact: `cuobjdump` shows four new device kernels
(`dsv41_hc_mean_kernel`, `dsv41_indexer_all_kernel`, the two Markov ones) and, more to the point,
**`dsv41_candidates_kernel` gains a parameter** (`...jjjj` to `...jjjjj`) - so candidate selection on
CUDA is genuinely different code, not the same op sequence. We have NOT profiled whether it fires on
our decode path, so that is a hypothesis for the divergence rather than a cause we have shown.

**What did not transfer, in one line:** the fused kernels are all Metal, and on the GB10 at this
regime the decode is bandwidth-bound (expert reads, not dispatch), so the mechanism that bought
you 1.8x on the M3 Ultra is bounded to a few percent here even once ported; the part of the
branch that helps CUDA is the scheduling, and `DS4_METAL_V41_DECODE_FLUSH_LAYERS=0` is the CUDA
reading of it (a flush is a blocking sync on this backend).

**Scope rule for this comment (navigator, 2026-09-20): SPARK ONLY.** The comment reports what worked on
the DGX Spark GB10 with CUDA. Our M5 Max runs of this branch are NOT reported here: that box was
contested (load 2.1 to 7.2 per cpu) and flipped power mode mid-session, so its ratio is not separable
from the box. The one M5 fact clean enough to carry, as a footnote only: the branch builds and runs on
Metal at a 25 GB expert cache on the q2 file, and the prefill logits are bit-identical to main.

DSpark on CUDA: not measured yet (needs the support GGUF from your converter; our V4.1 Q2 file was
built before a60edc61f).
