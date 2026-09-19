# DRAFT comment for antirez/ds4 PR #1073 - NOT POSTED

> ⛔ THIS FILE IS THE DELIVERABLE. Nothing below has been submitted anywhere: no
> `gh pr comment`, no `gh pr review`, no issue, nothing on antirez's repository. The navigator
> posts it by hand if he chooses, after the `[NOT YET MEASURED]` slots are filled from the KP1
> RESULT doc. Every number present was measured by lane KERNELPOOLSPARK on 2026-09-19/20 and is
> traceable to `BUILD_as-is-cuda_2026-09-20.md`; every slot marked `[NOT YET MEASURED]` is owed
> to the sealed KP1 round, which is queued behind another lane's lock. No slot is a guess.

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
| gen t/s, min across 4096/6144 | [NOT YET MEASURED] | [NOT YET MEASURED] | [NOT YET MEASURED] |
| prefill t/s (2048-token increment) | [NOT YET MEASURED] | [NOT YET MEASURED] | [NOT YET MEASURED] |
| 1-min load before / after | [NOT YET MEASURED] | [NOT YET MEASURED] | |
| greedy identity vs main, `--temp 0 --dump-logprobs`, 2 prompts | reference | [NOT YET MEASURED] | |

Sealed prediction, written before the run so it can be wrong in public: the header contributes
0.0 % on CUDA; the branch as a whole reads +1 .. +6 % from the removed syncs and the skipped
selection, and is greedy-identical to main on both prompts.

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
