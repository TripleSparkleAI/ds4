# BUILD - PR #1073 as-is on CUDA, DGX Spark GB10

Lane KERNELPOOLSPARK, 2026-09-19 (UTC; the lane is dated 09-20 in the navigator's day).
Box: DGX Spark GB10, driver 580.159.03, CUDA 13.0 (`nvcc` r13.0 build 36424714_0), Ubuntu 24.04,
20 cores. Target: `make cuda-spark` (`CUDA_ARCH=sm_121`, `-j20`). Load at start 0.17 1-min.
No model was loaded and no lock was taken; a build needs neither.

## The three builds

| worktree | sha | what | rc | ds4-server .text | delta vs main |
|---|---|---|---|---|---|
| `_worktrees/kp-main` | `8db1d1d1` | antirez main, fresh | **0** | 24,862,747 | - (byte-equal to R28/R29's tip) |
| `_worktrees/pr1073` | `af7c02b59` | PR #1073 head, untouched | **2** | no binary | - |
| `_worktrees/pr1073-cudafix` | `0bbca98ee` | af7c02b59 + 2 fix commits | **0** | 24,945,559 | **+82,812 B** (0.083 MB; inside the 0.2 MB seal gate) |

`ds4-bench` .text: main 24,577,474 -> fix 24,656,014 (+78,540 B). `ds4`: 24,637,169 -> 24,728,013
(+90,844 B). Wall: main 94 s, fix 87 s, both `-j20`.

## Why the as-is build fails, exactly

`nvcc` on `ds4_cuda.cu` (which `#include`s `ds4_deepseek41_cuda.cuh` at its end), six errors:

```
ds4_deepseek41_cuda.cuh(502): error: identifier "ds4_gpu_matmul_q8_0_bf16_tensor" is undefined
ds4_deepseek41_cuda.cuh(510): error: identifier "ds4_gpu_matmul_q8_0_bf16_tensor" is undefined
ds4_deepseek41_cuda.cuh(575): error: identifier "ds4_gpu_matmul_q8_0_bf16_tensor" is undefined
ds4_deepseek41_cuda.cuh(590): error: identifier "ds4_gpu_rms_norm_weight_bf16_tensor" is undefined
ds4_deepseek41_cuda.cuh(663): error: identifier "ds4_gpu_hc_weighted_sum_bf16_tensor" is undefined
ds4_deepseek41_cuda.cuh(664): error: identifier "ds4_gpu_rms_norm_weight_bf16_tensor" is undefined
```

Mechanism: commit `d8d1523b3` inserts the fused V4.1 entry points (`ds4_gpu_dsv41_project_pair_q8`,
`ds4_gpu_dsv41_shared_swiglu`, `ds4_gpu_dsv41_project_q`, `ds4_gpu_dsv41_norm_pair`,
`ds4_gpu_dsv41_hc_input`) into the header ABOVE the bf16 wrappers that commit `d95f8b610` appended
at its END (`ds4_gpu_rms_norm_weight_bf16_tensor`, `ds4_gpu_matmul_q8_0_bf16_tensor`,
`ds4_gpu_hc_weighted_sum_bf16_tensor`). The prototypes exist in `ds4_gpu.h`, but `ds4_cuda.cu`
never includes `ds4_gpu.h`, so the compiler has no declaration when it reaches the use. Metal
never compiles this header, which is why the branch's own tests and both Mac Studios never saw it.

With the header fixed (`577814f0d`, three `extern "C"` declarations at the top of the `.cuh`) the
compile passes and the LINK fails, five times each:

```
undefined reference to `ds4_gpu_tp_flag_fold_request'
undefined reference to `ds4_gpu_add_tensor_tp_flag'
```

Mechanism: both functions are implemented only in `ds4_metal.m`. Main guards its one use of
`ds4_gpu_add_tensor_tp_flag` (V4 FFN gate) with `#if defined(__APPLE__)`; the PR's V4.1 sites
in `ds41_attention_expand_tp` (`9f3892d2f`) and the shared-expert add in the routed path
(`867465cc6`) are unguarded. Both are `tp_world == 2` code that a one-box run never executes.
`0bbca98ee` adds the guards; the non-Apple branch of the add site calls `ds4_gpu_add_tensor`,
which the header comment on `ds4_gpu_add_tensor_tp_flag` names as its fallback.

## What the +82,812 B is

Not the fused kernels (none exist on CUDA). It is: the CUDA shims (compositions + `return 0`
stubs, ~430 header lines), the three real small kernels (`dsv41_indexer_all_kernel`,
`dsv41_hc_mean_kernel`, the two Markov draft kernels), and the 1,725 lines of `ds4.c` V4.1
plumbing (verify rows, decode queueing, DSpark control) compiled for every backend. None of it
is a GEMV.

## Suggested upstream fix (two hunks, both in the PR's own style)

1. In `ds4_deepseek41_cuda.cuh`, either move the three `*_bf16_tensor` definitions above their
   first use or add their declarations under the `#include "ds4_deepseek41_gpu.h"` line.
2. In `ds4.c`, wrap the two V4.1 TP calls as main wraps the V4 one, or give `ds4_cuda.cu` /
   `ds4_tp.c` non-Metal definitions of `ds4_gpu_tp_flag_fold_request` (no-op) and
   `ds4_gpu_add_tensor_tp_flag` (`ds4_gpu_add_tensor`).

Logs: `~/dwarfstar/.devlogs/kernelpoolspark_pr1073_build.log` (the rc=2 run),
`kernelpoolspark_builds2.log` (header-fix rc=2, link; main rc=0),
`kernelpoolspark_builds3.log` (both fixes, rc=0), on the spark.
