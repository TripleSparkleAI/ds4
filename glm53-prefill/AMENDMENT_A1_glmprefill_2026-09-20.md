# GLMPREFILL AMENDMENT A1 - V2's premise is false, checked BEFORE any run

Written **2026-09-20T09:12Z**, after the seal
(`PREREG_glmprefill_0e9cc2d-cuda-differential_2026-09-20.md`, sha256
`c140cfba844fc7200857c47b440c60bd9ffbb8ae2f101ca7bced836e68806a4b`, committed at `ec8085931`) and
**BEFORE cell 3355 had started: 3370 was still in `running/` with 1 h 46 m of CPU on its own python
and the model lock was free.** No prefill number exists yet on either arm.

## WHY THIS IS A SIDECAR AND NOT AN EDIT

The seal's own header says amendments append as A1, A2, ... Appending to that file would change its
sha256, and **the queued cell refuses unless the shipped `PREREG.md` hashes to exactly the value in
its `SEAL.txt`** - which is the mechanism that makes the seal binding on a box holding no git log of
ours. So the amendment lands beside the seal rather than inside it, and the seal file stays byte-for-
byte the thing the cell will hash. This is the estate's own sidecar pattern: a correction beside a
sealed stream, never a write into it.

## WHAT WAS READ

Seal V2 says:

> A counted run whose log contains `aligned artifact` is VOID: the first load of this file builds
> CUDA derived artifacts on disk and a run that pays that cost is a different machine.

**The "on disk" and "first load" halves are both false.** Read at `0e9cc2d43`, and the same code is
at `9d9e1296d` because this commit touches neither file:

- `ds4.c` calls `ds4_gpu_build_derived_artifacts(...)` from the ENGINE LOAD path, inside
  `#if !defined(__APPLE__) && !defined(DS4_ROCM_BUILD)` and under
  `backend == DS4_BACKEND_CUDA && !load_slice && !tp_shard && !e->ssd_streaming`. **Our
  configuration is resident CUDA, so the condition holds on every single load.**
- In `ds4_cuda.cu` the function returns early on `if (!g_derived_ranges.empty())`, and
  `g_derived_ranges` is a `static std::vector<cuda_derived_range>`. **That cache is PROCESS-LOCAL.**
- `cuda/mmq/ds4_repack.cu` is byte-identical at the two commits (`git diff --name-only 9d9e1296d
  0e9cc2d43 -- cuda/` is empty) and contains **zero** `O_WRONLY`, `O_CREAT`, `fopen`, `fwrite` or
  `pwrite` occurrences. **The builder writes no artifact file anywhere.**

=> the build is **per process, in memory, on every resident CUDA load, with no on-disk cache**. The
sentence V2 was written from is the GLM HOWTO's (`glm53/HOWTO_fire-the-glm-rounds.md`: *"The first
`./ds4` load of this file builds the CUDA aligned artifacts on disk ...; it is slow once"*), and that
sentence does not survive a read of the source.

## THE CONSEQUENCE, STATED BEFORE ANY NUMBER

**V2 as sealed would void all 24 runs**, because every one of them will carry the phrase. It must
not, and the reason is structural rather than a preference: in `ds4_bench.c` the engine and the
session are created before the frontier loop and `prefill_t0` is taken **inside** it, so the artifact
build is **outside the measured window** and is paid identically by both arms on every run.

**V2 IS THEREFORE NARROWED, IN ADVANCE:** a counted run is voided by `aligned artifact` only if the
phrase arrives with a build cost that differs materially between the arms, which the run log's own
wall time makes checkable. Otherwise the occurrence is COUNTED AND REPORTED and the run stands.
**Cycle 0 remains discarded whole regardless**, which is the other thing V2 was protecting.

⚠ **This narrowing is a criterion change and it is named as one.** It is legitimate here only because
no measurement exists to be flattered by it: the direction is fixed before the first t/s, and it is
recorded rather than applied silently. **What it does NOT do is un-say V2.** The sealed rule stands
as written; this file records that its premise was checked and found false before the cell ran, and
the result doc will cite this amendment by name wherever it relies on it.

## WHAT ELSE THE SAME READ SETTLES

- **The artifact build is another backend asymmetry, and it is not the one under test.** It is fenced
  out on Apple by `#if !defined(__APPLE__)`, so the M5 in issue 1029 never pays it at all. It is
  outside the prefill window on both of our arms, so it cannot be part of our differential either.
  Recorded so nobody later reads it as a candidate cause.
- **It is also fenced out under `--ssd-streaming`**, which is one more reason this cell runs resident
  rather than streamed: a streamed arm would differ from the issue's resident setup in the load path
  as well as in the attention path.
