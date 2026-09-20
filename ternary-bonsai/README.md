# TERNARY BONSAI - ternary PTQ for DeepSeek V4.1 Flash

> ⛔⛔ **WE DO NOT HOLD THE V4.1 FLASH FP16 SAFETENSORS, AND THE METHOD REQUIRES THEM.**
> The quant must come from FP16, not from an existing quant, or the result measures compounded
> error rather than ternary. **Nothing in this folder has been run.** Everything under
> `methods/`, `training/` and `explore/` is an unchecked box waiting on that precondition or on
> the one before it. Everything under `measured/` is a real measurement of a real file, with its
> job number and its box.
>
> ★ **And the biggest unpriced lever in the whole plan is free to check:** does the V4.1 FP16
> shard index permit a **per-layer SUBSET fetch**? If it does, step 1 costs a few shards and
> runs on a LOW box. If it does not, step 1 starts with a full checkpoint download. **Nobody has
> looked.** That check is `methods/01-P0-the-fp16-precondition.md`, box one.

This folder is the home of everything we know and everything we plan about **Bonsai-2-style
ternary post-training quantization of DeepSeek V4.1 Flash**, proposed as `antirez/ds4` issue
**1085** by **yjhsxdt-hub** on **2026-09-18T14:34Z**.

His proposal: take the V4.1 Flash FP16 safetensors, Hadamard-rotate to ternary `{-1, 0, +1}` at
about 1.72 bpw with imatrix calibration, and ship ternary kernels for the routed MoE path. His
claims: main weights resident on 256 GB Macs, and roughly 2.5x SSD-streaming decode on 128 GB.

His own caveat is the honest centre of it, and this folder is built around engaging it rather
than around agreeing with it:

> 98.2% retention is a 27B **dense** result; 755B **MoE** expert ternarization is unproven and
> router sensitivity is a real risk.

---

## ⚠ PTQ IS NOT TRAINING

Read this before `training/`. **Ternary PTQ is post-training quantization: the model is already
trained and no gradient step is taken.** `training/` exists because two things that touch PTQ
live in training territory - the **calibration corpus** an imatrix is built from, and the
**precision policy** the model's own designers chose - and because the navigator asked for the
training material to be here. It is **not** a plan to train anything. A reader who takes this
folder for a training plan has been misled by the folder name, and this paragraph is the fence.

## What is MEASURED, what is PLANNED, what is BLOCKED

Three marks, used everywhere in this folder and never mixed:

| mark | meaning | what it must carry |
|---|---|---|
| ✅ | **measured** | a number, a date, a box, and a provenance path |
| ⚠ | **planned** | an unchecked box and the precondition it waits on |
| ⛔ | **blocked** | what blocks it, and who can unblock it |

**No sentence in this folder may read as a result when it is a plan.** That is the discipline of
the folder, and it is the only thing that makes the measured half worth reading.

## Where things are

| path | what is in it | state |
|---|---|---|
| `wiki/` | the knowledge: the method, the ggml types, the MoE risk, the byte layout, the router, the box economics, calibration | mostly ✅, harvested and cited |
| `methods/` | the ladder P0 to P4, how the quant is actually produced, step by step | ⚠ all boxes unchecked |
| `training/` | the calibration and precision material, and the fence above | ⚠ plan, with ✅ source pages copied in |
| `explore/` | the JIMOTHY box-economics run: what a range of boxes costs and delivers | ⚠ not fired, no box rented |
| `measured/` | what we have actually measured, copied in whole with its provenance | ✅ |
| `artifacts/` | the local working artifacts, one named folder per kind, large files gitignored | empty, skeleton tracked |
| `HF_PUBLICATION_PLAN.md` | the stubbed plan for hosting the final versions on Hugging Face | ⛔ nothing published, licence undecided |

## The measured facts, in one place

Every one of these is read from a real file on a real box. **The provenance path is the
citation; this table is a finding aid, not the source.** Full statements with their arithmetic
are in `measured/ISSUE_1085_bonsai-style-ternary-ptq-for-v41-flash_what-we-measured_2026-09-20.md`.

| fact | figure | where it was measured |
|---|---|---|
| V4.1 Flash Q2 tensor table, total | **365,701,254,608 bytes** | job `1520-engram-table-geometry`, rented RTX 5090 |
| `blk.N.engram_embd.weight`, two I8 tensors | **202,758,032,400 bytes = 188.8 GiB = 55.44 %** | same job |
| everything else | 162,943,222,208 bytes = 151.8 GiB | same job |
| engram tensor dims | **(264, 384006168)** and **(264, 384016682)** | same job |
| vocab | **99,092** | same job |
| ggml type 142 `PQ2_0` | **2.125 bpw**, on mainline | `wiki/02-the-ggml-types.md` |
| ggml type 143 `PTQ1_0` | **1.75 bpw**, separate gguf. ⚠ **40 instructions of stub - HAND-READ, NEVER CHECKED** | `wiki/02-the-ggml-types.md` |
| `ffn_gate_inp.weight` dtype | **F16 on ds4flash, F32 on V4.1** | jobs `110/111-router-tensor-census` |
| `exp_probs_b.bias` absent at | **layers 0, 1, 2 of 43** | same jobs |
| router bias spread is per-layer | **p = 0.00100**, sealed | cell `121-router-bias-centred-null` |
| `mul_mat_q<142,64>` share of decode | **18.44 %** | `measured/MEASURED_DECODEPROFILE_*_2026-09-19.md` |
| the `<142>` family | **32.58 % of decode against 59.00 % of prefill** | same |
| expert-cache ask on the 5090 | **25GB plans 1,928, delivers 1,127; 18GB delivers the same 1,127** | `measured/` and the engine cells |

★ **His "189 GB" is 188.8 GiB and is exact once the unit is right.** ⚠ **And his 1.72 bpw is
1.75 in the type we hold**, which matters because the entire size argument multiplies that
number.

## The two things worth saying to the thread that nobody has said

1. ★ **Expert-selection agreement, not only NLL.** An MoE has two failure modes and only one is
   gentle. The chosen experts being slightly wrong is dense-like and NLL sees it. **A different
   expert being chosen is a different subnetwork computing the token**, and averaged over a
   corpus that can hide inside a perfectly good NLL. The cheap MoE-specific measurement is: same
   prompt, does the ternarized model route to the same 6 of 256 experts as FP16, layer by layer.
   **It needs no fast kernel and it does not exist anywhere yet.** `wiki/03-the-moe-risk.md`.
2. ⚠ **Naming type 143 out loud.** The issue's evidence paragraph says `PQ2_0` runs on mainline,
   which is true and is **not the 1.72 bpw thing**. Anyone aiming at "the ternary matmul" from
   that paragraph lands on **142**, a different type at a different width.
   `wiki/02-the-ggml-types.md`.

## Laws that bind this folder

- **Every number carries its stamps** - load, GPU, memory, driver, context, checkpoint, and the
  box's clock (the spark is UTC+10, the rented 5090 is UTC+0000).
- **A claim of "faster" or "retains N %" exists only after a measurement in `measured/`.**
- **Never "lossless"** about a ternary quant. It is an approximation; report divergence.
- **Code is cited by function name, never by line number.** A line number rots when the file grows.
- **Stamp a figure or derive it.** A figure written in the present tense is a claim about today.
- **Sources on disk, nothing quoted from memory.** `wiki/SOURCE_*` files carry their URL and the
  date they were read.
- **Nothing tracked carries a secret or a Hugging Face token.** Environment variable NAME only.
- **Nothing is submitted anywhere.** No issue comment, no pull request, no HF upload. This estate
  prepares and stops; a human posts.

## Provenance of this folder

Built 2026-09-20 by lane TERNARYBRANCH on branch `triple-ternary-bonsai-style-quant`, branched
from `origin/main` at `8db1d1d15` so it carries antirez's engine plus this one folder and none of
the research tree. The source material was gathered from branch
`autonomous/dspark-concurrency-and-spark-queue`; `training/README.md` and `wiki/00-INDEX.md` say
which files were brought in and which were deliberately left where they are.
