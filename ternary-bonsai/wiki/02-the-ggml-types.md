# ⛔ THE TYPE TRAP - 142 is not the ternary one

**This page exists because getting it wrong costs a week of kernel work aimed at the wrong type.**

## The two types

| ggml type | name | bpw | state |
|---|---:|---:|---|
| **142** | `PQ2_0` | **2.125** | on mainline llama.cpp; CUDA, Metal, ROCm, CPU. **This is what our decode profiles measured.** |
| **143** | `PTQ1_0` | **1.75** | **the ternary one, a separate gguf.** In this tree its kernel body is **40 instructions of stub** - ⚠ **HAND-READ, NEVER CHECKED**, see below. |

The bpw and the names are MEASURED / read from PrismML's own `MODEL-FORMATS.md`
(`SOURCE_prismml-model-formats_read-2026-09-18.md`, read 2026-09-18).

## ⛔⛔ THE "40 INSTRUCTIONS OF STUB" IS HAND-READ AND HAS NEVER BEEN CHECKED

⚠ **That figure is a claim WE made by reading the source by eye, and it is repeated all over this
estate as though it were measured.** It is not. The queued job written to check it says so in its
own header, verbatim:

> That 40 is a claim WE made, read by hand, and nothing has ever checked it. This dump checks it.
> **A warning in a brief is not a control; a count is.**

⇒ **cite it as HAND-READ, UNVERIFIED**, and do not put it in a table of measured facts without that
word. The job that would settle it is `070-sass-dump-ternary-143` in this repo's own task runner
(`experiments/task-runner/jobs/070-sass-dump-ternary-143/run.sh` on branch
`autonomous/dspark-concurrency-and-spark-queue`): a source clone, one translation unit, a static
`cuobjdump` disassembly, **no model and no lock**, so it is cheap and needs no rented box.

★ **And that job carries a discipline worth copying into every step of `../methods/`:** it reads
the **architecture from the card** rather than assuming one, because *"a SASS mix is per-architecture,
so a wrong arch answers a different question silently"* - `sm_120` on a consumer Blackwell 5090,
`sm_121` on the GB10. ⚠ It also **exits 3 with `PRECONDITION MISSING`** rather than producing a
number when `nvcc`, `cuobjdump` or `compute_cap` is absent. A job that cannot run says so instead
of reporting something.

⚠ **This folder's own state, honestly: the stub claim is repeated in `../README.md`'s summary table
and in `../artifacts/ternary-gguf/README.md`, both now marked HAND-READ.** It stays in the folder
because it is the best information we have and because it is load-bearing for the plan - it is why
P1 needs a dequantize-on-load path - but it is not a measurement.

## Why the trap is easy to fall into

Issue 1085's evidence paragraph says, correctly, that *"PQ2_0 (2-bit-slot packing) now runs on
mainline llama.cpp CPU/Metal/Vulkan/CUDA"*. **That sentence is true and it is about the wrong
type.** A reader who takes it as evidence that "the ternary matmul is on mainline" aims at 142.

⚠ **And the issue's 1.72 bpw is 1.75 in the type we hold.** Small, and it is the number the
entire size argument multiplies. PrismML's own table gives `PTQ1_0` as **1.75 bits/weight,
5.9 GB** for Bonsai 2 27B; the byte-exact file is **5,946,648,928 bytes** from the Hugging Face
API (`SOURCE_prismml-findings_read-2026-09-18.md` §1).

## Why 142 still matters, separately

On the GB10, MEASURED in `../measured/MEASURED_DECODEPROFILE_*_2026-09-19.md`:

| quantity | figure |
|---|---:|
| `mul_mat_q<142,64>` share of decode time | **18.44 %** |
| the whole `<142>` family, decode | **32.58 %** |
| the whole `<142>` family, prefill | **59.00 %** |

⚠ **A profile without `--cuda-graph-trace=node` cannot see the decode at all.** MEASURED: without
it, 150,106 rows all read `launchType=REGULAR`, `graphNodeId=NULL`, and every kernel after the
first `cudaGraphInstantiate` is missing. Any future ternary profile inherits that requirement.

## The third band, and why it is dangerous

CLAIMED by PrismML: a `Q2_0` band at **2.25 bpw / 7.6 GB**, published outside the main repo as
`Ternary-Bonsai-2-27B-Q2_0-prism-fork-required.gguf`.

⛔ **`PQ2_0` and `PTQ1_0` fail SAFELY on stock llama.cpp** because their type ids sit past
upstream's `GGML_TYPE_COUNT`. **`Q2_0` does not.** Upstream already knows type id 42 and the
`qwen35` architecture, so mainline loads that file with no warning and **outputs gibberish**.
The requirement is in the filename for exactly that reason.

## What this means for our plan

- ⛔ **Do not aim a kernel at 142.** `../methods/05-P4-the-kernel.md` says so in its own boxes.
- ⚠ **Something must read 143 before any quality number exists**, and in this tree that body is a
  stub. **A dequantize-on-load path is the cheapest thing that closes it**, and it is correct
  rather than fast, which is all step 1 needs. See `01-the-method-*.md` and
  `../methods/03-P1-the-quality-measurement.md`.
- **Name 143, `PTQ1_0` and 1.75 bpw out loud in any note to the thread**, so nobody builds
  against the wrong width.
