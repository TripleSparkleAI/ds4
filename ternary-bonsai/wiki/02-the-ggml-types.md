# ⛔ THE TYPE TRAP - 142 is not the ternary one

**This page exists because getting it wrong costs a week of kernel work aimed at the wrong type.**

## The two types

| ggml type | name | bpw | state |
|---|---:|---:|---|
| **142** | `PQ2_0` | **2.125** | on mainline llama.cpp; CUDA, Metal, ROCm, CPU. **This is what our decode profiles measured.** |
| **143** | `PTQ1_0` | **1.75** | **the ternary one, a separate gguf.** In this tree its kernel body is **40 instructions of stub.** |

MEASURED / read from the tree and from PrismML's own `MODEL-FORMATS.md`
(`SOURCE_prismml-model-formats_read-2026-09-18.md`, read 2026-09-18).

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
