# THE METHOD - Hadamard rotation, ternary weights, imatrix calibration

What the method is, which parts of it we have read from a source, and which parts we are guessing
at and have marked as guesses. ⚠ **We have not read the Bonsai method paper.** Everything CLAIMED
here comes from PrismML's own repository documentation on disk under `SOURCE_*`; where those
documents are silent, this page says UNKNOWN rather than filling the gap from general knowledge of
the quantization literature.

## The shape of it, as issue 1085 states it

| step | what it does | our state |
|---|---|---|
| 1 | take the **FP16 safetensors**, explicitly not an existing quant | ⛔ we do not hold them |
| 2 | **Hadamard-rotate** into a basis where the weight distribution suits ternary | ⚠ not run |
| 3 | quantize to ternary `{-1, 0, +1}` at about 1.72 bpw, **imatrix-calibrated** | ⚠ not run |
| 4 | ship **ternary kernels for the routed MoE path** | ⚠ not run, and it is a separate project |

⚠ **1.72 is 1.75 in the type we hold** - see `02-the-ggml-types.md`.

## Why the rotation is there at all

A ternary alphabet has three values. Quantizing a weight matrix to three values works when the
weights, in the basis you quantize in, are distributed so that three levels capture most of the
variance - and fails when a few large-magnitude directions dominate, because those directions get
crushed into the same `±1` as everything else. **A random orthogonal rotation spreads the energy
across the axes**, so no single axis carries an outlier the alphabet cannot express. A **Hadamard**
matrix is the cheap way to do that: it is orthogonal, its entries are `±1`, and the transform is a
butterfly rather than a matmul.

⚠ **The consequence, and it is the part that bites:** a rotated weight matrix is only correct if
the **activations are transformed to match**. PrismML says exactly this, verbatim, of all three of
its bands:

> "All three store their weights in a rotated basis and need the activation transform that only
> this demo's binaries carry, from the PrismML fork. There is no 'works everywhere' option the way
> group-64 `Q2_0` is for the previous generation."

⇒ **the rotation is not a preprocessing step you can throw away.** It becomes part of the file
format's contract, and a reader that does not apply the matching transform produces wrong numbers
rather than an error. `02-the-ggml-types.md` records the one band where that failure is silent.

## ★ The rotation and the calibration are COUPLED, and their own loader refuses a mismatch

MEASURED by reading PrismML's own scripts (`SOURCE_prismml-findings_read-2026-09-18.md`, the KV4
section). Their separate KV-bias calibration path:

- `./scripts/make_kv_bias.sh` runs `llama-kv-mean-center` over a **calibration corpus** and writes
  `<Model>-kv-bias.gguf`; the server picks it up automatically.
- **The server exports `LLAMA_ATTN_ROT_DISABLE=1` when a bias is present, because the bias is
  calibrated with K-rotation off.**
- ⛔ **"the loader refuses a mismatched bias by design."**

★ **That is the single most useful thing in the sources for our plan, and it is about a different
tensor.** It is a worked precedent for the shape of the problem we are walking into: **a
calibration artifact is only valid for the rotation state it was calibrated under**, and PrismML
thought it important enough to enforce in the loader rather than document in a README. ⇒ **an
imatrix built under one rotation is not an imatrix for another**, and our plan must treat the
rotation choice and the imatrix as one sealed pair rather than two independent settings.

⚠ That is an inference from a KV-cache mechanism to a weight-quantization mechanism. It is a
strong prior, not a measurement of the weight path. UNKNOWN whether their weight imatrix carries
the same coupling.

## What imatrix calibration is, and what it needs

An importance matrix records, over a **calibration corpus**, how much each weight actually
matters to the output, so the quantizer can spend its limited precision where it changes the
answer. Two things follow, and both are plan items rather than facts:

- ⚠ **The corpus choice is a real parameter**, not a formality. See
  `07-calibration-is-where-ptq-touches-training.md`.
- ⚠ **An imatrix is per-model and per-rotation**, so it is an artifact with provenance, which is
  why `../artifacts/imatrix/` exists with a tracked INDEX.

## What is genuinely unknown

- **UNKNOWN** which Hadamard construction and which block size PrismML uses.
- **UNKNOWN** whether the router is left in high precision in their pipeline. Our own measurements
  say the model ships its router at 8 to 16 times expert precision
  (`05-the-router-is-the-fragile-part.md`), so whether a ternary pipeline preserves that is the
  first question to ask of any implementation.
- **UNKNOWN** how the 98.2 % retention figure was measured, on what corpus, against what baseline.
  ⛔ It is a **27B dense** result and must never be carried to a 755B MoE.
- **UNKNOWN** whether ternary retention on any MoE has ever been measured by anybody.
