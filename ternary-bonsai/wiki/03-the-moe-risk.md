# ★ THE MoE RISK - the failure mode is discontinuous, so NLL alone will not see it

**This is the one genuinely additive thing this folder has to offer the thread.** The issue names
*"router sensitivity is a real risk"*, which is right. This page says why it is a sharper risk
than that phrase suggests, and what instrument follows from it.

## Two failure modes, and only one of them is gentle

A **dense** model degrades gradually under quantization because the error averages over every
weight in the path. That is why a 27B dense retention figure of 98.2 % is believable and why it
is also **not transferable**. An MoE has two modes:

| mode | what happens | does NLL see it |
|---|---|---|
| 1 · the chosen experts are slightly wrong | dense-like error, averaged over the path | **yes** |
| 2 · ⛔ **a different expert gets chosen** | the token is computed by a **different subnetwork** | **often not** |

Mode 2 is the one that matters. Each token takes **6 of 256** experts by a top-k argmax. A small
shift in router logits flips the selected set, and the result is not a small error, it is a
different answer. **Averaged over a corpus, a scattering of different answers can sit inside a
perfectly good NLL**, because the corpus mean is dominated by the tokens whose routing did not
flip.

⇒ **NLL and top-token agreement are not sufficient on their own.** They are necessary and they
are the wrong gate.

## The instrument that follows: EXPERT-SELECTION AGREEMENT

> Same prompt. Same layer. Does the ternarized model route to the same 6 of 256 experts as FP16?

Four properties, and all four are why this is worth building first:

- **It is cheap.** It needs no fast kernel, only a correct read of the quantised weights.
- **It is direct.** It measures the failure mode rather than a quantity the failure mode hides in.
- **It is per-layer.** Which matters, because the router is not uniform across layers
  (`05-the-router-is-the-fragile-part.md`).
- ⚠ **It does not exist.** Nothing in issue 1085 proposes it, and we have not found it in the
  quantization literature we have read. UNKNOWN whether somebody has built one elsewhere.

## ⚠ How to report it, because a pooled number hides the thing

**Report agreement PER LAYER, never pooled.** A pooled figure averages the layers that flip into
the layers that do not, which is the same mistake as reading NLL one level up. Suggested shape,
and it is a suggestion because no run has produced one:

| layer | tokens | exact-set agreement | top-1 expert agreement | Jaccard mean |
|---:|---:|---:|---:|---:|
| 0 | - | - | - | - |
| 1 | - | - | - | - |
| ... | - | - | - | - |

Three columns rather than one, because they fail differently: **exact-set** is the strict reading,
**top-1** isolates whether the single highest-weighted expert survived, and **Jaccard** says
whether a disagreement was one expert out of six or five.

⚠ **And it owes a NULL ARM.** The same pipeline with the Hadamard rotation applied but **no
ternarization** must score at or near perfect agreement, or the instrument is measuring its own
plumbing rather than the quantization. This estate has paid for that lesson: a null that shares a
model with the thing it tests cannot fail.

## What this does NOT say

⛔ **Nothing here is evidence that ternary breaks an MoE, and nothing here is evidence that it
works.** We have measured nothing about ternary retention on any MoE. This page is about the
object and about how to measure it. The measurement is `../methods/03-P1-the-quality-measurement.md`
and it has not been run.
