# TRAINING - the material, and the fence around it

> ⛔⛔ **PTQ IS NOT TRAINING.** Ternary post-training quantization takes a **finished** model and
> changes how its weights are stored. **No gradient step is taken, no optimizer runs, nothing is
> learned.** This directory is **not** a plan to train anything, and a reader who takes it for one
> has been misled by the word. This banner is the fence.

## Then why is it here

Two reasons, and both are real:

1. **The navigator asked for the training material to be in this folder**, so it is here rather than
   pointed at from somewhere else.
2. ★ **Two things that genuinely touch PTQ live in training territory**: the **calibration corpus**
   an imatrix is built from, and the **precision policy** the model's own designers chose per tensor
   class. The full argument is `../wiki/07-calibration-is-where-ptq-touches-training.md`.

## What we brought in, and why each one

Copied from `WIKI/training/` on branch `autonomous/dspark-concurrency-and-spark-queue`, into
`from-deepseek-training-wiki/`, unmodified.

| page | why it bears on ternary PTQ |
|---|---|
| `from-deepseek-training-wiki/01-pretraining-corpus.md` | ★ **what the model was trained on is the best available prior on what a calibration corpus should look like** - and knowing the data mix is how you avoid calibrating on your own test set |
| `from-deepseek-training-wiki/06-v4-flash-training-notes.md` | what is known about **this specific checkpoint**, as opposed to the family |
| `from-deepseek-training-wiki/10-fp8-mixed-precision-training.md` | ★ **the designers made a precision decision per tensor class, at scale, under pressure. What they kept HIGH is the first list to look at when deciding what ternary must not touch.** |
| `from-deepseek-training-wiki/17-deepseek-v2-mla-moe.md` | the MoE and router architecture the top-k argmax lives in - the mechanism behind `../wiki/03-the-moe-risk.md` |
| `from-deepseek-training-wiki/20-tokenizer-bpe-vocab-evolution.md` | vocab **99,092**, the denominator the engram dims were checked against in `../wiki/04-the-byte-layout-and-the-engram-ceiling.md` |
| `from-deepseek-training-wiki/27-aux-loss-free-balancing-training.md` | the router bias is a **trained** balancing device, which is why `exp_probs_b.bias` being absent at layers 0-2 is structural rather than incidental |

**Six pages of forty-two.**

## ⛔ What we deliberately LEFT, and why

The other **36 markdown pages**, the **17 arXiv PDFs**, the repo excerpts, the manifest and the
fetch script stay where they live. They are about **how DeepSeek trained the model** - DualPipe,
GRPO, RL infrastructure, distillation, the data pipeline, cluster config, the lineage - and **none
of it is reachable from a post-training quantization plan.**

★ **The reason this is a refusal rather than laziness:** copying a live wiki into a folder about a
different subject creates a second copy that then disagrees with the first, and a reader cannot tell
which is current. **One copy, where it lives; a curated few here, with a stated reason each.**

Reach the rest where they are:

```
git show autonomous/dspark-concurrency-and-spark-queue:WIKI/training/00-index.md
git show autonomous/dspark-concurrency-and-spark-queue:WIKI/training/<page>.md
```

## ⚠ The copies are FROZEN and the originals are LIVE

The six pages here are a snapshot taken **2026-09-20**. The originals keep being edited. ⇒ **for
anything load-bearing, read the original**, and treat these as the reason-annotated reading list
they are. **If they ever disagree, the original is right and this copy is stale.**

## The calibration plan

There is no separate calibration-plan file, deliberately. **The calibration boxes live in the step
that runs them**, `../methods/03-P1-the-quality-measurement.md`, so that a plan and its
preconditions cannot drift apart into two documents. The two boxes that matter most:

    [ ] the calibration corpus must be DISJOINT from the evaluation corpus, said in the seal
    [ ] the imatrix is VALID ONLY FOR THE ROTATION IT WAS CALIBRATED UNDER
        ^ PrismML's own loader REFUSES a mismatched calibration bias by design, for the KV path
        ^ ../wiki/01-the-method-hadamard-rotation-and-imatrix-calibration.md
