# ⚠ CALIBRATION - the one place PTQ touches training, and the fence

## The fence first

⛔ **TERNARY PTQ IS NOT TRAINING.** Post-training quantization takes a finished model and changes
how its weights are stored. No gradient step is taken, no optimizer runs, no data is learned from.
**A reader who takes this folder's `training/` directory for a training plan has been misled**, and
this page and `../training/README.md` are the two places that say so.

## Where they do touch, and it is exactly two places

### 1 · The calibration corpus

An imatrix is built by running a **corpus** through the model and recording which weights move the
output. That corpus is a real parameter with real consequences:

- A corpus unlike the deployment distribution spends precision in the wrong places.
- ⚠ A corpus that **overlaps the evaluation set** makes the retention figure a measurement of that
  overlap. This estate has already been caught by exactly this shape: a champion's gain measured as
  *"a property of a lexicon whose reference-token reachability exceeds about 0.966"*, and ours
  reached 0.9841 **only because it was the training half of the same corpus.** A lucky regime read
  as a result.
- ⛔ ⇒ **the calibration corpus and the evaluation corpus must be disjoint, and the seal must say
  so before the run.** `../methods/03-P1-the-quality-measurement.md`.

★ **This is why `../training/from-deepseek-training-wiki/01-pretraining-corpus.md` is in this folder.** Not
because we are going to train, but because **what the model was trained on is the best available
prior on what a calibration corpus should look like**, and because knowing the data mix is how you
avoid accidentally calibrating on your own test set.

### 2 · The precision policy the designers already chose

The model ships its router at 8 to 16 times its experts' precision
(`05-the-router-is-the-fragile-part.md`). That is a **training-time and architecture-time
decision**, and it is evidence about where the model is fragile that no amount of post-training
measurement gives you for free.

★ **This is why `../training/from-deepseek-training-wiki/10-fp8-mixed-precision-training.md` is in this
folder.** The people who trained this model made a precision decision per tensor class, under
pressure, at scale. **What they kept high is the first list to look at when deciding what ternary
must not touch.**

## The other three pages brought in, and why

| page | why it bears on PTQ |
|---|---|
| `../training/from-deepseek-training-wiki/17-deepseek-v2-mla-moe.md` | the MoE and router architecture the top-k argmax lives in - the mechanism behind `03-the-moe-risk.md` |
| `../training/from-deepseek-training-wiki/20-tokenizer-bpe-vocab-evolution.md` | vocab **99,092**, the denominator the engram dims were checked against in `04-the-byte-layout-*.md` |
| `../training/from-deepseek-training-wiki/27-aux-loss-free-balancing-training.md` | the router bias is a **trained** balancing device, which is why `exp_probs_b.bias` being absent at layers 0-2 is structural rather than incidental |
| `../training/from-deepseek-training-wiki/06-v4-flash-training-notes.md` | what is known about this specific checkpoint, as opposed to the family |

## What is NOT in this folder and why

⛔ The other **36 pages** of the DeepSeek training wiki, its 17 arXiv PDFs and its repo excerpts
stay where they are. They are about how DeepSeek trained the model - DualPipe, GRPO, RL
infrastructure, distillation, the data pipeline - and **none of it is reachable from a PTQ plan.**
Copying them here would put a second copy of a live wiki in a folder about something else, and the
two copies would then disagree.

Reach them where they live:

```
git show autonomous/dspark-concurrency-and-spark-queue:WIKI/training/00-index.md
git show autonomous/dspark-concurrency-and-spark-queue:WIKI/training/<page>.md
```
