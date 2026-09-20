# P1 - the quality measurement, the go/no-go

> ⚠ **NOT RUN.** Gated on P0. No ternary kernel work here, no residency claim, no speed number.

## The gate is NOT NLL

⚠ **A good NLL with poor expert agreement is the failure this step exists to catch**, and it is the
reading that would otherwise send somebody to write kernels. `../wiki/03-the-moe-risk.md`.

## The boxes

    [ ] SEAL FIRST: the arms, the layers, the measurand, the floor's source, the numeric
        predictions, and what VOIDs the cell. Committed before run 1, never edited; amendments
        append.
    [ ] pick the layers DELIBERATELY, not arbitrarily
        ^ layers 0, 1, 2 of 43 lack exp_probs_b.bias and differ structurally
        ^ the router bias spread is a per-layer property, p = 0.00100, sealed cell 121
        => the honest sample is TWO EARLY, TWO MIDDLE, TWO LATE, named in the seal before running
        => a sample that happens to include 0-2 measures something different from one that avoids
           them, and a sample that avoids them will look better for a reason unrelated to ternary
    [ ] Hadamard-rotate + imatrix-calibrate those layers' experts to ternary, offline
    [ ] ⚠ the calibration corpus must be DISJOINT from the evaluation corpus, and the seal says so
        ^ see ../wiki/07-calibration-is-where-ptq-touches-training.md - this estate has already
          been caught reading a train/test overlap as a result
    [ ] read type 143 PTQ1_0 by DEQUANTIZING TO F16 AT LOAD
        ^ the tree's 143 kernel body is 40 instructions of stub; a correct SLOW read closes it
        ^ correctness only. A fast kernel is P4 and is not needed for any number here.
    [ ] run the existing score harness - the same one that produced GLM Q4's
        avg_nll 0.300986170 / first_match 90 / avg_lcp 9.970
        against that QA section's reference 0.299917952 / 90 / 9.66
    [ ] ★ run the EXPERT-SELECTION AGREEMENT probe (methods/02), per layer, never pooled
    [ ] a NULL arm: the same pipeline with rotation but NO ternarization, so a zero is provable
    [ ] report divergence honestly. NEVER the word "lossless" about a ternary quant.

## What a pass and a fail look like, written down before the run

★ **Writing this before the run is the point.** A criterion rewritten after seeing the data is not
a criterion.

| outcome | reading |
|---|---|
| NLL close AND expert agreement high per layer | ⇒ the mechanism survives on this MoE. P3 is worth funding. |
| NLL close AND expert agreement collapses on some layers | ⇒ **the expected failure.** The layers that flip are the finding, and a per-layer precision policy is the next question rather than a kernel. |
| NLL close AND agreement collapses everywhere | ⇒ ternary does not survive this router. Report it; it is a real result and it is useful to the thread. |
| NLL bad | ⇒ nothing subtle happened. Check the rotation and the imatrix pairing first (`../wiki/01-the-method-*.md`). |
| the NULL arm also diverges | ⛔ the instrument is broken. No conclusion about ternary is available. Fix and re-run. |

⚠ **There is no row here for "it is faster".** P1 produces no speed number and any speed number
taken from a dequantize-on-load path would be meaningless.

## The stamps every number here carries

Load, GPU utilization, free memory, driver version, context length, checkpoint, quant, thread
count, and the box's clock. **The spark is UTC+10; the rented 5090 is UTC+0000.** A figure without
its stamps is not comparable to anything, including to itself on another day.
