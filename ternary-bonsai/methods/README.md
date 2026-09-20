# METHODS - how the quant is actually produced, step by step

> ⛔ **NOTHING HERE HAS BEEN RUN.** Every box is unchecked. No box has been rented for any of it,
> no JIMOTHY fleet has been raised, and no quantised file exists anywhere.

This is the ladder, expanded from the plan half of
`../measured/ISSUE_1085_bonsai-style-ternary-ptq-for-v41-flash_what-we-measured_2026-09-20.md`.
⚠ **That file is the origin and this directory is its expansion, not a second plan.** Where they
would disagree, the source file wins and this directory is wrong and must be fixed. The reason for
the rule: two plans for one job disagree eventually, and the one a reader happens to open becomes
the one that gets built.

| step | page | what it decides | gated on |
|---|---|---|---|
| **P0** | `01-P0-the-fp16-precondition.md` | can we get the weights at all, and for how much | nothing - **this is first** |
| **P1** | `03-P1-the-quality-measurement.md` | ★ the go/no-go. does ternary hold on an MoE | P0 |
| **P2** | `../explore/00-P2-the-jimothy-box-economics-explore.md` | what a range of boxes costs and delivers | nothing - **runs in parallel with P0** |
| **P3** | `04-P3-the-quantisation-run.md` | the real quantisation at scale | P1 passing, P2's table |
| **P4** | `05-P4-the-kernel.md` | the fast ternary matmul, residency and streaming | P1 and P3 both holding |

★ **P2 does not wait on P0.** It measures the fleet on a workload we already understand, so it can
run the moment somebody wants it, and its table is what P3 budgets against. ⚠ **And it must not run
on ternary** - mixing a fleet measurement with a model measurement gives one number that answers
neither.

## The ordering principle, and it is what makes step 1 small

★ **A quality measurement needs a CORRECT kernel, not a FAST one.** Dequantize ternary to F16 at
load and run the existing score harness. It will be slow, and it answers the only question P1
asks. **Residency and streaming speed need the fast kernel; the go/no-go decision does not.**

That single separation is what turns this from a months-long kernel project into a weekend
measurement, and it is the thing most worth saying to anyone reading issue 1085.

## What to do first, today, for free

    [ ] check whether the V4.1 FP16 shard index allows a per-layer SUBSET fetch   <- free, decisive
    [ ] write the expert-selection agreement probe against a model we already hold <- no rental
    [ ] name the layer sample and seal it                                          <- no rental

★ **All three cost nothing and the first one decides the shape of everything after it.**
