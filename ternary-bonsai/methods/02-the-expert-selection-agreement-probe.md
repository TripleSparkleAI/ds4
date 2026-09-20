# ★ THE EXPERT-SELECTION AGREEMENT PROBE - the instrument that does not exist yet

> ⚠ **NOT BUILT.** This page is the specification. **It needs no rented box and no FP16
> checkpoint** - it can be written and validated against a model we already hold, which makes it
> the cheapest genuinely useful thing on this whole board.

The theory is `../wiki/03-the-moe-risk.md`: an MoE's dangerous failure is a **different expert
being chosen**, which computes the token with a different subnetwork and can hide inside a good
NLL. This is the instrument that sees it.

## What it measures

> Same prompt. Same layer. Does the quantised model route to the same 6 of 256 experts as the
> reference?

## The boxes

    [ ] hook the router's top-k selection per layer per token, reference model
        ^ the tensor is ffn_gate_inp.weight; the selection is a top-k argmax over 256
        ^ cite the engine seam by FUNCTION NAME in the code, never by line number
    [ ] the same hook on the quantised model, same process where possible
    [ ] a fixed prompt set, named and sealed before the first run
    [ ] emit PER LAYER, never pooled:
          exact-set agreement    - all 6 identical
          top-1 agreement        - the single highest-weighted expert survived
          Jaccard mean           - was a disagreement one expert out of six, or five
    [ ] ⚠ a NULL ARM: rotation applied, NO ternarization. It must score at or near perfect.
    [ ] ⚠ a VACUITY CONTROL: a deliberately damaged selection must make the probe RED.
    [ ] a --selftest, wired into the repo's own gate, per the tool-building standard

## ⚠ Why the null arm and the vacuity control are not optional here

This estate's tenth measurement law, earned the hard way: **a null may not share a model with the
thing it tests, and a zero carries its power or it carries nothing.**

- **The null arm** separates "ternary changed the routing" from "our rotation plumbing changed the
  routing". Without it, a disagreement is unattributable.
- **The vacuity control** proves the probe can report disagreement at all. A probe that reads 100 %
  agreement because it is comparing a tensor with itself looks exactly like a great result.
- ⚠ **And a zero owes its power.** If the probe reports zero flips, the report must say how many
  flips it could have detected at that sample size. A clean zero at n too small is
  `UNDERPOWERED_ZERO`, not `clean`.

## ⚠ Two traps specific to this probe

1. **Same-process comparison, or the comparison is worthless.** Floating-point non-determinism
   across two model loads flips near-tie argmaxes on its own, and a router top-k is exactly a
   near-tie-prone argmax. **A cross-process A/B here can only disprove gross bugs; it cannot
   measure agreement.** This is the same law as this project's losslessness gate: same process,
   same config, pinned threads and seed.
   ★ **And note the sharpening: for a router, this is not merely good hygiene.** A top-k over 256
   logits has many near-ties by construction, so cross-process noise does not merely add scatter,
   it produces routing flips that look exactly like the signal.
2. ⚠ **A thread-count change changes BLAS reduction order**, which changes the logits, which can
   change the argmax. So the thread count is part of the seal, and it must not change mid-run.
   Read it from `/proc/<pid>/status Threads:` after a warm-up, never from `nproc` - `nproc` honours
   `OMP_NUM_THREADS` and reads back your own write.

## What it is worth even if ternary is never tried

★ **This probe is a general MoE-quantization instrument**, and we have not found one in the
literature we have read. It would apply to any quantization of any MoE, ours or anybody's. ⚠
UNKNOWN whether somebody has built one elsewhere and we have not found it; that claim is about our
search, not about the field.
