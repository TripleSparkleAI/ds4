# P4 - the kernel, which is a different project

> ⚠ **NOT RUN, AND GATED THREE TIMES.** Only if **P1** and **P3** both hold. This is where the
> residency and streaming claims live, and none of them can be made before this exists.

## The boxes

    [ ] ternary matmul for the ROUTED path, type 143, CUDA first because that is the box we own
    [ ] ⛔ DO NOT AIM AT TYPE 142. PQ2_0 at 2.125 bpw is a DIFFERENT TYPE at a different width.
        ^ ../wiki/02-the-ggml-types.md - this is the trap that costs a week
    [ ] the STREAMING win before the RESIDENCY win
        ^ streaming is byte-bound, so it should transfer across backends
        ^ and it helps every 128 GB user rather than only 256 GB owners
    [ ] G1 AT EVERY STEP: greedy token identity, SAME PROCESS, same config, against the
        unquantized read. A speed number before G1 is green is not a result.
    [ ] profile with --cuda-graph-trace=node or THE DECODE IS INVISIBLE
        ^ measured: without it, 150,106 rows all read launchType=REGULAR, graphNodeId=NULL,
          and every kernel after the first cudaGraphInstantiate is missing
    [ ] ⚠ READ THE ARCH FROM THE CARD, never assume one
        ^ sm_120 on a consumer Blackwell 5090, sm_121 on the GB10
        ^ a SASS mix is PER-ARCHITECTURE, so a wrong arch answers a different question SILENTLY
        ^ the pattern to copy: `nvidia-smi --query-gpu=compute_cap`, and REFUSE if it is absent
    [ ] ⚠ and count the 143 kernel body rather than quoting our own hand-read 40
        ^ ../wiki/02-the-ggml-types.md - that figure has never been checked by any instrument
        ^ a static cuobjdump disassembly settles it, with no model and no lock
    [ ] the activation transform must match the rotation the weights were quantised in
        ^ PrismML's own loader REFUSES a mismatched calibration bias by design, for the KV path
        ^ ../wiki/01-the-method-*.md - a mismatch here produces wrong numbers, not an error

## ⚠ The one honest thing to say about G1 here

**Greedy token identity against an unquantized read is the right gate and it is not a losslessness
claim.** A ternary quant is an approximation; it will diverge, and the question is where and by how
much, not whether. ⛔ **The word "lossless" never attaches to a ternary quant** in this folder or in
anything we publish. What G1 gives you is a *first divergence point*, which is a diagnostic.

⚠ **And a token comparison is not a distribution comparison.** This estate has measured a case where
tokens matched while the verified logit distribution was damaged - a copy-task fixture with margins
far too wide for the perturbation to flip anything. ⇒ **a ternary gate needs the distribution arm
too, and its fixture must have narrow margins somewhere**, or it is insensitive by construction.

## ⚠ The two claims this step would test, stated as the issue states them

| claim | what it needs |
|---|---|
| main weights resident on **256 GB Macs** | the fast kernel, Metal, and a 256 GB Mac. **We have neither.** |
| roughly **2.5x SSD-streaming decode** on 128 GB | the fast kernel and a streaming path. Byte-bound, so a CUDA measurement on the Spark should transfer. |

★ **The streaming half is the one we could actually measure**, on a box we own, and it is the half
that helps more people. ⇒ if P4 is ever funded, **streaming first**.
