# Issue 1085, Bonsai-style ternary PTQ for DeepSeek V4.1 Flash: what we already measured

> ⛔ **NOT SUBMITTED.** Nothing here has been posted, commented or pushed anywhere. This file is
> the working note and the prepared comment; a human posts it if he chooses.

`antirez/ds4` issue **1085**, opened **2026-09-18T14:34Z** by **yjhsxdt-hub**, no labels, no
comments. Title: *"Ternary PTQ for DeepSeek V4.1 Flash (Bonsai-2-style, ~1.72 bpw): could make
main weights resident on 256 GB Macs and ~2.5x SSD-streaming decode on 128 GB"*.

His proposal: take the V4.1 Flash **FP16 safetensors** (explicitly not an existing quant),
Hadamard-rotate to ternary `{-1, 0, +1}` at about 1.72 bpw with imatrix calibration, and ship
ternary kernels for the routed MoE path.

His own caveat, which is the honest centre of the issue and the thing worth engaging:

> 98.2% retention is a 27B **dense** result; 755B **MoE** expert ternarization is unproven and
> router sensitivity is a real risk. A first step could be small-scale: ternarize a few expert
> layers, measure NLL/top-token on the existing quality harness before committing to full kernels.

---

## 1 · His engram figure is right, and we confirm it independently

Measured by job `1520-engram-table-geometry` on the rented RTX 5090, reading the tensor table of
our own V4.1 Flash Q2 gguf. The whole table, tensor by tensor, offsets and residual walked:

| | bytes | GB | GiB |
|---|---:|---:|---:|
| tensor table total | 365,701,254,608 | 365.7 | 340.6 |
| **`blk.N.engram_embd.weight`, two I8 tensors** | **202,758,032,400** | 202.8 | **188.8** |
| everything else | 162,943,222,208 | 162.9 | 151.8 |

**He wrote 189 GB; it is 188.8 GiB.** Same quantity, the unit label is loose. So his ceiling on
total savings stands: **55.44 % of the file is a lookup table that cannot be ternarized**, and it
is **already I8**.

His sizing arithmetic also checks out in its own units: 294 GB at 4.5 bpw scaled to 1.72 bpw is
**112 GB**, exactly as he states.

⚠ **Our file cannot test his headline.** Ours is **Q2**, his residency claim is a **Q4 to ternary**
path. Our non-engram is 151.8 GiB at roughly 2.6 bpw, which ternary would take to about **102
GiB**. That is a different and smaller claim than the one the issue makes.

## 2 · The MoE risk is sharper than "router sensitivity", and it changes the instrument

A dense model degrades gradually under quantization because the error averages over every weight
in the path. **An MoE has two failure modes and only one of them is gentle.**

1. **The chosen experts are slightly wrong.** Dense-like. NLL sees it.
2. ⛔ **A different expert gets chosen.** Each token takes **6 of 256** experts by a top-k argmax.
   A small shift in router logits flips the selected set, and the token is then computed by a
   *different subnetwork*. That is not a small error, it is a different answer, and **averaged
   over a corpus it can sit inside a perfectly good NLL.**

⇒ **NLL and top-token agreement are not sufficient on their own.** The MoE-specific measurement is
**expert-selection agreement**: for the same prompt, does the ternarized model route to the same
experts as FP16, layer by layer? It is cheap, it needs no fast kernel, and **nothing in the issue
proposes it.**

★ This is the one genuinely additive thing we have to offer the thread.

## 3 · Three measured facts from our own tree that bear on the router

**a · The model's own designers already treat the router as precision-sensitive.**
Job `110/111-router-tensor-census` read `ffn_gate_inp.weight`, the router itself: **F16 on
ds4flash, F32 on V4.1**, while the experts around it are 2-bit. **The router ships at 8 to 16
times the experts' precision.** That is a strong prior about where the fragility is, from the
people who built the model.

⚠ It also corrects a claim in our own queue: row 4a asserted those tensors were F32 on both files.
On ds4flash `ffn_gate_inp.weight` is **F16**, and a probe written to the stated dtype would have
read an F16 tensor as F32.

**b · The router is not uniform across layers, so a uniform ternarization policy is already
wrong.** `exp_probs_b.bias` is **absent at layers 0, 1 and 2 of 43** on ds4flash. And sealed cell
`121-router-bias-centred-null` found the router bias spread **is a per-layer property**, p =
**0.00100** on both files, under a mean-preserving two-sided null with a vacuity control and a
positive control on a planted effect.

⇒ his *"ternarize a few expert layers"* is the right shape, **but which layers is not
arbitrary** - the first three differ structurally from the rest.

**c · The engram table is not what the obvious reading assumes.** Two tensors, dims **(264,
384006168)** and **(264, 384016682)** - *different row counts*, neither a power of two, against a
vocab of **99,092**. No power-of-two axis and no vocabulary power, so **not hashed n-gram
addressing** in the simple sense. Evidence against, from job `1520`.

## 4 · ⛔ The type trap, and it would cost somebody a week

His evidence paragraph says *"PQ2_0 (2-bit-slot packing) now runs on mainline llama.cpp
CPU/Metal/Vulkan/CUDA"*. That is true, and **PQ2_0 is not the 1.72 bpw thing.**

| ggml type | name | bpw | state |
|---|---|---:|---|
| **142** | `PQ2_0` | **2.125** | on mainline; this is what our decode profiles measured |
| **143** | `PTQ1_0` | **1.75** | **the ternary one, a separate gguf**; in our tree its kernel body is **40 instructions of stub** |

Anyone reading that paragraph and aiming at "the ternary matmul" lands on **142**, a different
type at a different bit width. Our own queue carries job `020/070-sass-dump-ternary-143` for
exactly this reason. **And his 1.72 bpw is 1.75 in the type we hold** - small, but it is the
number the entire size argument multiplies.

For scale on why 142 matters separately: on the GB10, `mul_mat_q<142,64>` is **18.44 %** of decode
time and the whole `<142>` family is **32.58 % of decode against 59.00 % of prefill**
(`MEASURED_DECODEPROFILE_*`, and note a profile without `--cuda-graph-trace=node` cannot see the
decode at all).

## 5 · Why step 1 is cheaper than the issue makes it sound

He writes *"offline, no kernels needed"* for step 1 and *"prototype ternary kernels"* for step 2.
The ordering is right and the reason is worth saying out loud, because it is what makes this a
weekend measurement rather than a months-long kernel project:

★ **A quality measurement needs a CORRECT kernel, not a FAST one.** Dequantize ternary to F16 at
load and run the existing score harness. It will be slow, and it answers the only question step 1
asks. **Residency and streaming speed need the fast kernel; the go/no-go decision does not.**

⚠ The one place *"no kernels needed"* is not literally true: something has to read type 143, and
in our tree that body is a stub. A dequantize-on-load path is the cheapest thing that closes it.

## 6 · What we can contribute, and what we cannot

**Can:**
- **The score harness against antirez's own reference figures.** Done this week on GLM Q4:
  `avg_nll` **0.300986170**, `first_match` **90**, `avg_lcp` **9.970**, against that QA section's
  reference **0.299917952 / 90 / 9.66**.
- **CUDA measurement on a DGX Spark GB10**, where he and Bonsai are on Metal and Mac. His
  streaming claim is byte-bound, so it should transfer across backends and is worth checking on
  both.
- **The expert-selection agreement instrument** from section 2, which does not exist yet anywhere.
- The byte-census tooling that produced section 1, and a two-box job queue that now runs engine
  cells under a model lock.

**Cannot, or not yet:**
- ⛔ **We do not hold the V4.1 FP16 safetensors**, and he is explicit that the quant must come from
  those rather than from an existing quant. That is a large download and a hard precondition.
- ⛔ **Our V4.1 file is Q2**, so it cannot test the Q4-to-ternary residency headline (section 1).
- We have measured nothing about ternary retention on any MoE. **Nothing in this file is evidence
  that ternary works here**; it is evidence about the object and about how to measure it.

## 7 · What we would add to his plan before a kernel is written

1. **Measure expert-selection agreement, not only NLL.** It is the MoE failure mode, and it is
   cheap.
2. **Choose the layers deliberately.** Layers 0 to 2 differ structurally from the rest.
3. **Name 143 / `PTQ1_0` / 1.75 bpw out loud**, so nobody builds against 142.
4. **Separate the correct-kernel question from the fast-kernel question** explicitly in the plan.

---

## Body

> ★ Everything below is the prepared comment for issue 1085. Everything above is working notes and
> does not get posted. Nothing here has been submitted.

@yjhsxdt-hub - four notes from a DGX Spark GB10, where we have been measuring the PrismML ternary
work and the V4.1 Flash byte layout this week.

**Your engram figure is right, and it checks out independently.** Reading our own V4.1 Flash Q2
tensor table: total **365,701,254,608** bytes, of which `blk.N.engram_embd.weight` is two I8
tensors totalling **202,758,032,400**. That is **188.8 GiB**, so your 189 is exact once the unit
is GiB. **55.44 % of the file is un-ternarizable lookup, already at I8**, which bounds total
savings just as you say. Your 294 GB at 4.5 bpw scaled to 1.72 also lands on 112 exactly.

**One type correction, because it would cost someone a week.** `PQ2_0` is ggml type **142 at 2.125
bpw** and is the one on mainline. The ternary type is **143, `PTQ1_0`, 1.75 bpw**, a separate
gguf. Anyone aiming at "the ternary matmul" from your evidence paragraph lands on 142, which is a
different width. Worth naming 143 and 1.75 explicitly in the proposal, since the size argument
multiplies that number.

**On your MoE caveat, which I think is the right caveat: the failure mode is discontinuous, so
NLL alone will not see it.** A dense model's quantization error averages over the path. An MoE
takes 6 of 256 experts by a top-k argmax, so a small router-logit shift flips the selected set and
the token is computed by a different subnetwork. That is not graceful degradation, and averaged
over a corpus it can hide inside a good NLL. ⇒ **the cheap MoE-specific measurement is
expert-selection agreement**: same prompt, does the ternarized model route to the same experts as
FP16, layer by layer. It needs no fast kernel.

**Two measurements that say the router is the fragile part, and that it is not uniform.** In our
census `ffn_gate_inp.weight` is **F16 on the Flash Q2 file and F32 on V4.1**, while the experts
around it are 2-bit, so the model already ships its router at 8 to 16 times expert precision. And
`exp_probs_b.bias` is **absent at layers 0, 1 and 2 of 43**, with the router bias spread measuring
as a per-layer property (p = 0.00100, two-sided mean-preserving null, vacuity control). ⇒ your
*"ternarize a few expert layers"* is the right first step, but **which** layers is not arbitrary.

**And one note on cost.** A quality measurement needs a *correct* ternary read, not a fast one:
dequantize to F16 on load and run the existing score harness. Residency and streaming speed need
the kernel; the go/no-go does not. Separating those two in the plan makes step 1 much smaller than
it looks.

Happy to run the CUDA half on the Spark if it is useful. We do not hold the V4.1 FP16 safetensors,
so we cannot produce the quant, but the harness, the byte census and an expert-agreement probe are
all here.

---

# THE PLAN: how this would actually be done

> ⛔ **STUBBED, NOT FIRED.** Nothing below has been run, no box has been rented for it, and no
> JIMOTHY fleet has been raised. This is the plan part of this file, at the navigator's word:
> *"put this just in md file as plan part, the jimothy explore."*

## P0 · The precondition nothing else can start without

⛔ **We do not hold the V4.1 Flash FP16 safetensors, and the issue requires them** - the quant must
come from FP16, not from an existing quant, or the result measures compounded error rather than
ternary. Everything below assumes that download is decided separately.

    [ ] price the download: bytes, hours, and where it lands (a Keep-class artifact, not the spark's nvme)
    [ ] decide whether a SUBSET of layers can be fetched rather than the whole checkpoint
        ^ this is the single biggest cost lever in the whole plan and it is unpriced

★ **Why the subset question matters more than anything else here.** Step 1 ternarizes *a few
expert layers*. If the safetensors can be fetched per-shard, step 1 costs a few shards instead of
a full checkpoint, and the whole plan becomes affordable on a LOW box. **Nobody has checked whether
the shard index allows it.** That check is free and is the first thing to do.

## P1 · The quality measurement, which needs a CORRECT kernel and not a fast one

The go/no-go. No ternary kernel work, no residency, no speed claim.

    [ ] pick the layers deliberately, NOT arbitrarily
        layers 0, 1, 2 of 43 lack exp_probs_b.bias and differ structurally (section 3b)
        the router bias spread is a per-layer property, p = 0.00100
        ⇒ the honest sample is: two early, two middle, two late, named in a seal before running
    [ ] Hadamard-rotate + imatrix-calibrate those layers' experts to ternary, offline
    [ ] read type 143 PTQ1_0 by DEQUANTIZING TO F16 AT LOAD
        ^ the tree's 143 kernel body is 40 instructions of stub; a correct slow read closes it
    [ ] run the existing score harness, the same one that produced GLM Q4's
        avg_nll 0.300986170 / first_match 90 / avg_lcp 9.970
    [ ] ★ run the EXPERT-SELECTION AGREEMENT probe (section 2), which does not exist yet:
        same prompt, same layer, does the ternarized model route to the same 6 of 256 experts
        report agreement per layer, not pooled - a pooled number hides the flips
    [ ] a NULL arm: the same pipeline with rotation but NO ternarization, so a zero is provable

⚠ **The gate is not NLL.** A good NLL with poor expert agreement is the failure this plan exists to
catch, and it is the reading that would otherwise send someone to write kernels.

## P2 · THE JIMOTHY EXPLORE: what a range of boxes actually costs and delivers

The navigator's words: *"send jimothy on some experiments with all size boxes ... once we figure
out a bit what we need to determine cost value and speed actual numbers from a range of boxes."*

⇒ **This is a MEASUREMENT of the fleet, not of the model.** Its product is a cost-per-answer table
that every later step budgets against, and it runs on a workload we already understand rather than
on ternary.

**The tiers, as JIMOTHY's own skill states them** (`docs/VAST_AI_OPERATIONS.md`, not invented here):

| tier | cards | rate | doctrine |
|---|---|---|---|
| **LOW** | RTX 3060-class | **$0.045 to $0.052/hr** | the default, about 90 % of rentals; whole-fleet economics run here |
| **MED** | 3070 / 3080 / 2080Ti | (skill's own figure) | only when a cell needs VRAM, or halving wall-time meets a deadline |
| **HIGH** | 4090 / A100 / H100-class | (skill's own figure) | **rare, pre-registered need only, never on vibes** |

⛔ **THE MONEY RULE, verbatim from the skill:** *"A Vast instance bills every second it exists,
running OR stopped (a stopped box still bills storage). `vastai destroy instance <id> -y` is the
only $0 state."* ⇒ destroy-immediately for one-shot cells, or a warm session with a **hard idle
cap**; never a box idle with no cap.

**The explore, stubbed:**

    [ ] ONE workload, held fixed across every box, chosen because we already know its answer
        candidate: the per-tensor checksum walk (a whole-file read) plus a fixed matmul batch
        ^ a workload whose right answer is known turns every box into a verify-first-time check
    [ ] rent ONE box per tier, by EXPLICIT LIST, never a formula and never "whatever is free"
    [ ] per box, record: card, driver, VRAM total and free, cgroup CPU quota (NOT nproc),
        cgroup memory limit (NOT /proc/meminfo), disk random-read rate, sequential rate
        ⚠ nproc honours OMP_NUM_THREADS, so it reads back your own write - use nproc --all
        ⚠ the machine's RAM is not the container's; read the cgroup
    [ ] per box, the measurand: SECONDS PER ANSWER and DOLLARS PER ANSWER, both stamped
    [ ] the deliverable: one table, cost per answer by tier, with the wall time beside it
    [ ] destroy every box in the same beat it finishes; pull before destroy

★ **What the table is FOR.** Every later step asks "which tier, and how many". Right now that
question has no numbers behind it, and the one figure we do hold says the obvious answer can be
wrong: on the 5090 the expert-cache ask **25GB is already saturated** - it plans 1,928 slots and
delivers **1,127**, and 18GB delivers the same 1,127. **The ask was never the constraint.** A
bigger box is not automatically a faster answer, and this table is how we stop guessing.

⚠ **It does not run on ternary and must not.** Mixing a fleet measurement with a model measurement
gives you one number that answers neither.

## P3 · THE JIMOTHY RUN for the quantization itself, stubbed

Only if P1's gate passes. This is the part that needs real compute.

    [ ] SEAL FIRST: the arms, the layers, the measurand, the floor's source, numeric predictions,
        and what VOIDs the cell. Committed before run 1 and never edited; amendments append.
    [ ] tier from P2's table, not from intuition. Ternary PTQ is a memory-and-bandwidth job,
        so the LOW default may well hold, and P2 is what says so.
    [ ] boxes by EXPLICIT LIST with DISJOINT host-shards: one layer range per box, named per box
        ^ JIMOTHY's law: explicit LISTS, never formulas - a modulus gets re-misread
    [ ] HOT AND SMOOTH: a rented box is busy with confirmed jobs or it is a named bad state.
        HOT-IDLE is fixed on sight: fire a cell or release the box.
    [ ] each box writes its own result and its own stamp; nothing is pooled until every shard lands
    [ ] pull before destroy, every box, no exception
    [ ] the no-loss ledger: landed / superseded / left-behind-and-shown

⚠ **The scaling trap, already paid for once in this project.** A rate measured at one workload size
multiplied to another is a PREDICTION. We once scaled a capture rate by 38.1x and the real cell
came in **3.85x cheaper**, which bought boxes nobody needed. **Label any extrapolated figure
`SCALED` and re-measure at the size you will actually run.**

## P4 · The kernel, which is a different project

Only if P1 and P3 both hold.

    [ ] ternary matmul for the ROUTED path, type 143, CUDA first because that is the box we own
    [ ] ⛔ do not aim at type 142 - PQ2_0 at 2.125 bpw is a different type (section 4)
    [ ] the streaming win before the residency win: streaming is byte-bound, so it should
        transfer across backends, and it helps every 128 GB user rather than only 256 GB owners
    [ ] G1 at every step: greedy token identity, same process, same config, against the
        unquantized read. A speed number before G1 is green is not a result.
    [ ] profile with --cuda-graph-trace=node or the decode is INVISIBLE
        ^ measured: without it, 150,106 rows all read launchType=REGULAR, graphNodeId=NULL
          and every kernel after the first cudaGraphInstantiate is missing

## What would stop this, honestly

- **P0 fails on cost** and the FP16 checkpoint is not worth fetching for a subset test.
- **P1's expert agreement collapses** while NLL looks fine. That is the expected failure and it is
  the whole reason to measure it.
- **P2 says the tier that can do P3 is HIGH**, at which point the arithmetic changes and the
  pre-registered-need bar applies.
- **The engram table stays 55.44 % of the file** no matter what, so even a perfect result caps the
  total saving where the issue says it does.

## What to do first, today, for free

    [ ] check whether the V4.1 FP16 shard index allows a per-layer subset fetch   <- free, decisive
    [ ] write the expert-selection agreement probe against a model we already hold <- no rental
    [ ] name the layer sample and seal it                                          <- no rental

★ **All three cost nothing and the first one decides the shape of everything after it.**
