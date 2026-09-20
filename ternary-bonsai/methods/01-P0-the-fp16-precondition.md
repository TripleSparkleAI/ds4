# P0 - the precondition nothing else can start without

> ⛔ **WE DO NOT HOLD THE V4.1 FLASH FP16 SAFETENSORS, AND THE METHOD REQUIRES THEM.** The quant
> must come from FP16, not from an existing quant, or the result measures compounded quantization
> error rather than ternary. Every step after this one assumes that download is decided separately,
> **and it is the navigator's decision, not a lane's.**

## ★ The one free check that decides the shape of everything after it

    [ ] read the V4.1 Flash safetensors SHARD INDEX and answer: can a per-layer SUBSET be fetched?
        ^ FREE. No download, no rental. It is a JSON file of a few hundred KB.
        ^ THIS IS THE SINGLE BIGGEST COST LEVER IN THE WHOLE PLAN AND IT IS UNPRICED.
    [ ] if YES: P1 costs a few shards, lands on a LOW box, and is a weekend
    [ ] if NO:  P1 starts with a full checkpoint download and is a different project

**Why it matters more than anything else here.** P1 ternarizes *a few expert layers*. If the
safetensors can be fetched per-shard, then step 1 needs those shards and nothing else. **Nobody has
checked whether the shard index allows it.**

⚠ **A `model.safetensors.index.json` maps parameter name to shard file**, so the question is
mechanically answerable: do the parameters of one block live in a small number of shards, or are
they interleaved across many? **Both answers are possible and we do not know which this checkpoint
is.** UNKNOWN, and cheap to resolve.

⚠ **And the index file itself is small and stays TRACKED** when it lands - see
`../.gitignore`, which ignores `*.safetensors` and not `*.json` for exactly this reason.

## The pricing, if the answer is NO

    [ ] price the download: bytes, hours, and the link's actual rate, not its advertised one
    [ ] decide where it lands: a Keep-class artifact, NOT the spark's nvme
        ^ THE KEEP is /media/hologram/M200/dwarfstar on the spark, external, 3.8 TB
        ^ the storage law: what cannot be remade lives in the Keep; everything else holds a recipe
    [ ] price the STORAGE, not only the transfer
        ^ JIMOTHY's measured trap: a 120 GB disk request more than DOUBLED a $0.0994 box to
          $0.2378/hr. Storage varies 6.5x between hosts. See ../wiki/06-the-box-economics.md
    [ ] a STONE at every path the artifact leaves, per the storage law - a file that vanishes is a
        mystery, a file that leaves a marker is a pointer
    [ ] the INDEX row in ../artifacts/fp16-source/INDEX.md: name, bytes, sha256, source, date,
        box, re-obtainable yes/no

⚠ **PULL BEFORE DESTROY, always**, and verify the destination sha **before** the source is touched.

## ⚠ What is NOT a substitute, and why

| tempting shortcut | why it fails |
|---|---|
| quantize from our existing Q2 gguf | ⛔ measures Q2-then-ternary compounded error. The issue is explicit that the source must be FP16. |
| quantize from a Q4 | same class of error, smaller. Still not the claim being tested. |
| test on Bonsai 2 27B instead | it is **dense**, and the entire open question is whether ternary survives an **MoE**. A green result there is the 98.2 % figure again. |
| test on a small MoE | ⚠ this one is not obviously wrong and is worth considering. It would test the MECHANISM cheaply. **But it does not test THIS model**, and the router facts in `../wiki/05-the-router-is-the-fragile-part.md` are facts about this model. UNKNOWN whether a small MoE's router is fragile in the same way. |

★ **The last row is the only honest cheap path if P0's download is refused**, and it should be
priced rather than dismissed. It would not settle the issue's claim; it would say whether the
mechanism is worth a download.

## Secrets

⛔ Any credential for fetching weights is an **environment variable NAME** in every tracked file.
Never the value, never in a log that gets committed, never in an INDEX row.
