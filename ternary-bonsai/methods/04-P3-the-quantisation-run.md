# P3 - the quantisation run at scale, on a JIMOTHY fleet

> ⚠ **NOT RUN, AND GATED TWICE.** It runs only if **P1's gate passes** and only with **P2's cost
> table** in hand. No box has been rented. This is the part that needs real compute.

## The boxes

    [ ] SEAL FIRST: the arms, the layers, the measurand, the floor's source, numeric predictions,
        and what VOIDs the cell. Committed before run 1 and never edited; amendments append.
    [ ] tier from P2's TABLE, not from intuition
        ^ ternary PTQ is a memory-and-bandwidth job, so the LOW default may well hold
        ^ P2 is what says so; without it this line is a guess
    [ ] ⚠ price the DISK, not only the card
        ^ a 120 GB disk request more than doubled a $0.0994 box to $0.2378/hr (measured, JIMOTHY)
        ^ this plan wants large disks, so storage is the likely cost driver here
    [ ] boxes by EXPLICIT LIST with DISJOINT host-shards: one layer range per box, named per box
        ^ JIMOTHY's law: explicit LISTS, never formulas - a modulus gets re-misread
    [ ] HOT AND SMOOTH: a rented box is busy with confirmed jobs or it is a named bad state
        ^ HOT-IDLE is fixed on sight: fire a cell or release the box
    [ ] each box writes its OWN result and its OWN stamp; nothing is pooled until every shard lands
    [ ] PULL BEFORE DESTROY, every box, no exception
    [ ] `vastai destroy instance <id> -y` in the same beat a box finishes - it is the only $0 state
    [ ] the no-loss ledger: landed / superseded / left-behind-and-shown
    [ ] every output artifact gets its INDEX row in ../artifacts/ before its box is destroyed
        ^ name, exact bytes, sha256, what it came from, the date, the box, re-obtainable yes/no
        ^ the artifact is gitignored, so the row IS the record. No row, no artifact.

## ⚠ The scaling trap, already paid for once in this project

A rate measured at one workload size and multiplied to another is a **PREDICTION**. This project
once scaled a capture rate by **38.1x** and the real cell came in **3.85x cheaper**, which bought
boxes nobody needed. **Label any extrapolated figure `SCALED` and re-measure at the size you will
actually run.**

★ And note the direction: that error was **pessimistic**. Being wrong in the safe direction is not
being right, and it is harder to catch because nothing breaks.

## ⚠ The determinism question, which decides whether shards are comparable

If different boxes quantise different layer ranges, the results are only one artifact if the boxes
are **instrument-equivalent**. This estate's own law: **bit-identity is a SAME-INSTRUMENT
property** - same GPU model plus determinism flags gives byte-identical output across different
boxes, and a different card class does not.

    [ ] decide and SEAL: one card class for the whole run, or per-shard stamps and no pooling
    [ ] if one class: pin it in the explicit box list and reject a substitution at provision time
    [ ] if mixed: every artifact carries its card and driver, and NO cross-shard number is quoted

⛔ **A fleet that silently mixes card classes produces one file out of several instruments.** That
is not automatically wrong, and it is automatically unreportable unless the stamps are per shard.

## What is deliberately NOT here

- ⛔ **No kernel work.** That is P4 and it is a different project.
- ⛔ **No speed number.** A quantisation run measures a quantisation, not a decode.
- ⛔ **No residency claim.** Residency needs the fast kernel and a box with the memory.
