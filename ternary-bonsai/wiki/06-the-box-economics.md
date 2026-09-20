# THE BOX ECONOMICS - the tiers and the money rule, quoted not invented

Every figure on this page is **quoted from JIMOTHY's own skill**, which is the source of truth for
fleet economics in this estate: `~/.claude/skills/jimothy/SKILL.md` and
`~/.claude/skills/jimothy/docs/VAST_AI_OPERATIONS.md`. ⛔ **Never invent a price.** And never
restate a limit as a number in a third place - JIMOTHY's own law is *"NEVER RESTATE A LIMIT. POINT
AT `--limits --json`"*, because a limit that exists in three files is three limits, and that has
already fired once in this estate.

## The card tiers, verbatim from `~/.claude/skills/jimothy/docs/VAST_AI_OPERATIONS.md`

> - **LOW (the default, ~90% of rentals):** RTX 3060-class, $0.045-0.052/hr - the char-nanoGPT
>   workhorse; whole-fleet economics run here. **Doctrine: rent LOW unless a measured reason says
>   otherwise.**
> - **MED (when a cell needs VRAM or halving wall-time meets a deadline):** 3070 / 3080 / 2080Ti /
>   A4000-class.
> - **HIGH (rare, pre-registered need only):** 4090 / A100 / H100-class - never rented on vibes;
>   any >50-box or high-tier standup shows the navigator the burn plan FIRST.

⚠ **MED and HIGH carry no rate in the skill's own tier table**, only card classes. This page does
not supply one. The live figure comes from the selector, not from prose:
`python3 ~/.claude/skills/jimothy/scripts/track0_fleet_value_selector.py --limits --json`.

★ **And the bands are not fixed forever.** The skill: *"The bands are re-derived from measured
$/cell (the living card-comparison table), not fixed forever - re-bench and move the boundaries
when the data says so."* ⇒ which is exactly what `../explore/` is for.

## ⛔ THE MONEY RULE, verbatim

> A Vast instance **bills every second it exists** - running OR stopped (a stopped box still bills
> storage). **`vastai destroy instance <id> -y` is the only $0 state.**

⇒ destroy-immediately for one-shot cells, or a warm session with a **hard idle cap**. Never a box
idle with no cap.

## ⚠ STORAGE IS NOT IN THE CEILING, and it is how a compliant box breaches

The skill's own worked case, and it is the trap most likely to hit a quantization run, because a
quantization run wants a **large disk**:

> A **120 GB disk request more than doubled** a `$0.0994` box to **`$0.2378/hr`** - *that* was the
> "3.7x breach" that named law 6; it was a **compliant host plus an oversized disk.** Storage
> varies **6.5x between hosts** (`$0.133-$0.867`/GB/month). **Rank on
> `dph_total + storage_cost x disk_needed` and request the smallest disk that fits (~30 GB).**

⛔ **This is the live hazard for this plan specifically.** The FP16 source shards are the largest
artifact in the whole project, so any box that holds them wants disk, and **the disk is where the
price goes rather than the card.** A plan that reasons only about the compute ceiling will price
this wrong. ⇒ `../methods/01-P0-the-fp16-precondition.md` box two, the subset question, is also a
**storage** question and not only a download-time one.

## ⚠ The value ceiling is not a CLI argument

> **THE VALUE CEILING is `$0.0650 / GPU / hr`, checked in code on the RAW price. There is no
> `--force` and no override** - exceeding it is a **navigator decision**, not a CLI argument.

## HOT AND SMOOTH

**HOT** = in the current JIMOTHY-managed fleet: provisioned, in the roster, under one drain.
**SMOOTH = BUSY** with confirmed jobs, short breaks only. ⛔ **HOT AND IDLE is a named bad state**
and is fixed on sight: fire a cell or release the box.

## The one figure of our own that bears on tier choice

✅ MEASURED on the rented 5090: the expert-cache ask **25GB is already saturated** - it plans
**1,928** slots and delivers **1,127**, and **18GB delivers the same 1,127**.

★ **The ask was never the constraint.** ⇒ **a bigger box is not automatically a faster answer**,
and this is the measured reason `../explore/` exists rather than an intuition about tiers.

## ⚠ The scaling trap, already paid for once in this project

A rate measured at one workload size and multiplied to another is a **PREDICTION**. This project
once scaled a capture rate by 38.1x and the real cell came in **3.85x cheaper**, which bought boxes
nobody needed. **Label any extrapolated figure `SCALED` and re-measure at the size you will
actually run.**

★ Note the direction: that error was **pessimistic**, and it was harder to catch for exactly that
reason, because nothing broke. **Being wrong in the safe direction is not being right.**
