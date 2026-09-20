# P2 - THE JIMOTHY EXPLORE: what a range of boxes actually costs and delivers

> ⛔ **STUBBED, NOT FIRED.** Nothing here has been run, no box has been rented, no fleet raised.

The navigator's words: *"send jimothy on some experiments with all size boxes ... once we figure
out a bit what we need to determine cost value and speed actual numbers from a range of boxes."*

⇒ **This is a MEASUREMENT OF THE FLEET, not of the model.** Its product is a **cost-per-answer
table** that every later step budgets against.

## ⚠ Before anything is rented: the choice of box is a TOOL CALL, not a judgement call

    [ ] python3 ~/.claude/skills/jimothy/scripts/track0_fleet_value_selector.py --plan <N>
    [ ] python3 ~/.claude/skills/jimothy/scripts/track0_fleet_value_selector.py --limits --json
        ^ the SINGLE SOURCE OF TRUTH for every limit. Never restate a limit as a number elsewhere.
    [ ] python3 ~/.claude/skills/jimothy/scripts/track0_fleet_provisioner.py --image-for <card>
        ^ never re-type an image tag; read it from --image-for
    [ ] --audit every fleet beat: over-ceiling boxes, HOT-AND-IDLE per host, burn, runway

⚠ **The value ceiling is `$0.0650 / GPU / hr`, checked in code on the RAW price, and there is no
`--force`.** Exceeding it is a navigator decision, not a CLI argument.

## The tiers, as JIMOTHY's own skill states them

Quoted from `~/.claude/skills/jimothy/docs/VAST_AI_OPERATIONS.md`. ⛔ **Not invented here, and no
price is supplied for a tier the skill does not price.**

| tier | cards | rate | doctrine |
|---|---|---|---|
| **LOW** | RTX 3060-class | **$0.045-0.052/hr** | the default, about 90 % of rentals; whole-fleet economics run here. Rent LOW unless a measured reason says otherwise. |
| **MED** | 3070 / 3080 / 2080Ti / A4000-class | ⚠ **the skill's tier table gives no rate** - read `--limits --json` | only when a cell needs VRAM, or halving wall-time meets a deadline |
| **HIGH** | 4090 / A100 / H100-class | ⚠ **no rate in the tier table** - read `--limits --json` | **rare, pre-registered need only, never on vibes.** Any >50-box or high-tier standup shows the navigator the burn plan FIRST. |

★ **And the bands move.** The skill: *"re-derived from measured $/cell ... re-bench and move the
boundaries when the data says so."* **This explore is that re-bench.**

## ⛔ THE MONEY RULE, verbatim from the skill

> A Vast instance **bills every second it exists** - running OR stopped (a stopped box still bills
> storage). **`vastai destroy instance <id> -y` is the only $0 state.**

⇒ **destroy-immediately for one-shot cells**, or a warm session with a **hard idle cap**. ⛔ Never a
box idle with no cap.

## ⚠ AND STORAGE IS NOT IN THE CEILING - which is the live hazard for THIS plan

> A **120 GB disk request more than doubled** a `$0.0994` box to **`$0.2378/hr`** ... Storage varies
> **6.5x between hosts** (`$0.133-$0.867`/GB/month). **Rank on
> `dph_total + storage_cost x disk_needed` and request the smallest disk that fits (~30 GB).**

⛔ **This plan wants large disks** - the FP16 shards are the biggest artifact in the project - so
**the disk is where the price goes rather than the card.** A tier table that prices only compute
will price this work wrong. ⇒ **the explore must record disk cost per box as its own column.**

## The explore, stubbed

    [ ] ONE workload, held FIXED across every box, chosen because we already know its answer
        candidate: the per-tensor checksum walk (a whole-file read) plus a fixed matmul batch
        ^ a workload whose right answer is known turns every box into a verify-first-time check
    [ ] rent ONE box per tier, by EXPLICIT LIST, never a formula and never "whatever is free"
    [ ] per box, record:
          card, driver version, VRAM total and free
          cgroup CPU quota        <- NOT nproc
          cgroup memory limit     <- NOT /proc/meminfo
          disk random-read rate, sequential rate
          disk SIZE requested and its $/hr contribution, separately from the card's
    [ ] per box, the measurand: SECONDS PER ANSWER and DOLLARS PER ANSWER, both stamped
    [ ] the deliverable: ONE TABLE, cost per answer by tier, with wall time beside it
    [ ] destroy every box in the same beat it finishes; PULL BEFORE DESTROY

## ⚠⚠ ASK WHAT YOU OWN, NOT WHAT YOU SEE - and the verification trap inside it

An environment reports the **HOST**; a container owns a **SHARE**. `nproc`, `os.cpu_count()`,
`/proc/meminfo` and `nvidia-smi` all answer a question you did not ask.

⛔ **And never verify the fix with `nproc`.** GNU `nproc` returns
`min(available_processors, $OMP_NUM_THREADS)` - **it honours the knob you just turned** - so a
reading taken after the export **reads back its own write** and cannot distinguish a working clamp
from a total no-op. MEASURED on an 8x box: `nproc` **64** before, **7** after, same command, same
64-core machine, seconds apart.

    [ ] use `nproc --all` for the machine
    [ ] read the CGROUP for the quota
    [ ] read /proc/<pid>/status Threads: AFTER a warm-up GEMM for whether the clamp bit
    [ ] ⚠ sched_getaffinity catches cpuset pinning but NOT a CFS quota - necessary, never sufficient

★ **A verification that cannot fail is not a verification.** The measured cost of getting this
wrong, on this project's own fleet: **1 worker 32.3 -> 185.6 GFLOP/s (5.74x)** and **2 workers
10.6 -> 306.2 (28.9x)**; the naive path **ANTI-SCALES**, so adding workers reduced total
throughput.

⚠ **And a thread-count change is a PRE-REGISTRATION EVENT** - it changes BLAS reduction order, so a
cell with a numeric anchor must not switch mid-run. Observe always, act on request.

## ⚠ The result table's own shape, so it cannot be quoted wrong

| box | tier | card | driver | disk GB | $/hr card | $/hr disk | owned cores | s/answer | $/answer | clock |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| - | - | - | - | - | - | - | - | - | - | - |

⚠ **Card and disk in separate columns**, because a single `$/hr` hides the whole storage finding.
⚠ **Owned cores, not machine cores.** ⚠ **A clock per row**, because the boxes are not in one
timezone: the spark is UTC+10, the rented 5090 is UTC+0000.

## ⛔ What would make this explore worthless

- Running it **on ternary**. Then a slow number has two possible causes.
- Running it **near the concurrency cap** on a shared box. A timing taken near the cap measures the
  box, not the code.
- Quoting a `$/answer` that was **extrapolated** from a smaller workload without the `SCALED` label.
