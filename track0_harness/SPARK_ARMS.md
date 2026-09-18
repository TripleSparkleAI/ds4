# spark_arms - one measurement round on the DGX Spark

`track0_harness/spark_arms.py` runs a round of `triple-*` arms against a tip: it resolves shas,
seals the rule, pushes to `sparkbox`, builds one detached worktree per arm, starts the driver
under a lane-named watchdog, and judges the CSVs by the sealed rule. stdlib only; runs from the
Mac over `ssh spark`, or on the spark itself.

    spark_arms.py run hitsfirst hotlist pool --name r4      # the one line
    spark_arms.py plan                                       # every carded branch, count + wall
    spark_arms.py --selftest                                 # 104 offline checks (count it, do not quote it)

## The lean defaults, each a measurement (2026-09-17, rounds 1-2)

| default | measured | restore |
|---|---|---|
| one TIP bracket per cycle | round 1, 41 runs: cycle-edge deltas agree with adjacent-bracket deltas at median diff 0.49 %, max 3.22 %, inside the in-run floor (median 0.52, max 4.13); all four verdicts unchanged. 7 arms: 33 runs, not 57 | `--adjacent` |
| frontiers 4096, 6144; 2048 is the discarded warmup; 8192 dropped | rounds 1-2, 59 runs: the min-across is unchanged in 56 of 59; when it moves the two-frontier min is higher by median 1.56 %, max 2.12 %, inside the floor. Saves ~37 s of a 159 s run. 6144 carries the min in 42 of 59 and is never dropped | `--frontiers 4096,6144,8192` |
| 5 s between runs | 15 s x 57 runs was 14 min of dead time | none |
| one warm-up TIP run before run 1, discarded by label (round 9 on) | the first run of a round reads low after the builds evict the 81 GB model from the page cache: r8_tip_1 8.61 t/s at 6144 vs 9.6-9.7 for every later r8 TIP run; r6_tip_1 9.45 vs 9.56-9.88; r4 pool_1 9.24 vs 10.34-10.47 (first runs also 6-8 s longer wall). Under the all-pairs floor a cold tip_1 inflates the floor (r8 partial 10.69 vs 2.7-3.4 on normal rounds) | `--no-warmup`, only for `--no-build` onto a warm box |

Not a lever, so not offered: model load and exit is ~12 s of a 159 s run, and the expert cache is
per-process device memory that fills the box, so a loaded model cannot be shared across binaries.
Net: 7 arms in 33 x ~122 s, about 70 min with sleeps, instead of ~165.

## Subcommands

| command | does | refuses when |
|---|---|---|
| `plan [ARM...]` | shas, arms, design, run count, wall | unknown branch, bad frontiers, cycles < 1 |
| `seal [ARM...] --name N` | writes `YYYY-MM-DD-N-PREREGISTERED-RULE.txt` with a PREDICTIONS block to fill | the file already exists |
| `build [ARM...] --name N` | pushes shas to `sparkbox` as `refs/arms/*`, detached worktrees, `make cuda-spark -j12`, logs rc and `.text` | remote is not the spark or the TripleSparkleAI fork; model lock held by another lane; an arm's `.text` is more than 0.2 MB off the tip's (named, never dropped silently) |
| `run [ARM...] --name N` | plan + seal check + build + start: takes `~/spark_model.lock/owner`, starts watchdog then driver with `setsid ... </dev/null`, verifies both alive with anchored `pgrep` | seal missing or NOT COMMITTED; seal shas differ from the plan; foreign lock; a driver of this lane already running |
| `status --name N` | END/OK/KILLED/REFUSED counts, last stamps, load, memavail, vitest count, watchdog CONTs | |
| `analyze --name N [--fetch] [--legacy-straggler]` | paired delta against the mean of the bracketing TIPs per frontier and min-across; median, sign count AND rate, SE, MAD, min, max; TIP floor median and max from ALL adjacent live TIP pairs; straggler rule on ARMS only at n >= 6, the no-drop reading beside each verdict; prints which rule it ran | no plan file |

Verdict at min-across, `floor_max` = this round's own TIP floor max: BEATS iff D >= floor_max and
the positive sign rate is 75 % or better; TIES iff |D| <= floor_max; LOSES iff D <= -floor_max;
otherwise PARTIAL. A killed run (rc != 0) is void and dropped, never reweighted.

## The straggler rule (round 7 on) and why it changed

Round 4's sealed rule (per arm on gen-min, |G - med| > 3 MAD, single pass, every arm including TIP,
from n=3) dropped 2 of 5 TIP controls at margins of 0.06 and 0.20 t/s against 3*MAD = 0.03: with
five points MAD collapses whenever three agree to the second decimal. The floor shrank from 2.71
to 0.21 and a +0.31 on one survivor read BEATS. So, from round 7:

- the rule applies to ARMS only; a TIP run is never dropped as a straggler
- the floor is computed from ALL adjacent live TIP pairs (contention drops still apply)
- an arm with fewer than 6 live repeats is not filtered at all; the analysis says
  `straggler rule ARM: n<6, not applied`
- both readings stay printed (kept, and no-drop beside every verdict)

`analyze --legacy-straggler` restores the round-4 rule so rounds 5 and 6, sealed under it, read as
sealed. The analysis header names the rule it ran. The seal text states the new rule.

## Per-arm environment switches (round 7 on)

An arm is `BRANCH` or `BRANCH@DS4_VAR=value[,DS4_VAR2=value]`, prefix optional:

    spark_arms.py plan --name r7 all-fastest-newtip@DS4_CUDA_HITS_FIRST=1 \
        all-fastest-newtip@DS4_PREFILL_READAHEAD_HOLD=0 \
        all-fastest-newtip@DS4_CUDA_HITS_FIRST=1,DS4_PREFILL_READAHEAD_HOLD=0 hitsfirst pool

- each (branch, switches) pair is its own arm with its own label: the branch label plus a suffix
  from the switches (`allfastestne+hits`, `allfastestne-hold`, `allfastestne+hits-hold`; a knob
  gives word+value, `marg12`). The analyzer judges each pair on its own.
- one worktree and one build per BRANCH; its switched arms share them
- the plan file carries the assignments as trailing columns; the one-script hands them to
  `env` for that run only (a plain arm and every TIP run the binary's default env); the START
  line records `switches=...`; the seal lists them under THE ARMS and `run` refuses a seal whose
  switches differ from the plan
- syntax is validated: `VAR=value` tokens, VAR matching `^DS4_[A-Z0-9_]+$`, value non-empty,
  no repeated VAR in one arm, no arm named twice (same branch AND same switches). Anything else
  is one `error:` line plus the usage, exit 2.

Check the switch's real name and default in the branch's `ds4_cuda.cu` before sealing: the
stacked tree at b8f8a5a0 reads `DS4_CUDA_HITS_FIRST` (off; `=1` turns it on) and
`DS4_PREFILL_READAHEAD_HOLD` (the readahead arm is OFF by default there since that commit; `=1`
or `DS4_CUDA_PREFETCH_SWEEP_ORDER=1` turns it on, `=0`/`no`/`false` forces it off).

## Per-arm bench arguments (round 11 on)

| spec | means | label | plan token | refused when |
|---|---|---|---|---|
| `BRANCH@[env];cache=<N>GB,gen=<N>,ctxmax=<N>` (`winners@;cache=60GB` is args only; quote the `;`) | after the `;`, overrides from a CLOSED SET: `cache` -> `--ssd-streaming-cache-experts` (default 90GB), `gen` -> `--gen-tokens` (128), `ctxmax` -> `--ctx-max` (the plan's); the one-script REPLACES that flag's value for that run only, never appends a second flag; START stamps `args=cache=..,gen=..,ctxmax=..`; the seal lists `args ...` under THE ARMS and `run` refuses a seal whose args differ; the arm's identity is (branch, switches, args) | `~c60`, `~g256`, `~x8192` appended to the label | `ARG:cache=60GB` after the env columns | unknown key, missing value, `cache` without `GB`, a repeated key, `ctxmax` below the highest frontier (the CSV could never carry it); one `error:` line, exit 2 |

Why it exists: on the 121 GB box a 90 GB expert cache leaves the page cache holding 2.9 GB of the
81 GB model, so every miss is an NVMe read; `cache=60GB` is a SETTING to measure against the tip
like any other arm, and it is a bench argument, not an env switch.

Every subcommand takes `--dry-run` (prints every script, touches nothing) and `--help`. A bad
parameter prints one `error:` line and the usage, exit 2, never a traceback. Branch names work
with or without the `triple-` prefix. Round and lane names: `--name r4` gives worktrees `r4-*`,
labels `r4_<arm>_<cycle>`, lane `ARMS-R4`, spark scripts `~/arms_r4_{one,driver,watchdog}.sh`,
logs `~/sweeps/RUNLOG_r4.txt` and `~/sweeps/BUILD_r4.log`.
