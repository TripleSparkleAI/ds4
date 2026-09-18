# MEASURED ROUTETRACE - the first real routing trace of DeepSeek-V4.1-Flash-Q2

Lane ROUTETRACE, branch `triple-jev-router`, 2026-09-18. What the engine's own
`DS4_CUDA_ROUTE_TRACE` sees when the model actually decodes on the GB10, and what
`train_next_layer.py` and `replay_three_tier.py` read off it.

**Every number below is measured on this box, on this day, from these traces.** Nothing is
scaled, predicted, or carried over from another checkpoint. Where a thing was not run, the
heading says NOT RUN.

---

## 1 - THE EXPERT BYTE SIZE - 9,953,280 B PER EXPERT

**This leads because it is the denominator for everything else: it decides whether a
forecaster is competing for the whole wait or for a residual.** Read from the GGUF header,
not derived from a total, and cross-checked against the engine's own allocator.

```
   ONE EXPERT OF DeepSeek-V4.1-Flash-Q2, AT ONE LAYER
   ──────────────────────────────────────────────────────────────
   ffn_gate_exps   IQ2_XXS   [5120, 2304, 384]   3,041,280 B  ████████
   ffn_up_exps     IQ2_XXS   [5120, 2304, 384]   3,041,280 B  ████████
   ffn_down_exps   Q2_K      [2304, 5120, 384]   3,870,720 B  ██████████
                                                 ───────────
                                                 9,953,280 B  = 9.49 MiB = 9.95 MB
```

Per tensor: `blk.N.ffn_gate_exps.weight` is 1,167,851,520 B for all 384 experts, so
1,167,851,520 / 384 = 3,041,280 B for one; same for `ffn_up_exps`; `ffn_down_exps` is
1,486,356,480 / 384 = 3,870,720 B. The engine reads exactly these three, as three preads,
on every miss.

**THE CROSS-CHECK, and it is independent.** The engine's own load line on this run reads
`8941 experts, 9.49 MiB each` for the 82.88 GiB dynamic cache. 9,953,280 B = 9.4921 MiB.
Two readings sharing no code - one from the file layout, one from the allocator - agree.

**The model's real shape, from its own KV:**

| KV key | value |
|---|---|
| `deepseek41.num_hidden_layers` | **40** |
| `deepseek41.n_routed_experts` | **384** |
| `deepseek41.num_experts_per_tok` | **6** |
| file size | 365,713,686,528 B |

Routing is **top-6 of 384 over 40 layers**, confirmed live in the trace (every layer line
carries exactly 6 ids). A token's routed demand is 6 x 40 x 9,953,280 = **2.389 GB if every
expert missed**; at the measured 95.7% hit rate it is **~102 MB of real SSD read per token**.

### THE TWO FIGURES THIS REPLACES

- **124 KB** (the original brief) is wrong for this model **by a factor of 80**. It is a
  preview-checkpoint figure and nothing in V4.1 Flash Q2 is 124 KB.
- **~9.1 MB** (the derived figure) is close and slightly low. The gap is that gate and up are
  IQ2_XXS while **down is Q2_K**, 27% fatter per element, so a single-quant derivation
  under-counts.
- **`ds4-gguf-census.py` MUST NOT BE USED FOR THE PER-EXPERT FIGURE ON THIS MODEL.** It
  prints `experts=256 top-8`, because it reads the **V4-Flash** KV names. Its routed TOTAL
  (152.882 GB) is right and its divisor is not, so dividing them gives **14.93 MB** - 50% too
  high. `152.882e9 / (40 x 384) = 9.953 MB` restores the agreement. The tool is correct about
  the object it was written for; the failure is that it reports a wrong expert count **without
  erroring**, so the wrong per-expert figure looks like a measurement.

---

## 2 - THE BOX, AND THE THREE RUNS

| | |
|---|---|
| host | `spark`, NVIDIA GB10, 20 cores, 121.69 GiB |
| driver | **580.159.03** (DRIVER-STAMP LAW) |
| kernel | 6.17.0-1018-nvidia |
| power | n/a (Linux; the M5 power-mode law does not apply) |
| model | `~/dwarfstar/gguf/DeepSeek-V4.1-Flash-Q2.gguf`, 365,713,686,528 B |
| regime | `--cuda --ssd-streaming --ssd-streaming-cache-experts 90GB`, promessi prompt |
| cache | 90 GiB target = 7.12 GiB prefill headroom + 82.88 GiB dynamic = **8941 slots** |
| build | `e3510c459`, `make cuda-spark -j12`, **0 errors** |

| run | argv | load at start | tokens | trace |
|---|---|---|---|---|
| WARMUP (discarded) | ctx 2048, gen 64 | 0.05 | - | deleted |
| **A** default | ctx-start 2048, ctx-max 6144, step 2048, gen 128 | 1.02 | 384 | 791,900 B |
| **B** long gen | ctx-start 2048, ctx-max 2048, **gen 1024** | 1.13 | 1024 | 2,071,679 B |
| **C** scout on | as A, plus `DS4_CUDA_SCOUT=1` | 1.06 | 384 | 2,302,635 B |

⚠ **The warm-up is discarded and its trace deleted.** The box rebooted at 21:08 local, so the
first pass over a 365 GB streaming model reads every expert cold. A cold run is a different
machine; it is not reported.

⚠ **THE TRACES WERE PRODUCED TWICE AND THE SECOND SET IS THE ONE REPORTED.** A waiter process
left running by a killed session re-acquired the lock at 12:02:45Z and re-fired the pipeline,
overwriting the first set mid-analysis. **This is recorded rather than hidden because one
reading was taken during the overwrite and was wrong:** trace B read `102,600 lookups` against
its true `245,760`, because the file was still being written. The replication reproduces the
first set's headline figures exactly (B: 0.988 over 245,760 both times), which is why the two
sets are interchangeable and why the bad reading is identifiable as a mid-write artifact
rather than a disagreement. **A number taken off a file another process holds open is not a
measurement.**

---

## 3 - HIT RATE TODAY

Read straight off the `m` lines - the engine's own LRU-with-hit-protection, unmodified.

| trace | lookups | misses | **hit rate** |
|---|---|---|---|
| **A** default | 92,160 | 3,944 | **0.957** |
| **B** long gen 1024 | 245,760 | 3,056 | **0.988** |
| **C** scout on | 92,160 | 3,935 | **0.957** |

```
   A default   ████████████████████████████████████████████████·· 0.957
   B long gen  ███████████████████████████████████████████████████ 0.988
   C scout on  ████████████████████████████████████████████████·· 0.957
                                                                  ↑ 1.000
   the headroom any forecaster is playing for:  A 4.3%   B 1.2%
```

★ **THE FIRST REAL FINDING: THE CACHE IS ALREADY WINNING, AND THE LONGER THE GENERATION THE
MORE IT WINS.** At 90 GB the cache holds 8,941 of 15,360 possible (layer, expert) pairs - 58%
of the whole routed model - so a warm decode rarely reaches the SSD at all. **B is the warm
case the brief asked for, and it is the best case: 98.8%.**

★ **C IS THE CONTROL THAT MATTERS: THE SCOUT CHANGED THE HIT RATE BY 0.000.** Same 92,160
lookups, 3,935 misses against A's 3,944 - a difference of 9 reads in 92,160. The engine's own
stats line says why: **`scout reads=84`** over the whole run. The scout found almost every
guess already resident or no victim to evict, so it spent nothing and bought nothing.

---

## 4 - THE POPULARITY CURVE

How many (layer, expert) pairs carry what share of demand.

| trace | distinct pairs | 50% of demand | 80% | 90% |
|---|---|---|---|---|
| **A** default | 7,513 | **357** | 1,756 | 3,027 |
| **B** long gen | 8,051 | **340** | 1,317 | 2,361 |

```
   demand covered
   50% ├── 357 pairs  ███                          3.4 GB   ( 4.0% of the cache)
   80% ├── 1,756      ███████████████             17.5 GB   (19.6%)
   90% ├── 3,027      ██████████████████████████  30.1 GB   (33.9%)
        the cache holds 8,941 pairs = 83 GB, so the 90% hot set fits THREE times over
```

★ **HALF THE DEMAND SITS IN 357 PAIRS - 3.4 GB, 4% of the cache we already pay for.** The hot
set is not merely small, it is small enough that a PINNED tier costs almost nothing. That is
the arithmetic the three-tier replay then confirms.

⚠ B's curve is *tighter* than A's on a *longer* run (340 pairs for 50%, against 357) while its
distinct-pair count is *higher* (8,051 against 7,513). A longer generation visits more experts
in total and concentrates harder on the few it keeps returning to.

---

## 5 - STICKINESS - **0.462**, AND JEVENGINE'S THRESHOLD IS NOT MET

The fraction of this token's routed ids that the previous token also routed to, at the same
layer.

| trace | stickiness |
|---|---|
| **A** default (384 tokens) | **0.462** |
| **B** long gen (1024 tokens) | **0.386** |
| **C** scout on | **0.462** |

★★ **THE ANSWER TO THE LANE'S OWN QUESTION IS NO.** JEVENGINE's README states: *"If the trace
shows stickiness at or above 90%, the finding is that the guess was never the problem and the
table is not loaded."* **Measured: 46.2%, and 38.6% on the long generation.** Stickiness is
**less than half** the threshold, so the guess IS a real problem - the scout's current
placeholder (the last token's ids, confidence 1.000) is wrong more often than it is right.

★ **CONFIRMED INDEPENDENTLY BY THE ENGINE ITSELF.** Trace C's stats line reads
`guess_hits=41974/90858` = **0.462** - the engine's own scoring of its own guess, matching the
offline measurement to three decimals, computed by different code on a different pass.

**Per layer, the spread is wide and structured:**

```
   layer  0  ·············· 0.201   ◀ lowest: the first layer is the least predictable
   layer 19  ················ 0.311
   layer  3  ················ 0.314
   layer  1  ················ 0.321
   layer 39  ·················· 0.350
                  ...
   layer 24  ······································ 0.580
   layer 29  ······································ 0.582
   layer 33  ······································· 0.594
   layer 26  ········································ 0.604
   layer 22  ·········································· 0.624  ◀ highest
```

⇒ **stickiness is a MIDDLE-LAYER property.** The early layers (0, 1, 3) and the last layer
(39) re-route hardest; layers 22-33 hold their experts best. A guesser that treats all 40
layers alike is leaving the structure on the table.

---

## 6 - LATENCY OR BANDWIDTH BOUND - **BANDWIDTH**, ON ALL THREE

Least-squares fit of the `w` lines (the hits-first pool wait per layer's miss batch).

| trace | fit | w lines | median misses |
|---|---|---|---|
| **A** default | `wait_us = -98 + 321 x n_miss` | 2,489 | 1.0 |
| **B** long gen | `wait_us = -197 + 322 x n_miss` | 2,472 | 1.0 |
| **C** scout on | `wait_us = -348 + 411 x n_miss` | 2,489 | 1.0 |

```
   the intercept ·  A  -98 us   ── essentially zero, and NEGATIVE on all three
   the per-miss  ·  A +321 us   ── the whole wait
                              ⇒ BANDWIDTH-BOUND: only FEWER BYTES pay
```

★ **THE VERDICT AND ITS CONSEQUENCE.** The intercept is at or below zero on every trace while
the per-miss term is 321-411 us, so **there is no fixed setup cost for a prefetch to hide
inside.** A forecaster that fills idle time buys nothing here; only one that **removes reads**
does. That is a direct argument for the PINNED tier (which removes reads permanently) over a
deeper lookahead (which reorders them).

★ **321 us for 9.95 MB is 31.0 GB/s of effective read**, which is the number to beat if anyone
optimises the read path rather than the routing.

⚠ The negative intercept is not physical; it is the fit's extrapolation to zero misses, where
the engine does not take the path at all. Read it as *"indistinguishable from zero"*, not as a
negative wait.

---

## 7 - THE THREE GUESS SOURCES, HELD-OUT

`train_next_layer.py`, last 20% of tokens held out in order, guess capped at K=8.

| trace | (a) sticky | (b) layer L to L+1 | (c) token-ahead | b-a | c-a |
|---|---|---|---|---|---|
| **A** default (307 train / 77 held) | 0.435 | 0.503 | **0.504** | +0.068 | **+0.069** |
| **B** long gen (819 / 205) | 0.326 | 0.694 | **0.852** | +0.367 | **+0.525** |
| **C** scout on | 0.435 | 0.503 | **0.504** | +0.068 | +0.069 |

```
   B, the long generation - where a trained guesser has 819 tokens to learn from:

   (a) sticky      ████████████████·················· 0.326
   (b) L -> L+1    ██████████████████████████████████ 0.694   +0.367
   (c) token-ahead █████████████████████████████████████████·· 0.852   +0.525
```

★★ **BOTH TRAINED SOURCES BEAT STICKINESS, AND ON THE LONG RUN THEY BEAT IT ENORMOUSLY.** On
B, the token-ahead head reaches **0.852 against stickiness's 0.326** - it more than doubles the
guess quality, on a held-out split, in order, with no teacher in the loop beyond the model's own
routing.

⚠ **THE DIFFERENCE BETWEEN A AND B IS THE TRAINING SET, NOT THE MODEL.** A has 307 training
tokens and both trained sources land near 0.50; B has 819 and (c) reaches 0.852. **Do not quote
A's +0.069 as the value of the method** - it is the value of the method *with 307 tokens of
history*. Equally, do not quote B's +0.525 as a general figure: it is one prompt, one context
length, one generation. **The honest statement is that the gain grows steeply with trace length
over the only range we have measured, and that range is 307 to 819 tokens.**

★ **THE `t` LINE WORKS: 384 of 384 and 1024 of 1024 tokens carry a token id** (237 and 196
distinct). Lane FORECASTLEARN's shape A (the token-id table) can run on these traces, which it
could not have done before this lane added the writer.

---

## 8 - HIT / MISS / WASTE PER SOURCE (trace C, the scout on)

| source | layers | HIT | MISS | WASTE | reads | hit/forecast | miss/routed |
|---|---|---|---|---|---|---|---|
| **sticky** | 2,468 | 6,137 | 8,671 | 8,671 | **84** | **0.414** | 0.586 |
| unknown | 12,469 | 35,861 | 38,953 | 38,953 | 0 | 0.479 | 0.521 |

HIT = forecast and routed · MISS = routed and not forecast (a wait) · WASTE = forecast and not
routed.

★ **THE SCOUT ISSUED 84 READS ACROSS 2,468 FORECAST LAYERS.** Its forecasts were almost
entirely free - every id already resident, or no eligible victim - which is exactly why C's hit
rate is identical to A's. **The scout as it stands is not wrong so much as inert.**

⚠ **The `unknown` rows are `o` lines with no matching `f`** - a forecast that found no miss
batch to ride, so it spent nothing. They are 83% of the forecast layers. This is the trace
format's own honest category, not a defect.

⚠ **WASTE EQUALS MISS EXACTLY on both rows** (8,671 and 38,953). That is a property of a
fixed-width top-6 guess against a fixed-width top-6 route: every id forecast and not routed is
matched by one routed and not forecast. It is arithmetic, not a finding.

**The worst windows** (token: MISS+WASTE summed across its 40 layers, out of a 480 ceiling):

```
   token 256 ····································· 380
   token 128 ································· 356
   token 376 ································· 356
   token 271 ······························· 342
   token  67 ······························ 340
                                    the median token sits far below these
```

⇒ the failures cluster on **particular tokens**, not uniformly - a token whose routing turns
over wholesale costs ~380 of a possible 480. Those are the tokens a token-ahead head would have
to catch, and they are the reason (c) beats (a).

---

## 9 - THE THREE-TIER REPLAY - WHAT IT WOULD SAVE

`replay_three_tier.py --slots 8941` (the box's true slot count), replayed on the held-out tail.

**Trace A, 77 tokens replayed:**

| policy | pinned | k | conf | reads/token | **crit misses/token** | hit rate |
|---|---|---|---|---|---|---|
| **LRU (today), replayed** | - | - | - | 48.82 | **48.82** | 0.797 |
| tiers | 0.25 | 1 | 0.50 | 48.39 | **2.99** | 0.988 |
| tiers | 0.50 | 1 | 0.50 | 24.27 | **1.38** | 0.994 |
| tiers | **0.75** | **1** | **0.50** | **12.12** | **0.52** | **0.998** |

**Trace B (long gen), 205 tokens replayed:**

| policy | pinned | k | conf | reads/token | **crit misses/token** | hit rate |
|---|---|---|---|---|---|---|
| **LRU (today), replayed** | - | - | - | 13.51 | **13.51** | 0.944 |
| tiers | 0.50 | 1 | 0.50 | 6.48 | **0.14** | 0.999 |
| tiers | **0.75** | **1** | **0.50** | **2.34** | **0.03** | **1.000** |

```
   TRACE A - critical misses per token, the thing a token actually waits for

   LRU today      ████████████████████████████████████████████████ 48.82
   pinned 0.25    ███                                               2.99
   pinned 0.50    █                                                 1.38
   pinned 0.75    ▌                                                 0.52
                  └─ 94x fewer waits at the best tier split
```

★ **THE BEST TIER SIZES: PINNED 0.75 OF THE CACHE, LOOKAHEAD k=1, ADMISSION CONFIDENCE 0.50.**
That is **6,706 of 8,941 slots pinned** to the hot set, with the remaining 2,235 serving
staging and pass-through.

★ **WHAT IT SAVES, on the replayed tail: critical misses fall 48.82 to 0.52 per token (94x) on
A, and 13.51 to 0.03 (450x) on B.** Total reads fall 4.0x and 5.8x - so it is not merely
re-ordering the reads, it is **removing** them, which is the only thing that pays on a
bandwidth-bound wait (section 6).

★ **k=1 IS ENOUGH, AND k=2 AND k=4 ARE WORSE.** At pinned 0.75, k=1 gives 0.52 crit misses and
k=4 gives 10.45 - **twenty times worse**. A deeper lookahead evicts more than it saves. Whoever
builds this should not reach for a longer horizon.

⚠⚠ **THREE FENCES ON EVERY NUMBER IN THIS SECTION.**

1. **THE LOOKAHEAD IS AN ORACLE.** It knows t+1..t+k from the trace. These are an **UPPER
   BOUND on what any guesser can buy at that tier split** - they say what the tiers are worth,
   not what a guesser achieves. Section 7 says the best real guesser we measured reaches 0.852
   on B, not 1.000.
2. **THE REPLAYED LRU IS NOT TODAY'S HIT RATE.** The replay starts cold on the held-out tail,
   so its LRU reads 0.797 where the live engine reads 0.957. **Compare tiers against the
   replayed LRU (same cold start, same tail), never against the live 0.957.** The like-for-like
   claim is 0.797 to 0.998 on A.
3. **PINNING 75% OF THE CACHE IS A REAL COST, NOT A FREE WIN.** It holds 6,706 slots against
   one workload's hot set. This trace is one prompt at one context; a pinned set chosen from it
   is fitted to it. **The replay does not measure what happens when the workload changes**, and
   that is the question any deployment must answer first.

---

## 10 - WHAT THIS SAYS ABOUT THE JEV ROUTER

Read together, the measurements point one way and it is not the way the branch was aimed.

- **The guess is genuinely weak** (stickiness 0.462, below the 0.90 threshold), so the hook's
  premise is sound: there IS room to guess better, and both trained sources do.
- **But the demand for a better guess is small**, because the cache already answers 95.7% of
  lookups (98.8% warm). The headroom is 4.3% of lookups, and the scout currently spends 84
  reads to chase it.
- **And the wait is bandwidth-bound**, so a better-timed read buys nothing; only a removed read
  pays.
- ⇒ **the measured lever is the PINNED TIER, not the forecaster.** Pinning 357 pairs (3.4 GB,
  4% of the cache) covers half of all demand; the replay's best split removes 94x of the
  critical waits. **The forecaster is the smaller half of this branch's value and the tiers are
  the larger half** - which inverts the branch's own emphasis.

⚠ **NO t/s CLAIM IS MADE OR IMPLIED.** Every number here is a count of reads, waits and
hit rates. **Not one of these runs measured tokens per second against a control**, and a
replay is not a build. Whether any of this moves decode throughput on the GB10 is **NOT RUN**.

---

## WHAT IS NOT RUN

- **Any t/s A/B.** No speed claim exists for the jev router, the tiers, or the scout.
- **The jev table in the engine.** `DS4_CUDA_JEV_ROUTER` was never loaded: every run reports
  `jev_hits=0/0`. The tables exported here (`jev_A_default.table`, `jev_B_long1024.table`,
  `jev_C_scout.table`) have never been fed back to the engine.
- **The token-ahead head in the engine.** Source (c) is offline only; the engine has no `ahead`
  source, as the trace format's own comment says.
- **Any workload but this one.** One prompt (promessi), one model, one box, one day.
- **The G1 gate on any of it.** Reads-only by construction, but not gate-proven here.

## PROVENANCE

Traces on the spark at `~/track1/`, sha256 in `~/track1/routetrace_SHA256SUMS.txt`:

```
be2abdf53969e7e5d63ca572ba8d2aef047b076a672c4ece1b1f95bec0443226  routetrace_A_default.txt
d1f8f974d63ce8f454e515f72d07a78ddea58776a22e3c838517b2476dccd24a  routetrace_B_long1024.txt
a934b99ef363a1d939e2537b326f8d6fefa6c44ce5546071baf3c0ec036ab1ec  routetrace_C_scout.txt
a0b09c6963c29122197334aec25994a1621404f3442629be6cdc6f4df92068f6  routetrace_stamps.txt
```

Analysis outputs beside them: `FINAL_train_<label>.out`, `FINAL_replay_<label>.out`,
`jev_<label>.table`. Per-run stamps (load, driver, gpu, memavail, argv) in
`routetrace_stamps.txt`. Engine logs in `routetrace_<label>.log`.

⚠ **numpy is not installed on the spark's system python**; both tools exit 2 there with
`numpy is required`. The analysis ran under `~/track4/venv/bin/python3` (numpy 2.5.1). The
pipeline's own inline analysis pass failed for this reason and was re-run by hand; that is why
the `train rc=2 / replay rc=2` lines appear in `routetrace_stamps.txt` above the good output.

## INSTRUMENT NOTES

- `train_next_layer.py --selftest` **13/13** (was 10; three added for the `t` line, including
  that a `t` line moves neither (a) nor (b) by **exact** equality).
- `replay_three_tier.py --selftest` **11/11**, its planted trace carrying `t` lines.
- `tests/test_jev_router_logic` **33/33** · `tests/test_scout_logic` **40/40** · Mac `make` rc 0.
- **Two clean CUDA builds, `grep -ci error` = 0 on both** (`33b8d00ba` as landed, and
  `e3510c459` with the `t` line). The branch's `.cu` edits had only ever been compiled on the
  Mac, where `ds4_cuda.cu` is not built, so a first-build failure was expected and briefed for.
  There was none, either time; that is why this branch carries no compile-fix commit.
- **No number in this file comes from a synthetic trace.** The selftests are instrument checks
  on planted data and are not routing facts.
