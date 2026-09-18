# MEASURED ROUTETRACE - the first real routing trace of DeepSeek-V4.1-Flash-Q2

Lane ROUTETRACE, branch `triple-jev-router`, 2026-09-18. What the engine's own
`DS4_CUDA_ROUTE_TRACE` sees when the model actually decodes, and what
`train_next_layer.py` and `replay_three_tier.py` read off it.

> **STATE OF THIS DOCUMENT.** Sections 1 to 3 are MEASURED and complete. Sections
> 4 onward are **NOT RUN**: the traced bench runs are queued behind round 21,
> which took the model lock at 11:15:00Z. Every heading below carries its own
> state. No number in this file is predicted, scaled or carried over from another
> checkpoint; where a figure does not exist yet the heading says NOT RUN and the
> table is empty.

---

## 1 - THE EXPERT BYTE SIZE, MEASURED FROM THE FILE - 9,953,280 B PER EXPERT

**This is the number that decides what a forecaster is competing for**, so it
leads. It is read from the GGUF header, not derived from a total and not carried
from another model.

```
   ONE EXPERT OF DeepSeek-V4.1-Flash-Q2, AT ONE LAYER
   ────────────────────────────────────────────────────────────
   ffn_gate_exps   IQ2_XXS   [5120, 2304, 384]   3,041,280 B  ████████
   ffn_up_exps     IQ2_XXS   [5120, 2304, 384]   3,041,280 B  ████████
   ffn_down_exps   Q2_K      [2304, 5120, 384]   3,870,720 B  ██████████
                                                 ───────────
                                                 9,953,280 B  = 9.49 MiB = 9.95 MB
```

**The derivation, per tensor:** `blk.N.ffn_gate_exps.weight` is 1,167,851,520 B
for all 384 experts, so 1,167,851,520 / 384 = 3,041,280 B for one. Same for
`ffn_up_exps`. `ffn_down_exps` is 1,486,356,480 / 384 = 3,870,720 B. The three
sum to **9,953,280 B**, and the engine reads exactly these three, as three
preads, on every miss (`ds4_cuda.cu`, the gate / up / down task triple).

**The cross-check, and it is an independent one.** The engine's own load line
reports `8941 experts, 9.49 MiB each` for the 82.88 GiB dynamic cache. 9,953,280 B
= 9.4921 MiB. Two readings that share no code: one from the file layout, one from
the allocator's own arithmetic. They agree.

**The model's real shape, from its own KV, not from a default:**

| KV key | value |
|---|---|
| `deepseek41.num_hidden_layers` | **40** |
| `deepseek41.n_routed_experts` | **384** |
| `deepseek41.num_experts_per_tok` | **6** |
| `deepseek41.n_shared_experts` | 1 |
| file size | 365,713,686,528 B |

So routing is **top-6 of 384 over 40 layers**, and a token's routed demand is
6 x 40 x 9,953,280 = **2.389 GB if every expert missed**.

### ⚠ THE TWO FIGURES THIS REPLACES, AND WHY EACH WAS WRONG

- **124 KB** (the brief's figure) is a **preview-checkpoint number** and is wrong
  for this model **by a factor of 80**. Nothing in V4.1 Flash Q2 is 124 KB.
- **~9.1 MB** (the derived figure) is close and slightly low; the measured value
  is **9.95 MB**. The gap is that gate and up are IQ2_XXS while **down is Q2_K**,
  which is 27 % fatter per element, so a single-quant derivation under-counts.
- ⛔ **`ds4-gguf-census.py` MUST NOT BE USED FOR THE PER-EXPERT FIGURE ON THIS
  MODEL.** It prints `experts=256 top-8`, because it reads the **V4-Flash** KV
  names; the real KV says 384 and top-6. Its routed TOTAL (152.882 GB) is right
  and its divisor is not, so dividing them gives **14.93 MB per expert** - 50 %
  too high. `152.882e9 / (40 x 384) = 9.953 MB` restores the agreement.
  ★ The tool is correct about the object it was written for. The failure is that
  it reports a wrong expert count **without erroring**, so the wrong per-expert
  figure looks like a measurement. Read the header for this model.

### WHAT THE NUMBER MEANS FOR A FORECASTER

At 9.95 MB per expert, **one avoided miss saves 9.95 MB of SSD read**. Whether
that is the whole wait or a residual is section 6's question (the latency /
bandwidth fit), and it is **NOT RUN**. The figure is recorded here so that
section is read against a measured denominator rather than a remembered one.

---

## 2 - THE `t` LINE: THE TRACE NOW CARRIES THE TOKEN ID - BUILT, IN THE BINARY

The writer emitted no token id, so lane FORECASTLEARN's token-id table (shape A)
could not run on a real trace at all. Added at `e3510c459`:

```
t <token> <token_id>     once per token, immediately before that token's first
                         layer line; written only when a single-token embed has
                         actually run
```

- **The hook** is `cuda_route_trace_note_token(token)`, called from
  `ds4_gpu_embed_token_hc_tensor` and `ds4_gpu_embed_token_quant_tensor`. The
  q8_0 entry delegates to `_quant_tensor`, so **two hooks cover all three**
  single-token entries, and exactly one runs per decode step on CUDA.
- **A missing `t` line means NOT AVAILABLE, never id 0.** A prefill batch never
  fabricates one, and the release path clears the id.
- **No position field, deliberately.** The absolute KV position is **not
  reachable at the embed seam** - the entries do not carry one. Rather than ship
  a third field that would only repeat the first, `<token>` is documented as what
  it is: the trace's own counter, which is the position within the traced decode
  stream, per process, and the key every other line already uses.
- **Verified in the built binary:** `strings ds4-bench | grep "t %llu %d"` returns
  the format string on the spark binary built at 21:13 local.

---

## 3 - THE BUILD: TWO CLEAN CUDA BUILDS, NO COMPILE FIXES WERE NEEDED

The branch's `.cu` edits had been compiled only on the Mac, where `ds4_cuda.cu` is
not built at all, so the first spark build was expected to fail and the lane was
briefed to fix it.

**It did not fail, either time.**

| build | tree | `grep -ci error` | result |
|---|---|---|---|
| 1 | `33b8d00ba` (JEVENGINE's, as landed) | **0** | rc 0, all five binaries linked |
| 2 | `e3510c459` (plus this lane's `t` line) | **0** | rc 0, all five binaries linked |

`make cuda-spark -j12`, nvcc for `sm_121a`, on the GB10. **Zero compile-fix
commits exist on this branch because none were needed.**

### THE BOX, STAMPED

| | |
|---|---|
| host | `spark`, NVIDIA GB10, 20 cores, 121.69 GiB |
| driver | **580.159.03** (DRIVER-STAMP LAW) |
| kernel | 6.17.0-1018-nvidia |
| model | `~/dwarfstar/gguf/DeepSeek-V4.1-Flash-Q2.gguf`, 365,713,686,528 B |
| regime | `--cuda --ssd-streaming --ssd-streaming-cache-experts 90GB`, promessi prompt |
| cache | 90 GiB target -> 7.12 GiB prefill headroom + 82.88 GiB dynamic = **8610 slots** |

⚠ **The box rebooted at 21:08 local**, so the page cache is cold and a 365 GB
streaming model reads every expert from the SSD on its first pass. The run script
therefore fires a **discarded warm-up** before the measured traces and deletes its
trace; a cold run is a different machine (round 9's finding) and is not reported.

---

## 4 - HIT RATE TODAY - **NOT RUN**

Read straight off the `m` lines of trace A. Empty until the runs land.

| trace | tokens | layer-expert demands | hits | hit rate |
|---|---|---|---|---|
| A default | NOT RUN | | | |
| B long gen 1024 | NOT RUN | | | |

## 5 - THE POPULARITY CURVE - **NOT RUN**

Hot-set size covering 50 / 80 / 90 % of demand, as a fraction of the **8610**
real slots. The interesting quantity is whether the 50 % set fits in a tier much
smaller than the cache.

## 6 - LATENCY OR BANDWIDTH BOUND - **NOT RUN**

Least-squares fit of the `w` lines, `wait_us = a + b * n_miss`. A large intercept
`a` against `b x (median misses)` says a prefetch filling idle time can pay; a
small `a` says only fewer bytes pay. **Read this against section 1's 9.95 MB.**

## 7 - STICKINESS PER LAYER - **NOT RUN**

★ **This is the decision the lane exists to make.** If stickiness is **at or above
90 %**, the finding is that **the guess was never the problem** and the jev table
is not worth loading. Below it, the successor table has something to win.

## 8 - THE THREE GUESS SOURCES, HELD-OUT - **NOT RUN**

`train_next_layer.py`, last 20 % of tokens held out in order.

| source | held-out hit rate |
|---|---|
| (a) sticky | NOT RUN |
| (b) layer L -> L+1 | NOT RUN |
| (c) token-ahead | NOT RUN |

`t`-line coverage will be reported in the same table; a trace showing
`n_tid 0` would mean shape A cannot run on it.

## 9 - HIT / MISS / WASTE PER SOURCE, AND THE WORST WINDOWS - **NOT RUN**

From trace C's `f` and `o` lines (the scout on). ⚠ Trace C is a **different cache
regime** from trace A - the scout issues real reads - so its hit rate is not
trace A's and the two must not be quoted as one number.

## 10 - THE THREE-TIER REPLAY - **NOT RUN**

`replay_three_tier.py --slots 8610`, sweeping pinned fraction, lookahead k and
admission confidence, against the LRU control. ⚠ The lookahead is an **ORACLE**
bounded by the trace: its numbers are an **upper bound** on what any guesser can
buy at that tier split, not what a guesser achieves.

---

## WHAT IS OWED

1. The three traced runs (A, B, C) plus the discarded warm-up. **Queued**; the
   waiter holds no lock and acquires only after three consecutive free polls, so
   round 21 and any priority lane claim it first.
2. Sections 4 to 10, from those traces.
3. The trace file copied to `~/track1/` on the spark with its **sha256** recorded.

## INSTRUMENT NOTES

- `train_next_layer.py --selftest` **13/13** (was 10; three added for the `t`
  line, including that a `t` line moves neither (a) nor (b) by **exact**
  equality - a line carrying no routing may not move a score).
- `replay_three_tier.py --selftest` **11/11**, its planted trace now carrying
  `t` lines, which is the evidence that the reader ignores them.
- `tests/test_jev_router_logic` **33/33**, `tests/test_scout_logic` **40/40**,
  Mac `make` rc 0.
- ⚠ **No number in this file comes from a synthetic trace.** The selftests above
  are instrument checks on planted data and are not routing facts.
