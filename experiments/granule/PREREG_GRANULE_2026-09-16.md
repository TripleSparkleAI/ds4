# PREREG - GRANULE

**Sealed before the instrument was run.** Written 2026-09-16, committed as the first
commit of `triple-granule` with `README.md` untouched. The numbers in
`MEASURED_GRANULE_*.md` were produced by running `granule_discrimination.py` at the
commit named there, against the constants fixed in this file. Nothing in this file
was edited after the run.

Register: **RESEARCH**. No t/s, no "competes", no "lossless". No GPU, no model, no
network, no judge.

---

## 0 - THE QUESTION

From `experiments/track3-semiotic-codebook/NEW_IDEA_THE_CEREBELLAR_FRONT_CACHE_2026-09-01.md`,
section 6, build item 3, which the document names **GRANULE**:

> "**GRANULE** - does the SDR expansion make the cheap lookup more discriminative
> than a plain hash? The one place our existing `wikis/WIKI_SDR/canonical/` work
> becomes load-bearing."

and section 1, which is the claim under test:

> "Granule cells are the most numerous neuron type in the brain, and their whole job
> is **expansion recoding**: throw the input into a hugely higher-dimensional sparse
> space so that similar patterns become separable. **That is the SDR, and it is why
> the SDR is load-bearing rather than decorative** - it is the thing that makes a
> cheap lookup discriminative."

The question is a question because it may be a clean null.

## 1 - THE HONEST PRIOR, STATED FIRST

**Our own measurements are hostile to this family, and a null here is a real result.**

- **`WIKI/theory/167-the-content-control-a-frozen-random-table-performs-like-a-real-one-2026-08-31.md`
  line 68-70**: for a table injected into a frozen backbone, the **presence** effect is
  **exactly reproducible (sd 0 across seeds)** while the **content** effect's sign flips
  between seeds and its seed-to-seed sd **exceeds its own mean**. The presence effect is
  **27x-45x** the content effect (line 70). **A better addressing function may buy far
  less than it looks like it should**, because what dominates is whether the memory is
  there at all, not what is in it.
- **`experiments/track3-semiotic-codebook/MEASURED_track3_T3-69_the-encoder-was-never-the-axis-the-vote-was_2026-07-28.md`
  line 108**: "even taking the CI's upper bound, **+0.0174**, the encoder axis is **19x
  too small** to account for the -0.3256 the question is about", and line 31: **"AND THE
  AXIS IS THE VOTE."** The encoder was not the axis.
- **`experiments/track3-semiotic-codebook/MEASURED_track3_T3-58_the-counted-shortlist-holds-and-the-cost-claim-does-not_2026-07-27.md`
  line 127-130**: "**THE EXPLORATORY \"FOUR TIMES CHEAPER\" IS REFUTED** ... That is a tie,
  not a design win. T3-57's 4x came from a K chosen after the answer was known." A
  counted shortlist held while a cost claim did not.

So the prior direction is: **the plain hash wins, or the comparison is a null.** Any
SDR win has to survive that prior, and the null is recorded as a result of the same
standing as a win.

## 2 - WHAT IS BUILT, AND WHAT WAS NOT REBUILT

The existing machinery in `wikis/WIKI_SDR/canonical/` (surveyed first, per the brief)
already contains every piece this needs, and the harness **imports it read-only**
(`G_CANON_READONLY`, sha256 of the whole tree before and after):

| piece | file | what is used |
|---|---|---|
| sparse binary vector, random draw, union, overlap | `canonical/sdr.py` | `SDR.random(n,w,rng)` = "random SDR with exactly w active bits"; `SDR.union` = "bitwise OR ... the union property" |
| exact false-match rate | `canonical/sdr.py` | `false_match_probability(n,w,theta)`, "exact probability that a random w-of-n SDR overlaps a fixed one in >= theta bits", hypergeometric, exact integer arithmetic |
| address encoders and the sparse-default density | `canonical/address_encoders.py` | `encoder_registry()`'s `n_active` default, "~2% of n_bits (a sparse SDR-style density)" (line 272) |
| address-width / capacity theory | `canonical/sizing.py` | `recommend_address_width`, `critical_distance`, the 1/sqrt(d) noise floor |
| bit-identical SDR backends | `canonical/sdr_reps.py` | the representation layer, not needed here (uint64 packing is enough at these widths) |

**Not rebuilt.** No second SDR implementation, no second encoder, no second sampler.

## 3 - THE CORPUS, NAMED

The track-3 corpus, committed in the main work tree:
`experiments/track3-semiotic-codebook/corpus_artifacts/code_tokens.npy` (**1,324,191
int64 token ids**, `ds4.c`, tokenizer `models/nanochat/base-d20/tokenizer.pkl`, vocab
65,536) and `book_tokens.npy` (**314,997 tokens**, Collins *The Woman in White*).
Manifest: `experiments/track3-semiotic-codebook/CORPUS_MANIFEST.json`. Both streams are
run; **code is primary**, book is the replication.

Keys are **3-grams of token ids** - Engram's own addressing, which is what makes the
row prefetchable before layer 1 runs (`WIKI/theory/167` line 35-37: "Addresses are
3-grams of **token ids** ... Hashed to **65,536 slots** by multiplicative XOR").

No synthetic data, no external download, no model.

## 4 - THE TWO ADDRESS FUNCTIONS, AND THE ONE READOUT

```
  P(W)  PLAIN HASH                            G(m,k)  SDR EXPANSION
  ------------                                -----------------
  token t0 -> * C0                            token t0 -> SDR.random(m, k/3)
  token t1 -> * C1                            token t1 -> SDR.random(m, k/3)
  token t2 -> * C2                            token t2 -> SDR.random(m, k/3)
        |                                           |
        v                                           v
      XOR      (multiplicative XOR)               UNION      (bitwise OR)
        |                                           |
        v                                           v
  mix64 -> W bits                             exactly k of m bits active
        |                                           |
        +-------------------+-----------------------+
                            |
                            v
           ONE READOUT, IDENTICAL FOR BOTH ARMS:
           distance = popcount(store ^ query)   (numpy.bitwise_count, uint64)
           HIT iff distance <= D,  D fitted per arm on CALIB, frozen for REPORT
```

The readout is a **Hamming radius** for both arms, because that is the only form that
means the same thing for a dense hash code and a sparse code. An "agreeing bits"
similarity is **not** used: for a sparse code it is dominated by shared zeros and reads
`m` for any two codes at all, which would make the comparison meaningless. This was
caught in the instrument's own selftest before any run (`self_distance_zero_sdr`).

## 5 - THE TWO BUDGET NORMALIZATIONS

A "same budget" claim is ambiguous, so **both** readings are run and both are reported.

- **A - MATCHED BYTES.** Both addresses are `W` bits wide, so each table entry spends
  the same bytes. This is the deployment question: a front cache budgets bytes.
  `W in {12, 16, 20, 24, 32, 64, 128, 1024}`, `k = 3 * max(1, round(0.02*W/3))`.
- **B - MATCHED ENTROPY (primary).** Both arms can address `2**b` distinct states:
  `log2 C(m,k) = b` for the SDR, `b` bits for the hash. The SDR's raw address is then
  `m > b` bits - **expansion recoding proper, more neurons at the same information**.
  `b in {12, 16, 20, 24, 28}`, `k = 3` pinned (one active bit per token). A diagnostic
  `k = 6` variant at `b in {12, 16, 20, 24}` is also run: more active bits must lose
  more to a confusable key, and if it does not, the confusable metric is not measuring
  the bits.

## 6 - THE METRICS, FROZEN

Frozen constants (`granule_discrimination.py` section "FROZEN EXPERIMENT CONSTANTS",
mirrored here):

```
N_STORE = 2048     Q_CLASS = 1024     CHUNK = 128
CALIB_SPAN  = (0, 400000)          REPORT_SPAN = (400000, 800000)
ARM_A_WIDTHS = (12, 16, 20, 24, 32, 64, 128, 1024)
ARM_B_ENTROPY = (12, 16, 20, 24, 28)      ARM_B_K6_ENTROPY = (12, 16, 20, 24)
D_GRID_POINTS = 40    SESOI = 0.010    EQUAL_RATE = 0.05    SEED = 20260916
```

Classes, all disjoint from the store key set and **asserted** (`G_SPLIT_DISJOINT`):

- **hit** - the first 1024 distinct REPORT 3-grams that are also store keys.
- **FMR_rand** - the next 1024 distinct REPORT 3-grams that are **not** store keys.
- **FMR_conf** - for 1024 store keys, **exactly one of the three tokens replaced** with
  a different token id, keeping only rows that are not store keys. Index-aligned with
  the source key, which is what makes `NEAR_ENTRY` measurable.
- CALIB mirrors of all three, over `[300000, 400000)`, used for the radius fit only.

Per arm and budget:

```
TMR          fraction of hit queries with argmin == own entry and distance <= D
FMR_rand     fraction of unrelated queries with distance <= D
FMR_conf     fraction of confusable queries with distance <= D
DISCR        TMR - max(FMR_rand, FMR_conf)
NEAR_ENTRY   fraction of confusable queries whose nearest entry IS the key perturbed
NEAR_FAR     FMR_conf - NEAR_ENTRY
```

**`NEAR_ENTRY` is the load-bearing column and it is the same number as `FMR_conf` read
under the opposite policy.** In a front cache a near-miss must stay a miss
(`NEW_IDEA` line 67, "A cache you cannot miss cleanly is not a cache"), so `FMR_conf` is
the cost. If a tolerant policy were taken instead, `NEAR_ENTRY` is what it buys. A
tolerance whose errors are structured (`NEAR_ENTRY ~ FMR_conf`) is worth having; one
whose errors land at random (`NEAR_ENTRY ~ FMR_conf / N`) is not.

Radius selection: `D` fitted on CALIB by maximizing `DISCR`, from a ladder spanning
`[0, p99 of the confusable best-distance]`. **Ties at equal CALIB discrimination break
toward the SMALLER `D`** - a compact radius is the hash-friendly readout, so an SDR win
cannot be an artifact of the tie-break. Declared now, not chosen later.

## 7 - THE SEALED VERDICT RULE

`SESOI = 0.010`, the house `SESOI_strict` (`MEASURED_track3_T3-58` line 120:
"RESOLVED-NULL against the strict SESOI of 0.010"), applied to `DISCR` (a difference of
two rates, so directly in rate units).

On **ARM B, k = 3** (the fair expansion reading), 5 budgets x 2 streams = 10 contrasts,
`need = ceil(0.8 * 10) = 8`:

| outcome | rule |
|---|---|
| **SDR WINS** | `SDR_DISCR - HASH_DISCR > +0.010` at **>= 8 of 10** contrasts |
| **NULL** | `abs(SDR_DISCR - HASH_DISCR) <= 0.010` at **>= 8 of 10** contrasts |
| **PLAIN HASH WINS** | `SDR_DISCR - HASH_DISCR < -0.010` at **>= 8 of 10** contrasts |
| UNRESOLVED | none of the above |

ARM A is reported beside it as the deployment reading and adjudicates nothing on its own.

## 8 - THE SEALED PREDICTIONS, WRITTEN BEFORE THE RUN

**P1 (primary, the null): at matched ENTROPY the plain hash and the expansion are the
same on discrimination.** `abs(delta) <= 0.010` at >= 8 of 10 ARM B contrasts. The
mechanism, stated in advance so it can be wrong: at matched entropy **both arms can
address the same 2**b states**, so their D = 0 identity-collision rate is the same
`N / 2**b`, and each arm's own fitted radius rejects a confusable key, because a
confusable shares 2 of 3 tokens and so sits at roughly 2k/3 in the sparse arm, above
any radius that still holds `TMR ~ 0.99`. **Prediction: a NULL, and the expansion buys
nothing on discrimination.**

**P2: at matched BYTES the plain hash wins by more than the SESOI at every budget
below 64.** The expansion at `W` bits holds only `log2 C(W,k)` address states against
the hash's `2**W`, i.e. `~0.10-0.13 W` bits of address entropy for the same bytes.

**P3 (the SDR's win, in the other column): the expansion's false routings are
STRUCTURED and the hash's are not.** At the radius where `FMR_conf` first reaches 0.05,
the sparse arm's `NEAR_ENTRY / FMR_conf` will exceed the hash's by **more than 3x**, and
`NEAR_ENTRY` itself will exceed the hash's by more than the SESOI, at >= 8 of 10
contrasts. The hash also has a near-miss rate at a radius above zero - but its argmin is
a **random** entry, so its tolerance buys nothing. **This is the prediction that makes
the null readable: if the expansion does not win THIS, the instrument is blind to the
very mechanism it is testing and no null from it may be quoted.**

**P4 (the analytic cross-check, and it can fail): the measured `FMR_rand` at the fitted
radius will agree with the closed form to within 20 percent relative.** For the hash,
`1 - (1 - sum_{d<=D} C(W,d) 2**-W)**N`. For the sparse arm at `D = 0` only, the owned
form `1 - (1 - false_match_probability(m,k,k))**N`. **For D > 0 no analytic is claimed
for the sparse arm** - the union of three random token codes is not a uniform k-of-m
draw and there is no exact closed form here - and the harness records an explicit
`null` rather than a number in that case. If P4 fails at D = 0, **the instrument is
wrong, not the theory.**

## 9 - THE CONTROLS, ALL IN THE SAME RUN

| control | what it must show |
|---|---|
| **PC_identity** (negative) | the same codes on both sides read identically - the metric can report "no difference" |
| **PC_sparse_near** (positive) | a 1-token perturbation of a sparse code stays near (`median distance <= 4` at m=1024, k=6) |
| **PC_dense_far** (positive) | the same perturbation of a hash code does not (`median distance >= 450`) |
| **PC_unrelated_far** (positive) | an unrelated sparse query does not land near (`median distance >= 8`) |
| **PC_overfull** (positive) | a 16-slot table with 256 keys saturates at `FMR ~ 1` |
| **G_CANON_READONLY** (gate) | sha256 of `wikis/WIKI_SDR/canonical/` identical before and after |
| **G_SPLIT_DISJOINT** (gate) | no query class intersects the store key set |
| **G_CANON_UNION** (gate) | the packed 3-gram address equals `canonical.SDR.union` on the first 200 rows |
| **G_DETERMINISM** (gate) | same seed gives identical addresses and distances |

Every null quotes the positive the identical instrument returned in the same run.

## 10 - BUILD GATE: NOT APPLICABLE

**This work is CPU-only and offline.** numpy plus the repo's own canonical SDR toolkit,
on the local token streams. No GPU is touched, no CUDA build is needed, no model is
loaded, and the Spark is running an unrelated series and is not touched. There is
therefore **no build gate to copy and none is claimed** - a CUDA build gate here would
be a borrowed criterion, not a check.

## 11 - THE SWITCH: NONE

This is an offline experiment, not engine code. **There is no engine switch and no knob
is added to `ds4.c` or any other engine file.** Nothing in this branch changes the
engine, its binaries, or its behaviour. No `DS4_*` environment variable is introduced.

## 12 - WHERE IT WOULD BE USED IF IT WORKED

The front-cache addresses we actually have: a **hot-list address** (the expert hot list,
`triple-hotlist`), an **Engram hash address** (the 3-gram to 65,536 slots step,
`WIKI/theory/167` line 37), and a **draft-table address** (the suffix-lookup drafter).
The comparison is at those budgets, over those key types.

---

# AMENDMENT-1 - PRE-RUN INSTRUMENT REPAIRS (no measurement existed)

**Filed before any number was produced, and it changes no prediction, no metric
definition, no threshold and no ladder.** Four build failures were hit by running the
harness, and the repairs are recorded here rather than made silently, per the T3-58
precedent (`MEASURED_track3_T3-58` PART 0: "One build decision is recorded ... because
it would otherwise have been silent"). Its first run raised; it did not measure.

1. **Per-stream spans.** `book_tokens.npy` is 314,997 tokens, so the code stream's
   `REPORT_SPAN = (400000, 800000)` ran off the end of it (`slice too short for
   3-grams`). Spans are now per stream: code `calib (0, 400000)` / `report (400000,
   1324191)`; book `calib (0, 100000)` / `report (100000, 314997)`. The splits remain
   disjoint by position and the disjointness gate is unchanged.
2. **Query class size is a ceiling, not a demand.** The hit class is bounded by how
   many store keys actually RECUR in the held-out span, which is a property of the
   corpus and not of the design: the code stream yields 1,043 recurring keys, the book
   stream 477. Classes are now `min(Q_CLASS, available)` with a floor of `Q_MIN = 256`,
   and the actual `n` per class is reported with every rate. `Q_CLASS = 1024` is
   unchanged as the ceiling.
3. **The split gate was wrong, and it fired on the hit class.** `G_SPLIT_DISJOINT`
   asserted that every class is disjoint from the store - but a HIT is by definition a
   key that IS in the store. The gate now applies to the three non-store classes
   (`fr`, `fc`, `c_fr`, `c_fc`) and a separate `G_HITS_IN_STORE` asserts the opposite
   containment for the hit class.
4. **A saturated table is not an error.** At the smallest budgets two store keys can
   share an address, so the nearest entry to a hit query is legitimately a different
   index and no CALIB hit finds its own entry. That is a measured property of a dense
   table, so the fit now reports `TMR` at every radius instead of raising, and records
   `n_calib_hits_self`. No budget, ladder or threshold moved.

# AMENDMENT-2 - CODE BROUGHT BACK TO THE SEALED DEFINITION OF `NEAR_ENTRY`

**A 2.2 s pipeline smoke on three of the twenty budgets was run before the run of
record, and it found `NEAR_ENTRY` reading the wrong query set.** As implemented it
read the similarity of the SOURCE keys (which are in the store) rather than of the
PERTURBED queries, so it could exceed `FMR_conf` - impossible, since a nearest entry
cannot be both the perturbed-from entry and not be a false routing.

The **sealed definition is unchanged**: section 6 says "`NEAR_ENTRY` - fraction of
confusable queries whose nearest entry IS the key perturbed". Section 6 also says
"`NEAR_FAR` = `FMR_conf` - `NEAR_ENTRY`", which is only coherent if the two are read
off the SAME argmin array. The fix makes the code do exactly that: both are now taken
from one `best_of` call over the confusable queries. `NEAR_ENTRY` is a subset of
`FMR_conf` **by construction** after the repair, and that is asserted in the selftest
(`planted_near_structure_exists`).

**No prediction, no metric definition, no SESOI, no verdict rule and no ladder was
changed by either amendment.** No pre-registered number was moved. The three pilot
budgets produced by the broken build are named in the MEASURED record and were not
used for anything.

**A second mismatch, same class, repaired here.** P3 (section 8) is stated **at the
radius where `FMR_conf` first reaches `EQUAL_RATE = 0.05`**, which is a point on the
REPORT curve and not the CALIB-fitted radius. The first build adjudicated P3 at the
fitted radius, where both arms sit at `D = 0` and `NEAR_ENTRY` is ~0 for both by
construction (a one-token perturbation is never at distance 0 from its source key).
The adjudication now reads the pre-registered quantity off the curve
(`structure_at_rate`), which is where P3 always said to read it. The fitted-radius
`NEAR_ENTRY` is still reported, labelled as the degenerate reading it is.

**A third mismatch, same class.** The P3 ratio test read `NEAR_ENTRY_share_of_conf`
for truthiness, so a share of exactly `0.0` - which is what the plain hash produces,
and the whole point of the comparison - was read as MISSING rather than as a measured
zero. Tested as `is not None` so a measured zero is compared as a zero. The repair can
only move the count TOWARD P3 holding, and every raw `NEAR_ENTRY` behind it is in the
verdict rows so a reader can check the count independently.

