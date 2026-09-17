# MEASURED - GRANULE: the sparse-distributed expansion is NOT the more discriminative address

**Branch `triple-granule`, 2026-09-16. CPU only, offline, no GPU, no model, no network,
no judge. Wall 10.0 s on this box.**

**PREREG sealed first.** `PREREG_GRANULE_2026-09-16.md` was committed at `11dc10e92`
with `README.md` untouched, **before any number was produced**. The predictions, the
metrics, the two budget normalizations, the `SESOI` and the verdict rule in that file
were not moved. Two amendments are recorded in it: **AMENDMENT-1** (four crash-driven
instrument repairs) and **AMENDMENT-2** (three places where the code did not implement
the sealed text). Neither touched a prediction, a metric definition, a threshold or a
ladder; every one of them is a repair to the instrument and each is named below with
the raw numbers left visible so a reader can check the count independently.

Question, from `experiments/track3-semiotic-codebook/NEW_IDEA_THE_CEREBELLAR_FRONT_CACHE_2026-09-01.md`
section 6, build item 3:

> "**GRANULE** - does the SDR expansion make the cheap lookup more discriminative than a
> plain hash? The one place our existing `wikis/WIKI_SDR/canonical/` work becomes
> load-bearing."

---

## THE ANSWER

> ### VERDICT: `PLAIN_HASH_WINS`.
>
> On the **primary arm, matched ADDRESS ENTROPY with `k = 3`**, the plain hash is more
> discriminative at **10 of 10 contrasts** (5 budgets x 2 streams), every one by more
> than the pre-registered `SESOI = 0.010`. The deltas are **-0.0414 to -0.2629**.
> `0` SDR wins, `0` nulls.
>
> On the **secondary arm, matched BYTES**, it is **16 of 16**, deltas **-0.0294 to
> -1.6357**.
>
> ### AND THE SEALED PREDICTION P1 WAS A NULL. IT FAILED.
>
> P1 predicted `abs(delta) <= 0.010` at 8 of 10. It measured a hash win at 10 of 10.
> **A prediction that fails in the direction the prior predicted is still a failed
> prediction, and it is recorded as one.**
>
> ### THE MECHANISM IS NAMED, AND IT IS NOT "HASHES ARE BETTER".
>
> The expansion's **effective address space plateaus near `2**15`** while the hash's
> tracks its nominal width. Section 5 has the measurement and the cause: with one active
> bit per token, a token id is reduced to one of `m` bits, and the table's 2048 keys are
> drawn from only **589 distinct token ids**, so **61 of the 2048 keys are unavoidable
> duplicate addresses** at every width from `m = 467` to `m = 1174`. The hash has **zero**
> collisions among the same 2048 keys from `b = 20` up.

---

## 1 · WHAT WAS RUN

Two address functions over the same key type, read through the **same** readout.

```
  P(W)  PLAIN HASH                            G(m,k)  SDR EXPANSION
  ------------                                -----------------
  token t0 -> * C0 = 0x2545F4914F6CDD1D       token t0 -> SDR.random(m, k/3)
  token t1 -> * C1 = 0x9E3779B97F4A7C15       token t1 -> SDR.random(m, k/3)
  token t2 -> * C2 = 0xC2B2AE3D27D4EB4F       token t2 -> SDR.random(m, k/3)
        |                                           |
        v   XOR the three (multiplicative XOR)      v   UNION the three (bitwise OR)
        |                                           |
        v   splitmix64 finalizer, one per word      v   exactly k of m bits active
  W-bit code                                  m-bit code, k << m
        |                                           |
        +---------------------+---------------------+
                              |
                              v
                 ONE READOUT, IDENTICAL FOR BOTH ARMS
                 distance = popcount(store ^ query)      (numpy.bitwise_count, uint64)
                 HIT iff distance <= D, D fitted per arm on CALIB, frozen for REPORT
```

Key = **3-gram of token ids**, Engram's own addressing (`WIKI/theory/167-...-2026-08-31.md`
line 35: "Addresses are 3-grams of **token ids** ... Hashed to **65,536 slots** by
multiplicative XOR", line 37). The plain hash is that multiplicative-XOR form.

**Corpus, and it is the track-3 corpus, committed in the main work tree and only read
here:** `experiments/track3-semiotic-codebook/corpus_artifacts/code_tokens.npy`
(1,324,191 int64 token ids, `ds4.c`, tokenizer `models/nanochat/base-d20/tokenizer.pkl`,
vocab 65,536) and `book_tokens.npy` (314,997 tokens, Collins *The Woman in White*),
manifest `experiments/track3-semiotic-codebook/CORPUS_MANIFEST.json`. **Code is
primary, book is the replication.** No synthetic data, no download, no model.

Store `N = 2048` entries. Query classes, all disjoint from the store and asserted:
`hits` 1024 (code) / 477 (book) recurring keys, `FMR_rand` 1024 never-stored unrelated
keys, `FMR_conf` 1024 / 477 keys with **exactly one of the three tokens replaced**.
Radius `D` is fitted per arm on the disjoint CALIB classes only, ties toward the
**smaller** `D` (the hash-friendly choice, declared in the PREREG section 6).

`G_CANON_READONLY` PASS: sha256 of `wikis/WIKI_SDR/canonical/` identical before and
after (`15d9a3...` recorded in the run JSON). `G_SPLIT_DISJOINT`, `G_HITS_IN_STORE`,
`G_CANON_UNION` (the packed address equals `canonical.SDR.union` row for row on the
first 200 rows), `G_DETERMINISM`, `G_NEAR_SUBSET_CONF` all PASS. Selftest **21/21**.

## 2 · THE PRIMARY ARM: MATCHED ADDRESS ENTROPY, `k = 3`

`log2 C(m,k) = b` for the expansion, `b` bits for the hash: the same number of
addressable states, the expansion spending more raw bits to get there. This is
expansion recoding as the document defines it.

| stream | `b` | `m` | raw bits hash/SDR | SDR nominal entropy | hash TMR | hash FMR_rand | hash FMR_conf | SDR TMR | SDR FMR_rand | SDR FMR_conf | `dDISCR` |
|---|---|---|---|---|---|---|---|---|---|---|---|
| code | 12 | 31 | 12 / 31 | 12.13 | 0.7744 | 0.3789 | 0.4082 | 0.6592 | 0.4551 | 0.4531 | **-0.1621** |
| code | 16 | 75 | 16 / 75 | 16.04 | 0.9824 | 0.0430 | 0.0332 | 0.8975 | 0.1221 | 0.1133 | **-0.1641** |
| code | 20 | 186 | 20 / 186 | 20.01 | 1.0000 | 0.0010 | 0.0020 | 0.9395 | 0.0410 | 0.0352 | **-0.0996** |
| code | 24 | 467 | 24 / 467 | 24.01 | 1.0000 | 0.0000 | 0.0000 | 0.9580 | 0.0156 | 0.0088 | **-0.0576** |
| code | 28 | 1174 | 28 / 1174 | 28.00 | 1.0000 | 0.0000 | 0.0000 | 0.9619 | 0.0078 | 0.0000 | **-0.0459** |
| book | 12 | 31 | 12 / 31 | 12.13 | 0.7799 | 0.3828 | 0.3753 | 0.6017 | 0.4492 | 0.4675 | **-0.2629** |
| book | 16 | 75 | 16 / 75 | 16.04 | 0.9748 | 0.0244 | 0.0356 | 0.8491 | 0.0957 | 0.0985 | **-0.1887** |
| book | 20 | 186 | 20 / 186 | 20.01 | 0.9979 | 0.0020 | 0.0021 | 0.9308 | 0.0312 | 0.0398 | **-0.1048** |
| book | 24 | 467 | 24 / 467 | 24.01 | 1.0000 | 0.0000 | 0.0000 | 0.9455 | 0.0156 | 0.0126 | **-0.0701** |
| book | 28 | 1174 | 28 / 1174 | 28.00 | 1.0000 | 0.0000 | 0.0000 | 0.9665 | 0.0078 | 0.0063 | **-0.0414** |

**`dDISCR` = SDR DISCR minus hash DISCR, `DISCR` = TMR - max(FMR_rand, FMR_conf).**
All ten are negative and all ten exceed the `SESOI` of 0.010 in magnitude.
The verdict rule (`PREREG` section 7) is applied mechanically: `PLAIN_HASH_WINS`.

The `k = 6` diagnostic behaved as `PREREG` section 5 said it would, and it buys the
expansion most of a budget back by shrinking `m`: `dDISCR` -0.1211 / -0.0898 / -0.0518
on code and -0.1405 / -0.0704 / -0.0419 on book at `b` of 16 / 20 / 24, and it is the
only place anywhere in this run where the expansion is not behind, at `b = 12`
(`+0.0029` code, `+0.0201` book). **At `b = 12` both arms are at the collision floor
(`m = 15` for `k = 6`, so 2048 keys over at most `C(15,6) = 5005` states) and the hash
is not a clean reference there. It is reported and it is not claimed as a win.**

## 3 · THE SECONDARY ARM: MATCHED BYTES

Both addresses `W` bits wide, so each table entry spends the same bytes. The deployment
reading: a front cache budgets bytes.

| stream | `W` | `k` | hash TMR | hash FMR_rand | SDR TMR | SDR FMR_rand | SDR nominal entropy | `dDISCR` |
|---|---|---|---|---|---|---|---|---|
| code | 12 | 3 | 0.7744 | 0.3789 | 0.1016 | 0.9883 | 7.78 | **-1.2529** |
| code | 16 | 3 | 0.9824 | 0.0430 | 0.2285 | 0.9248 | 9.13 | **-1.6357** |
| code | 20 | 3 | 1.0000 | 0.0010 | 0.3848 | 0.8125 | 10.15 | **-1.4258** |
| code | 24 | 3 | 1.0000 | 0.0000 | 0.5088 | 0.6680 | 10.98 | **-1.1592** |
| code | 32 | 3 | 1.0000 | 0.0000 | 0.6816 | 0.4414 | 12.28 | **-0.7598** |
| code | 64 | 3 | 1.0000 | 0.0000 | 0.8838 | 0.1475 | 15.35 | **-0.2637** |
| code | 128 | 3 | 1.0000 | 0.0000 | 0.9307 | 0.0674 | 18.38 | **-0.1367** |
| code | 1024 | 21 | 1.0000 | 0.0000 | 0.9678 | 0.0059 | 144.23 | **-0.0381** |
| book | 12 | 3 | 0.7799 | 0.3828 | 0.1216 | 0.9902 | 7.78 | **-1.2734** |
| book | 16 | 3 | 0.9748 | 0.0244 | 0.2201 | 0.9082 | 9.13 | **-1.6273** |
| book | 20 | 3 | 0.9979 | 0.0020 | 0.3983 | 0.7646 | 10.15 | **-1.3774** |
| book | 24 | 3 | 1.0000 | 0.0000 | 0.4654 | 0.6104 | 10.98 | **-1.1449** |
| book | 32 | 3 | 1.0000 | 0.0000 | 0.6101 | 0.4482 | 12.28 | **-0.8512** |
| book | 64 | 3 | 1.0000 | 0.0000 | 0.8239 | 0.1602 | 15.35 | **-0.3501** |
| book | 128 | 3 | 1.0000 | 0.0000 | 0.8994 | 0.0537 | 18.38 | **-0.1543** |
| book | 1024 | 21 | 1.0000 | 0.0000 | 0.9706 | 0.0000 | 144.23 | **-0.0294** |

`SDR nominal entropy` = `log2 C(W,k)`: how many address states the expansion's code
could hold if its addresses were drawn uniformly from all `k`-subsets of `W` bits.
They are not, and section 5 is why.

## 4 · THE ONE COLUMN THE EXPANSION WINS, AND WHAT IT COSTS

`NEAR_ENTRY` is read at the radius where `FMR_conf` first reaches `EQUAL_RATE = 0.05`,
which is where `PREREG` P3 said to read it.

| stream | `b` | hash `FMR_conf` | hash `NEAR_ENTRY` | hash share | SDR `FMR_conf` | SDR `NEAR_ENTRY` | SDR share |
|---|---|---|---|---|---|---|---|
| code | 12 | 0.4082 | 0.0000 | 0.000 | 0.4531 | 0.0215 | 0.047 |
| code | 16 | 0.4063 | 0.0010 | 0.002 | 0.1133 | 0.0098 | 0.086 |
| code | 20 | 0.3135 | 0.0000 | 0.000 | 0.2227 | 0.0391 | 0.175 |
| code | 24 | 0.2393 | 0.0000 | 0.000 | 0.1621 | 0.0332 | 0.205 |
| code | 28 | 0.1416 | 0.0000 | 0.000 | 0.1367 | 0.0332 | 0.243 |
| book | 12 | 0.3753 | 0.0000 | 0.000 | 0.4675 | 0.0231 | 0.049 |
| book | 16 | 0.3941 | 0.0000 | 0.000 | 0.0985 | 0.0147 | 0.149 |
| book | 20 | 0.3166 | 0.0000 | 0.000 | 0.1950 | 0.0398 | 0.204 |
| book | 24 | 0.2264 | 0.0000 | 0.000 | 0.1321 | 0.0294 | 0.222 |
| book | 28 | 0.1593 | 0.0000 | 0.000 | 0.0629 | 0.0168 | 0.267 |

**P3 HELD, and it is the reason the negative above is readable at all.** The expansion's
false routings carry structure: `NEAR_ENTRY` exceeds the hash's by more than the `SESOI`
at **9 of 10** contrasts, and its share of its own confusions is more than 3x the hash's
at **10 of 10** (the hash's share is a measured `0.000` in 9 of 10 cases). The
pre-registered readability condition `structure_positive_control_readable` is therefore
`True`, which means the instrument **can** resolve the mechanism it was pointed at.

**And the honest bound on that win.** The structure covers only **0.047 to 0.267** of the
expansion's own false routings; the rest still land on an arbitrary entry. In absolute
terms the expansion is routing 1.0 to 4.0 percent of confusable queries to the right
entry, against the hash's 0.0 to 0.1 percent. **`NEAR_ENTRY` is the same number as
`FMR_conf` under the other policy** (`NEW_IDEA` line 67: "A cache you cannot miss cleanly
is not a cache. It is a bug"), so this is not a free good: it is 1 to 4 percent of
never-stored keys being answered with a stored key's entry, structured rather than
arbitrary. The hash does not have that failure mode at all.

## 5 · WHY THE EXPANSION LOSES: THE EFFECTIVE ADDRESS SPACE PLATEAUS

`1 - TMR` is the rate at which a HIT query's nearest entry is a different index, which
happens exactly when another stored key shares its address. For a table of `N` over a
space of `S` states, only the earlier-index half of a colliding pair is observed, so
`S = 0.5 * N / (1 - TMR)`. `1 - TMR` has `n = 1024` so it is measurable on a base near 1
with no floor problem, unlike an `FMR_rand` that reads `0/1024`.

| stream | `b` | hash `1-TMR` | hash effective bits | SDR `1-TMR` | SDR effective bits | SDR **nominal** bits | shortfall |
|---|---|---|---|---|---|---|---|
| code | 12 | 0.2256 | 12.15 | 0.3408 | 11.55 | 12.13 | 0.58 |
| code | 16 | 0.0176 | 15.83 | 0.1025 | 13.29 | 16.04 | **2.76** |
| code | 20 | 0.0000 | > 20.0 | 0.0605 | 14.05 | 20.01 | **5.96** |
| code | 24 | 0.0000 | > 20.0 | 0.0420 | 14.57 | 24.01 | **9.43** |
| code | 28 | 0.0000 | > 20.0 | 0.0381 | 14.71 | 28.00 | **13.29** |
| book | 12 | 0.2201 | 12.18 | 0.3983 | 11.33 | 12.13 | 0.81 |
| book | 16 | 0.0252 | 15.31 | 0.1509 | 12.73 | 16.04 | **3.32** |
| book | 20 | 0.0021 | 18.90 | 0.0692 | 13.85 | 20.01 | **6.16** |
| book | 24 | 0.0000 | > 20.0 | 0.0545 | 14.20 | 24.01 | **9.81** |
| book | 28 | 0.0000 | > 20.0 | 0.0335 | 14.90 | 28.00 | **13.11** |

**The hash's effective space tracks its width; the expansion's stops near `2**15`,
whatever `m` is.** At `W = 1024` in the matched-BYTES arm the expansion's effective space
is **14.96 bits** against a nominal **144.23** (code; 15.09 against 144.23 on book), and
the hash still reads `1 - TMR = 0.0000`.

**The cause, measured directly rather than argued.** The store's 2048 keys are drawn from
**589 distinct token ids**. With one active bit per token, a token id is reduced to one
of `m` bits, and two distinct token ids that share a bit make two distinct 3-grams
indistinguishable. Counting the distinct address codes among the store's own 2048 keys:

| `b` | `m` | `log2 C(m,k)` | hash distinct codes | SDR distinct codes | duplicate SDR codes |
|---|---|---|---|---|---|
| 12 | 31 | 12.13 | 1595 | 1411 | 637 |
| 16 | 75 | 16.04 | 2012 | 1844 | 204 |
| 20 | 186 | 20.01 | **2048** | 1952 | 96 |
| 24 | 467 | 24.01 | **2048** | 1980 | 68 |
| 28 | 1174 | 28.00 | **2048** | 1987 | **61** |
| `b=24, k=6` | 51 | 24.01 | **2048** | 1987 | **61** |

**61 of the 2048 keys are unavoidable duplicate addresses at every width from `m = 467`
to `m = 1174`, and also at `k = 6`.** Widening the expansion by 2.5x, and doubling the
per-token active count, does not remove one of them, because the duplication is in the
token-to-bit map and not in the width. The hash reaches 2048 distinct codes from `b = 20`
up. **That is the whole result: at matched entropy the expansion has fewer usable
addresses than the hash, not more, and it cannot close the gap by expanding.**

## 6 · THE SEALED PREDICTIONS, SCORED

| # | sealed claim | outcome | measured |
|---|---|---|---|
| **P1** | ratio null on the primary arm, `abs(delta) <= 0.010` at >= 8 of 10 | **FAILED** | hash wins 10 of 10, deltas -0.0414 to -0.2629 |
| **P2** | hash wins matched BYTES by > SESOI at every budget below 64 | **HELD** | 16 of 16, deltas -0.0294 to -1.6357 |
| **P3** | the expansion's false routings are structured, share > 3x the hash's and `NEAR_ENTRY` > hash + SESOI at >= 8 of 10 | **HELD** | 10 of 10 on the ratio, 9 of 10 on the absolute |
| **P4** | measured `FMR_rand` within 20 percent relative of the closed form | **FAILED, both arms, two different reasons** | see section 7 |

**P4 is the most useful failure here and it fails in both directions for different causes.**

*The hash.* Where the closed form is above the resolution floor it agrees:
at `b = 12`, measured 0.3789 against `1 - (1 - 2**-12)**2048 = 0.3935`, that is
**-3.7 percent** with a 1-standard-error of **3.9 percent**. At `b = 16` it is
**+39.7 percent** with a 1-SE of **17.5 percent** (2.3 SE high). The sealed 20 percent
**relative** tolerance was under-powered by construction: at `p = 0.03` and `n = 1024`,
one standard error is already 17.5 percent relative, and at `b = 20` it is 70.7 percent.
**The hash did not disagree with the theory; the tolerance could not have resolved a
disagreement at those budgets.** That is a pre-registration error, not a physics result.

*The expansion.* The owned closed form **understates the measured collision rate, and the
error grows with `m`**:

| stream | `b` | SDR `m` | measured `FMR_rand` | `1 - (1 - fmp(m,k,k))**N` | ratio |
|---|---|---|---|---|---|
| code | 12 | 31 | 0.4551 | 0.3660 | 1.24x |
| code | 16 | 75 | 0.1221 | 0.0299 | **4.09x** |
| code | 20 | 186 | 0.0410 | 0.0019 | **21x** |
| code | 24 | 467 | 0.0156 | 0.00012 | **129x** |
| code | 28 | 1174 | 0.0078 | 0.0000080 | **1025x** |

`canonical.sdr.false_match_probability(n, w, theta)` is exact for **two independent
`w`-of-`n` draws** (`wikis/WIKI_SDR/canonical/sdr.py` line 272: "Exact probability that
a random w-of-n SDR overlaps a fixed one in >= theta bits", and the identity event is
`overlap == w`). **The expansion's addresses are not independent `k`-of-`m` draws**, and
this branch is the measurement that says so: the per-token sub-codes come from a
`k_tok`-of-`m` alphabet with far fewer members than the token vocabulary, so the composed
addresses are clustered. Any future sizing that reads `log2 C(m,k)` as an address budget
will overstate the expansion by the shortfall in section 5.

## 7 · LIMITATIONS, STATED

1. **One expansion construction, not all of them.** The address is the union of three
   random per-token codes, which is the canonical SDR operation the document names and
   which `canonical.SDR.union` implements. A construction with a larger per-token
   alphabet (`k_tok` of `m` with `m` large) would not suffer the token-to-bit collapse of
   section 5, and the `k = 6` diagnostic is the direction that points. **This branch does
   not test that construction**, and the `k = 6` arm at `b = 12` is the only point in the
   whole run where the expansion is ahead.
2. **The verdict is ARM B at `k = 3`, as sealed.** The `k = 6` diagnostic is reported and
   adjudicates nothing.
3. **The hit class is corpus-bounded.** The code stream yields 1,043 recurring store keys
   and the book stream 477, so the hit class is 1024 and 477 rather than the sealed
   ceiling of 1024 for both (AMENDMENT-1 item 2). Every rate is reported with its class
   size; `1 - TMR` at `n = 477` has a 1-SE of about 0.021 at `p = 0.04`.
4. **`NEAR_ENTRY` is not a second win.** It is the same number as `FMR_conf` read under
   the near-miss-tolerant policy, and it is 1 to 4 percent, not 20 percent.
5. **The radius fitted to `D = 0` on both arms at most budgets.** That is the arms'
   choice, not a constraint: the ladder spans `[0, p99 of the confusable best-distance]`
   and the hash's advantage is largest at the tight radius. The consequence is that
   `NEAR_ENTRY` at the FITTED radius is ~0 for both arms by construction (a one-token
   perturbation is never at distance 0 from its source), which is why P3 is adjudicated
   off the curve at `FMR_conf = 0.05`.
6. **One key type, one value semantics.** Keys are 3-grams of token ids and the metric is
   address-level, which is what a front cache decides on. Values are not scored, so
   nothing here speaks to whether a retrieved entry is *correct* in content terms.
7. **Two streams, one tokenizer.** Both corpora are tokenized with
   `models/nanochat/base-d20/tokenizer.pkl` (vocab 65,536) and the two streams agree on
   every verdict, which is a replication and not a generalisation.

## 8 · WHAT THIS HANDS THE FRONT CACHE

1. **Do not buy discrimination with expansion recoding.** At a matched budget the address
   function that uses every bit as a bit is more discriminative than the one that spends
   bits on a sparse code, on both readings of "matched budget", on both corpora, at every
   budget, by 0.03 to 1.64 in `TMR - max(FMR)`.
2. **The expansion's one real property is that its errors are structured**, and in a
   front cache that property *is* a false-positive rate. `NEW_IDEA` section 3 law 2
   ("a miss must be FREE") is the reason: a structured miss still costs the price of a
   hit if the cache answers with it.
3. **`log2 C(m,k)` is not an address budget.** An expansion built from a per-token code
   over a small alphabet holds far fewer usable addresses than its nominal combinatorial
   count, and the gap widens with width. Measure the effective space, do not derive it.
4. **The three real front-cache addresses stay as they are.** A hot-list address
   (`triple-hotlist`), an Engram hash address (`WIKI/theory/167` line 37) and a
   draft-table address are all plain-hash-shaped, and this branch's result is that the
   shape is the right one for discrimination.

## 9 · WHAT THIS DOES NOT TOUCH

**No engine change and no switch.** There is no `DS4_*` variable, no file added to
`ds4.c`, and no knob anywhere. This is an offline experiment over committed token
streams; the engine is byte-identical to `triple-antirez-tip-latest`.

**No GPU, no model, no network, no judge, no Spark.** The box's token streams and
numpy. `G_CANON_READONLY` held.

**BUILD GATE: NOT APPLICABLE.** CPU-only and offline, so no CUDA gate was copied and
none is claimed. A build gate here would be a borrowed criterion, not a check.

## 10 · PROVENANCE

| commit | what |
|---|---|
| `11dc10e92` | **`UP`** - the harness, the `.gitignore` and the **sealed PREREG**. `README.md` untouched. |
| `9c0c66cb9` | `UP:` pre-run instrument repairs, AMENDMENT-1 and AMENDMENT-2 part 1. No measurement existed. |
| `497ed4d3b` | `UP:` P3 adjudication moved onto the REPORT curve, AMENDMENT-2 part 2. No measurement existed. |
| `466ad0ed8` | `UP:` P3 ratio test reads a measured zero as zero, AMENDMENT-2 part 3. No measurement existed. |
| *(the run)* | **`python3 experiments/granule/granule_discrimination.py --run`, at `466ad0ed8`, wall 10.0 s**, exit 0, selftest **21/21 PASS**. Log and JSON committed beside this file. |

Hashes of the two artifacts that ride with the numbers:

```
  PREREG_GRANULE_2026-09-16.md  sha256 c3008810c30391949e2a031f87b0674eb2255b935e9f5c088b0a3cc545791547
  granule_discrimination.py     sha256 81f6ccf9b26bbd98d6f1e2e722dc96abab537c25af27245609c183a6299b81a7
  wikis/WIKI_SDR/canonical/     sha256 83de49d11e8113bfc09d49c63c3dc084be953324404077b71d969f4fa0f834d7
```

`prereg_constants.canon_sha256_before` and `canon_sha256_after` are both
`83de49d1...` in `granule_results.json`, which is the `G_CANON_READONLY` proof:
the canonical toolkit was imported read-only and nothing under it moved.

Every rate in this file is on a class of `n = 1024` (code) or `n = 477` (book) except
the `FMR_rand` classes, which are `n = 1024` on both. `1 - TMR` is quoted with its
standard error where it decides a claim.

**The prediction was written first, and it was wrong.** The prior was right, and the
mechanism is named.
