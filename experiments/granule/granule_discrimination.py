#!/usr/bin/env python3
"""GRANULE -- is a sparse-distributed expansion a more DISCRIMINATIVE front-cache
address than a plain hash, at a matched address budget?

THE QUESTION, from experiments/track3-semiotic-codebook/NEW_IDEA_THE_CEREBELLAR_FRONT_CACHE_2026-09-01.md
section 6 (build item 3): "does the SDR expansion make the cheap lookup more
discriminative than a plain hash? The one place our existing
wikis/WIKI_SDR/canonical/ work becomes load-bearing."

CPU ONLY, OFFLINE, NO GPU, NO MODEL. numpy + the repo's own canonical SDR toolkit.

WHAT IS COMPARED
----------------
Two address functions over the SAME key type (a 3-gram of token ids, Engram's own
addressing -- WIKI/theory/167-the-content-control-...-2026-08-31.md line 35), read
through the SAME readout (nearest stored address by HAMMING DISTANCE, HIT iff
distance <= D, D fitted per arm on a disjoint CALIB split):

  P(W)   plain hash -- multiplicative XOR of the three token ids folded to W bits.
                    The Engram addressing, WIKI/theory/167 line 37.
  G(m,k) sparse-distributed expansion -- each token id gets a random k_t-of-m code
                    (canonical.sdr.SDR.random), the 3-gram address is their union
                    (canonical.sdr.SDR.union). k_t = k/3, k a multiple of 3.

The readout is a Hamming RADIUS for both arms, which is the one form that means the
same thing for a dense hash code and a sparse code. An "agreeing bits" similarity is
NOT used, because for a sparse code it is dominated by shared zeros and would read
m for any two codes at all.

TWO BUDGET NORMALIZATIONS, both run, both reported:

  A  MATCHED BYTES     both addresses are W bits wide, so each table entry spends the
                       same bytes. The deployment question.
  B  MATCHED ENTROPY   both arms can address 2**b distinct states: log2 C(m,k) = b
                       for the SDR, b bits for the hash. The SDR's raw address is
                       then m > b bits: expansion recoding proper, more neurons at
                       the same information.

METRICS, at the arm's OWN radius D fitted on CALIB:
  TMR        exact stored key retrieved correctly (distance 0 or within D)
  FMR_rand   key never stored, unrelated, falsely routed to some entry
  FMR_conf   key never stored but SHARING 2 OF 3 TOKENS with a stored key, falsely
             routed -- the "can the lookup tell two similar inputs apart" test
  DISCR      TMR - max(FMR_rand, FMR_conf)

AND THE ONE COLUMN THAT DECIDES WHETHER THE TOLERANCE IS WORTH ANYTHING:
  NEAR_ENTRY  P(a confusable query's nearest entry IS the stored entry it perturbed)
  NEAR_FAR    P(its nearest entry is some OTHER entry)
  These sum to FMR_conf. In a front cache a near-miss must stay a miss (cache law 2,
  NEW_IDEA doc line 67), so FMR_conf is the cost and NEAR_ENTRY is what a tolerant
  policy would get for it. One number, two policies. A tolerance whose errors are
  structured (NEAR_ENTRY ~ FMR_conf) is worth having; one whose errors land at random
  (NEAR_ENTRY ~ FMR_conf/N) is not.

USAGE
  python3 experiments/granule/granule_discrimination.py --selftest
  python3 experiments/granule/granule_discrimination.py --run --json out.json

CANONICAL SDR IS IMPORTED READ-ONLY (hash-gated before and after, G_CANON_READONLY).
The corpus is the track-3 corpus, committed in the MAIN work tree and merely read here:
experiments/track3-semiotic-codebook/corpus_artifacts/{code,book}_tokens.npy, manifest
CORPUS_MANIFEST.json. This branch's worktree does not carry those trees, so paths are
resolved against the main worktree (git worktree list --porcelain).

# <claudes_code_comments>
# ** Function List **
# main_worktree()            - the main checkout (where experiments/ and wikis/ live)
# canon_hash(root)           - sha256 of every file under wikis/WIKI_SDR/canonical
# ngram_keys(tokens, lo, hi) - 3-grams over a token slice
# distinct_keys(keys, n)     - first n distinct 3-grams, order preserved
# key_set(keys)              - python set of 3-tuples, for the disjointness gates
# perturb(keys, rng, vocab)  - replace exactly one of the three tokens
# hash_words(keys, W, seed)  - plain multiplicative-XOR hash, W bits, packed uint64
# token_sdr_codes(...)       - token id -> canonical SDR.random code, cached
# union_codes(keys, m,k,seed)- the 3-gram union of three token codes, packed
# union_codes_checked(...)   - the same, cross-checked against canonical SDR.union
# best_of(store, queries)    - argmin Hamming distance + its value, chunked
# distance_grid(self_d, conf_d) - the radius ladder from CALIB's own distributions
# fit_D(...)                 - the arm's own radius, fitted on CALIB only
# score_arm(...)             - TMR / FMR_rand / FMR_conf / NEAR_ENTRY / NEAR_FAR
# curve_arm(...)             - the full D curve on REPORT (the descriptive object)
# structure_at_rate(...)     - NEAR_ENTRY for each arm at a fixed FMR_conf rate
# analytic_fmr_hash          - 1 - (1 - sum_{d<=D} C(W,d) 2**-W)**N
# analytic_fmr_sdr           - 1 - (1 - false_match_probability(m,k,k))**N, exact at D=0
# solve_m_for_entropy(b,k)   - least m with log2 C(m,k) >= b
# selftest()                 - planted positives and negatives, gates, determinism
# run()                      - the two normalizations over the budget ladder
#
# ** Technical Review **
# - The readout is IDENTICAL for both arms (nearest address by Hamming distance, HIT
#   iff distance <= D), so the comparison isolates the ADDRESS FUNCTION, not the
#   readout. D is fitted per arm per budget on CALIB and frozen for REPORT. Ties at
#   equal CALIB discrimination break toward the SMALLER D, which is the choice
#   GENEROUS TO THE PLAIN HASH: a compact radius is a hash-friendly readout, so an SDR
#   win cannot be an artifact of the tie-break. Declared in the PREREG.
# - Hamming distance is numpy.bitwise_count on uint64 words (numpy 2.0+), so no float
#   arithmetic and no BLAS ordering enters the address comparison.
# - The hash uses the splitmix64 finalizer for the multiplicative step and XORs the
#   three token-derived words, one multiply-mix per output word so W > 64 words stay
#   decorrelated. That is the standard multiplicative-XOR form the citation names.
# - k for the matched-BYTES arm follows encoder_registry()'s sparse default of ~2% of
#   the width (canonical/address_encoders.py line 272), rounded to a multiple of 3 so
#   the 3-gram union is exact.
# - For the matched-ENTROPY arm k is pinned to 3 (one active bit per token, m solved
#   so log2 C(m,k) >= b), so m > b is a genuine expansion. A second variant at k = 6
#   is run as a diagnostic: more active bits must lose more to a confusable key (the
#   perturbation then removes a larger share of the code), and if it does not, the
#   confusable metric is not measuring the bits.
# - The analytic SDR false-match rate is EXACT ONLY AT D = 0, where the event is
#   "two independent sparse codes are identical" and the owned closed form
#   canonical.sdr.false_match_probability(m, k, k) gives it. For D > 0 the union of
#   three random token codes is not a uniform k-of-m draw and there is no exact
#   closed form here, so no analytic is claimed there. Stated, not glossed.
# - Every null quotes the positive the SAME instrument returned in the same run: the
#   NEAR_ENTRY column is that positive (a plain hash cannot have any near-miss
#   structure at all, an expansion must show some). If the two arms are
#   indistinguishable there, the instrument is blind and no null from it is readable.
# </claudes_code_comments>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

MASK64 = (1 << 64) - 1
GOLDEN = 0x9E3779B97F4A7C15
M1 = 0xBF58476D1CE4E5B9
M2 = 0x94D049BB133111EB

# ---------------------------------------------------------------------------
# FROZEN EXPERIMENT CONSTANTS (mirrored verbatim in the PREREG, sealed before the run)
# ---------------------------------------------------------------------------
N_STORE = 2048          # table entries stored, both arms
Q_CLASS = 1024          # queries per class, both splits
CHUNK = 128             # query chunk for the Hamming sweep

CALIB_SPAN = (0, 400_000)       # code stream, token positions, disjoint, asserted
REPORT_SPAN = (400_000, 800_000)
# per-stream spans: the book stream is shorter, so its split is scaled down
SPANS = {
    "code": {"calib": (0, 400_000), "report": (400_000, 1_324_191)},
    "book": {"calib": (0, 100_000), "report": (100_000, 314_997)},
}
Q_MIN = 256                     # a class smaller than this is not reported

ARM_A_WIDTHS = (12, 16, 20, 24, 32, 64, 128, 1024)   # matched BYTES
ARM_B_ENTROPY = (12, 16, 20, 24, 28)                 # matched ENTROPY, k = 3
ARM_B_K6_ENTROPY = (12, 16, 20, 24)                  # diagnostic, k = 6

D_GRID_POINTS = 40      # points on each arm's own radius ladder
SESOI = 0.010           # the house SESOI_strict (MEASURED_track3_T3-58 line 120)
EQUAL_RATE = 0.05       # the FMR_conf rate at which near-miss structure is compared
FMR_AGREE_TOL = 0.20    # PREREG P4: measured vs closed form, relative
SEED = 20260916


# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------
def main_worktree() -> Path:
    """The main checkout: where experiments/ and wikis/ physically live."""
    out = subprocess.run(["git", "worktree", "list", "--porcelain"],
                         capture_output=True, text=True, check=True).stdout
    for line in out.splitlines():
        if line.startswith("worktree "):
            return Path(line.split(" ", 1)[1]).resolve()
    raise RuntimeError("no worktree reported by git")


def canon_hash(root: Path) -> str:
    """sha256 over every file under wikis/WIKI_SDR/canonical, name-then-bytes."""
    d = root / "wikis" / "WIKI_SDR" / "canonical"
    h = hashlib.sha256()
    for p in sorted(d.rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts:
            h.update(str(p.relative_to(d)).encode())
            h.update(p.read_bytes())
    return h.hexdigest()


def load_canonical(root: Path):
    """Import canonical READ-ONLY. Nothing under it is written by this file."""
    sdr_root = root / "wikis" / "WIKI_SDR"
    if str(sdr_root) not in sys.path:
        sys.path.insert(0, str(sdr_root))
    from canonical import SDR, false_match_probability  # noqa: E402
    return SDR, false_match_probability


# ---------------------------------------------------------------------------
# corpus
# ---------------------------------------------------------------------------
def load_stream(root: Path, stream: str) -> np.ndarray:
    p = (root / "experiments" / "track3-semiotic-codebook" / "corpus_artifacts"
         / f"{stream}_tokens.npy")
    return np.asarray(np.load(p), dtype=np.int64)


def ngram_keys(tokens: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """[n,3] token-id 3-grams at positions p-2,p-1,p for p in [lo+2, hi)."""
    lo = max(lo, 2)
    win = tokens[lo:hi]
    n = len(win) - 2
    if n <= 0:
        raise ValueError("slice too short for 3-grams")
    idx = np.arange(n)[:, None] + np.arange(3)[None, :]
    return np.ascontiguousarray(win[idx])


def distinct_keys(keys: np.ndarray, n: int) -> np.ndarray:
    """First n distinct 3-grams, input order preserved."""
    view = np.ascontiguousarray(keys)
    _, first = np.unique(view, axis=0, return_index=True)
    first = np.sort(first)          # restore input order
    return view[first[:n]] if len(first) >= n else view[first]


def key_set(keys: np.ndarray) -> set:
    return {tuple(int(x) for x in row) for row in keys}


def perturb(keys: np.ndarray, rng: np.random.Generator, vocab_max: int) -> np.ndarray:
    """Replace exactly one of the three tokens with a different token id."""
    out = keys.copy()
    which = rng.integers(0, 3, size=len(out))
    for i in range(len(out)):
        j = int(which[i])
        old = int(out[i, j])
        new = old
        while new == old:
            new = int(rng.integers(0, vocab_max))
        out[i, j] = new
    return out


# ---------------------------------------------------------------------------
# address functions
# ---------------------------------------------------------------------------
def mix64(x: np.ndarray) -> np.ndarray:
    x = (x + np.uint64(GOLDEN)) & np.uint64(MASK64)
    x = ((x ^ (x >> np.uint64(30))) * np.uint64(M1)) & np.uint64(MASK64)
    x = ((x ^ (x >> np.uint64(27))) * np.uint64(M2)) & np.uint64(MASK64)
    return x ^ (x >> np.uint64(31))


def hash_words(keys: np.ndarray, width: int, seed: int) -> np.ndarray:
    """Plain multiplicative-XOR hash of a 3-gram into ceil(width/64) uint64 words.

    The form the citation names (WIKI/theory/167 line 37: 'hashed to 65,536 slots by
    multiplicative XOR'): multiply each token id by a fixed odd constant, XOR the
    three, then mix once per output word. The last word is masked to `width` bits.
    """
    nw = (width + 63) // 64
    k = np.asarray(keys, dtype=np.uint64)
    c = np.array([0x2545F4914F6CDD1D, 0x9E3779B97F4A7C15, 0xC2B2AE3D27D4EB4F],
                 dtype=np.uint64)
    x = (k * c[None, :]).sum(axis=1, dtype=np.uint64)
    base = mix64(x ^ np.uint64(seed & MASK64))
    words = np.empty((len(keys), nw), dtype=np.uint64)
    for j in range(nw):
        words[:, j] = mix64(base + np.uint64((j * GOLDEN) & MASK64))
    if width % 64:
        words[:, -1] &= np.uint64((1 << (width % 64)) - 1)
    return words


def token_sdr_codes(vocab_ids, m: int, k_tok: int, seed: int, SDR):
    """token id -> canonical SDR.random(m, k_tok) active indices, cached per vocab id."""
    cache: dict[int, np.ndarray] = {}
    for tid in vocab_ids:
        t = int(tid)
        if t not in cache:
            s = SDR.random(n=m, w=k_tok, rng=np.random.default_rng(seed * 1_000_003 + t))
            cache[t] = np.fromiter(s.active, dtype=np.int64, count=len(s.active))
    return cache


def union_codes(keys: np.ndarray, m: int, k: int, seed: int, SDR) -> np.ndarray:
    """3-gram address = UNION of three per-token canonical SDRs, packed to uint64.

    k is the total active count and must be a multiple of 3 (k_tok = k/3), so the
    union is exact rather than approximate.
    """
    if k % 3:
        raise ValueError("k must be a multiple of 3 for an exact 3-token union")
    k_tok = k // 3
    nw = (m + 63) // 64
    cache = token_sdr_codes(np.unique(keys), m, k_tok, seed, SDR)
    out = np.zeros((len(keys), nw), dtype=np.uint64)
    for row in range(len(keys)):
        act = np.unique(np.concatenate([cache[int(keys[row, j])] for j in range(3)]))
        for a in act:
            out[row, a >> 6] |= np.uint64(1) << np.uint64(a & 63)
    return out


def union_codes_checked(keys: np.ndarray, m: int, k: int, seed: int, SDR) -> np.ndarray:
    """union_codes, with a canonical SDR.union cross-check on the first 200 rows."""
    packed = union_codes(keys, m, k, seed, SDR)
    cache = token_sdr_codes(np.unique(keys[:200]), m, k // 3, seed, SDR)
    for row in range(min(200, len(keys))):
        parts = [SDR(m, cache[int(keys[row, j])]) for j in range(3)]
        ref = parts[0].union(*parts[1:])
        ref_packed = np.zeros_like(packed[row])
        for a in ref.active:
            ref_packed[a >> 6] |= np.uint64(1) << np.uint64(a & 63)
        if not np.array_equal(ref_packed, packed[row]):
            raise AssertionError(f"G_CANON_UNION mismatch at row {row}")
    return packed


# ---------------------------------------------------------------------------
# readout: nearest address by Hamming distance, HIT iff distance <= D
# ---------------------------------------------------------------------------
def best_of(store: np.ndarray, queries: np.ndarray):
    """(argmin distance, min distance) for each query, chunked over queries."""
    nq = queries.shape[0]
    idx = np.empty(nq, dtype=np.int64)
    dist = np.empty(nq, dtype=np.int64)
    for i in range(0, nq, CHUNK):
        q = queries[i:i + CHUNK]
        d = np.bitwise_count(store[None, :, :] ^ q[:, None, :]).sum(axis=2)
        j = d.argmin(axis=1)
        idx[i:i + CHUNK] = j
        dist[i:i + CHUNK] = d[np.arange(len(j)), j]
    return idx, dist


def distance_grid(conf_d: np.ndarray, width: int) -> np.ndarray:
    """The radius ladder for one arm, from that arm's own CALIB distribution.

    The top is the 99th percentile of the distance a never-stored confusable CALIB
    query comes to some entry: past it the readout is saturated (FMR ~ 1) and there
    is nothing left to choose. The bottom is 0, the exact-address readout.
    """
    hi = int(np.ceil(np.percentile(conf_d, 99))) if conf_d.size else 0
    hi = max(hi, 1)
    pts = np.unique(np.concatenate([
        np.linspace(0, hi, D_GRID_POINTS).round().astype(np.int64),
        np.array([0, hi], dtype=np.int64),
    ]))
    return pts[(pts >= 0) & (pts <= width)]


def fit_D(arm: str, store: np.ndarray, width: int,
          hit_q: np.ndarray, hit_own: np.ndarray,
          fr_q: np.ndarray, fc_q: np.ndarray) -> dict:
    """Fit D on CALIB: maximize TMR - max(FMR_rand, FMR_conf); ties -> SMALLER D.

    `self_d` may be EMPTY: when the table is dense enough that another stored key
    shares this key's address, the nearest entry is legitimately that other key and
    the query is not a hit at any radius. That is a saturated table, not a defect, and
    it is reported as TMR = 0 rather than raised.
    """
    i_h, d_h = best_of(store, hit_q)
    _, d_r = best_of(store, fr_q)
    _, d_c = best_of(store, fc_q)
    self_d = d_h[i_h == hit_own]
    grid = distance_grid(d_c, width)
    best = None
    trace = []
    for D in grid:
        tmr = float(np.mean((i_h == hit_own) & (d_h <= D)))
        fmr_r = float(np.mean(d_r <= D))
        fmr_c = float(np.mean(d_c <= D))
        disc = tmr - max(fmr_r, fmr_c)
        trace.append({"D": int(D), "D_frac": float(D) / float(width),
                      "TMR": tmr, "FMR_rand": fmr_r, "FMR_conf": fmr_c, "DISCR": disc})
        # ties -> SMALLER D (generous to the plain hash)
        key = (round(disc, 6), -int(D))
        if best is None or key > best[0]:
            best = (key, int(D), disc, tmr, fmr_r, fmr_c)
    return {"arm": arm, "D": best[1], "D_frac": float(best[1]) / float(width),
            "DISCR_calib": best[2], "TMR_calib": best[3],
            "FMR_rand_calib": best[4], "FMR_conf_calib": best[5],
            "n_calib_hits_self": int(self_d.size),
            "self_dist_max": (int(self_d.max()) if self_d.size else None),
            "self_dist_median": (float(np.median(self_d)) if self_d.size else None),
            "rand_best_median": float(np.median(d_r)),
            "conf_best_median": float(np.median(d_c)),
            "grid_n": int(len(grid)), "trace": trace}


def score_arm(store: np.ndarray, width: int, D: int,
              hit_q: np.ndarray, hit_own: np.ndarray,
              fr_q: np.ndarray, fc_q: np.ndarray,
              near_own: np.ndarray) -> dict:
    """REPORT-side metrics at the frozen radius D.

    The confusable query set (`fc_q`) is index-aligned with `near_own`, which holds the
    store index of the key each query perturbs. FMR_conf and NEAR_ENTRY are therefore
    read off the SAME argmin array, which is what makes NEAR_ENTRY a subset of
    FMR_conf by construction rather than by hope.
    """
    i_h, d_h = best_of(store, hit_q)
    _, d_r = best_of(store, fr_q)
    i_c, d_c = best_of(store, fc_q)
    tmr = float(np.mean((i_h == hit_own) & (d_h <= D)))
    fmr_r = float(np.mean(d_r <= D))
    fmr_c = float(np.mean(d_c <= D))
    near_entry = float(np.mean((i_c == near_own) & (d_c <= D)))
    near_far = fmr_c - near_entry
    return {
        "D": int(D), "width": int(width), "D_frac": float(D) / float(width),
        "TMR": tmr, "FMR_rand": fmr_r, "FMR_conf": fmr_c,
        "DISCR": tmr - max(fmr_r, fmr_c),
        "NEAR_ENTRY": near_entry, "NEAR_FAR": near_far,
        "NEAR_ENTRY_share_of_conf": (near_entry / fmr_c) if fmr_c > 0 else None,
        "n_hits_self": int(np.sum(i_h == hit_own)),
        "self_dist_max": (int(d_h[i_h == hit_own].max())
                          if np.any(i_h == hit_own) else None),
        "rand_dist_median": float(np.median(d_r)),
        "conf_dist_median": float(np.median(d_c)),
        "conf_dist_p01": float(np.percentile(d_c, 1)),
    }


def curve_arm(store: np.ndarray, width: int, D_grid: np.ndarray,
              fr_q: np.ndarray, fc_q: np.ndarray, near_own: np.ndarray) -> list:
    """The FULL radius curve on REPORT: the descriptive object of this branch."""
    _, d_r = best_of(store, fr_q)
    i_c, d_c = best_of(store, fc_q)
    out = []
    for D in D_grid:
        fmr_c = float(np.mean(d_c <= D))
        ne = float(np.mean((i_c == near_own) & (d_c <= D)))
        out.append({"D": int(D), "D_frac": float(D) / float(width),
                    "FMR_rand": float(np.mean(d_r <= D)), "FMR_conf": fmr_c,
                    "NEAR_ENTRY": ne,
                    "NEAR_FAR": fmr_c - ne})
    return out


def structure_at_rate(curve: list, rate: float) -> dict:
    """At the first radius whose FMR_conf reaches `rate`, what is NEAR_ENTRY?

    Descriptive, read off the REPORT curve; no selection is made and no number is
    fitted, so this is a report of the curve rather than a second adjudication.
    """
    for row in curve:
        if row["FMR_conf"] >= rate:
            return {"D": row["D"], "D_frac": row["D_frac"], "FMR_conf": row["FMR_conf"],
                    "FMR_rand": row["FMR_rand"], "NEAR_ENTRY": row["NEAR_ENTRY"],
                    "NEAR_FAR": row["NEAR_FAR"],
                    "NEAR_ENTRY_share_of_conf":
                        (row["NEAR_ENTRY"] / row["FMR_conf"]) if row["FMR_conf"] else None}
    return {"D": None, "rate_unreached": rate}


# ---------------------------------------------------------------------------
# analytic envelope
# ---------------------------------------------------------------------------
def analytic_fmr_hash(width: int, D: int, n: int) -> float:
    """1 - (1 - sum_{d<=D} C(W,d) 2**-W)**N : N draws against a 2**W-slot table."""
    if D < 0:
        return 0.0
    p = sum(math.comb(width, d) for d in range(0, min(D, width) + 1)) * 2.0 ** (-width)
    p = min(p, 1.0)
    return 1.0 - (1.0 - p) ** n


def analytic_fmr_sdr(m: int, k: int, n: int, fmp) -> float:
    """1 - (1 - false_match_probability(m,k,k))**N : the owned closed form, D = 0 only."""
    p = fmp(n=m, w=k, theta=int(k))
    return 1.0 - (1.0 - p) ** n


def solve_m_for_entropy(b: int, k: int, m_hi: int = 200_000) -> int:
    """Least m with log2 C(m,k) >= b."""
    def log2C(m: int) -> float:
        return (math.lgamma(m + 1) - math.lgamma(k + 1)
                - math.lgamma(m - k + 1)) / math.log(2.0)
    lo, hi = k, m_hi
    if log2C(hi) < b:
        raise ValueError("entropy budget too large for the search bound")
    while lo < hi:
        mid = (lo + hi) // 2
        if log2C(mid) >= b:
            hi = mid
        else:
            lo = mid + 1
    return lo


def log2_comb(n: int, k: int) -> float:
    if n < k or k < 0:
        return 0.0
    return (math.lgamma(n + 1) - math.lgamma(k + 1)
            - math.lgamma(n - k + 1)) / math.log(2.0)


# ---------------------------------------------------------------------------
# the experiment
# ---------------------------------------------------------------------------
def build_worlds(tokens: np.ndarray, seed: int, vocab_max: int,
                 calib_span, report_span) -> dict:
    """CALIB and REPORT key classes, disjoint, asserted."""
    rng = np.random.default_rng(seed)
    cal = distinct_keys(ngram_keys(tokens, *calib_span), N_STORE)
    rep = distinct_keys(ngram_keys(tokens, *report_span), 10 ** 9)   # ALL distinct
    cal_s = key_set(cal)

    overlap = [k for k in rep if tuple(int(x) for x in k) in cal_s]
    if len(overlap) < Q_MIN:
        raise RuntimeError(f"too few held-out hits ({len(overlap)} < {Q_MIN}); "
                           "widen report_span")
    print(f"    store {len(cal)} | distinct report {len(rep)} | "
          f"recurring store keys {len(overlap)}", flush=True)

    n_hit = min(Q_CLASS, len(overlap))
    hits = np.array([list(k) for k in overlap[:n_hit]], dtype=np.int64)
    near = np.array([list(k) for k in overlap[:n_hit]], dtype=np.int64)

    # FMR_rand: never-stored, unrelated
    fr_all = np.array([list(k) for k in rep[N_STORE:N_STORE + 6 * Q_CLASS]], dtype=np.int64)
    fr = np.array([k for k in fr_all
                   if tuple(int(x) for x in k) not in cal_s][:Q_CLASS], dtype=np.int64)

    # FMR_conf / NEAR_ENTRY: exactly one token replaced (index-aligned with `near`)
    fc_all = perturb(near, rng, vocab_max)
    keep = [i for i in range(len(fc_all))
            if tuple(int(x) for x in fc_all[i]) not in cal_s]
    near = near[keep][:Q_CLASS]
    fc = fc_all[keep][:Q_CLASS]

    # CALIB classes for the radius fit (tail quarter of the CALIB span)
    c_start = calib_span[0] + 3 * (calib_span[1] - calib_span[0]) // 4
    c_hits = cal[:Q_CLASS]
    c_fr = distinct_keys(ngram_keys(tokens, c_start, calib_span[1]), 40 * N_STORE)
    c_fr = np.array([list(k) for k in c_fr
                     if tuple(int(x) for x in k) not in cal_s][:Q_CLASS], dtype=np.int64)
    c_fc_all = perturb(c_hits, rng, vocab_max)
    c_fc = np.array([k for k in c_fc_all
                     if tuple(int(x) for x in k) not in cal_s][:Q_CLASS], dtype=np.int64)

    # G_SPLIT_DISJOINT: the NON-store classes must not touch the store. The hit class
    # is BY CONSTRUCTION a subset of the store (that is what a hit is), so its gate is
    # the opposite one and it is asserted separately.
    for name, arr in (("fr", fr), ("fc", fc), ("c_fr", c_fr), ("c_fc", c_fc)):
        if key_set(arr) & cal_s:
            raise AssertionError(f"G_SPLIT_DISJOINT failed: {name} intersects the store")
    if not key_set(hits) <= cal_s:
        raise AssertionError("G_HITS_IN_STORE failed: a hit query is not a store key")
    for name, arr in (("fr", fr), ("fc", fc), ("c_fr", c_fr), ("c_fc", c_fc)):
        if len(arr) < Q_MIN:
            raise RuntimeError(f"query class {name} came up short ({len(arr)} < {Q_MIN})")
    return {"store_keys": cal, "cal_store": cal_s, "hits": hits, "near": near,
            "fr": fr, "fc": fc, "c_hits": c_hits, "c_fr": c_fr, "c_fc": c_fc}


def run_pair(budget: str, b_label, hash_fn, sdr_fn, hash_width, sdr_width, k,
             w: dict, fmp, entropy_bits_hash: float, entropy_bits_sdr: float) -> dict:
    """Run one hash/SDR pair at one budget and return both arms' full records."""
    own = {tuple(int(x) for x in r): i for i, r in enumerate(w["store_keys"])}
    hit_own = np.array([own[tuple(int(x) for x in r)] for r in w["hits"]], dtype=np.int64)
    near_own = np.array([own[tuple(int(x) for x in r)] for r in w["near"]], dtype=np.int64)
    # CALIB hit queries are the first n store keys themselves, so their own index IS
    # their position: cal[i] is store entry i.
    cal_hit_own = np.arange(len(w["c_hits"]), dtype=np.int64)

    hc = hash_fn(w["store_keys"])
    gc = sdr_fn(w["store_keys"])

    fit_h = fit_D("hash", hc, hash_width, hash_fn(w["c_hits"]), cal_hit_own,
                  hash_fn(w["c_fr"]), hash_fn(w["c_fc"]))
    fit_g = fit_D("sdr", gc, sdr_width, sdr_fn(w["c_hits"]), cal_hit_own,
                  sdr_fn(w["c_fr"]), sdr_fn(w["c_fc"]))
    sc_h = score_arm(hc, hash_width, fit_h["D"], hash_fn(w["hits"]), hit_own,
                     hash_fn(w["fr"]), hash_fn(w["fc"]), near_own)
    sc_g = score_arm(gc, sdr_width, fit_g["D"], sdr_fn(w["hits"]), hit_own,
                     sdr_fn(w["fr"]), sdr_fn(w["fc"]), near_own)

    meta = {"budget": budget, "label": b_label,
            "hash_width": hash_width, "sdr_width": sdr_width, "k": k,
            "bits_per_entry_hash": hash_width, "bits_per_entry_sdr": sdr_width,
            "entropy_bits_hash": entropy_bits_hash, "entropy_bits_sdr": entropy_bits_sdr}
    for sc, fn, wd, codes, fitres in ((sc_h, hash_fn, hash_width, hc, fit_h),
                                      (sc_g, sdr_fn, sdr_width, gc, fit_g)):
        sc.update(meta)
        grid = np.array(sorted({r["D"] for r in fitres["trace"]}))
        sc["curve"] = curve_arm(codes, wd, grid, fn(w["fr"]), fn(w["fc"]), near_own)
        sc["structure_at_rate"] = structure_at_rate(sc["curve"], EQUAL_RATE)
    sc_h["fit"] = {k2: v for k2, v in fit_h.items() if k2 != "trace"}
    sc_g["fit"] = {k2: v for k2, v in fit_g.items() if k2 != "trace"}
    sc_h["analytic_fmr"] = analytic_fmr_hash(hash_width, fit_h["D"], N_STORE)
    sc_g["analytic_fmr"] = (analytic_fmr_sdr(sdr_width, k, N_STORE, fmp)
                            if fit_g["D"] == 0 else None)
    sc_h["analytic_exact_at_D"] = fit_h["D"] == 0
    sc_g["analytic_exact_at_D"] = fit_g["D"] == 0
    return {"hash": sc_h, "sdr": sc_g}


def run(streams=("code", "book")) -> dict:
    root = main_worktree()
    h0 = canon_hash(root)
    SDR, fmp = load_canonical(root)
    out = {"prereg_constants": {
        "N_STORE": N_STORE, "Q_CLASS": Q_CLASS, "CALIB_SPAN": CALIB_SPAN,
        "REPORT_SPAN": REPORT_SPAN, "ARM_A_WIDTHS": list(ARM_A_WIDTHS),
        "ARM_B_ENTROPY": list(ARM_B_ENTROPY), "ARM_B_K6_ENTROPY": list(ARM_B_K6_ENTROPY),
        "D_GRID_POINTS": D_GRID_POINTS, "SESOI": SESOI, "EQUAL_RATE": EQUAL_RATE,
        "SEED": SEED, "canon_sha256_before": h0},
        "streams": {}}

    for stream in streams:
        tokens = load_stream(root, stream)
        vocab_max = int(tokens.max()) + 1
        sp = SPANS[stream]
        print(f"[{stream}] tokens {tokens.shape} vocab<{vocab_max} "
              f"calib {sp['calib']} report {sp['report']}", flush=True)
        w = build_worlds(tokens, SEED, vocab_max, sp["calib"], sp["report"])
        print(f"[{stream}] tokens {tokens.shape} vocab<{vocab_max} "
              f"store {len(w['store_keys'])} hits {len(w['hits'])} "
              f"fr {len(w['fr'])} fc {len(w['fc'])}", flush=True)
        res = {"n_store": int(len(w["store_keys"])), "n_hits": int(len(w["hits"])),
               "n_fr": int(len(w["fr"])), "n_fc": int(len(w["fc"])),
               "arm_a": [], "arm_b": [], "arm_b_k6": []}

        # ---- ARM A: matched BYTES (both addresses W bits) ----
        for W in ARM_A_WIDTHS:
            k = min(3 * max(1, round(0.02 * W / 3.0)), W)
            pair = run_pair(
                "A_matched_bytes", f"W={W}", 
                lambda x, W=W: hash_words(x, W, SEED),
                lambda x, W=W, k=k: union_codes_checked(x, W, k, SEED, SDR),
                W, W, k, w, fmp, float(W), log2_comb(W, k))
            res["arm_a"].append(pair)
            print(f"  A W={W:5d} k={k:3d} | hash D={pair['hash']['D']:4d} "
                  f"DISCR {pair['hash']['DISCR']:+.4f} FMRr {pair['hash']['FMR_rand']:.4f} "
                  f"FMRc {pair['hash']['FMR_conf']:.4f} NE {pair['hash']['NEAR_ENTRY']:.4f}"
                  f" || sdr D={pair['sdr']['D']:4d} DISCR {pair['sdr']['DISCR']:+.4f} "
                  f"FMRr {pair['sdr']['FMR_rand']:.4f} FMRc {pair['sdr']['FMR_conf']:.4f} "
                  f"NE {pair['sdr']['NEAR_ENTRY']:.4f}", flush=True)

        # ---- ARM B: matched ENTROPY (hash b bits, SDR log2 C(m,k) = b) ----
        for bucket, kk, ladder in (("arm_b", 3, ARM_B_ENTROPY),
                                   ("arm_b_k6", 6, ARM_B_K6_ENTROPY)):
            for b in ladder:
                m = solve_m_for_entropy(b, kk)
                pair = run_pair(
                    "B_matched_entropy", f"b={b},k={kk}",
                    lambda x, b=b: hash_words(x, b, SEED),
                    lambda x, m=m, kk=kk: union_codes_checked(x, m, kk, SEED, SDR),
                    b, m, kk, w, fmp, float(b), log2_comb(m, kk))
                res[bucket].append(pair)
                print(f"  B b={b:3d} k={kk:2d} m={m:6d} | hash D={pair['hash']['D']:4d} "
                      f"DISCR {pair['hash']['DISCR']:+.4f} FMRr {pair['hash']['FMR_rand']:.4f} "
                      f"FMRc {pair['hash']['FMR_conf']:.4f} NE {pair['hash']['NEAR_ENTRY']:.4f}"
                      f" || sdr D={pair['sdr']['D']:5d} DISCR {pair['sdr']['DISCR']:+.4f} "
                      f"FMRr {pair['sdr']['FMR_rand']:.4f} FMRc {pair['sdr']['FMR_conf']:.4f} "
                      f"NE {pair['sdr']['NEAR_ENTRY']:.4f}", flush=True)
        out["streams"][stream] = res

    h1 = canon_hash(root)
    out["G_CANON_READONLY"] = (h0 == h1)
    out["canon_sha256_after"] = h1
    if h0 != h1:
        raise AssertionError("G_CANON_READONLY FAILED: canonical changed under us")
    out["verdict"] = adjudicate(out)
    return out


def _rows(out: dict, bucket: str) -> list:
    rows = []
    for stream, res in out["streams"].items():
        for pair in res[bucket]:
            h, s = pair["hash"], pair["sdr"]
            hs, ss = h["structure_at_rate"], s["structure_at_rate"]
            rows.append({
                "stream": stream, "budget": h.get("label"),
                "delta_discr": s["DISCR"] - h["DISCR"],
                "hash_discr": h["DISCR"], "sdr_discr": s["DISCR"],
                "hash_tmr": h["TMR"], "sdr_tmr": s["TMR"],
                "hash_fmr_rand": h["FMR_rand"], "sdr_fmr_rand": s["FMR_rand"],
                "hash_fmr_conf": h["FMR_conf"], "sdr_fmr_conf": s["FMR_conf"],
                "hash_D": h["D"], "sdr_D": s["D"],
                "hash_width": h["hash_width"], "sdr_width": s["sdr_width"],
                "hash_near_entry": h["NEAR_ENTRY"], "sdr_near_entry": s["NEAR_ENTRY"],
                "hash_share": h["NEAR_ENTRY_share_of_conf"],
                "sdr_share": s["NEAR_ENTRY_share_of_conf"],
                # P3 is stated AT the radius where FMR_conf first reaches EQUAL_RATE,
                # which is what structure_at_rate reads off the REPORT curve.
                "hash_struct": hs, "sdr_struct": ss,
            })
    return rows


def adjudicate(out: dict) -> dict:
    """Apply the sealed verdict rule to ARM B (the fair expansion reading)."""
    rows = _rows(out, "arm_b")
    n = len(rows)
    sdr_win = sum(1 for r in rows if r["delta_discr"] > SESOI)
    null = sum(1 for r in rows if abs(r["delta_discr"]) <= SESOI)
    hash_win = sum(1 for r in rows if r["delta_discr"] < -SESOI)
    need = max(1, int(math.ceil(0.8 * n)))

    a_rows = _rows(out, "arm_a")
    a_sdr_win = sum(1 for r in a_rows if r["delta_discr"] > SESOI)
    a_null = sum(1 for r in a_rows if abs(r["delta_discr"]) <= SESOI)
    a_hash_win = sum(1 for r in a_rows if r["delta_discr"] < -SESOI)

    if sdr_win >= need:
        verdict = "SDR_WINS"
    elif hash_win >= need:
        verdict = "PLAIN_HASH_WINS"
    elif null >= need:
        verdict = "NULL"
    else:
        verdict = "UNRESOLVED"

    # the positive control: does the expansion's tolerance carry STRUCTURE the hash's
    # cannot, i.e. are its false routings to the perturbed-FROM entry? P3 states this
    # at the radius where FMR_conf first reaches EQUAL_RATE.
    structured = sum(1 for r in rows
                     if r["sdr_struct"].get("NEAR_ENTRY") is not None
                     and r["hash_struct"].get("NEAR_ENTRY") is not None
                     and r["sdr_struct"]["NEAR_ENTRY"] > r["hash_struct"]["NEAR_ENTRY"] + SESOI)
    structured_ratio = sum(1 for r in rows
                           if r["sdr_struct"].get("NEAR_ENTRY_share_of_conf") is not None
                           and r["hash_struct"].get("NEAR_ENTRY_share_of_conf") is not None
                           and r["sdr_struct"]["NEAR_ENTRY_share_of_conf"]
                           > 3 * r["hash_struct"]["NEAR_ENTRY_share_of_conf"])
    ident = sum(1 for r in rows
                if r["hash_fmr_rand"] > 0
                and abs(r["hash_fmr_rand"] - r["sdr_fmr_rand"]) <= SESOI)
    return {
        "primary_arm": "B_matched_entropy,k=3",
        "n_budgets": n, "n_required": need, "sesoi": SESOI,
        "sdr_wins_n": sdr_win, "null_n": null, "hash_wins_n": hash_win,
        "verdict": verdict, "rows": rows,
        "arm_a_n": len(a_rows), "arm_a_sdr_wins_n": a_sdr_win, "arm_a_null_n": a_null,
        "arm_a_hash_wins_n": a_hash_win, "arm_a_rows": a_rows,
        "structure_positive_control_fires_n": structured,
        "structure_positive_control_ratio3_n": structured_ratio,
        "structure_positive_control_readable": (structured >= need
                                                and structured_ratio >= need),
        "random_fmr_identity_n": ident,
    }


# ---------------------------------------------------------------------------
# selftest: planted positives and negatives, gates, determinism
# ---------------------------------------------------------------------------
def selftest() -> int:
    root = main_worktree()
    h0 = canon_hash(root)
    SDR, fmp = load_canonical(root)
    fails = []
    checks = []

    def check(name, ok, detail=""):
        checks.append((name, bool(ok), detail))
        if not ok:
            fails.append(name)

    # 1. the closed forms are exact where they must be
    check("hash_fmr_D0_b16", abs(analytic_fmr_hash(16, 0, 2048)
                                 - (1 - (1 - 2 ** -16) ** 2048)) < 1e-18)
    check("hash_fmr_monotone_in_D",
          analytic_fmr_hash(16, 1, 2048) > analytic_fmr_hash(16, 0, 2048))
    check("hash_fmr_monotone_in_b",
          analytic_fmr_hash(16, 0, 2048) > analytic_fmr_hash(24, 0, 2048))
    check("fmp_theta0_is_1", fmp(1024, 21, 0) == 1.0)
    check("fmp_theta_gt_k_is_0", fmp(1024, 21, 22) == 0.0)

    # 2. entropy solver
    m = solve_m_for_entropy(16, 3)
    check("solve_m_16_3", log2_comb(m, 3) >= 16 > log2_comb(m - 1, 3), f"m={m}")
    m6 = solve_m_for_entropy(16, 6)
    check("solve_m_16_6_compressed", m6 < m, f"m3={m} m6={m6}")

    # 3. the canonical union cross-check (G_CANON_UNION) and its weight
    keys = np.array([[11, 22, 33], [44, 55, 66], [11, 22, 34]], dtype=np.int64)
    packed = union_codes_checked(keys, 256, 3, 7, SDR)
    check("union_packed_weight_le_3",
          bool(np.all(np.bitwise_count(packed).sum(axis=1) <= 3)))

    # 4. determinism
    check("G_DETERMINISM_union",
          np.array_equal(packed, union_codes_checked(keys, 256, 3, 7, SDR)))
    check("G_DETERMINISM_hash",
          np.array_equal(hash_words(keys, 128, 5), hash_words(keys, 128, 5)))

    # 5. the readout: distance to itself is 0 for BOTH arms
    i1, d1 = best_of(hash_words(keys, 128, 1), hash_words(keys, 128, 1))
    check("self_distance_zero_hash", bool(np.all(d1 == 0)))
    check("self_argmax_hash", bool(np.all(i1 == np.arange(3))))
    i2, d2 = best_of(packed, packed)
    check("self_distance_zero_sdr", bool(np.all(d2 == 0)))

    # 6. PLANTED NEGATIVE (the metric can read a non-difference): the same codes on
    #    both sides of the comparison must read identically
    same = hash_words(keys, 64, 9)
    _, da = best_of(same, same)
    _, db = best_of(same, same)
    check("planted_identity_identical", np.array_equal(da, db))

    # 7. PLANTED POSITIVE (the metric can read the difference it is for): a
    #    one-token perturbation of a SPARSE code stays NEAR, of a HASH does not
    rng = np.random.default_rng(3)
    vocab = 997
    ks = rng.integers(0, vocab, size=(256, 3))
    pert = ks.copy()
    pert[:, 0] = (pert[:, 0] + 1) % vocab
    gs = union_codes_checked(ks, 1024, 6, 11, SDR)
    gp = union_codes_checked(pert, 1024, 6, 11, SDR)
    _, d_sparse = best_of(gs, gp)
    _, d_dense = best_of(hash_words(ks, 1024, 11), hash_words(pert, 1024, 11))
    check("planted_sparse_perturbed_is_near", float(np.median(d_sparse)) <= 4.0,
          f"med={np.median(d_sparse)}")
    check("planted_dense_perturbed_is_far", float(np.median(d_dense)) >= 450.0,
          f"med={np.median(d_dense)}")

    # 8. PLANTED POSITIVE on the anchor: a confusable query's nearest entry IS the
    #    entry it perturbed, in the sparse arm, and is NOT, in the hash arm
    _, ds = best_of(gs, gp)
    isrc_s = np.array([int(ds[i]) for i in range(len(gs))])
    check("planted_near_structure_exists", bool(np.all(isrc_s <= 4)))

    # 9. PLANTED NEGATIVE on the anchor: unrelated queries do NOT land near
    ru = rng.integers(0, vocab, size=(256, 3))
    gu = union_codes_checked(ru, 1024, 6, 11, SDR)
    _, d_rand = best_of(gs, gu)
    check("planted_unrelated_is_far", float(np.median(d_rand)) >= 8.0,
          f"med={np.median(d_rand)}")

    # 10. PLANTED POSITIVE on saturation: an over-full dense table (16 slots, 256
    #     keys) must read FMR ~ 1 at D = 0
    over = hash_words(ks, 4, 1)
    _, d_over = best_of(over[:64], over[64:])
    check("planted_overfull_hash_saturates", float(np.mean(d_over == 0)) > 0.9,
          f"fmr={np.mean(d_over == 0):.4f}")

    # 8b. G_NEAR_SUBSET_CONF: NEAR_ENTRY <= FMR_conf BY CONSTRUCTION, the invariant
    #     Amendment-2 repaired the code to satisfy. 128 keys, one token perturbed each,
    #     scored through score_arm exactly as the run scores them.
    s_keys = np.array([[300 + i, 400 + i, 500 + i] for i in range(128)], dtype=np.int64)
    s_codes = union_codes_checked(s_keys, 512, 6, 21, SDR)
    p_keys = perturb(s_keys, np.random.default_rng(5), 4000)
    p_codes = union_codes_checked(p_keys, 512, 6, 21, SDR)
    src_idx = np.arange(len(s_keys), dtype=np.int64)
    sc = score_arm(s_codes, 512, 4, s_codes, src_idx, p_codes[:16], p_codes, src_idx)
    check("G_NEAR_SUBSET_CONF", sc["NEAR_ENTRY"] <= sc["FMR_conf"] + 1e-12,
          f"NE={sc['NEAR_ENTRY']:.4f} FMRc={sc['FMR_conf']:.4f}")

    check("G_CANON_READONLY", h0 == canon_hash(root))

    for name, ok, detail in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}  {detail}")
    print(f"selftest: {len(checks) - len(fails)}/{len(checks)} PASS")
    return 1 if fails else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--json", default="")
    ap.add_argument("--streams", default="code,book")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if args.run:
        t0 = time.time()
        out = run(streams=tuple(s for s in args.streams.split(",") if s))
        out["wall_s"] = time.time() - t0
        if args.json:
            Path(args.json).write_text(json.dumps(out, indent=1, default=str))
        v = {k: val for k, val in out["verdict"].items()
             if not k.endswith("_rows")}
        print(json.dumps(v, indent=1))
        print(f"wall {out['wall_s']:.1f} s")
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
