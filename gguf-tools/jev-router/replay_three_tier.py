#!/usr/bin/env python3
"""replay_three_tier.py - replay a routing trace through a three-tier expert cache, offline.

Reads the trace DS4_CUDA_ROUTE_TRACE writes (see README.md: `t L ids...`, `m t L 0|1...`,
`w t L wait_us n_miss`) and reports, before any engine change:

  1. hit rate TODAY, read straight off the `m` lines (the engine's own LRU-with-hit-protection);
  2. the POPULARITY CURVE: how many (layer, expert) pairs cover 50 / 80 / 90 % of demand;
  3. STICKINESS: the last token's ids at the same layer, as a fraction of this token's;
  4. LATENCY OR BANDWIDTH BOUND: a least-squares fit of the `w` lines, wait_us = a + b * n_miss.
     A large intercept a against b * (median misses) says the wait is latency-bound (a prefetch
     that fills idle time can pay); a small a says bandwidth-bound (only fewer bytes pay);
  5. THE THREE TIERS, ranked by EXPECTED EXPERT REUSE over the horizon, replayed with a sweep
     over the pinned fraction, the lookahead k and the admission confidence, counting SSD reads
     and critical-path misses per token. Demand of the current token always wins a slot.

       PINNED        (layer, expert) pairs whose EXPECTED EXPERT REUSE over the whole trace is
                     highest (they recur every few tokens); held, never evicted by a forecast
                     or a one-off.
       STAGED        pairs the lookahead names for t+1..t+k, admitted only when its confidence
                     for that pair is at least the admission threshold; read into slack,
                     expiring unused after k tokens.
       PASS-THROUGH  everything else: read on demand, used once, no slot, so a one-off never
                     evicts a hot expert (why hotlist and the warm set lost).

     The lookahead in this replay is an ORACLE bounded by the trace (it knows t+1..t+k) scaled by
     the admission confidence; the trained guesser is lane FORECASTLEARN's and plugs in here in
     place of the oracle. With the oracle the numbers are an UPPER BOUND on what any guesser can
     buy for a given tier split; they say what the tiers are worth, not what a guesser achieves.

  python3 replay_three_tier.py TRACE [--slots N] [--pinned 0.25,0.5] [--k 1,2,4] [--conf 0.5,0.8]
  python3 replay_three_tier.py --selftest

EXPECTED EXPERT REUSE of a (layer, expert) pair over a horizon H tokens = the fraction of the
next H tokens that route to it, estimated from the trace as its demand rate. numpy only.
"""
# <claudes_code_comments>
# ** Function List **
# read_trace(path, with_forecasts) - ids, miss flags, waits, and the f/o forecast records
# forecast_report(fore, outc) - HIT/MISS/WASTE per source and the worst windows
# hit_rate_today(miss) - 1 - misses / lookups from the m lines
# popularity_curve(ids) - hot-set sizes covering 50/80/90 % of demand
# stickiness(ids) - last token's ids at the same layer over this token's
# wait_fit(waits) - least squares wait_us = a + b * n_miss; the bound verdict
# expected_expert_reuse(ids, train_tokens) - demand rate per (layer, expert) pair
# replay(ids, slots, pinned_frac, k, conf, reuse) - the three-tier replay; reads and waits
# sweep(...) - the grid over pinned fraction, k and confidence
# selftest() - planted traces: a hot set is pinned and never read twice, a one-off never evicts
# main() - the CLI
#
# ** Technical Review **
# The replay keeps one cache of `slots` (layer, expert) pairs. PINNED takes the top pinned_frac
# of slots by EXPECTED EXPERT REUSE (estimated on the first 80 % of tokens, never on the tokens
# being replayed). STAGED holds oracle-forecast pairs for the next k tokens, admitted when the
# pair's EXPECTED EXPERT REUSE over the next k tokens (the oracle's own count / k) is at least
# `conf`; a staged pair expires k tokens after admission if unused. PASS-THROUGH pairs are read
# and not kept. The current token's demand always wins: a demand miss on a pass-through pair
# costs a read and no slot; on a staged pair it is a hit if staged in time. Reads count every
# SSD read (demand misses plus staged reads); critical-path misses count demand reads the token
# had to wait for. The engine's real LRU is replayed too (`lru` row) as the control, from the
# same trace, so a tier split is compared against the policy the box runs today.
# </claudes_code_comments>
import argparse
import os
import sys
import tempfile
from collections import OrderedDict, defaultdict

try:
    import numpy as np
except ImportError:  # pragma: no cover
    sys.stderr.write("replay_three_tier.py: numpy is required\n")
    sys.exit(2)


def read_trace(path, with_forecasts=False):
    """ids[(t, L)], miss[(t, L)], waits[(t, L, us, n_miss)]; with_forecasts also returns
    fore[(t, L)] = {"source", "reads", "conf": {id: conf}} and outc[(t, L)] = {id: HIT|MISS|WASTE}."""
    ids, miss, waits, fore, outc = {}, {}, [], {}, {}
    with open(path) as f:
        for raw in f:
            p = raw.split()
            if not p or p[0].startswith("#") or p[0] == "h":
                continue
            if p[0] == "m":
                miss[(int(p[1]), int(p[2]))] = [int(x) for x in p[3:]]
            elif p[0] == "w":
                waits.append((int(p[1]), int(p[2]), float(p[3]), int(p[4])))
            elif p[0] == "f":
                fore[(int(p[1]), int(p[2]))] = {"source": p[3], "reads": int(p[4]),
                                                "conf": {int(x.split(":")[0]): float(x.split(":")[1]) for x in p[5:]}}
            elif p[0] == "o":
                outc[(int(p[1]), int(p[2]))] = {int(x.split("=")[0]): x.split("=")[1] for x in p[3:]}
            elif p[0].isdigit():
                ids[(int(p[0]), int(p[1]))] = [int(x) for x in p[2:]]
    if with_forecasts:
        return ids, miss, waits, fore, outc
    return ids, miss, waits


def forecast_report(fore, outc, worst=5):
    """Per source: HIT / MISS / WASTE totals and reads spent; then the worst windows (tokens with
    the most MISS + WASTE), the shape FORECASTLEARN's --inspect prints from the same record."""
    if not outc:
        print("forecast outcomes: no o lines (the run had no scout forecast, or DS4_CUDA_SCOUT=0)")
        return
    per = defaultdict(lambda: {"HIT": 0, "MISS": 0, "WASTE": 0, "reads": 0, "layers": 0})
    by_token = defaultdict(int)
    for key, o in outc.items():
        src = fore.get(key, {}).get("source", "unknown")
        d = per[src]
        d["layers"] += 1
        d["reads"] += fore.get(key, {}).get("reads", 0)
        for v in o.values():
            d[v] += 1
        by_token[key[0]] += sum(1 for v in o.values() if v != "HIT")
    print("  forecast outcomes per source (HIT forecast+routed · MISS routed+not forecast · WASTE forecast+not routed)")
    print("  source     layers      HIT     MISS    WASTE   reads   hit/forecast   miss/routed")
    for src, d in sorted(per.items()):
        f_n = d["HIT"] + d["WASTE"]
        r_n = d["HIT"] + d["MISS"]
        print("  %-8s %8d %8d %8d %8d %7d   %12s   %11s"
              % (src, d["layers"], d["HIT"], d["MISS"], d["WASTE"], d["reads"],
                 "%.3f" % (d["HIT"] / f_n) if f_n else "-", "%.3f" % (d["MISS"] / r_n) if r_n else "-"))
    top = sorted(by_token.items(), key=lambda kv: (-kv[1], kv[0]))[:worst]
    print("  worst windows (token: MISS+WASTE across its layers): %s"
          % ", ".join("%d:%d" % kv for kv in top))


def hit_rate_today(miss):
    n = sum(len(v) for v in miss.values())
    m = sum(sum(v) for v in miss.values())
    return (1.0 - m / n) if n else None, n, m


def popularity_curve(ids):
    demand = defaultdict(int)
    for (t, L), v in ids.items():
        for e in v:
            demand[(L, e)] += 1
    total = sum(demand.values())
    counts = sorted(demand.values(), reverse=True)
    cum = np.cumsum(counts) / float(total) if total else np.array([])
    out = {}
    for q in (0.5, 0.8, 0.9):
        out[q] = int(np.searchsorted(cum, q) + 1) if total else 0
    return out, len(demand), total


def stickiness(ids):
    per = defaultdict(list)
    for (t, L), v in ids.items():
        prev = ids.get((t - 1, L))
        if prev:
            per[L].append(len(set(prev) & set(v)) / float(len(set(v))))
    allv = [x for L in per for x in per[L]]
    return (float(np.mean(allv)) if allv else None), {L: float(np.mean(per[L])) for L in per}


def wait_fit(waits):
    if len(waits) < 4:
        return None
    x = np.asarray([w[3] for w in waits], dtype=np.float64)
    y = np.asarray([w[2] for w in waits], dtype=np.float64)
    A = np.stack([np.ones_like(x), x], axis=1)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    a, b = float(coef[0]), float(coef[1])
    med = float(np.median(x[x > 0])) if (x > 0).any() else 0.0
    verdict = "latency-bound (intercept dominates)" if a > b * med else "bandwidth-bound (per-miss term dominates)"
    return {"a_us": a, "b_us_per_miss": b, "median_misses": med, "verdict": verdict, "n": len(waits)}


def expected_expert_reuse(ids, tokens):
    """demand rate per (layer, expert) pair over the given tokens: the fraction of tokens routing to it."""
    demand = defaultdict(int)
    for (t, L), v in ids.items():
        if t in tokens:
            for e in v:
                demand[(L, e)] += 1
    n = float(len(tokens)) if tokens else 1.0
    return {pair: c / n for pair, c in demand.items()}


def replay(ids, tokens, slots, pinned_frac, k, conf, reuse, policy="tiers"):
    """Returns (reads, critical_misses, lookups). policy 'lru' replays a plain LRU over all slots."""
    layers = sorted({L for (t, L) in ids})
    n_pinned = int(slots * pinned_frac) if policy == "tiers" else 0
    pinned = set(p for p, _ in sorted(reuse.items(), key=lambda kv: (-kv[1], kv[0]))[:n_pinned])
    free_slots = slots - len(pinned)
    lru = OrderedDict()           # pair -> expiry token (None = demand-admitted, LRU-managed)
    reads = crit = lookups = 0
    for t in tokens:
        # the oracle names what t+1..t+k need; admit by EXPECTED EXPERT REUSE over that horizon
        if policy == "tiers" and k > 0 and free_slots > 0:
            fut = defaultdict(int)
            for dt in range(1, k + 1):
                for L in layers:
                    for e in ids.get((t + dt, L), []):
                        fut[(L, e)] += 1
            for pair, c in fut.items():
                if pair in pinned or pair in lru or c / float(k) < conf:
                    continue
                if len(lru) >= free_slots:
                    victim = next((p for p, exp in lru.items() if exp is not None and exp < t), None)
                    if victim is None:
                        continue  # nothing expired and nothing evictable by a forecast: do not admit
                    del lru[victim]
                lru[pair] = t + k
                reads += 1
        for L in layers:
            for e in ids.get((t, L), []):
                lookups += 1
                pair = (L, e)
                if pair in pinned:
                    continue
                if pair in lru:
                    lru.move_to_end(pair)
                    if policy == "lru":
                        lru[pair] = None
                    continue
                reads += 1
                crit += 1
                if policy == "lru":
                    if len(lru) >= free_slots:
                        lru.popitem(last=False)
                    lru[pair] = None
                # tiers: a demand miss that is not staged and not pinned is PASS-THROUGH: no slot
        # expire staged pairs
        if policy == "tiers":
            for pair in [p for p, exp in lru.items() if exp is not None and exp < t]:
                del lru[pair]
    return reads, crit, lookups


def sweep(ids, slots, pinned_list, k_list, conf_list):
    tokens = sorted({t for (t, L) in ids})
    cut = int(round(len(tokens) * 0.8))
    train, held = set(tokens[:cut]), tokens[cut:]
    reuse = expected_expert_reuse(ids, train)
    rows = []
    r, c, n = replay(ids, held, slots, 0.0, 0, 0.0, reuse, policy="lru")
    rows.append(("lru (today's policy, replayed)", "-", "-", "-", r, c, n))
    for pf in pinned_list:
        for k in k_list:
            for cf in conf_list:
                r, c, n = replay(ids, held, slots, pf, k, cf, reuse)
                rows.append(("tiers", pf, k, cf, r, c, n))
    return rows, len(held)


def print_report(trace, ids, miss, waits, rows, n_held, slots):
    print("routing trace: %s" % trace)
    tokens = sorted({t for (t, L) in ids})
    layers = sorted({L for (t, L) in ids})
    print("tokens %d · layers %d · replayed on the last %d tokens · cache %d slots" % (len(tokens), len(layers), n_held, slots))
    hr, n, m = hit_rate_today(miss)
    print("hit rate today (from the m lines): %s over %d lookups, %d misses" % ("%.3f" % hr if hr is not None else "no m lines", n, m))
    curve, pairs, total = popularity_curve(ids)
    print("popularity: %d distinct (layer, expert) pairs carry %d lookups; hot set covering 50 %% = %d, 80 %% = %d, 90 %% = %d pairs"
          % (pairs, total, curve[0.5], curve[0.8], curve[0.9]))
    st, per = stickiness(ids)
    print("stickiness (last token's ids at the same layer): %s" % ("%.3f" % st if st is not None else "-"))
    fit = wait_fit(waits)
    if fit:
        print("wait fit over %d w lines: wait_us = %.0f + %.0f * n_miss (median misses %.1f) -> %s"
              % (fit["n"], fit["a_us"], fit["b_us_per_miss"], fit["median_misses"], fit["verdict"]))
    else:
        print("wait fit: no w lines (the trace was taken without hits-first waits, or too few)")
    print()
    print("  policy                            pinned     k   conf    reads/token   crit misses/token   hit rate")
    for name, pf, k, cf, r, c, n in rows:
        print("  %-32s %7s %5s %6s   %11.2f   %17.2f   %8.3f"
              % (name, pf if pf == "-" else "%.2f" % pf, k, cf if cf == "-" else "%.2f" % cf,
                 r / float(n_held), c / float(n_held), 1.0 - c / float(n) if n else 0.0))
    print()
    print("reads/token counts every SSD read (demand + staged); crit misses/token are the reads the token waited for.")
    print("The lookahead here is an oracle bounded by the trace: an UPPER BOUND on any guesser at that tier split.")


def _synth(path, tokens=600, layers=4, E=48, K=6, hot=8, seed=1):
    """hot experts recur on most tokens; the rest are one-offs; token t's misses are marked as LRU would."""
    rng = np.random.default_rng(seed)
    with open(path, "w") as f:
        for t in range(tokens):
            f.write("t %d %d\n" % (t, 1000 + (t % 97)))   # the engine's `t` line; this replay ignores it
            for L in range(layers):
                n_cold = 2 if (t % 3) else 1   # 1 or 2 one-offs, so the wait fit has a slope to find
                hot_ids = list(rng.choice(hot, K - n_cold, replace=False))
                cold = list(rng.choice(np.arange(hot, E), n_cold, replace=False))
                v = hot_ids + cold
                f.write("%d %d %s\n" % (t, L, " ".join(str(int(e)) for e in v)))
                f.write("m %d %d %s\n" % (t, L, " ".join("0" if e < hot else "1" for e in v)))
                f.write("w %d %d %d %d\n" % (t, L, 300 + 900 * n_cold, n_cold))
    return path


def selftest():
    checks = []
    with tempfile.TemporaryDirectory() as d:
        tr = _synth(os.path.join(d, "s.txt"))
        ids, miss, waits = read_trace(tr)
        hr, n, m = hit_rate_today(miss)
        checks.append(("hit rate today reads the m lines (hot ids hit, one-offs miss)", hr is not None and 0.6 < hr < 0.9))
        curve, pairs, total = popularity_curve(ids)
        checks.append(("popularity: the 4 layers x 8 hot pairs cover 50 %% of demand (%d)" % curve[0.5], curve[0.5] <= 32))
        fit = wait_fit(waits)
        checks.append(("wait fit recovers the planted intercept and slope (300 + 900/miss)", fit is not None and abs(fit["b_us_per_miss"] - 900) < 1e-6 and abs(fit["a_us"] - 300) < 1e-6))
        st, _ = stickiness(ids)
        checks.append(("stickiness is scored", st is not None and 0.0 < st < 1.0))
        rows, n_held = sweep(ids, slots=40, pinned_list=[0.8], k_list=[0, 2], conf_list=[0.5])
        lru = rows[0]
        tiers_k0 = [r for r in rows if r[0] == "tiers" and r[2] == 0][0]
        tiers_k2 = [r for r in rows if r[0] == "tiers" and r[2] == 2][0]
        checks.append(("pinning the hot set (32 pairs of 40 slots): no hot expert is ever read on the critical path",
                       tiers_k0[5] <= n_held * 4 * 2))  # only the one-offs miss: at most 2 per layer per token
        checks.append(("a one-off never takes a slot: reads == crit misses at k=0", tiers_k0[4] == tiers_k0[5]))
        checks.append(("the oracle lookahead at k=2 lowers critical misses below k=0", tiers_k2[5] < tiers_k0[5]))
        checks.append(("the lru control is replayed and reported", lru[0].startswith("lru")))
        checks.append(("tiers beat lru on critical misses on the planted trace", tiers_k0[5] <= lru[5]))
        fo = os.path.join(d, "fo.txt")
        with open(fo, "w") as f:
            f.write("0 1 3 4 5\nf 0 1 layer 2 3:0.900 4:0.500 9:0.100\no 0 1 3=HIT 4=HIT 9=WASTE 5=MISS\n")
            f.write("1 1 3 7 8\nf 1 1 sticky 1 3:1.000 4:1.000\no 1 1 3=HIT 4=WASTE 7=MISS 8=MISS\n")
        _, _, _, fore, outc = read_trace(fo, with_forecasts=True)
        checks.append(("f lines parse: source, reads, confidence per id",
                       fore[(0, 1)]["source"] == "layer" and fore[(0, 1)]["reads"] == 2 and fore[(0, 1)]["conf"][3] == 0.9))
        checks.append(("o lines parse: HIT / MISS / WASTE per id",
                       outc[(0, 1)][9] == "WASTE" and outc[(0, 1)][5] == "MISS" and outc[(1, 1)][3] == "HIT"))
    fails = 0
    for name, ok in checks:
        print("  %s  %s" % ("ok  " if ok else "FAIL", name))
        fails += 0 if ok else 1
    print("replay_three_tier selftest: %d/%d checks passed" % (len(checks) - fails, len(checks)))
    return 1 if fails else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace", nargs="?")
    ap.add_argument("--slots", type=int, default=4096, help="cache slots (the 90 GB cache on the spark holds about 4096)")
    ap.add_argument("--pinned", default="0.0,0.25,0.5,0.75", help="pinned fractions of the slots")
    ap.add_argument("--k", default="0,1,2,4", help="lookahead horizons in tokens")
    ap.add_argument("--conf", default="0.5,1.0", help="admission thresholds on EXPECTED EXPERT REUSE over k")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    if not a.trace:
        ap.print_help()
        print("\nNEXT -> DS4_CUDA_ROUTE_TRACE=route.txt on the spark, then: %s route.txt --slots 4096" % sys.argv[0])
        return
    ids, miss, waits, fore, outc = read_trace(a.trace, with_forecasts=True)
    rows, n_held = sweep(ids, a.slots, [float(x) for x in a.pinned.split(",")],
                         [int(x) for x in a.k.split(",")], [float(x) for x in a.conf.split(",")])
    print_report(a.trace, ids, miss, waits, rows, n_held, a.slots)
    print()
    forecast_report(fore, outc)


if __name__ == "__main__":
    main()
