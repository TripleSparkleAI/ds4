#!/usr/bin/env python3
"""wordfinisher_union - adjacent-token expert union, offline, from a handoff probe log.

# <claudes_code_comments>
# ** Function List **
# parse_rows(path) - tolerant parse of the probe's `HO ...` stderr rows
# steps_by_layer(rows) - group rows into per-layer ordered steps
# score(rows, k_max) - the three numbers: union growth, miss reuse, k-pass saving
# main() - print the table
#
# ** Technical Review **
# Reads lane PREFILLHANDOFF's committed probe dump (see its `471eebb47`), whose rows carry
# `call`, `layer`, the selected expert ids `L=`, the hits `H=` and the misses `M=`. Rows are
# grouped per layer and ordered by `call`, so consecutive rows in a layer are consecutive
# steps for that layer.
#
# It answers finding 3 without any box time. Three quantities, and the middle one is the one
# that decides the lane:
#   UNION GROWTH   |union(S_t..S_t+k-1)| / (k * |S_t|). 1.0 means k tokens select k disjoint
#                  sets and a k-token verify pass unions nothing; 1/k means they select the
#                  same set and the pass is free after the first token.
#   MISS REUSE     fraction of step t+1's MISSES that were in step t's SELECTED set. High means
#                  the persistent expert arena already collects the overlap, so batching adds
#                  nothing on the expert-read axis. Near zero means the misses are genuinely new
#                  experts and k of them union to nothing either. Only the middle pays.
#   K-PASS SAVING  1 - |union of k steps' MISSES| / sum of k steps' miss counts. This is the
#                  bytes a k-token verify pass actually avoids against k single passes.
# </claudes_code_comments>
"""
import sys, re, collections

ROW = re.compile(r"\bcall=(\d+).*?\blayer=(\d+)")

def _ids(line, tag):
    m = re.search(r"\s" + tag + r"=([0-9,:?]*)", line)
    if not m:
        return []
    out = []
    for p in m.group(1).split(","):
        p = p.strip()
        if not p or p == "?":
            continue
        out.append(p.split(":")[-1])
    return [int(x) for x in out if x.lstrip("-").isdigit()]

def parse_rows(path, nsel_only=6):
    """Decode rows only by default.

    nsel is the probe's slot_count. A decode row has nsel=6, one token's top-6 for one
    layer, and `L=` prints the last 6 of selected_ids, so for nsel=6 it is the COMPLETE
    selected set. A prefill row (nsel in the thousands) has a truncated L= and its
    residency is filled by a second path the probe does not watch, so it is excluded:
    the union metric is only defined where L= is complete.
    """
    rows = []
    with open(path, "r", errors="replace") as f:
        for line in f:
            if "HO " not in line or "layer=" not in line:
                continue
            m = ROW.search(line)
            if not m:
                continue
            ns = re.search(r"\bnsel=(\d+)", line)
            if nsel_only is not None and (not ns or int(ns.group(1)) != nsel_only):
                continue
            rows.append({
                "call": int(m.group(1)),
                "layer": int(m.group(2)),
                "nsel": int(ns.group(1)) if ns else -1,
                "sel": _ids(line, "L"),
                "hit": _ids(line, "H"),
                "miss": _ids(line, "M"),
            })
    return rows

def steps_by_layer(rows):
    by = collections.defaultdict(list)
    for r in sorted(rows, key=lambda r: r["call"]):
        by[r["layer"]].append(r)
    return by

def score(rows, k_max=8):
    by = steps_by_layer(rows)
    print(f"layers={len(by)}  rows={len(rows)}")
    print(f"{'k':>2} {'union_growth':>13} {'miss_reuse':>11} {'k_pass_saving':>14} {'steps':>8}")
    for k in range(2, k_max + 1):
        gn = gd = 0
        mn = md = 0
        sn = sd = 0
        steps = 0
        for _, seq in by.items():
            for i in range(len(seq) - k + 1):
                win = seq[i:i + k]
                sel = [set(w["sel"]) for w in win]
                if not sel[0]:
                    continue
                u = set()
                for s in sel:
                    u |= s
                gn += len(u); gd += sum(len(s) for s in sel)
                mu = set()
                tot = 0
                for w in win:
                    mu |= set(w["miss"]); tot += len(w["miss"])
                sn += len(mu); sd += tot
                steps += 1
        # miss reuse is a pairwise quantity, computed once
        for _, seq in by.items():
            for i in range(len(seq) - 1):
                nxt = set(seq[i + 1]["miss"])
                if not nxt:
                    continue
                mn += len(nxt & set(seq[i]["sel"])); md += len(nxt)
        growth = gn / gd if gd else 0.0
        reuse = mn / md if md else 0.0
        saving = 1.0 - (sn / sd) if sd else 0.0
        print(f"{k:>2} {growth:>13.4f} {reuse:>11.4f} {saving:>14.4f} {steps:>8,}")
    print("\nunion_growth 1/k = k tokens select the same experts, a k-pass unions perfectly.")
    print("union_growth 1.0 = disjoint, a k-pass unions nothing and this lane is dead.")
    print("miss_reuse high = the persistent arena already collects the overlap.")

def main():
    if len(sys.argv) < 2:
        print("usage: wordfinisher_union.py <probe-log> [...]")
        return 2
    for p in sys.argv[1:]:
        rows = parse_rows(p)
        print(f"\n=== {p} (decode rows only, nsel=6) ===")
        if not rows:
            print("no decode HO rows found; is this the probe's stderr log?")
            continue
        # CROSS-CHECK against the engine's own counters before reporting anything.
        tm = sum(len(r["miss"]) for r in rows)
        th = sum(len(r["hit"]) for r in rows)
        tu = sum(len(r["sel"]) for r in rows)
        bad = [r["call"] for r in rows if len(r["sel"]) != 6]
        print(f"ENGINE CROSS-CHECK  rows={len(rows)}  selected={tu}  hits={th}  misses={tm}"
              f"  rows_without_6_selected={len(bad)}")
        score(rows)
    return 0

if __name__ == "__main__":
    sys.exit(main())
