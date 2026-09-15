#!/usr/bin/env python3
"""Answer the PREFILLHANDOFF lane's three questions from a DS4_PROBE_HANDOFF log.

Reads the `HO ` lines the probe build prints on stderr and replays the expert
cache exactly, because every membership change is in the log: a miss inserts a
gate, an eviction removes one, a hit changes nothing.

  Q1  what fraction of the prompt's experts is still resident when the first
      decode token starts
  Q2  how much the first 32 decode tokens miss, against the steady state
  Q3  of those early misses, what fraction was loaded during prefill and then
      evicted

Usage:  track_handoff_probe.py RUN.log [--early 32] [--json OUT.json]
"""
import argparse
import json
import re
import sys

LINE = re.compile(
    r"^HO call=(\d+) layer=(\d+) nsel=(\d+) uniq=(\d+) hits=(\d+) "
    r"evict=(\d+) res=(\d+) cap=(\d+) stamp=(\d+) "
    r"L=([^ ]*) H=([^ ]*) M=([^ ]*) E=([^ ]*)$"
)


def ids(field):
    """Parse a trailing-comma id list into ints."""
    return [int(x) for x in field.split(",") if x]


def pairs(field):
    """Parse a trailing-comma layer:expert list into (layer, expert) tuples."""
    out = []
    for item in field.split(","):
        if not item or item == "?":
            continue
        layer, _, expert = item.partition(":")
        out.append((int(layer), int(expert)))
    return out


def parse(path):
    calls = []
    for raw in open(path, errors="replace"):
        m = LINE.match(raw.strip())
        if not m:
            continue
        (call, layer, nsel, uniq, hits, evict, res, cap, stamp,
         last, hit_ids, miss_ids, evicted) = m.groups()
        calls.append({
            "call": int(call), "layer": int(layer), "nsel": int(nsel),
            "uniq": int(uniq), "hits": int(hits), "evict": int(evict),
            "res": int(res), "cap": int(cap), "stamp": int(stamp),
            "last": ids(last), "hit": ids(hit_ids), "miss": ids(miss_ids),
            "evicted": pairs(evicted),
        })
    return calls


def split_phases(calls):
    """Prefill calls carry a whole batch of tokens; a decode call carries one.

    Classified by nsel rather than by position, so a run with no prefill or a
    prompt of one token is still read correctly rather than silently mislabelled.
    """
    top_k = min((c["nsel"] for c in calls), default=0)
    prefill = [c for c in calls if c["nsel"] > top_k]
    boundary = None
    for i, c in enumerate(calls):
        if prefill and c["call"] > prefill[-1]["call"]:
            boundary = i
            break
    decode = calls[boundary:] if boundary is not None else []
    return top_k, prefill, decode


def group_tokens(decode):
    """A decode token is one sweep over the MoE layers, so a layer that does not
    advance starts a new token."""
    tokens, current, prev = [], [], None
    for c in decode:
        if prev is not None and c["layer"] <= prev:
            tokens.append(current)
            current = []
        current.append(c)
        prev = c["layer"]
    if current:
        tokens.append(current)
    return tokens


def replay(calls, upto=None):
    """Reconstruct the resident set, returning it at index `upto`."""
    resident = set()
    for i, c in enumerate(calls):
        if upto is not None and i >= upto:
            break
        for pair in c["evicted"]:
            resident.discard(pair)
        for expert in c["miss"]:
            resident.add((c["layer"], expert))
    return resident


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--early", type=int, default=32)
    ap.add_argument("--json")
    args = ap.parse_args()

    calls = parse(args.log)
    if not calls:
        sys.exit("no HO lines in %s (was DS4_PROBE_HANDOFF set?)" % args.log)

    top_k, prefill, decode = split_phases(calls)
    if not prefill:
        sys.exit("no prefill calls found; this log cannot answer Q1 or Q3")
    if not decode:
        sys.exit("no decode calls found; run with -n greater than 0")

    n_prefill = len(prefill)
    boundary = len(calls) - len(decode)
    resident_at_boundary = replay(calls, boundary)

    # Q1. Two populations, because they answer different questions: everything
    # the prompt touched, and the last prompt token's own experts, which are the
    # ones decode is most likely to want next.
    touched = set()
    for c in prefill:
        for expert in c["hit"] + c["miss"]:
            touched.add((c["layer"], expert))
    last_token = set()
    for c in prefill:
        for expert in c["last"][:top_k]:
            last_token.add((c["layer"], expert))

    survived = touched & resident_at_boundary
    last_survived = last_token & resident_at_boundary

    tokens = group_tokens(decode)
    early = tokens[: args.early]
    steady = tokens[args.early:]

    def miss_rate(group):
        if not group:
            return None, 0, 0
        misses = sum(len(c["miss"]) for t in group for c in t)
        lookups = sum(c["uniq"] for t in group for c in t)
        return misses / len(group), misses, lookups

    early_per_token, early_misses, early_lookups = miss_rate(early)
    steady_per_token, steady_misses, steady_lookups = miss_rate(steady)

    # Q3. An early miss is "evicted prefill work" when the prompt touched that
    # expert and it is no longer resident, so the read is one we already paid for.
    was_prefilled = 0
    for t in early:
        for c in t:
            for expert in c["miss"]:
                if (c["layer"], expert) in touched:
                    was_prefilled += 1

    cap = calls[0]["cap"]
    report = {
        "calls": len(calls), "top_k": top_k, "capacity_slots": cap,
        "prefill_calls": n_prefill,
        "prefill_tokens": prefill[0]["nsel"] // top_k if top_k else 0,
        "decode_tokens": len(tokens),
        "q1_touched": len(touched),
        "q1_resident_at_boundary": len(resident_at_boundary),
        "q1_survived": len(survived),
        "q1_survived_frac": len(survived) / len(touched) if touched else 0.0,
        "q1_last_token_experts": len(last_token),
        "q1_last_token_survived": len(last_survived),
        "q1_last_token_frac":
            len(last_survived) / len(last_token) if last_token else 0.0,
        "q2_early_tokens": len(early),
        "q2_early_misses_per_token": early_per_token,
        "q2_steady_tokens": len(steady),
        "q2_steady_misses_per_token": steady_per_token,
        "q2_early_hit_rate":
            1 - early_misses / early_lookups if early_lookups else None,
        "q2_steady_hit_rate":
            1 - steady_misses / steady_lookups if steady_lookups else None,
        "q3_early_misses": early_misses,
        "q3_early_misses_prefilled": was_prefilled,
        "q3_prize_frac": was_prefilled / early_misses if early_misses else 0.0,
    }

    print("PREFILLHANDOFF probe, %s" % args.log)
    print("  capacity %d slots, top-%d routing, %d prefill tokens, %d decode tokens"
          % (cap, top_k, report["prefill_tokens"], len(tokens)))
    print()
    print("Q1  the prompt touched %d expert triples; %d are resident at the first"
          % (len(touched), len(survived)))
    print("    decode token = %.1f%%. Arena held %d of %d slots at the boundary."
          % (100 * report["q1_survived_frac"], len(resident_at_boundary), cap))
    print("    Of the LAST prompt token's %d experts, %d survive = %.1f%%."
          % (len(last_token), len(last_survived),
             100 * report["q1_last_token_frac"]))
    print()
    if steady_per_token is None:
        print("Q2  only %d decode tokens; no steady state to compare against."
              % len(tokens))
    else:
        print("Q2  first %d decode tokens miss %.1f experts/token; steady state %.1f."
              % (len(early), early_per_token, steady_per_token))
        print("    hit rate %.4f early against %.4f steady."
              % (report["q2_early_hit_rate"], report["q2_steady_hit_rate"]))
    print()
    print("Q3  of %d early misses, %d were loaded during prefill and then evicted"
          % (early_misses, was_prefilled))
    print("    = %.1f%%. That is the size of the prize."
          % (100 * report["q3_prize_frac"]))

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(report, fh, indent=2)
        print("\nwrote %s" % args.json)


if __name__ == "__main__":
    main()
