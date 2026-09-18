#!/usr/bin/env python3
"""g1_compare.py - compare two `ds4 --dump-logprobs` runs: sidecar OFF vs ON.

# <claudes_code_comments>
# ** Function List **
# load(path) - the steps list of a --dump-logprobs JSON
# compare(off, on) - first divergence index, identical-token fraction, top-20 drift
# selftest() - synthetic step lists: identical, diverge at 3, planted drift
# main(argv) - CLI
#
# ** Technical Review **
# Greedy identity is not the gate for a quant change; this prints what the gate needs:
# (a) the step index of the first selected-token divergence (-1 if none over the shared
# length), (b) the fraction of positions with identical selected tokens, and (c) over the
# steps BEFORE the first divergence (where both runs saw the same prefix, so the logits
# are comparable), the max and mean abs logprob difference across the tokens the two
# top-20 lists share, plus the top-20 set overlap. The drift is what to hold against the
# q2 noise floor from gguf-tools/quality-testing.
# </claudes_code_comments>
"""
import json
import sys


def load(path):
    with open(path) as f:
        return json.load(f)["steps"]


def compare(off, on):
    n = min(len(off), len(on))
    first = -1
    same = 0
    for i in range(n):
        if off[i]["selected"]["id"] == on[i]["selected"]["id"]:
            same += 1
        elif first < 0:
            first = i
    comparable = n if first < 0 else first
    max_abs, sum_abs, count, overlap_sum = 0.0, 0.0, 0, 0.0
    for i in range(comparable):
        a = {t["token"]["id"]: t["logprob"] for t in off[i]["top_logprobs"]}
        b = {t["token"]["id"]: t["logprob"] for t in on[i]["top_logprobs"]}
        shared = set(a) & set(b)
        overlap_sum += len(shared) / max(1, len(a))
        for tok in shared:
            d = abs(a[tok] - b[tok])
            max_abs = max(max_abs, d)
            sum_abs += d
            count += 1
    return {
        "steps_off": len(off), "steps_on": len(on), "compared": n,
        "first_divergence": first,
        "identical_fraction": same / n if n else 0.0,
        "comparable_prefix_steps": comparable,
        "top20_max_abs_logprob_drift": max_abs,
        "top20_mean_abs_logprob_drift": sum_abs / count if count else 0.0,
        "top20_mean_overlap": overlap_sum / comparable if comparable else 0.0,
    }


def _steps(ids, lp_shift=0.0):
    return [{"selected": {"id": i}, "top_logprobs": [
        {"token": {"id": i}, "logprob": -0.1 + lp_shift}, {"token": {"id": i + 1000}, "logprob": -2.0}]}
        for i in ids]


def selftest():
    ok = True
    r = compare(_steps([1, 2, 3, 4]), _steps([1, 2, 3, 4]))
    ok &= r["first_divergence"] == -1 and r["identical_fraction"] == 1.0 and r["top20_max_abs_logprob_drift"] == 0
    r = compare(_steps([1, 2, 3, 4, 5]), _steps([1, 2, 3, 9, 5]))
    ok &= r["first_divergence"] == 3 and abs(r["identical_fraction"] - 0.8) < 1e-9 and r["comparable_prefix_steps"] == 3
    r = compare(_steps([1, 2, 3]), _steps([1, 2, 3], lp_shift=0.25))  # planted drift must show
    ok &= abs(r["top20_max_abs_logprob_drift"] - 0.25) < 1e-9
    print("g1_compare selftest:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv == ["--selftest"]:
        return selftest()
    if len(argv) != 2:
        print("usage: g1_compare.py OFF.json ON.json | --selftest")
        return 2
    for k, v in compare(load(argv[0]), load(argv[1])).items():
        print(f"  {k:32s} {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
