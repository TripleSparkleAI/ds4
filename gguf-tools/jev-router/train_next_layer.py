#!/usr/bin/env python3
"""train_next_layer.py - score three guess sources for the scout on one routing trace, export one.

Reads a routing trace written by the engine under DS4_CUDA_ROUTE_TRACE (one line per token per
MoE layer: `token layer id0 .. idK`, plus an optional `h token v0 .. vD` line per token carrying
the final hidden state when the engine logs it). Splits the tokens in ORDER (the last 20 % held
out; routing is a time series and a shuffled split leaks the neighbourhood), fits on the first
80 %, and prints one table with the held-out hit rate per layer and averaged for:

  (a) sticky   the last token's experts at the same layer (the scout's guess today, the baseline)
  (b) layer    layer L's routed ids of THIS token -> this token's top-K at L+1 (a per-layer
               successor table; with router logits in the trace it would be a logistic head)
  (c) ahead    token t's whole routing signature (all layers' ids; or its hidden state when the
               trace carries one) -> ALL of token t+1's top-K at every layer, one pass

Hit rate = |guess ∩ actual| / |actual| per (token, layer), averaged over held-out tokens. The
guess is capped at K ids so a source cannot win by guessing more than the scout may read.

Exports the (b) table as `ds4-jev-router 1` text for ds4_jev_router_logic.h, and, when (c) beats
(a), a second table with the token-ahead head, so the engine hook can take either.

  python3 train_next_layer.py TRACE [--out table.txt] [--out-ahead ahead.txt] [--k 8]
  python3 train_next_layer.py --synth trace.txt [--tokens 4000] [--sticky-prob 0.5]
  python3 train_next_layer.py --selftest

numpy only. No number a synthetic trace produces is a routing fact.
"""
# <claudes_code_comments>
# ** Function List **
# read_trace(path) - parse trace lines into per-token id arrays and optional hidden states
# split_in_order(n_tokens, frac) - held-out = last frac of tokens, in order
# hit_rate(guess, actual) - |guess ∩ actual| / |actual|
# score_sticky(tok, held) - source (a): last token's ids at the same layer
# fit_layer_table(tok, train, E, K) - source (b): per-(L, e) successor counts L -> L+1
# predict_layer(table, ids, sticky_ids, w, K) - the engine's predict, in numpy, for (b)
# score_layer(...) - held-out hit rate of (b), per layer
# fit_ahead(tok, hid, train, E, K) - source (c): per target layer, a linear head from the signature
# score_ahead(...) - held-out hit rate of (c), per layer
# export_table(path, table, ...) - write the ds4-jev-router 1 text table
# synth_trace(path, ...) - a planted-structure trace for the selftest
# selftest() - the instrument reads what is planted
# main() - the CLI
#
# ** Technical Review **
# One pass over the trace builds tok[t][L] = sorted top-K ids (K = n_expert_used) and, when
# present, hid[t] = the hidden-state vector. All three sources are scored on the SAME held-out
# tokens with the SAME hit-rate definition and the SAME cap, so the table's rows are comparable.
# (b) is a co-occurrence table normalised per source expert (P(e' at L+1 | e at L)), scored by
# summing over the token's ids at L and adding a sticky bonus w for the last token's L+1 ids; w
# and the min score are picked on the TRAIN split by a small grid, never on held-out. (c) is a
# ridge-regularised linear head per target layer from the signature (a 0/1 bag over
# layers x experts, or the hidden state) to a 0/1 target over experts; the top-K scores are the
# guess. The synthetic trace plants a permutation from L to L+1 (so (b) has something to learn)
# and a token-to-token stickiness probability (so (a) reads near it); selftest asserts both.
# </claudes_code_comments>
import argparse
import os
import sys
import tempfile

try:
    import numpy as np
except ImportError:  # pragma: no cover
    sys.stderr.write("train_next_layer.py: numpy is required\n")
    sys.exit(2)


def read_trace(path):
    """Returns (tok, hid): tok is a dict token -> dict layer -> list of ids; hid token -> array or None."""
    tok, hid = {}, {}
    with open(path) as f:
        for raw in f:
            parts = raw.split()
            if not parts or parts[0].startswith("#"):
                continue
            if parts[0] == "h":
                t = int(parts[1])
                hid[t] = np.asarray([float(x) for x in parts[2:]], dtype=np.float32)
                continue
            if not parts[0].isdigit():
                continue  # m / w / f / o records belong to replay_three_tier.py
            t, layer = int(parts[0]), int(parts[1])
            ids = [int(x) for x in parts[2:]]
            tok.setdefault(t, {})[layer] = ids
    return tok, hid


def split_in_order(n_tokens, frac=0.2):
    cut = int(round(n_tokens * (1.0 - frac)))
    cut = max(1, min(n_tokens - 1, cut))
    return list(range(cut)), list(range(cut, n_tokens))


def hit_rate(guess, actual):
    if not actual:
        return None
    a = set(actual)
    return len(a.intersection(guess)) / float(len(a))


def score_sticky(tok, held, layers):
    per = {L: [] for L in layers}
    for t in held:
        if t - 1 not in tok:
            continue
        for L in layers:
            if L in tok[t] and L in tok[t - 1]:
                per[L].append(hit_rate(tok[t - 1][L], tok[t][L]))
    return per


def fit_layer_table(tok, train, E, layers):
    """counts[L][e, e'] = tokens routing to e at L and e' at L+1; rows normalised on demand."""
    counts = {}
    for L in layers:
        if L + 1 not in layers:
            continue
        c = np.zeros((E, E), dtype=np.float64)
        for t in train:
            if L in tok[t] and L + 1 in tok[t]:
                for e in set(tok[t][L]):
                    for e2 in set(tok[t][L + 1]):
                        c[e, e2] += 1.0
        counts[L] = c
    return counts


def _normalise(counts):
    probs = {}
    for L, c in counts.items():
        row = c.sum(axis=1, keepdims=True)
        row[row == 0] = 1.0
        probs[L] = c / row
    return probs


def predict_layer(probs, L, ids, sticky_ids, w, min_score, K):
    if L not in probs:
        return []
    score = np.zeros(probs[L].shape[1], dtype=np.float64)
    any_row = False
    for e in set(ids):
        if probs[L][e].any():
            any_row = True
            score += probs[L][e]
    if not any_row:
        return []
    for e in set(sticky_ids or []):
        score[e] += w
    order = np.lexsort((np.arange(len(score)), -score))  # best score first, lower id on ties
    out = [int(e) for e in order if score[e] >= min_score and score[e] > 0.0]
    return out[:K]


def score_layer(tok, probs, tokens, layers, w, min_score, K):
    per = {L: [] for L in layers}
    for t in tokens:
        for L in layers:
            if L not in tok[t] or L + 1 not in tok[t]:
                continue
            sticky = tok[t - 1].get(L + 1) if t - 1 in tok else None
            g = predict_layer(probs, L, tok[t][L], sticky, w, min_score, K)
            if not g and sticky:
                g = sticky  # the engine falls back to stickiness on an empty row
            per[L + 1].append(hit_rate(g, tok[t][L + 1]))
    return per


def pick_weights(tok, probs, train, layers, K):
    """Grid over the sticky bonus and the minimum score on the TRAIN split only."""
    best, best_v = (0.0, 0.0), -1.0
    tail = train[len(train) // 2:]  # fit-half/pick-half inside train; held-out untouched
    for w in (0.0, 0.25, 0.5, 1.0):
        for m in (0.0, 0.05, 0.15):
            per = score_layer(tok, probs, tail, layers, w, m, K)
            v = _avg(per)
            if v is not None and v > best_v:
                best, best_v = (w, m), v
    return best


def _avg(per):
    vals = [x for L in per for x in per[L] if x is not None]
    return float(np.mean(vals)) if vals else None


def _signature(tok, t, layers, E, hid):
    if hid and t in hid:
        return hid[t]
    x = np.zeros(len(layers) * E, dtype=np.float32)
    for i, L in enumerate(layers):
        for e in tok[t].get(L, []):
            x[i * E + e] = 1.0
    return x


def fit_ahead(tok, hid, train, E, layers, ridge=1.0):
    """Per target layer L: a ridge linear head from token t's signature to token t+1's ids at L."""
    X, Ys = [], {L: [] for L in layers}
    for t in train:
        if t + 1 not in tok:
            continue
        X.append(_signature(tok, t, layers, E, hid))
        for L in layers:
            y = np.zeros(E, dtype=np.float32)
            for e in tok[t + 1].get(L, []):
                y[e] = 1.0
            Ys[L].append(y)
    if not X:
        return None
    X = np.asarray(X, dtype=np.float64)
    G = X.T @ X + ridge * np.eye(X.shape[1])
    Ginv = np.linalg.inv(G)
    heads = {}
    for L in layers:
        Y = np.asarray(Ys[L], dtype=np.float64)
        heads[L] = Ginv @ (X.T @ Y)  # [dim, E]
    return heads


def score_ahead(tok, hid, heads, tokens, layers, E, K):
    per = {L: [] for L in layers}
    if heads is None:
        return per
    for t in tokens:
        if t - 1 not in tok:
            continue
        x = _signature(tok, t - 1, layers, E, hid).astype(np.float64)
        for L in layers:
            if L not in tok[t]:
                continue
            with np.errstate(all="ignore"):  # Accelerate-numpy warns spuriously on 0/1 inputs
                s = x @ heads[L]
            if not np.isfinite(s).all():
                per[L].append(0.0)  # a head that returns non-finite scores guesses nothing
                continue
            order = np.lexsort((np.arange(E), -s))
            per[L].append(hit_rate([int(e) for e in order[:K]], tok[t][L]))
    return per


def export_table(path, counts, E, n_layers, K, w, min_score):
    probs = _normalise(counts)
    with open(path, "w") as f:
        f.write("ds4-jev-router 1\n")
        f.write("layers %d experts %d k %d sticky %.4f min %.4f\n" % (n_layers, E, K, w, min_score))
        for L in sorted(probs):
            for e in range(E):
                row = probs[L][e]
                if not row.any():
                    continue
                top = np.lexsort((np.arange(E), -row))[:K]
                pairs = ["%d %.4f" % (int(e2), row[e2]) for e2 in top if row[e2] > 0.0]
                f.write("%d %d %s\n" % (L, e, " ".join(pairs)))


def synth_trace(path, tokens=3000, layers=8, E=64, K=6, sticky_prob=0.5, perm_noise=0.15, seed=7):
    """Plant: layer L+1's ids are a fixed permutation of layer L's ids (with noise); token t's
    layer-0 ids keep each of token t-1's with probability sticky_prob."""
    rng = np.random.default_rng(seed)
    perms = [rng.permutation(E) for _ in range(layers)]
    prev0 = list(rng.choice(E, K, replace=False))
    with open(path, "w") as f:
        f.write("# synthetic trace, planted permutation + stickiness %.2f\n" % sticky_prob)
        for t in range(tokens):
            keep = [e for e in prev0 if rng.random() < sticky_prob]
            pool = [e for e in range(E) if e not in keep]
            ids0 = keep + list(rng.choice(pool, K - len(keep), replace=False))
            cur = ids0
            for L in range(layers):
                f.write("%d %d %s\n" % (t, L, " ".join(str(int(e)) for e in sorted(cur))))
                nxt = [int(perms[L][e]) for e in cur]
                for i in range(len(nxt)):
                    if rng.random() < perm_noise:
                        nxt[i] = int(rng.integers(E))
                nxt = list(dict.fromkeys(nxt))
                while len(nxt) < K:
                    c = int(rng.integers(E))
                    if c not in nxt:
                        nxt.append(c)
                cur = nxt
            prev0 = ids0
    return path


def run(trace, out=None, out_ahead=None, K=8, quiet=False):
    tok, hid = read_trace(trace)
    tokens = sorted(tok)
    if len(tokens) < 10:
        raise SystemExit("train_next_layer.py: fewer than 10 tokens in %s" % trace)
    # re-index tokens densely in order
    tok = {i: tok[t] for i, t in enumerate(tokens)}
    hid = {i: hid[t] for i, t in enumerate(tokens) if t in hid} if hid else {}
    layers = sorted({L for t in tok for L in tok[t]})
    E = 1 + max(e for t in tok for L in tok[t] for e in tok[t][L])
    n_layers = 1 + max(layers)
    train, held = split_in_order(len(tok))
    counts = fit_layer_table(tok, train, E, layers)
    probs = _normalise(counts)
    w, m = pick_weights(tok, probs, train, layers, K)
    a = score_sticky(tok, held, layers)
    b = score_layer(tok, probs, held, layers, w, m, K)
    heads = fit_ahead(tok, hid, train, E, layers)
    c = score_ahead(tok, hid, heads, held, layers, E, K)
    res = {"layers": layers, "E": E, "K": K, "w": w, "min": m, "n_train": len(train), "n_held": len(held),
           "a": a, "b": b, "c": c, "avg_a": _avg(a), "avg_b": _avg(b), "avg_c": _avg(c),
           "c_signature": "hidden state (%d dims)" % len(next(iter(hid.values()))) if hid else
                          "routing signature (%d layers x %d experts, 0/1)" % (len(layers), E)}
    if not quiet:
        print_table(res, trace)
    if out:
        export_table(out, counts, E, n_layers, K, w, m)
        if not quiet:
            print("wrote (b) table: %s" % out)
    if out_ahead and heads is not None and res["avg_c"] is not None and res["avg_a"] is not None \
            and res["avg_c"] > res["avg_a"]:
        np.savez(out_ahead, **{"L%d" % L: heads[L] for L in heads})
        if not quiet:
            print("wrote (c) heads: %s (numpy; no engine loader for it yet, see NOTES)" % out_ahead)
    return res


def print_table(res, trace):
    print("routing trace: %s" % trace)
    print("tokens: %d train, %d held out (in order, the last 20 %%) · experts %d · guess cap K=%d"
          % (res["n_train"], res["n_held"], res["E"], res["K"]))
    print("(b) picked on train: sticky bonus %.2f, min score %.2f · (c) input: %s"
          % (res["w"], res["min"], res["c_signature"]))
    print()
    print("  layer   (a) sticky   (b) layer L->L+1   (c) token-ahead     b-a      c-a")
    for L in res["layers"]:
        a = _avg({L: res["a"][L]})
        b = _avg({L: res["b"][L]})
        c = _avg({L: res["c"][L]})
        f = lambda v: ("%.3f" % v) if v is not None else "  -  "
        d = lambda x, y: ("%+.3f" % (x - y)) if (x is not None and y is not None) else "  -   "
        print("  %5d   %10s   %16s   %15s   %s   %s" % (L, f(a), f(b), f(c), d(b, a), d(c, a)))
    a, b, c = res["avg_a"], res["avg_b"], res["avg_c"]
    print("  %5s   %10.3f   %16.3f   %15.3f   %+.3f   %+.3f"
          % ("avg", a or 0.0, b or 0.0, c or 0.0, (b or 0.0) - (a or 0.0), (c or 0.0) - (a or 0.0)))
    print()
    if a is not None and a >= 0.90:
        print("FINDING: stickiness is already %.1f %% on held-out tokens; the guess was never the"
              " problem and a predictor is not worth the hot path." % (100.0 * a))
    elif c is not None and a is not None and c > a and c >= (b or 0.0):
        print("FINDING: (c) token-ahead beats sticky by %+.3f; the hook takes (c) for the next token's"
              " misses and (b) as the per-layer correction." % (c - a))
    elif b is not None and a is not None and b > a:
        print("FINDING: (b) beats sticky by %+.3f and (c) does not; the hook takes (b)." % (b - a))
    else:
        print("FINDING: neither head beats sticky on this trace; the table is not worth loading.")


def selftest():
    fails = 0
    with tempfile.TemporaryDirectory() as d:
        tr = synth_trace(os.path.join(d, "synth.txt"), tokens=2000, sticky_prob=0.5)
        out = os.path.join(d, "table.txt")
        res = run(tr, out=out, K=6, quiet=True)
        a, b, c = res["avg_a"], res["avg_b"], res["avg_c"]
        checks = [
            ("(a) reads near the planted stickiness at layer 0", abs(_avg({0: res["a"][0]}) - 0.5) < 0.12),
            ("(b) beats (a) on the planted permutation", b is not None and b > a + 0.2),
            ("(b) is high where the permutation is planted", b > 0.6),
            ("(c) is scored (not None)", c is not None),
            ("the exported table loads back as text with the header", open(out).readline().strip() == "ds4-jev-router 1"),
            ("held-out is the last 20 % in order", res["n_held"] == 400 and res["n_train"] == 1600),
        ]
        # a trace with NO structure: (b) must not beat (a) by more than noise
        rng = np.random.default_rng(3)
        flat = os.path.join(d, "flat.txt")
        with open(flat, "w") as f:
            for t in range(1500):
                for L in range(4):
                    f.write("%d %d %s\n" % (t, L, " ".join(str(int(e)) for e in rng.choice(64, 6, replace=False))))
        rf = run(flat, K=6, quiet=True)
        checks.append(("no structure: (b) within 0.05 of (a)", abs(rf["avg_b"] - rf["avg_a"]) < 0.05))
        checks.append(("no structure: (a) near chance 6/64", abs(rf["avg_a"] - 6.0 / 64.0) < 0.04))
        # a hidden-state line per token is read and used by (c)
        hs = os.path.join(d, "hid.txt")
        with open(hs, "w") as f, open(tr) as src:
            for line in src:
                f.write(line)
                p = line.split()
                if p and p[0] != "#" and p[1] == "0":
                    v = np.zeros(64); v[[int(x) for x in p[2:]]] = 1.0  # the hidden state IS the layer-0 ids
                    f.write("h %s %s\n" % (p[0], " ".join("%.1f" % x for x in v)))
        rh = run(hs, K=6, quiet=True)
        checks.append(("hidden-state line is read for (c)", rh["c_signature"].startswith("hidden state (64")))
        checks.append(("(c) from a hidden state that encodes layer 0 beats sticky at layer 1",
                       _avg({1: rh["c"][1]}) > _avg({1: rh["a"][1]})))
    for name, ok in checks:
        print("  %s  %s" % ("ok  " if ok else "FAIL", name))
        fails += 0 if ok else 1
    print("train_next_layer selftest: %d/%d checks passed" % (len(checks) - fails, len(checks)))
    return 1 if fails else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace", nargs="?", help="routing trace from DS4_CUDA_ROUTE_TRACE")
    ap.add_argument("--out", help="write the (b) table for DS4_CUDA_JEV_ROUTER")
    ap.add_argument("--out-ahead", help="write the (c) heads (.npz) when (c) beats sticky")
    ap.add_argument("--k", type=int, default=8, help="guess cap (the scout reads at most 8)")
    ap.add_argument("--synth", help="write a planted synthetic trace here and exit")
    ap.add_argument("--tokens", type=int, default=4000)
    ap.add_argument("--sticky-prob", type=float, default=0.5)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    if a.synth:
        synth_trace(a.synth, tokens=a.tokens, sticky_prob=a.sticky_prob)
        print("wrote synthetic trace: %s  NEXT -> python3 %s %s --out table.txt" % (a.synth, sys.argv[0], a.synth))
        return
    if not a.trace:
        ap.print_help()
        print("\nNEXT -> DS4_CUDA_ROUTE_TRACE=route.txt ./ds4-bench ... on the spark, then: %s route.txt --out table.txt"
              % sys.argv[0])
        return
    run(a.trace, out=a.out, out_ahead=a.out_ahead, K=a.k)


if __name__ == "__main__":
    main()
