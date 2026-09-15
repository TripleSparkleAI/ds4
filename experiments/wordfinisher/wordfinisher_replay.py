#!/usr/bin/env python3
"""wordfinisher_replay - offline acceptance rate of a lookup drafter, no GPU, no model.

# <claudes_code_comments>
# ** Function List **
# load_tokens(path) - read a ds4 --dump-tokens file into a list of ints
# replay(toks, n, draft_len, cache_cap, full_context) - one arm; returns pass stats
# main() - sweep n x draft_len x corpus, print the table
#
# ** Technical Review **
# Simulates EXACT-ACCEPTANCE speculative decoding with a lookup drafter over a fixed
# token stream. At each verify pass the drafter proposes `draft_len` tokens from the
# continuation stored against the last `n` tokens; the pass accepts the longest
# matching PREFIX of that proposal, plus the one token the target emits anyway.
# Accepted length per pass is therefore 1 + matched, and the stream advances by that
# amount - which is what makes mean-accepted-length the real speedup denominator.
# Two drafter variants: a size-limited LRU cache keyed on the suffix (our design) and
# an unbounded full-context search (the ceiling). The table is updated with the TRUE
# continuation after every pass, so the drafter only ever sees tokens already emitted.
# </claudes_code_comments>
"""
import sys, collections

def load_tokens(path):
    toks = []
    with open(path, 'r', errors='replace') as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            first = s.split(None, 1)[0]
            try:
                toks.append(int(first))
            except ValueError:
                continue
    return toks

def replay(toks, n, draft_len, cache_cap=None, full_context=False):
    """Return (passes, emitted, accepted_lengths_hist, proposals_made, proposals_hit)."""
    N = len(toks)
    table = collections.OrderedDict()   # suffix tuple -> tuple of next draft_len tokens
    passes = 0
    emitted = 0
    hist = collections.Counter()
    proposals = 0
    proposal_hits = 0
    i = 0
    while i < N:
        # the target emits toks[i] regardless; that is the free token of the pass
        accepted = 1
        if i >= n:
            key = tuple(toks[i - n + 1:i + 1])   # suffix ENDING at the token just emitted
            prop = table.get(key)
            if prop is not None:
                if cache_cap is not None:
                    table.move_to_end(key)
                proposals += 1
                m = 0
                while m < len(prop) and i + 1 + m < N and prop[m] == toks[i + 1 + m]:
                    m += 1
                if m > 0:
                    proposal_hits += 1
                accepted += m
        passes += 1
        hist[accepted] += 1
        emitted += accepted
        # learn: record the true continuation for every suffix inside the accepted span
        for j in range(i, min(i + accepted, N)):
            if j >= n - 1:
                k = tuple(toks[j - n + 1:j + 1])
                v = tuple(toks[j + 1:j + 1 + draft_len])
                if v:
                    table[k] = v
                    if cache_cap is not None:
                        table.move_to_end(k)
                        if len(table) > cache_cap:
                            table.popitem(last=False)
        i += accepted
    return passes, emitted, hist, proposals, proposal_hits

def main():
    corpora = sys.argv[1:]
    if not corpora:
        print("usage: wordfinisher_replay.py <tokfile> [<tokfile> ...]")
        return 2
    for path in corpora:
        toks = load_tokens(path)
        print(f"\n=== {path}  tokens={len(toks):,}  distinct={len(set(toks)):,} ===")
        print(f"{'n':>2} {'draft':>5} {'cache':>9} {'passes':>9} {'emitted':>9} "
              f"{'mean_acc':>8} {'speedup':>7} {'prop%':>6} {'hit%':>6}")
        for cap_label, cap in (("1M", 1_000_000), ("64k", 65_536), ("unbounded", None)):
            for n in (2, 3, 4):
                for L in (2, 4, 8):
                    p, e, hist, prop, hit = replay(toks, n, L, cache_cap=cap)
                    mean_acc = e / p if p else 0.0
                    prop_rate = 100.0 * prop / p if p else 0.0
                    hit_rate = 100.0 * hit / prop if prop else 0.0
                    print(f"{n:>2} {L:>5} {cap_label:>9} {p:>9,} {e:>9,} "
                          f"{mean_acc:>8.4f} {mean_acc:>7.3f}x {prop_rate:>6.1f} {hit_rate:>6.1f}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
