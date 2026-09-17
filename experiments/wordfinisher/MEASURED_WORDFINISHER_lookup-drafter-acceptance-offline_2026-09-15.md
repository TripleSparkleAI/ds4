# MEASURED - lookup-drafter acceptance, offline, no GPU and no lock

Date 2026-09-15. Lane WORDFINISHER. Tree `origin/main` at `9139e2ae5`, branch `triple-word-finisher`.

## What was measured

The acceptance rate of a suffix-lookup drafter under EXACT acceptance: a drafted token is
committed only when it matches the token the target actually emits, so the output stream is
byte-identical to unaided decode by construction.

Method: replay a fixed token stream. At each verify pass the drafter proposes `draft_len` tokens
from the continuation stored against the last `n` tokens; the pass commits the longest matching
PREFIX of that proposal plus the one token the target emits anyway; the stream advances by the
accepted length; the true continuation is then learned. `mean_acc` is mean committed tokens per
verify pass, which is the speedup ceiling before any verify cost is charged.

Tokeniser: `./ds4 -m DeepSeek-V4.1-Flash-Q2.gguf --raw --dump-tokens --prompt-file <f>`. This is a
TOKENISER-ONLY load, measured at 0.07 s wall and 206 MiB RSS with zero major page faults, so it
needs no GPU and no model lock.

Corpora: `speed-bench/promessi_sposi.txt` whole (419,509 tokens, non-repetitive) and
`ds4_cuda.cu` first 900,000 bytes (270,886 tokens, repetitive).

## Results, cache 64k entries

    corpus  n  draft   passes    mean_acc  propose%  hit%
    prose   1      8  359,096      1.1682      98.0  13.0
    prose   2      8  358,802      1.1692      68.4  18.2
    prose   3      8  394,454      1.0635      21.6  20.8
    prose   4      8  410,627      1.0216       5.9  24.1
    code    1      8  131,647      2.0577      96.7  39.7
    code    2      8  114,182      2.3724      75.7  52.2
    code    2     16  106,030      2.5548      73.9  50.5
    code    3      8  119,428      2.2682      53.4  60.8
    code    4      8  130,692      2.0727      38.1  66.5

Accepted-length distribution, share of verify passes:

    prose  n=2 L=8    1 token 88%   2  9%   3  2%   4  1%   5+ ~0%
    code   n=2 L=8    1 token 60%   2 14%   3  7%   4  4%   5  3%   6+ 2%

## Design answers, each with the measurement behind it

**Suffix length n=2, bracketed on both sides.** The hit rate rises monotonically with n and does
not matter, because the propose rate collapses faster. n=1 saturates the propose rate at 98% and
loses 15% of mean accepted length on code. There is an interior optimum and it is n=2.

**Cache 64k entries, LRU, and the eviction policy is close to irrelevant.** 64k, 1M and unbounded
agree to the third decimal on code and to about 0.007 on prose. A size-limited cache costs
essentially nothing against a perfect one.

**Draft length 8.** Going 2 to 8 moves code 1.775 to 2.372 and moves prose 1.158 to 1.169. Length
past 8 buys little: 16 adds 0.18 on code and 0.0001 on prose.

## Rejected alternatives

Longer suffix keys (n of 3 or 4) are rejected on the propose-rate collapse above, not on taste.
A larger or unbounded cache is rejected because it is measurably worth nothing. Eviction policies
beyond LRU are rejected for the same reason. Draft lengths past 16 are rejected as unmeasurable
against the verify width they cost.

## What these numbers are not

Natural text, not model output. No long greedy transcript exists on the box; every
`--dump-logprobs` dump there is the 32-token gate, too short for statistics. Greedy output is
generally more repetitive than natural prose, so the prose row is a LOWER bound for prose-like
generation and the code row is the honest figure for code-like generation.

The owed arm is one long real generation replayed identically, which is a single `-n 512` run
under the model lock.

No speedup is claimed. `mean_acc` is a ceiling that charges nothing for the verify pass, and the
verify pass's expert-read cost is the open question that decides whether the ratios survive.

## Finding 3, measured, and it closes the lane

Source, with its chain of custody. Lane PREFILLHANDOFF's committed handoff probe, run by them
under the model lock; I booked nothing. The file I scored, on the Spark:

    /home/hologram/handoff-run-probe-final/long_probe.log   sha256 1bde97f7b98b15711fcbeb283454b17ba3fa472cf7a2f57a7a280290f2430acf
    /home/hologram/handoff-run-probe-final/short_probe.log  sha256 8a545a2ee78823c3c837215a26dbb173d0afbf2af75599d0eadfb501fbf38574

The logs are another lane's artifact and are deliberately not committed here; that lane banks its
own runs. The verdict does not rest on the file being retrievable: the four engine counters below
are printed by the engine itself on every run, so any repeat run re-derives the cross-check.

Scored decode rows only (`nsel=6`, where the probe's `L=` is the complete top-6). No booking was made and
no probe of mine was written. 40 layers, 63 decode tokens.

Validated against the engine's own decode counters before concluding anything, all four exact:
rows 2,520, selected 15,120, hits 13,805, misses 1,315. Recomputed through a second, separately
written code path that imports none of the scorer; every figure reproduces.

    k   union_growth   experts/layer   miss_union_saving
    2         0.8055            9.67             0.0000
    4         0.6525           15.66             0.0000
    8         0.5185           24.89             0.0000

    adjacent routing overlap   38.90%
    miss reuse                 0 of 1,219

Selections union usefully and the bytes do not. Four adjacent tokens touch 15.66 experts per layer
instead of 24, a real 35 percent reduction, but the experts that union are already resident, so
they are hits and the saving falls on bytes nobody was going to read. The misses are disjoint: the
union of k steps' misses equals their sum exactly, so a k-token verify pass reads the same miss
bytes as k single passes.

The pre-committed kill condition was union growth above about 0.85 at k=4, or miss reuse above
about 0.8. Measured, 0.6525 and 0.0000. Neither fired. The condition was aimed at the union of
selections; the quantity that decides the lane is the union of misses. The line was mis-specified
and the correct quantity kills the lane outright. Recorded as such rather than as a threshold
honoured.

## Verdict

Closed, no patch. Acceptance is real - 2.37x on code, 1.17x on prose with 88 percent of passes
committing one token - and it does not matter on this model, because the cost it would amortise
does not amortise. The expert-read saving of a k-token verify pass is zero.

Not claimed: that speculative decoding is worthless here. Committing k tokens in one pass still
saves per-pass fixed costs, which is a different lever and is unmeasured.

The constraint that outlives the lane: the batched verifier and one-token decode run different
floating-point reduction orders, upstream does not promise byte-identical output from the batched
verifier, and the exact variant handles two tokens only. Exact argmax acceptance is not sufficient
on its own.

## Reproduce

The token files are regenerable in about 0.3 s and are deliberately not committed. Make them
first, on a box that holds the GGUF. This is a tokeniser-only load: no GPU, no model lock.

    D=~/dwarfstar/gguf/DeepSeek-V4.1-Flash-Q2.gguf
    head -c 1329139 speed-bench/promessi_sposi.txt > prose.txt
    head -c 900000  ds4_cuda.cu                    > code.txt
    ./ds4 -m $D --raw --dump-tokens --prompt-file prose.txt > prose.tok
    ./ds4 -m $D --raw --dump-tokens --prompt-file code.txt > code.tok

Then the replay, which needs no box at all:

    python3 wordfinisher_replay.py prose.tok code.tok

Finding 3, once lane PREFILLHANDOFF's handoff probe log exists:

    python3 wordfinisher_union.py <probe-stderr-log>
