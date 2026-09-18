# THE CARD STANDARD v3 - small, credible, generated (2026-09-18)

The navigator, 2026-09-17: *"better results tables, clear, in ASCII tables. Simplify the text of all
READMEs. Standard, small, credible cards for antirez to read."* v3 replaces v2. The v2 defects it
fixes: a 27-star banner, a card box holding 80 lines of prose, three rounds stuffed into one GEN
field, and blurb sections nobody asked for. A v3 card is under 45 lines and a person reads it once.

## THE SIX PARTS, in order, nothing else below the banner

1. **THE BANNER**, one line, never a star row:
   `✦ TRIPLESPARKLE ✦  this branch's card is below; antirez's README above is unchanged`
2. **THE BOX**, right edge open, at most 14 lines: `BRANCH` + one status word · `WHAT` (1 to 3
   lines, plain) · `SWITCH` (one line) · `OUTPUT` (one line) · `BASE`.
3. **RESULTS**: one ASCII table, one row per sealed round the branch was in, newest first, columns
   exactly `round  variant  date  arm t/s  tip t/s  delta  sign  floor  verdict  n`, then one footer naming the
   measurand once, then the prefill line where known, then one source line per distinct result file (two rows of one round citing
   one file print one source line), in the order first seen. `variant` is blank unless the row's
   `variant` field is set; two rows of one round that differ only by a switch or a bench argument
   (`QD=16`, `cache=60GB`, `hits ON`) MUST carry it, or the table prints them identically. A branch never
   in a sealed round prints `RESULTS  none yet - never in a sealed round` and no table.
4. **NOTES**: at most 5 bullets, one line each, plain English: what it does, why, the honest caveat,
   what is owed.
5. **ALSO TRIED**, optional: at most 5 lines, one per OTHER `triple-*` branch a maintainer reading
   only this card might want to explore: `branch  best sealed delta  one line <= 84 chars`. It
   points; it never argues. Absent when the card has nothing to point at.
6. **NOTHING ELSE.** No figures, no history sections, no per-card ASCII art.

## THE RULES ON WORDS AND NUMBERS

- Status words, exactly: `POSITIVE` · `WORTH ZERO` · `NEGATIVE` · `NOT YET` · `CONTROL` · `TOOLING`.
- Verdict words, exactly: `BEATS` · `TIES` · `LOSES` · `PARTIAL`. The verdict is the sealed rule's
  word from the result file, never re-judged on the card.
- Every number is CURRENT and comes from a sealed round's result file, which the row cites. A number
  a later sealed round superseded stays in its own row; the table is a record, newest first.
- `OUTPUT` is `greedy-identical to the tip: G1 short <sha16> · long <sha16>` or `not gated`. A sha
  taken on a tree the branch is no longer on is `not gated`; the old sha may live in a note.
- No em-dashes anywhere. The regular hyphen only.
- Right edges open: no `┐ ┘ ╗ ╝ ┤ ╣` in the rendered part.

## THE TWO SOURCE FILES, and the tool

Everything below the banner is GENERATED from two files the branch carries at its root:

- `card.json` - the schema, pinned for every lane (no other field names):
  ```
  {"branch": "triple-x", "status": "POSITIVE",
   "what": ["line 1", "line 2"],                       1 to 3 lines, each <= 66 chars
   "switch": "DS4_X=0 turns it off",                    one line <= 66 chars; "none" is legal
   "output": {"gated": true, "short": "<sha16>", "long": "<sha16>"}   or {"gated": false}
   "base": {"branch": "triple-tip-2026-09-16", "sha": "12997e9c"},
   "results": [ {"round": "r9", "date": "2026-09-17", "arm_tps": 10.41, "tip_tps": 9.42,
                 "delta_pct": 12.09, "sign": "4/4", "floor_pct": 17.27, "verdict": "TIES", "n": 4,
                 "file": "2026-09-17-R9-RESULT-....md", "note": "optional, <= 80 chars",
                 "variant": "QD=16"} ],                  optional, <= 22 chars, rendered as a column; a note is never rendered
   "prefill": "r9 89.0 vs tip 85.6",                     or ""
   "tried": [ {"branch": "triple-pool", "best": "+9.6%", "line": "<= 84 chars"} ]}   optional, <= 5
  ```
  `output.long` may be omitted when only the short prompt was gated. `tried` may be omitted; when
  present each `branch` must be a `triple-*` name that exists as a local branch (the tool checks
  `git for-each-ref` in the worktree's repo), `best` is a signed percent (`+9.6%`) or `-`, and
  `line` is 1 to 84 chars. `index` ignores `tried`; only the branch's own README carries it.
- `NOTES.md` - 1 to 5 lines, each starting `- `, each <= 110 chars, no headings.

The tool is `track0_harness/card_tool.py` (stdlib, `--selftest`):

```
python3 track0_harness/card_tool.py render <worktree>   # rewrite README from the banner up to the index heading (EOF if none)
python3 track0_harness/card_tool.py lint   <worktree>   # the same checks + rendered == render(card.json) + the index section is index()'s
python3 track0_harness/card_tool.py index  <all-fastest-worktree>   # the index from every triple-* HEAD
```

A hand edit to the rendered part is a defect: `lint` reds on any drift from `render(card.json)`.
On `triple-all-fastest` the README also carries the index section (`## The index: every branch card`)
below its own card: `render` owns only the region from the banner up to that heading and leaves the
index byte-identical; `lint` compares that same region, and reds if the index section is present but
lacks its `Regenerated <stamp>, N cards.` line or carries a number of card boxes other than N.
`index` reads each branch's `card.json` at its HEAD with `git show`, groups by status, prints a
one-screen summary table (`branch  status  best sealed delta  latest round  verdict  rows`, sorted by
status then delta; `rows` is the number of sealed rounds on the card, the evidence depth) and then each branch's box and results table, never its notes. A branch with no
`card.json` yet falls back to its v2 box, marked `(v2 card, not yet converted)`.
`rebuild_index.py` on `triple-all-fastest` is a two-line shim onto `index`. `index` refuses, in one line, a
README with no banner: it never invents one; `render` the card first.

## THE TEMPLATE, as the tool renders it

```
✦ TRIPLESPARKLE ✦  this branch's card is below; antirez's README above is unchanged

  ┌──────────────────────────────────────────────────────────────────────
  │  BRANCH   triple-example                                    POSITIVE
  │  WHAT     one to three plain lines saying what the lever does
  │           and where it acts
  │  SWITCH   DS4_EXAMPLE=0 turns it off
  │  OUTPUT   greedy-identical to the tip: G1 short 0123456789abcdef · long fedcba9876543210
  │  BASE     triple-tip-2026-09-16 @12997e9c
  └──────────────────────────────────────────────────────────────────────

  RESULTS
  round  variant                date        arm t/s  tip t/s    delta  sign   floor  verdict  n
  r9     QD=16                  2026-09-17    10.41     9.42  +12.09%   4/4   17.27  TIES     4
  r4                            2026-09-17    10.37     9.65   +6.47%   4/4    2.71  BEATS    4
  gen tokens/s at min across ctx 4096 and 6144 · DGX Spark GB10 · each repeat against its bracketing tip runs · floor = max adjacent tip pair
  prefill: reported, never a verdict - r9 89.0 vs tip 85.6
  r9  2026-09-17-R9-RESULT-....md
  r4  2026-09-17-R4-RESULT-....md

NOTES
- what it does, in one line
- the honest caveat, in one line

ALSO TRIED
  triple-pool            +9.6%   parallel SSD reads; fast alone, loses under CPU load
  triple-winners         -       the combination; not above hitsfirst alone
```
