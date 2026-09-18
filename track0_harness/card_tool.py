#!/usr/bin/env python3
"""card_tool - render, lint and index the v3 branch cards (CARD_STANDARD_v3.md)."""
# <claudes_code_comments>
# ** Function List **
# die(msg) - one-line error + usage, exit 2
# load_card(root) - read + validate card.json, return dict
# load_notes(root) - read + validate NOTES.md, return list of bullets
# validate(card, branches=None) - every schema refusal incl. tried, raises CardError
# _local_branches(root) - local branch names of the repo holding root, None if not git
# check_text(text, where) - em-dash and closing-edge refusals on any rendered text
# render_box(card) - the 14-line-max box, right edge open
# render_results(card) - the RESULTS table (variant column, blank when absent), footer, prefill line, source lines
# render_tried(card) - the ALSO TRIED block (<= 5 sibling branches), '' when absent
# render_tail(card, notes) - banner + box + results + notes + also-tried, the README's generated part
# split_readme(text) - (upstream part, generated part) at the banner or any legacy star row
# split_tail(tail) - (card part, index part) at the index heading; index part '' when absent
# check_index(index) - the index part carries its 'Regenerated ... N cards' line and N boxes
# cmd_render(root) / cmd_lint(root) - the two per-branch commands
# parse_v2_box(readme) - fallback: the v2 box + status from a branch not yet converted
# build_index(entries) - summary table + per-branch boxes from {branch: card|v2 box}
# cmd_index(root, dry) - reads every triple-* HEAD via git show, rewrites the index section;
#   refuses a README with no banner, never invents one
# selftest() - golden render, every refusal proven RED (tried included), index on two fixture cards
# ** Technical Review **
# The README tail is GENERATED: render() writes from the banner up to the index heading
# (to EOF when there is none; the index section, on triple-all-fastest, is left byte-identical),
# lint() reds if that same region differs from render(card.json), and reds if an index section
# is present but was not produced by index() (its count line missing or not matching its boxes).
# Source lines under the table are one per distinct result file, first seen first. Refusals (unknown status/verdict, WHAT > 3 lines, > 5
# notes, note > 110 chars, row variant > 22 chars, missing/non-numeric row fields, em-dash, closing right edge,
# tried > 5 entries / a branch that is not a local triple-* branch / a line > 84 chars) live
# in validate()/check_text() and are shared by render, lint and index. index() never reads a
# working tree: `git show <branch>:card.json` at HEAD, falling back to the v2 box parse so the
# unconverted branches still appear, marked. Numbers are printed with fixed decimals and are
# never computed here - the card carries what the sealed result file says.
# </claudes_code_comments>
import json, os, re, subprocess, sys

BANNER = "✦ TRIPLESPARKLE ✦  this branch's card is below; antirez's README above is unchanged"
STATUSES = ["POSITIVE", "WORTH ZERO", "NEGATIVE", "NOT YET", "CONTROL", "TOOLING"]
VERDICTS = ["BEATS", "TIES", "LOSES", "PARTIAL"]
FOOTER = ("gen tokens/s at min across ctx 4096 and 6144 · DGX Spark GB10 · "
          "each repeat against its bracketing tip runs · floor = max adjacent tip pair")
EDGE = "─" * 70
CLOSERS = "┐┘╗╝┤╣"
EMDASH = "—–"
INDEX_HEAD = "## The index: every branch card"
TITLES = {"POSITIVE": "POSITIVELY MEASURED", "WORTH ZERO": "MEASURED CORRECT, WORTH ZERO ON THE CLOCK",
          "NEGATIVE": "NEGATIVELY MEASURED OR KILLED", "NOT YET": "NOT YET MEASURED",
          "CONTROL": "THE CONTROL", "TOOLING": "TOOLING, NOT A LEVER", "UNTAGGED": "UNTAGGED"}
USAGE = """usage: card_tool.py render <worktree> | lint <worktree> | index <all-fastest-worktree> [--dry] | --selftest
  render  rewrite <worktree>/README.md from the banner up to the index heading (EOF if none) out of card.json + NOTES.md, print it
  lint    same checks on the README as it stands, plus rendered part == render(card.json), plus the index section (if any) is index()'s; exit 1 with reasons
  index   every triple-* branch's card.json at its HEAD -> the index section of the stack README
examples:
  python3 track0_harness/card_tool.py render _worktrees/dw-triple-hitsfirst
  python3 track0_harness/card_tool.py lint   _worktrees/dw-triple-hitsfirst
  python3 track0_harness/card_tool.py index  _worktrees/dw-triple-all-fastest --dry
standard: CARD_STANDARD_v3.md"""


class CardError(Exception):
    pass


def die(msg):
    print("error: " + msg, file=sys.stderr); print(USAGE, file=sys.stderr); sys.exit(2)


# ---------------------------------------------------------------- validation
def check_text(text, where):
    for i, line in enumerate(text.split("\n"), 1):
        if any(c in line for c in EMDASH):
            raise CardError("%s line %d: em-dash or en-dash" % (where, i))
        if any(c in line for c in CLOSERS):
            raise CardError("%s line %d: closing right edge" % (where, i))


def _num(row, k):
    v = row.get(k)
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise CardError("results row %s: field %s is not a number (%r)" % (row.get("round", "?"), k, v))
    return v


def validate(card, branches=None):
    for k in ("branch", "status", "what", "switch", "output", "base", "results"):
        if k not in card:
            raise CardError("card.json: missing field %s" % k)
    if card["status"] not in STATUSES:
        raise CardError("card.json: unknown status word %r" % card["status"])
    what = card["what"]
    if not isinstance(what, list) or not 1 <= len(what) <= 3:
        raise CardError("card.json: WHAT must be 1 to 3 lines, got %d" % (len(what) if isinstance(what, list) else -1))
    for w in what:
        if not isinstance(w, str) or len(w) > 66:
            raise CardError("card.json: WHAT line over 66 chars: %r" % (w,))
    if not isinstance(card["switch"], str) or not card["switch"] or len(card["switch"]) > 66:
        raise CardError("card.json: SWITCH must be one line <= 66 chars")
    out = card["output"]
    if not isinstance(out, dict) or "gated" not in out:
        raise CardError("card.json: output must be {gated: bool, ...}")
    if out["gated"]:
        if not re.fullmatch(r"[0-9a-f]{16}", str(out.get("short", ""))):
            raise CardError("card.json: output.short must be a 16-hex sha")
        if "long" in out and out["long"] and not re.fullmatch(r"[0-9a-f]{16}", str(out["long"])):
            raise CardError("card.json: output.long must be a 16-hex sha")
    base = card["base"]
    if not isinstance(base, dict) or not base.get("branch") or not base.get("sha"):
        raise CardError("card.json: base needs branch and sha")
    if not isinstance(card["results"], list):
        raise CardError("card.json: results must be a list")
    for row in card["results"]:
        for k in ("round", "date", "arm_tps", "tip_tps", "delta_pct", "sign", "floor_pct", "verdict", "n", "file"):
            if k not in row:
                raise CardError("results row %s: missing field %s" % (row.get("round", "?"), k))
        for k in ("arm_tps", "tip_tps", "delta_pct", "floor_pct", "n"):
            _num(row, k)
        if row["verdict"] not in VERDICTS:
            raise CardError("results row %s: verdict %r not in %s" % (row["round"], row["verdict"], "/".join(VERDICTS)))
        if not re.fullmatch(r"\d+/\d+", str(row["sign"])):
            raise CardError("results row %s: sign must look like 4/4" % row["round"])
        if "note" in row and len(str(row["note"])) > 80:
            raise CardError("results row %s: note over 80 chars" % row["round"])
        if "variant" in row:
            if not isinstance(row["variant"], str) or len(row["variant"]) > 22:
                raise CardError("results row %s: variant must be a string <= 22 chars (%r)" % (row["round"], row["variant"]))
    if not isinstance(card.get("prefill", ""), str):
        raise CardError("card.json: prefill must be a string")
    tried = card.get("tried", [])
    if not isinstance(tried, list) or len(tried) > 5:
        raise CardError("card.json: tried must be a list of at most 5 entries")
    if tried:
        if branches is None:
            raise CardError("card.json: tried names branches but no local branch list is available (not a git worktree?)")
        for t in tried:
            if not isinstance(t, dict) or set(t) != {"branch", "best", "line"}:
                raise CardError("card.json: tried entry needs exactly branch, best, line: %r" % (t,))
            if not str(t["branch"]).startswith("triple-") or t["branch"] not in branches:
                raise CardError("card.json: tried branch %r is not a local triple-* branch" % (t["branch"],))
            if not (t["best"] == "-" or re.fullmatch(r"[+-]\d+(\.\d+)?%", str(t["best"]))):
                raise CardError("card.json: tried best must be a signed percent like +9.6%% or '-' (%r)" % (t["best"],))
            if not isinstance(t["line"], str) or not t["line"] or len(t["line"]) > 84:
                raise CardError("card.json: tried line must be 1 to 84 chars: %r" % (t["line"],))
    check_text(json.dumps(card, ensure_ascii=False), "card.json")
    return card


def _local_branches(root):
    """The local branch names of the repo holding <root>, or None when it is not a git worktree."""
    try:
        return set(_git(root, "for-each-ref", "--format=%(refname:short)", "refs/heads/").split())
    except (subprocess.CalledProcessError, OSError):
        return None


def validate_notes(notes):
    if not 1 <= len(notes) <= 5:
        raise CardError("NOTES.md: 1 to 5 bullets, got %d" % len(notes))
    for n in notes:
        if not n.startswith("- "):
            raise CardError("NOTES.md: every line starts with '- ': %r" % n[:40])
        if len(n) > 110:
            raise CardError("NOTES.md: bullet over 110 chars: %r" % n[:40])
    check_text("\n".join(notes), "NOTES.md")
    return notes


def load_card(root):
    p = os.path.join(root, "card.json")
    if not os.path.exists(p):
        raise CardError("no card.json in " + root)
    try:
        card = json.load(open(p, encoding="utf-8"))
    except ValueError as e:
        raise CardError("card.json does not parse: %s" % e)
    return validate(card, _local_branches(root))


def load_notes(root):
    p = os.path.join(root, "NOTES.md")
    if not os.path.exists(p):
        raise CardError("no NOTES.md in " + root)
    notes = [l.rstrip() for l in open(p, encoding="utf-8").read().split("\n") if l.strip()]
    return validate_notes(notes)


# ---------------------------------------------------------------- rendering
def render_box(card):
    L = ["  ┌" + EDGE]
    L.append("  │  BRANCH   %-50s%s" % (card["branch"], card["status"]))
    L.append("  │  WHAT     " + card["what"][0])
    for w in card["what"][1:]:
        L.append("  │           " + w)
    L.append("  │  SWITCH   " + card["switch"])
    o = card["output"]
    if o["gated"]:
        s = "greedy-identical to the tip: G1 short " + o["short"]
        if o.get("long"):
            s += " · long " + o["long"]
    else:
        s = "not gated"
    L.append("  │  OUTPUT   " + s)
    L.append("  │  BASE     %s @%s" % (card["base"]["branch"], card["base"]["sha"]))
    L.append("  └" + EDGE)
    assert len(L) <= 14
    return "\n".join(L)


def render_results(card):
    rows = card["results"]
    if not rows:
        return "  RESULTS  none yet - never in a sealed round"
    L = ["  RESULTS",
         "  round  variant                date        arm t/s  tip t/s    delta  sign   floor  verdict  n"]
    for r in rows:
        L.append("  %-6s %-22s %-10s %8.2f %8.2f  %+6.2f%%  %4s  %6.2f  %-7s  %d" % (
            r["round"], r.get("variant", ""), r["date"], r["arm_tps"], r["tip_tps"], r["delta_pct"], r["sign"],
            r["floor_pct"], r["verdict"], r["n"]))
    L.append("  " + FOOTER)
    if card.get("prefill"):
        L.append("  prefill: reported, never a verdict - " + card["prefill"])
    seen = []
    for r in rows:
        if r["file"] not in seen:
            seen.append(r["file"])
            L.append("  %-3s %s" % (r["round"], r["file"]))
    return "\n".join(L)


def render_tried(card):
    """The ALSO TRIED block: one line per entry, after the notes; '' when the card has none."""
    tried = card.get("tried", [])
    if not tried:
        return ""
    return "\n".join(["", "", "ALSO TRIED"] + ["  %-22s %-6s  %s" % (t["branch"], t["best"], t["line"]) for t in tried])


def render_tail(card, notes):
    return "\n".join([BANNER, "", "```", render_box(card), "", render_results(card), "```", "",
                      "NOTES"] + list(notes)) + render_tried(card) + "\n"


def render_index_card(card):
    return "\n".join([render_box(card), "", render_results(card)])


def split_readme(text):
    """Return (upstream, tail). The cut is the first v3 banner, legacy T R I P L E banner or star row."""
    lines = text.split("\n")
    cut = None
    for i, l in enumerate(lines):
        s = l.strip().strip("*").strip()
        if s == BANNER or "T R I P L E S P A R K L E" in s or (s and set(s) <= set("✦✧ ")):
            cut = i; break
    if cut is None:
        up = lines
        tail = ""
    else:
        up = lines[:cut]
        tail = "\n".join(lines[cut:])
    while up and not up[-1].strip():
        up.pop()
    if up and up[-1].strip() == "---":
        up.pop()
    while up and not up[-1].strip():
        up.pop()
    return "\n".join(up) + "\n", tail


def split_tail(tail):
    """Return (card part, index part). The index part starts at the INDEX_HEAD line, '' when absent."""
    lines = tail.split("\n")
    for i, l in enumerate(lines):
        if l.strip() == INDEX_HEAD:
            return "\n".join(lines[:i]), "\n".join(lines[i:])
    return tail, ""


REGEN_RE = re.compile(r"Regenerated .+?, (\d+) cards\.")


def check_index(index):
    """An index section must be index()'s: its count line present and matching its card boxes."""
    m = REGEN_RE.search(index)
    if not m:
        raise CardError("index section carries no 'Regenerated ..., N cards.' line: not produced by index")
    n = int(m.group(1))
    boxes = sum(1 for l in index.split("\n") if l.startswith("  ┌"))
    if boxes != n:
        raise CardError("index section says %d cards but carries %d card boxes: hand edit or stale index" % (n, boxes))


def cmd_render(root):
    card, notes = load_card(root), load_notes(root)
    tail = render_tail(card, notes)
    check_text(tail, "rendered")
    rp = os.path.join(root, "README.md")
    up, old = split_readme(open(rp, encoding="utf-8").read())
    _, index = split_tail(old)
    open(rp, "w", encoding="utf-8").write(up + "\n" + tail + ("\n" + index if index else ""))
    print(tail, end="")
    return 0


def cmd_lint(root):
    reasons = []
    try:
        card, notes = load_card(root), load_notes(root)
    except CardError as e:
        print("RED  " + str(e)); return 1
    text = open(os.path.join(root, "README.md"), encoding="utf-8").read()
    _, tail = split_readme(text)
    if not tail:
        reasons.append("README.md carries no banner")
    else:
        try:
            check_text(tail, "README tail")
        except CardError as e:
            reasons.append(str(e))
        cardpart, index = split_tail(tail)
        if cardpart.rstrip("\n") != render_tail(card, notes).rstrip("\n"):
            reasons.append("README card part differs from render(card.json): hand edit or stale render")
        if index:
            try:
                check_index(index)
            except CardError as e:
                reasons.append(str(e))
    for r in reasons:
        print("RED  " + r)
    if not reasons:
        print("GREEN  %s: card.json + NOTES.md + README card part agree%s" % (
            card["branch"], "; index section present and index()'s" if index else ""))
    return 1 if reasons else 0


# ---------------------------------------------------------------- the index
def parse_v2_box(readme):
    m = re.search(r"( *┌──.*?\n)(.*?)( *└──[─]*)", readme, re.S)
    if not m:
        return None
    box = m.group(0)
    tag = "UNTAGGED"
    bm = re.search(r"BRANCH\s+\S+\s+(?:CORRECT, )?(POSITIVE|WORTH ZERO|NEGATIVE|NOT YET|CONTROL|TOOLING)", box)
    if bm:
        tag = bm.group(1)
    gm = re.search(r"GEN\s+([+-]\d+\.\d+) %", box)
    lm = re.search(r"LATEST\s+round (\d+) · \S+ · (\w+)", box)
    return {"v2": True, "status": tag, "box": box,
            "delta": float(gm.group(1)) if gm else None,
            "round": ("r" + lm.group(1)) if lm else "-", "verdict": lm.group(2) if lm else "-"}


def _summary(branch, e):
    if e.get("v2"):
        d = e["delta"]
        return (branch, e["status"], d, e["round"], e["verdict"], True, None)
    rows = e["results"]
    if not rows:
        return (branch, e["status"], None, "-", "-", False, 0)
    best = max(rows, key=lambda r: r["delta_pct"])
    return (branch, e["status"], best["delta_pct"], rows[0]["round"], rows[0]["verdict"], False, len(rows))


def build_index(entries, stamp="now"):
    order = {s: i for i, s in enumerate(STATUSES + ["UNTAGGED"])}
    summ = [_summary(b, e) for b, e in entries.items()]
    summ.sort(key=lambda t: (order.get(t[1], 99), -(t[2] if t[2] is not None else -1e9), t[0]))
    L = [INDEX_HEAD, "",
         "One card per branch, generated by `track0_harness/card_tool.py index` from each branch's",
         "`card.json` at its HEAD (CARD_STANDARD_v3.md), grouped by the status the card declares.",
         "Regenerate, never hand-edit. Regenerated %s, %d cards." % (stamp, len(entries)), "",
         "```", "  branch                           status      best delta  latest  verdict  rows",]
    for b, st, d, rd, vd, v2, nr in summ:
        ds = "%+6.2f%%" % d if d is not None else "      -"
        rs = "%4d" % nr if nr is not None else "   -"
        L.append("  %-32s %-11s %10s  %-6s  %-7s  %s%s" % (b, st, ds, rd, vd, rs, "  (v2 card, not yet converted)" if v2 else ""))
    L += ["```", ""]
    for g in STATUSES + ["UNTAGGED"]:
        members = sorted(b for b, e in entries.items() if e["status"] == g)
        if not members:
            continue
        L += ["## " + TITLES[g], ""]
        for b in members:
            e = entries[b]
            if e.get("v2"):
                L += ["### %s (v2 card, not yet converted)" % b, "", "```", e["box"].rstrip("\n"), "```", ""]
            else:
                L += ["### " + b, "", "```", render_index_card(e), "```", ""]
    out = "\n".join(L) + "\n"
    check_text(out, "index")
    return out


def _git(root, *args):
    return subprocess.check_output(["git", "-C", root] + list(args), text=True, stderr=subprocess.DEVNULL)


def cmd_index(root, dry):
    rp = os.path.join(root, "README.md")
    src = open(rp, encoding="utf-8").read()
    if not split_readme(src)[1]:
        raise CardError("%s carries no banner: render the card first, index never invents one" % rp)
    branches = [b for b in _git(root, "for-each-ref", "--format=%(refname:short)", "refs/heads/").split()
                if b.startswith("triple-")]
    entries = {}
    for b in branches:
        try:
            card = validate(json.loads(_git(root, "show", b + ":card.json")), set(branches))
            entries[b] = card
            continue
        except subprocess.CalledProcessError:
            pass
        except (CardError, ValueError) as e:
            print("skip %s: %s" % (b, e), file=sys.stderr); continue
        try:
            readme = _git(root, "show", b + ":README.md")
        except subprocess.CalledProcessError:
            continue
        v2 = parse_v2_box(readme)
        if v2:
            entries[b] = v2
    stamp = subprocess.check_output(["date", "-u", "+%Y-%m-%d %H:%MZ"], text=True).strip()
    new = build_index(entries, stamp)
    i = src.find(INDEX_HEAD)
    if i < 0:
        raise CardError("index heading not found in " + rp)
    if dry:
        print(new); print("-- would write %d cards" % len(entries), file=sys.stderr)
    else:
        open(rp, "w", encoding="utf-8").write(src[:i] + new)
        print("wrote %s: %d cards (%d v3, %d v2)" % (rp, len(entries),
              sum(1 for e in entries.values() if not e.get("v2")), sum(1 for e in entries.values() if e.get("v2"))))
    return 0


# ---------------------------------------------------------------- selftest
FIXTURE = {
    "branch": "triple-example", "status": "POSITIVE",
    "what": ["one to three plain lines saying what the lever does", "and where it acts"],
    "switch": "DS4_EXAMPLE=0 turns it off",
    "output": {"gated": True, "short": "0123456789abcdef", "long": "fedcba9876543210"},
    "base": {"branch": "triple-tip-2026-09-16", "sha": "12997e9c"},
    "results": [
        {"round": "r9", "date": "2026-09-17", "arm_tps": 10.41, "tip_tps": 9.42, "delta_pct": 12.09,
         "sign": "4/4", "floor_pct": 17.27, "verdict": "TIES", "n": 4, "file": "2026-09-17-R9-RESULT-....md",
         "variant": "QD=16"},
        {"round": "r4", "date": "2026-09-17", "arm_tps": 10.37, "tip_tps": 9.65, "delta_pct": 6.47,
         "sign": "4/4", "floor_pct": 2.71, "verdict": "BEATS", "n": 4, "file": "2026-09-17-R4-RESULT-....md"}],
    "prefill": "r9 89.0 vs tip 85.6"}
FIXTURE_NOTES = ["- what it does, in one line", "- the honest caveat, in one line"]
GOLDEN = """✦ TRIPLESPARKLE ✦  this branch's card is below; antirez's README above is unchanged

```
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
```

NOTES
- what it does, in one line
- the honest caveat, in one line
"""


def selftest():
    import copy, tempfile
    n = [0, 0]

    def ok(cond, name):
        n[0] += 1
        if not cond:
            n[1] += 1; print("  FAIL " + name)

    TRIED_BRANCHES = {"triple-pool", "triple-winners", "triple-a", "triple-b", "triple-c", "triple-d"}
    TRIED = [{"branch": "triple-pool", "best": "+9.6%", "line": "parallel SSD reads; fast alone, loses under CPU load"},
             {"branch": "triple-winners", "best": "-", "line": "the combination; not above hitsfirst alone"}]

    def red(name, mut, notes=None):
        c = copy.deepcopy(FIXTURE); mut(c)
        try:
            validate(c, TRIED_BRANCHES); validate_notes(notes or FIXTURE_NOTES); check_text(render_tail(c, notes or FIXTURE_NOTES), "r")
            ok(False, "refusal not raised: " + name)
        except CardError as e:
            ok(True, name); print("  RED  %-34s %s" % (name, e))

    # golden
    card = validate(copy.deepcopy(FIXTURE))
    ok(render_tail(card, FIXTURE_NOTES) == GOLDEN, "golden render")
    ok(len(render_box(card).split("\n")) <= 14, "box <= 14 lines")
    ok("  r9     QD=16                  2026-09-17" in GOLDEN and "  r4                            2026-09-17" in GOLDEN,
       "variant column: set on r9, blank on r4, between round and date")
    cv = copy.deepcopy(FIXTURE); cv["results"][0]["variant"] = "v" * 22
    ok("r9     " + "v" * 22 + " 2026-09-17" in render_results(validate(cv)), "variant at exactly 22 chars renders")
    ok(not any(c in GOLDEN for c in CLOSERS), "golden has no closing edge")
    # refusals, each proven red
    red("unknown status word", lambda c: c.__setitem__("status", "SUPERSEDED"))
    red("WHAT over 3 lines", lambda c: c.__setitem__("what", ["a", "b", "c", "d"]))
    red("WHAT line over 66 chars", lambda c: c.__setitem__("what", ["x" * 67]))
    red("SWITCH over 66 chars", lambda c: c.__setitem__("switch", "y" * 67))
    red("results row missing field", lambda c: c["results"][0].pop("floor_pct"))
    red("results row non-numeric number", lambda c: c["results"][0].__setitem__("arm_tps", "10.41"))
    red("verdict outside the set", lambda c: c["results"][0].__setitem__("verdict", "WINS"))
    red("bad sign", lambda c: c["results"][0].__setitem__("sign", "four"))
    red("row note over 80 chars", lambda c: c["results"][0].__setitem__("note", "z" * 81))
    red("row variant over 22 chars", lambda c: c["results"][0].__setitem__("variant", "v" * 23))
    red("row variant not a string", lambda c: c["results"][0].__setitem__("variant", 16))
    red("planted em-dash in WHAT", lambda c: c.__setitem__("what", ["a lever — with a dash"]))
    red("planted en-dash in switch", lambda c: c.__setitem__("switch", "DS4_X=0 – off"))
    red("planted closing edge in WHAT", lambda c: c.__setitem__("what", ["boxed ┘"]))
    red("bad G1 sha", lambda c: c["output"].__setitem__("short", "nothex"))
    red("over 5 notes", lambda c: None, ["- 1", "- 2", "- 3", "- 4", "- 5", "- 6"])
    red("note over 110 chars", lambda c: None, ["- " + "n" * 109])
    red("note without dash", lambda c: None, ["plain line"])
    red("planted em-dash in a note", lambda c: None, ["- a note — with a dash"])
    red("planted closing edge in a note", lambda c: None, ["- a box ╝"])
    # tried: renders after the notes; every refusal red
    ct = copy.deepcopy(FIXTURE); ct["tried"] = TRIED
    tt = render_tail(validate(ct, TRIED_BRANCHES), FIXTURE_NOTES)
    ok(tt.startswith(GOLDEN.rstrip("\n") + "\n\nALSO TRIED\n"), "tried block comes right after the notes")
    cl = copy.deepcopy(ct); cl["tried"] = [dict(TRIED[0], branch="triple-a" + "b" * 17)]
    ok("  triple-a" + "b" * 17 + " +9.6%   parallel" in render_tried(validate(cl, TRIED_BRANCHES | {"triple-a" + "b" * 17})),
       "a 25-char branch name keeps one space before best")
    ok("  triple-pool            +9.6%   parallel SSD reads; fast alone, loses under CPU load\n" in tt
       and "  triple-winners         -       the combination; not above hitsfirst alone\n" in tt,
       "tried lines: branch padded to 22 + a space, best to 6, then the line")
    ok(render_tail(validate(copy.deepcopy(FIXTURE), TRIED_BRANCHES), FIXTURE_NOTES) == GOLDEN, "no tried: no block, golden unchanged")
    ok("ALSO TRIED" not in render_index_card(validate(ct, TRIED_BRANCHES)), "index card ignores tried")
    red("tried over 5 entries", lambda c: c.__setitem__("tried", [dict(TRIED[0], branch="triple-" + x) for x in "abcd"] + TRIED))
    red("tried branch not a local branch", lambda c: c.__setitem__("tried", [dict(TRIED[0], branch="triple-nonesuch")]))
    red("tried branch not triple-*", lambda c: c.__setitem__("tried", [dict(TRIED[0], branch="main")]))
    red("tried line over 84 chars", lambda c: c.__setitem__("tried", [dict(TRIED[0], line="l" * 85)]))
    red("tried best not a signed percent", lambda c: c.__setitem__("tried", [dict(TRIED[0], best="9.6")]))
    red("tried entry with a stray field", lambda c: c.__setitem__("tried", [dict(TRIED[0], extra=1)]))
    red("planted em-dash in a tried line", lambda c: c.__setitem__("tried", [dict(TRIED[0], line="fast — alone")]))
    try:
        validate(ct, None); ok(False, "tried with no branch list refuses")
    except CardError as e:
        ok("no local branch list" in str(e), "tried with no branch list refuses")
    ok(validate(ct, TRIED_BRANCHES)["tried"][0]["best"] == "+9.6%" and validate(
        dict(ct, tried=[dict(TRIED[0], best="+10.25%")]), TRIED_BRANCHES), "tried best accepts +9.6% and +10.25%")
    # ungated + no results
    c2 = copy.deepcopy(FIXTURE); c2["output"] = {"gated": False}; c2["results"] = []; c2["prefill"] = ""
    t = render_tail(validate(c2), FIXTURE_NOTES)
    ok("OUTPUT   not gated" in t and "RESULTS  none yet - never in a sealed round" in t and "round  variant" not in t, "ungated, no rounds")
    # long optional
    c3 = copy.deepcopy(FIXTURE); del c3["output"]["long"]
    ok("· long" not in render_box(validate(c3)), "output.long optional")
    # split + render + lint round trip on a temp worktree
    d = tempfile.mkdtemp()
    up = "# upstream\n\nlast upstream line\n\n---\n\n\n**✦   ✧   ✦**\n\n**✦ ✧ ✦   T R I P L E S P A R K L E   ✦ ✧ ✦**\n\nold card\n"
    open(os.path.join(d, "README.md"), "w").write(up)
    json.dump(FIXTURE, open(os.path.join(d, "card.json"), "w"))
    open(os.path.join(d, "NOTES.md"), "w").write("\n".join(FIXTURE_NOTES) + "\n")
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        cmd_render(d)
    txt = open(os.path.join(d, "README.md")).read()
    ok(txt == "# upstream\n\nlast upstream line\n\n" + GOLDEN, "render cuts legacy star row + --- and appends golden")
    with contextlib.redirect_stdout(io.StringIO()):
        cmd_render(d)
    ok(open(os.path.join(d, "README.md")).read() == txt, "render is idempotent")
    with contextlib.redirect_stdout(io.StringIO()):
        ok(cmd_lint(d) == 0, "lint GREEN after render")
    open(os.path.join(d, "README.md"), "a").write("hand edit\n")
    with contextlib.redirect_stdout(io.StringIO()):
        ok(cmd_lint(d) == 1, "lint RED on a hand edit (drift)")
    open(os.path.join(d, "README.md"), "w").write(txt.replace("└──", "└──┘"))
    with contextlib.redirect_stdout(io.StringIO()):
        ok(cmd_lint(d) == 1, "lint RED on a planted closing edge in README")
    open(os.path.join(d, "README.md"), "w").write(txt.replace("- what it does", "- what it does — x"))
    with contextlib.redirect_stdout(io.StringIO()):
        ok(cmd_lint(d) == 1, "lint RED on a planted em-dash in README")
    # tried round trip on a real git worktree: render, lint GREEN, a bogus branch RED through load_card
    d4 = tempfile.mkdtemp()
    subprocess.run(["git", "-C", d4, "init", "-q"], check=True)
    subprocess.run(["git", "-C", d4, "-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-q", "--allow-empty", "-m", "x"], check=True)
    subprocess.run(["git", "-C", d4, "branch", "triple-pool"], check=True)
    subprocess.run(["git", "-C", d4, "branch", "triple-winners"], check=True)
    json.dump(ct, open(os.path.join(d4, "card.json"), "w"))
    open(os.path.join(d4, "NOTES.md"), "w").write("\n".join(FIXTURE_NOTES) + "\n")
    open(os.path.join(d4, "README.md"), "w").write("# upstream\n")
    with contextlib.redirect_stdout(io.StringIO()):
        cmd_render(d4)
        ok(cmd_lint(d4) == 0, "lint GREEN on a worktree whose card names two real local branches")
    ok(open(os.path.join(d4, "README.md")).read().endswith("\nALSO TRIED\n" + tt.split("ALSO TRIED\n")[1]), "rendered README ends with the tried block")
    json.dump(dict(ct, tried=[dict(TRIED[0], branch="triple-nonesuch")]), open(os.path.join(d4, "card.json"), "w"))
    with contextlib.redirect_stdout(io.StringIO()):
        ok(cmd_lint(d4) == 1, "lint RED when a tried branch does not exist locally")
    # index on two fixture cards + one v2 fallback
    v2 = parse_v2_box("x\n```\n  ┌──────\n  │\n  │  BRANCH     triple-old                              WORTH ZERO\n"
                      "  │  LATEST     round 5 · 2026-09-17 · TIES · +0.10 % ·\n  │  GEN        +0.10 %  floor 4.00 %  sign 2/4  n=4\n  └──────\n```\n")
    ok(v2 and v2["status"] == "WORTH ZERO" and v2["delta"] == 0.10 and v2["round"] == "r5", "v2 box fallback parse")
    cb = copy.deepcopy(FIXTURE); cb["branch"] = "triple-second"; cb["results"] = cb["results"][1:]
    idx = build_index({"triple-example": card, "triple-second": validate(cb), "triple-old": v2}, "2026-09-18")
    lines = idx.split("\n")
    ok(lines[0] == INDEX_HEAD, "index heading")
    si = [l for l in lines if l.startswith("  triple-")]
    ok(si[0].startswith("  triple-example") and si[1].startswith("  triple-second") and "not yet converted" in si[2],
       "summary sorted by status then delta, v2 marked")
    ok("+12.09%" in si[0] and "r9" in si[0] and "TIES" in si[0], "summary carries best delta, latest round, verdict")
    ok("### triple-example" in idx and "### triple-old (v2 card, not yet converted)" in idx, "per-branch sections")
    ok("NOTES" not in idx, "index carries no notes")
    ok("## POSITIVELY MEASURED" in idx and "## MEASURED CORRECT, WORTH ZERO ON THE CLOCK" in idx, "index groups")
    ok("verdict  rows" in idx and re.search(r"^  triple-example .*TIES\s+2$", idx, re.M)
       and re.search(r"^  triple-second .*BEATS\s+1$", idx, re.M) and re.search(r"^  triple-old .*\s-  \(v2", idx, re.M),
       "summary rows column: sealed-round count, '-' for v2")
    # a README carrying a card AND an index section: render keeps the index, lint judges both
    d2 = tempfile.mkdtemp()
    json.dump(FIXTURE, open(os.path.join(d2, "card.json"), "w"))
    open(os.path.join(d2, "NOTES.md"), "w").write("\n".join(FIXTURE_NOTES) + "\n")
    open(os.path.join(d2, "README.md"), "w").write("# upstream\n\n" + GOLDEN + "\n" + idx)
    before = open(os.path.join(d2, "README.md")).read()
    with contextlib.redirect_stdout(io.StringIO()):
        cmd_render(d2)
    after = open(os.path.join(d2, "README.md")).read()
    ok(after == before, "render on a README with an index leaves it byte-identical (index kept)")
    ok(split_tail(split_readme(after)[1])[1] == idx, "index part is the index verbatim")
    with contextlib.redirect_stdout(io.StringIO()):
        ok(cmd_lint(d2) == 0, "lint GREEN with card + index present")
    open(os.path.join(d2, "README.md"), "w").write(after.replace("3 cards.", "4 cards."))
    with contextlib.redirect_stdout(io.StringIO()):
        ok(cmd_lint(d2) == 1, "lint RED when the index count line is tampered")
    open(os.path.join(d2, "README.md"), "w").write(after.replace("Regenerated 2026-09-18, 3 cards.", "hand-written"))
    with contextlib.redirect_stdout(io.StringIO()):
        ok(cmd_lint(d2) == 1, "lint RED when the index has no Regenerated line")
    open(os.path.join(d2, "README.md"), "w").write(after.replace("- what it does, in one line", "- what it does"))
    with contextlib.redirect_stdout(io.StringIO()):
        ok(cmd_lint(d2) == 1, "lint RED on card-part drift with the index present")
    # index refuses a README with no banner, before it touches git
    d3 = tempfile.mkdtemp()
    open(os.path.join(d3, "README.md"), "w").write("# upstream only\n")
    try:
        cmd_index(d3, True); ok(False, "index refuses a README with no banner")
    except CardError as e:
        ok("no banner" in str(e), "index refuses a README with no banner")
    # two rows of one round citing one file: one source line, first seen first
    c4 = copy.deepcopy(FIXTURE)
    c4["results"] = [dict(c4["results"][0]), dict(c4["results"][0], arm_tps=10.50), c4["results"][1]]
    r4 = render_results(validate(c4))
    ok(r4.count("2026-09-17-R9-RESULT-....md") == 1 and r4.count("2026-09-17-R4-RESULT-....md") == 1
       and r4.index("R9-RESULT") < r4.index("R4-RESULT") and r4.count("r9     QD=16                  2026-09-17") == 2,
       "two rows one file: two table rows, one source line, first-seen order")
    print("selftest: %d checks, %d failed" % (n[0], n[1]))
    return 1 if n[1] else 0


def main(argv):
    if not argv or argv[0] in ("-h", "--help"):
        print(USAGE); return 0
    if argv[0] == "--selftest":
        return selftest()
    if len(argv) < 2:
        die("missing <worktree>")
    cmd, root = argv[0], argv[1]
    if not os.path.isdir(root):
        die("not a directory: " + root)
    try:
        if cmd == "render":
            return cmd_render(root)
        if cmd == "lint":
            return cmd_lint(root)
        if cmd == "index":
            return cmd_index(root, "--dry" in argv)
    except CardError as e:
        die(str(e))
    die("unknown command " + cmd)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
