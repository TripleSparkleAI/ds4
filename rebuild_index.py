#!/usr/bin/env python3
# <claudes_code_comments>
# ** Function List **
# branches() - every triple-* branch name
# card(b) - that branch README card box and its declared status tag
# main() - rewrite everything from the index heading down, grouped by tag
# ** Technical Review **
# Reads each branch at its HEAD via git show, never the working tree, so the index
# reflects committed cards only. Replaces the tail of README.md from "## The index"
# onward; --dry prints instead. Run from any checkout of the fork.
# </claudes_code_comments>
"""Rebuild the card index at the tail of triple-all-fastest's README from every carded branch's
own card box at that branch's HEAD. Usage: rebuild_index.py <stack-README-path> [--dry]"""
import subprocess, sys, re
ROOT = subprocess.check_output(['git','rev-parse','--show-toplevel'],text=True).strip()
GROUPS = ["POSITIVE", "WORTH ZERO", "NEGATIVE", "NOT YET", "CONTROL", "TOOLING", "UNTAGGED"]
TITLES = {"POSITIVE":"POSITIVELY MEASURED", "WORTH ZERO":"MEASURED CORRECT, WORTH ZERO ON THE CLOCK",
          "NEGATIVE":"NEGATIVELY MEASURED OR KILLED",
          "NOT YET":"NOT YET MEASURED", "CONTROL":"THE CONTROL", "TOOLING":"TOOLING, NOT A LEVER",
          "UNTAGGED":"UNTAGGED - the card carries no status"}
def branches():
    out = subprocess.check_output(["git","-C",ROOT,"for-each-ref","--format=%(refname:short)","refs/heads/"],text=True).split()
    return [b for b in out if b.startswith("triple-")]
def card(b):
    try: txt = subprocess.check_output(["git","-C",ROOT,"show",f"{b}:README.md"],text=True)
    except subprocess.CalledProcessError: return None
    if "T R I P L E" not in txt: return None
    m = re.search(r"( *┌──.*?\n)(.*?)( *└──[─]*\n)", txt, re.S)
    if not m: return None
    box = m.group(0)
    tag = "UNTAGGED"
    bm = re.search(r"BRANCH\s+\S+\s+(?:CORRECT, )?(POSITIVE|WORTH ZERO|NEGATIVE|NOT YET|CONTROL|TOOLING)", box)
    if bm: tag = bm.group(1)
    return tag, box
def main():
    path = sys.argv[1]; dry = "--dry" in sys.argv
    cards = {}
    for b in branches():
        c = card(b)
        if c: cards[b] = c
    lines = ["## The index: every branch card",
             "",
             "One card per branch, the box copied from that branch's own README at its HEAD by",
             "`rebuild_index.py`, grouped by the status the card itself declares. Regenerate,",
             "never hand-edit: a hand-edited row drifts from the branch it describes.",
             f"Regenerated {subprocess.check_output(['date','-u','+%Y-%m-%d %H:%MZ'],text=True).strip()}, {len(cards)} cards.",
             ""]
    for g in GROUPS:
        members = sorted(b for b,(t,_) in cards.items() if t==g)
        if not members: continue
        lines += [f"## {TITLES[g]}", ""]
        for b in members:
            lines += [f"### {b}", "", "```", cards[b][1].rstrip("\n"), "```", ""]
    new = "\n".join(lines)+"\n"
    src = open(path).read()
    i = src.find("## The index: every branch card")
    assert i > 0, "index heading not found"
    out = src[:i] + new
    if dry:
        print(new); print(f"-- would replace {len(src)-i} chars with {len(new)}; cards: {len(cards)}", file=sys.stderr)
    else:
        open(path,"w").write(out); print(f"wrote {path}: {len(cards)} cards")
main()
