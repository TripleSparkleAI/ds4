#!/usr/bin/env python3
"""Resolve git conflict markers in one file from an ordered choice list.

# <claudes_code_comments>
# ** FUNCTION LIST **
# main - parse conflict markers, apply one choice per hunk, rewrite the file
#
# ** TECHNICAL REVIEW **
# Called by build_stack.sh once per conflicted file after a lever is applied --3way.
# argv[1] is the file; argv[2:] are the choices, in hunk order: ours | theirs | both |
# bothrev. ours is the pre-existing side (base plus earlier levers), theirs the incoming
# lever, both keeps ours then theirs, bothrev the reverse. A fifth choice, hand:<file>,
# replaces the hunk with the RECORDED text of <file> under stack.patches/ (resolved against
# this script's own directory) - for the hunks where neither side nor their concatenation is
# a program, because two levers rewrote one region and the merge is a third text. The file is
# data in the tree, so the build still reproduces from the manifest alone. Diff3 bases
# (||||||| blocks) are skipped: the manifest records a decision, not a three-way replay.
#
# It prints "<path> <nhunks>" so the caller can advance its choice cursor across several
# conflicted files in one lever. Exit 3 means a hunk had no choice - the caller turns that
# into the loud STOP. Exit 3 is deliberate and is the whole safety property: an unrecorded
# conflict must never be resolved by a default, because a silent default is how a curated
# tree stops matching its description.
# </claudes_code_comments>
"""
import os
import sys

def main():
    if len(sys.argv) < 2:
        sys.stderr.write("usage: stack_resolve.py <file> [choice ...]\n")
        return 1
    path, choices = sys.argv[1], sys.argv[2:]
    text = open(path, encoding='utf-8', errors='surrogateescape').read()
    lines = text.split('\n')
    out, i, n = [], 0, 0
    while i < len(lines):
        if lines[i].startswith('<<<<<<<'):
            ours, theirs = [], []
            i += 1
            while i < len(lines) and not lines[i].startswith(('|||||||', '=======')):
                ours.append(lines[i]); i += 1
            if i < len(lines) and lines[i].startswith('|||||||'):
                i += 1
                while i < len(lines) and not lines[i].startswith('======='):
                    i += 1
            i += 1
            while i < len(lines) and not lines[i].startswith('>>>>>>>'):
                theirs.append(lines[i]); i += 1
            i += 1
            anchor = next((l.strip() for l in ours + theirs if l.strip()), '(blank line)')
            if n >= len(choices):
                sys.stderr.write("hunk %d unrecorded: %s\n" % (n, anchor[:70]))
                return 3
            ch = choices[n]
            if ch == 'ours':      out += ours
            elif ch == 'theirs':  out += theirs
            elif ch == 'both':    out += ours + theirs
            elif ch == 'bothrev': out += theirs + ours
            elif ch.startswith('hand:'):
                here = os.path.dirname(os.path.abspath(__file__))
                hp = os.path.join(here, 'stack.patches', ch[5:])
                if not os.path.isfile(hp):
                    sys.stderr.write("hunk %d hand file missing: %s\n" % (n, hp))
                    return 3
                out += open(hp, encoding='utf-8').read().rstrip('\n').split('\n')
            else:
                sys.stderr.write("hunk %d bad choice %r: %s\n" % (n, ch, anchor[:70]))
                return 3
            n += 1
        else:
            out.append(lines[i]); i += 1
    open(path, 'w', encoding='utf-8', errors='surrogateescape').write('\n'.join(out))
    print("%s %d" % (path, n))
    return 0

if __name__ == '__main__':
    sys.exit(main())
