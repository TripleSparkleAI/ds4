#!/usr/bin/env python3
"""Print one `resolve` line per conflict hunk in a file, with its anchor as a comment.

# <claudes_code_comments>
# ** FUNCTION LIST **
# main - walk conflict markers and print a resolve stub per hunk
#
# ** TECHNICAL REVIEW **
# Used only by build_stack.sh --survey, to enumerate the decisions a manifest still owes
# instead of stopping at the first one. argv: <onDiskPath> <repoRelativePath> <leverName>.
# It DECIDES NOTHING: every line it prints carries the placeholder ??? where a choice goes,
# so a survey's output cannot be pasted into a manifest and mistaken for a derivation. The
# anchor is the hunk's first non-blank line from either side, whitespace collapsed, so a
# reader can find the place in the file without a line number - line numbers in a 31,000
# line translation unit rot the moment anything above them moves.
# </claudes_code_comments>
"""
import sys, re

def main():
    path, rel, lever = sys.argv[1], sys.argv[2], sys.argv[3]
    lines = open(path, encoding='utf-8', errors='surrogateescape').read().split('\n')
    i, n = 0, 0
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
            anchor = re.sub(r'\s+', ' ', anchor)[:64]
            print("    resolve %s %02d ???   # [%s] %s" % (rel, n, lever, anchor))
            n += 1
        else:
            i += 1
    print("    # %s: %d hunks owed in %s" % (lever, n, rel))
    return 0

if __name__ == '__main__':
    sys.exit(main())
