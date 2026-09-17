#!/usr/bin/env bash
# <claudes_code_comments>
# ** FUNCTION LIST **
# die                    - print to stderr and exit 1
# loud                   - print a framed STOP message naming file and hunk, then exit 2
# parse_manifest         - read the manifest into the LINES array, stripping comments
# resolve_pathspec       - turn the manifest's `paths` line into git pathspec arguments
# lever_diff             - emit one lever's code-only diff against lever-base
# resolve_conflicts      - drive stack_resolve.py over every unmerged file for one lever
# build                  - the whole build: fresh detached worktree, base, levers, local, patch
# do_check               - build, then diff the result against a target sha and report
# main                   - argument handling
#
# ** TECHNICAL REVIEW **
# The mechanism is `git apply --3way` of a per-lever diff, NOT `git cherry-pick`. The reason is
# measurable rather than stylistic: the ten lever branches do not share one base. Four of them
# (margin, draincut, pagecache, engram-lead) sit on the stack's own base line and six were
# rebased onto triple-tip-2026-09-16 on 2026-09-17, and each carries card and bench commits that
# are not part of any build. A cherry-pick would need the commit list per branch and would carry
# that documentation into the tree; a diff of the branch tip against a stated lever-base carries
# exactly the code the lever contributes, once, with no history to curate. The cost is that a
# conflict arrives as markers rather than as a rebase state, which is why resolution is a
# manifest line and not an interactive step.
#
# Flow: build() adds a FRESH detached worktree under _worktrees/stack-build-<stamp> - never onto
# a branch, never into an existing worktree, because a reproduction that writes into somebody's
# lane is not a reproduction. Each `lever` line produces a diff which is applied --3way. After
# each lever, every unmerged file is handed to stack_resolve.py with that lever's ordered
# resolve choices. A conflict hunk with no choice makes the resolver exit 3 and build() calls
# loud(), which names the file and the hunk's first non-blank line and stops. It never guesses:
# an unrecorded conflict is a fact about two levers touching one place and it belongs in the
# manifest.
#
# `lever-base` may appear more than once and rebinds the base for the levers that follow it.
# That is not decoration: the ten lever branches genuinely do not share one base once six of
# them are rebased onto a new tip, and a manifest that pretended otherwise would carry the whole
# tip delta into four levers' diffs as noise.
#
# `local` lines apply a stack-only commit's own diff. The `patch` line applies a recorded
# hand-edit last. Both exist because the stack contains code no lever branch carries, and the
# honest way to reproduce it is to record it as data.
#
# do_check() compares the built tree against a target sha twice: over the manifest's declared
# `paths` (the build surface, which is what must be byte-identical and what gates the exit code)
# and over the whole tree (reported always, gating nothing). Documentation is outside the build
# surface on purpose - the stack's own README says README.md was excluded from every lever
# patch - so a full-tree difference in docs is a fact to print, not a failure to hide.
#
# Constants: RC 1 usage or environment, 2 unrecorded conflict, 3 check failed on the build
# surface. --dry-run prints the plan and writes nothing.
# </claudes_code_comments>
set -uo pipefail

SELF_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="$(git -C "$SELF_DIR" rev-parse --show-toplevel 2>/dev/null)"
[ -n "${REPO:-}" ] || { echo "not inside a git repository" >&2; exit 1; }
COMMON="$(git -C "$SELF_DIR" rev-parse --git-common-dir)"
ROOT="$(cd "$(dirname "$COMMON")" && pwd)"

die()  { echo "build_stack: $*" >&2; exit 1; }

loud() {
  local file="$1" hunk="$2" lever="$3" anchor="$4"
  {
    echo
    echo "  STOP. UNRECORDED CONFLICT."
    echo "  ---------------------------------------------------------------"
    echo "  lever : $lever"
    echo "  file  : $file"
    echo "  hunk  : $hunk"
    echo "  at    : $anchor"
    echo
    echo "  This build has no resolve line for that hunk and it will not guess."
    echo "  A conflict is a fact about two levers touching one place. Decide it"
    echo "  and write it down, immediately after the lever's own line:"
    echo
    echo "      resolve $file $hunk ours|theirs|both|bothrev"
    echo
    echo "  ours   = the tree so far, base plus every earlier lever"
    echo "  theirs = the incoming lever"
    echo "  both   = ours then theirs; bothrev = the reverse"
    echo "  ---------------------------------------------------------------"
  } >&2
  exit 2
}

MANIFEST=""; DRY=0; CHECK=""; KEEP=0; SURVEY=0
declare -a LINES=()

parse_manifest() {
  local l
  while IFS= read -r l || [ -n "$l" ]; do
    l="${l%%$'\t'#*}"
    case "$l" in \#*|"") continue ;; esac
    LINES+=("$l")
  done < "$MANIFEST"
}

field() { echo "$1" | awk -v n="$2" '{print $n}'; }

BASE=""; LEVER_BASE=""; declare -a PATHSPEC=()

resolve_pathspec() {
  local l
  for l in "${LINES[@]}"; do
    case "$l" in
      base*)       BASE="$(field "$l" 2)" ;;
      lever-base*) LEVER_BASE="$(field "$l" 2)" ;;
      paths*)      read -r -a PATHSPEC <<< "$(echo "$l" | cut -d' ' -f2-)" ;;
    esac
  done
  [ -n "$BASE" ] || die "manifest has no base line"
  [ ${#PATHSPEC[@]} -gt 0 ] || PATHSPEC=('*')
  [ -n "$LEVER_BASE" ] || LEVER_BASE="$BASE"
}

lever_diff() {  # <sha> -> stdout
  git -C "$REPO" diff "$LEVER_BASE" "$1" -- "${PATHSPEC[@]}"
}

BUILD=""

resolve_conflicts() {  # <lever-name> <choices...>
  local lever="$1"; shift
  local -a choices=("$@")
  local f n=0 rc
  if [ "$SURVEY" -eq 1 ]; then
    for f in $(git -C "$BUILD" diff --name-only --diff-filter=U); do
      python3 "$SELF_DIR/stack_list_conflicts.py" "$BUILD/$f" "$f" "$lever"
      # keep going with `theirs` throughout: a SURVEY produces a conflict list,
      # never a tree anybody may build on. It is labelled as such in its banner.
      local cnt; cnt="$(grep -c '<<<<<<<' "$BUILD/$f" || true)"
      local -a all=(); local k
      for ((k=0;k<cnt;k++)); do all+=("theirs"); done
      python3 "$SELF_DIR/stack_resolve.py" "$BUILD/$f" "${all[@]+"${all[@]}"}" >/dev/null
      git -C "$BUILD" add -- "$f"
    done
    return 0
  fi
  for f in $(git -C "$BUILD" diff --name-only --diff-filter=U); do
    local -a mine=()
    # choices are consumed in file order across the lever's conflicted files
    mine=("${choices[@]:$n}")
    out="$(python3 "$SELF_DIR/stack_resolve.py" "$BUILD/$f" "${mine[@]+"${mine[@]}"}" 2>&1)"
    rc=$?
    if [ "$rc" -ne 0 ]; then
      if [ "$rc" -eq 3 ]; then
        local hunk anchor
        hunk="$(echo "$out" | sed -n 's/.*hunk \([0-9]*\) .*/\1/p' | head -1)"
        anchor="$(echo "$out" | head -1 | cut -d' ' -f4-)"
        loud "$f" "$(printf '%02d' "${hunk:-0}")" "$lever" "${anchor:-unknown}"
      fi
      die "resolver failed on $f: $out"
    fi
    n=$(( n + $(echo "$out" | head -1 | awk '{print $2}') ))
    git -C "$BUILD" add -- "$f"
  done
}

build() {
  resolve_pathspec
  local stamp; stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  BUILD="$ROOT/_worktrees/stack-build-$stamp"
  echo "manifest  $MANIFEST"
  echo "base      $BASE"
  echo "leverbase $LEVER_BASE"
  echo "paths     ${PATHSPEC[*]}"
  echo "build dir $BUILD"
  [ "$SURVEY" -eq 1 ] && { echo; echo "  *** SURVEY MODE: this enumerates conflicts. The tree it leaves is NOT a build"; echo "  *** and must not be measured. Nothing here is a resolution decision."; }
  echo

  local l kind
  if [ "$DRY" -eq 1 ]; then
    echo "--- plan (dry run, nothing is written) ---"
    for l in "${LINES[@]}"; do
      kind="$(field "$l" 1)"
      case "$kind" in
        lever-base) LEVER_BASE="$(field "$l" 2)"
                 printf '  base   %s\n' "$LEVER_BASE" ;;
        lever)   printf '  apply  %-32s %s  default=%s  (%s diff lines)\n' \
                   "$(field "$l" 2)" "$(field "$l" 3)" "$(field "$l" 4)" \
                   "$(lever_diff "$(field "$l" 3)" | wc -l | tr -d ' ')" ;;
        resolve) printf '    resolve %s hunk %s -> %s\n' "$(field "$l" 2)" "$(field "$l" 3)" "$(field "$l" 4)" ;;
        local)   printf '  local  %-32s %s\n' "$(field "$l" 2)" "$(field "$l" 3)" ;;
        patch)   printf '  patch  %s\n' "$(field "$l" 2)" ;;
      esac
    done
    return 0
  fi

  [ -e "$BUILD" ] && die "$BUILD already exists"
  git -C "$REPO" worktree add --detach "$BUILD" "$BASE" >/dev/null 2>&1 \
    || die "could not create a detached worktree at $BUILD"
  git -C "$BUILD" rev-parse --abbrev-ref HEAD | grep -qx HEAD \
    || die "refusing to build: the worktree is on a branch"

  local cur_lever="" ; declare -a pending=()
  flush() {
    [ -n "$cur_lever" ] || return 0
    resolve_conflicts "$cur_lever" ${pending[@]+"${pending[@]}"}
    cur_lever=""; pending=()
  }

  for l in "${LINES[@]}"; do
    kind="$(field "$l" 1)"
    case "$kind" in
      base|paths) : ;;
      lever-base) LEVER_BASE="$(field "$l" 2)" ;;
      lever)
        flush
        cur_lever="$(field "$l" 2)"; pending=()
        local sha; sha="$(field "$l" 3)"
        printf '  %-32s %s ... ' "$cur_lever" "${sha:0:9}"
        lever_diff "$sha" > "$BUILD/.lever.diff"
        if [ ! -s "$BUILD/.lever.diff" ]; then echo "EMPTY DIFF"; rm -f "$BUILD/.lever.diff"; continue; fi
        git -C "$BUILD" apply --3way --whitespace=nowarn ".lever.diff" 2>/dev/null
        rm -f "$BUILD/.lever.diff"
        if [ -n "$(git -C "$BUILD" diff --name-only --diff-filter=U)" ]; then
          echo "conflicts in: $(git -C "$BUILD" diff --name-only --diff-filter=U | tr '\n' ' ')"
        else
          echo "clean"
        fi
        ;;
      resolve) pending+=("$(field "$l" 4)") ;;
      local)
        flush
        local lsha; lsha="$(field "$l" 2)"
        printf '  local %-26s %s ... ' "$(field "$l" 3)" "${lsha:0:9}"
        git -C "$REPO" diff "$lsha^" "$lsha" -- "${PATHSPEC[@]}" > "$BUILD/.local.diff"
        git -C "$BUILD" apply --3way --whitespace=nowarn ".local.diff" 2>/dev/null
        rm -f "$BUILD/.local.diff"
        [ -n "$(git -C "$BUILD" diff --name-only --diff-filter=U)" ] \
          && loud "$(git -C "$BUILD" diff --name-only --diff-filter=U | head -1)" "--" "local $lsha" "no resolve lines are defined for a local commit"
        echo "clean"
        ;;
      patch)
        flush
        local p; p="$(field "$l" 2)"
        printf '  patch %-32s ... ' "$p"
        [ -f "$SELF_DIR/$p" ] || die "patch file $p is missing"
        git -C "$BUILD" apply --whitespace=nowarn "$SELF_DIR/$p" || die "patch $p did not apply"
        echo "applied"
        ;;
      *) die "unknown manifest line kind: $kind" ;;
    esac
  done
  flush
  git -C "$BUILD" add -A -N >/dev/null 2>&1
  if [ "$SURVEY" -eq 1 ]; then
    echo
    echo "SURVEY COMPLETE. Every hunk above is a decision owed to the manifest."
    git -C "$REPO" worktree remove --force "$BUILD" >/dev/null 2>&1 \
      && echo "(survey worktree removed: it was never a build)"
    return 0
  fi
  echo
  echo "built at $BUILD"
}

do_check() {
  local target="$1"
  git -C "$REPO" rev-parse --verify "$target^{commit}" >/dev/null 2>&1 || die "no such sha: $target"
  build || return $?
  echo
  echo "=== CHECK against $target ==="
  local surface full
  surface="$(git -C "$BUILD" diff --stat "$target" -- "${PATHSPEC[@]}")"
  full="$(git -C "$BUILD" diff --stat "$target")"
  echo "--- the build surface (${PATHSPEC[*]}) ---"
  if [ -z "$surface" ]; then
    echo "  EMPTY. byte-identical."
  else
    echo "$surface"
  fi
  echo "--- the whole tracked tree (reported, gates nothing) ---"
  if [ -z "$full" ]; then echo "  EMPTY."; else echo "$full"; fi
  echo
  if [ -z "$surface" ]; then
    echo "PASS: the manifest reproduces $target on its build surface."
    [ -n "$full" ] && echo "NOTE: the tree differs outside the build surface, listed above."
    rc=0
  else
    echo "FAIL: the manifest does NOT reproduce $target. The files above are what the"
    echo "      hand-curated tree contains that no manifest line accounts for."
    rc=3
  fi
  if [ "$KEEP" -eq 0 ]; then
    git -C "$REPO" worktree remove --force "$BUILD" >/dev/null 2>&1 && echo "(build worktree removed; --keep to retain it)"
  else
    echo "(kept: $BUILD)"
  fi
  return $rc
}

main() {
  while [ $# -gt 0 ]; do
    case "$1" in
      --dry-run) DRY=1 ;;
      --survey)  SURVEY=1; KEEP=0 ;;
      --keep)    KEEP=1 ;;
      --check)   shift; [ $# -gt 0 ] || die "--check needs a sha"; CHECK="$1" ;;
      -h|--help)
        echo "usage: build_stack.sh <manifest> [--dry-run] [--check <sha>] [--keep]"
        echo
        echo "  builds the manifest's stack into a FRESH detached worktree under _worktrees/."
        echo "  --dry-run  print the plan, write nothing"
        echo "  --check    build, then assert the result matches <sha> on the build surface"
        echo "  --keep     do not remove the build worktree after a check"
        echo "  --survey   enumerate EVERY conflict hunk instead of stopping at the first."
        echo "             it resolves with theirs throughout purely to keep walking, so its"
        echo "             tree is a survey artifact and must never be built on or measured."
        echo
        echo "  it STOPS on any conflict with no resolve line. it never guesses."
        exit 0 ;;
      -*) die "unknown option $1" ;;
      *)  [ -z "$MANIFEST" ] && MANIFEST="$1" || die "more than one manifest given" ;;
    esac
    shift
  done
  [ -n "$MANIFEST" ] || die "usage: build_stack.sh <manifest> [--dry-run] [--check <sha>]"
  [ -f "$MANIFEST" ] || die "no such manifest: $MANIFEST"
  command -v python3 >/dev/null || die "python3 is required for conflict resolution"
  parse_manifest
  if [ -n "$CHECK" ]; then do_check "$CHECK"; else build; fi
}

main "$@"
