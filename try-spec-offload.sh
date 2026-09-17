#!/bin/sh
# ===========================================================================
#  try-spec-offload.sh - the DSpark-versus-offload arm runner.
#
#  A sibling of try.sh, same house conventions: POSIX sh, no python, no jq,
#  a named arm out of environment and command-line switches, ONE fixed test
#  set run every time, a markdown-table result file, and the model unloaded
#  on every exit path.
#
#  THE QUESTION THIS RUNS
#  ----------------------
#  Does speculative decoding PAY on a box whose weights STREAM FROM DISK
#  rather than being resident?  Two axes, four legs:
#
#      residency   cold  (raw streaming, cache deliberately evicted)
#      residency   warm  (working set ALREADY resident at a fixed context)
#      spec        on    (--dspark with the support GGUF)
#      spec        off   (the same argv, the verify path killed)
#
#  The legs are run INTERLEAVED, on ONE binary vintage, because that is the
#  measurement law: a delta against a control from another build is not a
#  delta.  See the branch README for the arm definition and the citations.
#
#    ./try-spec-offload.sh --arm cold --on dspark     # cold, DSpark on
#    ./try-spec-offload.sh --arm cold --off dspark    # cold, DSpark off
#    ./try-spec-offload.sh --arm warm --on dspark     # WARM, DSpark on
#    ./try-spec-offload.sh --arm warm --off dspark    # WARM, DSpark off
#    ./try-spec-offload.sh --arm warm --on dspark --prime yes
#    ./try-spec-offload.sh --evict                    # cold leg: evict pages
#    ./try-spec-offload.sh --list                     # branches and builds
#    ./try-spec-offload.sh --dry-run                  # print the plan, load nothing
#
#  The four orders that matter, run in this sequence on one vintage:
#    1. cold / dspark off      3. warm / dspark off
#    2. cold / dspark on       4. warm / dspark on
#  then the whole block AGAIN, in the same order, for the repeat.
#
#  The Spark is a shared box and it is currently serialized to an unrelated
#  series: this script does NOT run itself, and it takes no lock, until that
#  lane is free.  It unloads the model when it ends, when nothing has run for
#  IDLE_SECS, and from a watchdog ON the Spark if this machine dies.
# ===========================================================================

set -u

# Numbers must not follow the terminal's locale: a comma decimal point would
# corrupt the recorded tok/s cells and mis-sort the summary.  C everywhere.
LC_ALL=C
export LC_ALL

# ---- knobs ----------------------------------------------------------------
SPARK="${SPARK:-spark}"
REMOTE_ROOT="${REMOTE_ROOT:-dwarfstar}"
MODEL="${MODEL:-gguf/DeepSeek-V4.1-Flash-Q2.gguf}"
MODEL_ID="${MODEL_ID:-ds4}"
# The DSpark drafter.  Flag name on the server is --mtp-model; the file is the
# support GGUF the downloader puts in gguf/ (docs/SPECULATIVE_DECODING.md:20).
DSPARK_MODEL="${DSPARK_MODEL:-gguf/DeepSeek-V4-Flash-DSpark-support-0731.gguf}"
PORT="${PORT:-8000}"
# A FIXED context, because the arms compare against each other and context is
# the largest single lever on a streaming box.  Same default as try.sh.
CTX="${CTX:-8321}"
SSD_CACHE="${SSD_CACHE:-90GB}"
IDLE_SECS="${IDLE_SECS:-30}"
CURL_MAX="${CURL_MAX:-600}"
DEFAULT_BRANCH="${DEFAULT_BRANCH:-triple-spec-under-offload}"
BUILD_DIR="${BUILD_DIR:-}"
FORK="${FORK:-sparkle}"
OUT_DIR="${OUT_DIR:-try-results}"
INDEX="${INDEX:-INDEX.md}"
MAX_SHORT="${MAX_SHORT:-128}"
MAX_LONG="${MAX_LONG:-256}"
# The cold leg's eviction.  A "cold" leg that is not actually cold measures the
# page cache, not the lever (WIKI/theory/48-decode-speedup-levers.md:229-234).
EVICT_FILE="${EVICT_FILE:-gguf/DeepSeek-V4-PRO-433G.gguf}"
EVICT_GB="${EVICT_GB:-112}"
PRIME="${PRIME:-no}"          # warm leg: run the fixed prime prompt first?
PRIME_TOKENS="${PRIME_TOKENS:-256}"

SSH_OPTS="-o ConnectTimeout=15 -o ServerAliveInterval=10 -o ServerAliveCountMax=3"

# $SPARK is an ssh alias: ssh resolves it from ~/.ssh/config, curl does not.
HTTP_HOST="${HTTP_HOST:-$(ssh $SSH_OPTS -G "$SPARK" 2>/dev/null | awk '$1 == "hostname" { print $2; exit }')}"
[ -n "$HTTP_HOST" ] || HTTP_HOST="$SPARK"

ROOT=$(cd "$(dirname "$0")" 2>/dev/null && pwd) || ROOT="."
[ -d "$ROOT/$OUT_DIR" ] || mkdir -p "$ROOT/$OUT_DIR"

# ---- colour ---------------------------------------------------------------
# Warm metals for structure: gold, amber, rust, ember.  Exactly ONE cool
# accent: teal, and it is reserved for measured numbers, nothing else.
if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then
	GOLD=$(printf '\033[38;5;220m')
	AMBER=$(printf '\033[38;5;214m')
	RUST=$(printf '\033[38;5;166m')
	EMBER=$(printf '\033[38;5;130m')
	TEAL=$(printf '\033[38;5;44m')
	DIM=$(printf '\033[2m')
	RST=$(printf '\033[0m')
else
	GOLD=''; AMBER=''; RUST=''; EMBER=''; TEAL=''; DIM=''; RST=''
fi

say() { [ "${QUIET:-0}" = 1 ] || printf '%s\n' "$*"; }
gold() { printf '%s%s%s\n' "$GOLD" "$*" "$RST"; }
amber() { printf '%s%s%s\n' "$AMBER" "$*" "$RST"; }
rust() { printf '%s%s%s\n' "$RUST" "$*" "$RST"; }
ember() { printf '%s%s%s\n' "$EMBER" "$*" "$RST"; }
teal() { printf '%s%s%s\n' "$TEAL" "$*" "$RST"; }
warn() { printf '%s %s%s\n' "$(rust '!!')" "$RUST" "$*$RST" >&2; }
die() { warn "$*"; exit 1; }

now_epoch() { date +%s; }
mtime() { stat -f %m "$1" 2>/dev/null || stat -c %Y "$1" 2>/dev/null || echo 0; }
short_sha() { printf '%s' "$1" | cut -c1-9; }

# JSON string escaping with POSIX tools only (no python3, no jq).
json_str() {
	printf '%s' "$1" |
		sed -e 's/\\/\\\\/g' -e 's/"/\\"/g' -e 's/\t/\\t/g' |
		sed -e 's/$/\\n/' | tr -d '\n' | sed -e 's/\\n$//'
}

# ---- the banner -----------------------------------------------------------
banner() {
	printf '\n'
	printf '        %s.%s         %s*%s\n' "$EMBER" "$RST" "$GOLD" "$RST"
	printf '          %s\\    /%s\n' "$RUST" "$RST"
	printf '   %s*%s    %s.  \\  /  .%s    %s*%s\n' "$GOLD" "$RST" "$AMBER" "$RST" "$GOLD" "$RST"
	printf '             %s\\/%s\n' "$GOLD" "$RST"
	printf '     %sD W A R F S T A R%s   %s.spec-offload.%s\n' "$GOLD" "$RST" "$RUST" "$RST"
	printf '             %s/\\%s      %sdoes spec-decode pay when the weights stream?%s\n' "$GOLD" "$RST" "$DIM" "$RST"
	printf '   %s*%s    %s.  /  \\  .%s    %s*%s   %sfour legs: cold and warm, dspark off and on%s\n' "$GOLD" "$RST" "$AMBER" "$RST" "$GOLD" "$RST" "$DIM" "$RST"
	printf '          %s/    \\%s\n' "$RUST" "$RST"
	printf '        %s*%s         %s.%s\n\n' "$GOLD" "$RST" "$EMBER" "$RST"
}

# ---- arguments ------------------------------------------------------------
BRANCH="$DEFAULT_BRANCH"
BRANCH_SET=0
ARM="warm"
TOGGLES=""
DRY=0; QUIET=0; LIST=0; EVICT=0

while [ $# -gt 0 ]; do
	case "$1" in
		--branch|--build-branch) BRANCH="${2:-}"; BRANCH_SET=1; shift 2 ;;
		--dir)     BUILD_DIR="${2:-}"; shift 2 ;;
		--arm)     ARM="${2:-}"; shift 2 ;;
		--on)      TOGGLES="$TOGGLES $2:on"; shift 2 ;;
		--off)     TOGGLES="$TOGGLES $2:off"; shift 2 ;;
		--prime)   PRIME="${2:-}"; shift 2 ;;
		--evict)   EVICT=1; shift ;;
		--quiet|-q) QUIET=1; shift ;;
		--list)    LIST=1; shift ;;
		--dry-run) DRY=1; shift ;;
		-h|--help) sed -n '2,38p' "$0"; exit 0 ;;
		-*) printf 'unknown option: %s\n' "$1" >&2; exit 2 ;;
		*) BRANCH="$1"; BRANCH_SET=1; shift ;;
	esac
done

# ---- discovery: which branches exist on the fork --------------------------
# Nothing here is a hardcoded list of arms.  Any branch whose name starts with
# "triple-" appears in the menu the moment it is pushed, with no code change.
discover_branches() {
	git -C "$ROOT" ls-remote --heads "$FORK" 2>/dev/null |
		sed -n 's|^\([0-9a-f]*\)[[:space:]]*refs/heads/\(triple-.*\)$|\1 \2|p' |
		sort -k2
}

# One ssh round trip: every ds4-server worktree on the Spark, with its branch
# and HEAD, so the menu can say what is already built.  Read-only.
remote_worktrees() {
	ssh $SSH_OPTS "$SPARK" "
		for d in \$HOME/$REMOTE_ROOT/_worktrees/*/; do
			[ -d \"\$d\" ] || continue
			n=\$(basename \"\$d\")
			b=\$(git -C \"\$d\" rev-parse --abbrev-ref HEAD 2>/dev/null || echo none)
			s=\$(git -C \"\$d\" rev-parse HEAD 2>/dev/null || echo none)
			x=no
			[ -x \"\$d/ds4-server\" ] && x=yes
			c=\$(git -C \"\$d\" log -1 --format=%ct 2>/dev/null || echo 0)
			printf '%s|%s|%s|%s|%s\n' \"\$n\" \"\$b\" \"\$s\" \"\$x\" \"\$c\"
		done" </dev/null 2>/dev/null
}

wt_field() { printf '%s\n' "${WT_ROWS:-}" | awk -F'|' -v d="$1" -v n="$2" '$1==d{print $n; exit}'; }

# Which worktree on the Spark serves this branch?  Same rules as try.sh:
# try-arms.txt, then a worktree named after the branch, then one checked out on
# it, then the nearest tb-* build (a GUESS, and the runner says so).
arm_for_slug() {
	awk -v s="$1" '$1==s && $2!="" {print $2; exit}' "$ROOT/try-arms.txt" 2>/dev/null
}

resolve_dir() { # $1=slug $2=branch tip sha
	for d in ${WT_ALL:-}; do
		[ "$d" = "$1" ] && [ "$(wt_field "$d" 4)" = yes ] && { printf '%s' "$d"; return 0; }
	done
	for d in ${WT_ALL:-}; do
		[ "$(wt_field "$d" 4)" = yes ] || continue
		[ "$(wt_field "$d" 2)" = "$1" ] && { printf '%s' "$d"; return 0; }
	done
	best=""; bestd=""; bestct=0
	for d in ${WT_ALL:-}; do
		[ "$(wt_field "$d" 4)" = yes ] || continue
		case "$d" in tb-*) ;; *) continue ;; esac
		s=$(wt_field "$d" 3)
		git -C "$ROOT" cat-file -e "$s^{commit}" 2>/dev/null || continue
		git -C "$ROOT" merge-base --is-ancestor "$s" "$2" 2>/dev/null || continue
		n=$(git -C "$ROOT" rev-list --count "$s..$2" 2>/dev/null) || continue
		[ -n "$n" ] || continue
		ct=$(wt_field "$d" 5); ct=${ct:-0}
		if [ -z "$best" ] || [ "$n" -lt "$best" ] || { [ "$n" = "$best" ] && [ "$ct" -gt "$bestct" ]; }; then
			best=$n; bestd=$d; bestct=$ct
		fi
	done
	[ -n "$bestd" ] && printf '%s' "$bestd"
	return 0
}

pick_dir_auto() {
	d=$(arm_for_slug "$1")
	[ -n "$d" ] && [ "$(wt_field "$d" 4)" = yes ] && { printf '%s' "$d"; return 0; }
	resolve_dir "$1" "$2"
}

# ---- the levers and their switches ----------------------------------------
# Two tables, because this branch's levers live in TWO places and try.sh's
# single table is environment-only.  The format is identical (six fields) and
# the runner prints both; the split is stated rather than hidden.
#
# Fields: lever|switch|env-when-on|env-when-off|has-an-off-switch|note
#
# ENV levers.  Verified against the source in this tree.
#   lever        switch                  default  off
#   -----------  ----------------------  -------  --------------------------
#   spec-verify  DS4_MTP_SPEC_DISABLE    on       =1  (kills the verify path)
#
# ARGV levers.  These are command-line flags, not environment; they are what
# the ARM is loaded with, and there is no env to flip them afterwards.
#   lever        argv                          default  off
#   -----------  ----------------------------  -------  --------------------
#   dspark       --dspark --mtp-model <file>   OFF      both flags omitted
#   streaming    --ssd-streaming               on       flag omitted
#   cold         --ssd-streaming-cold          OFF      flag omitted
#   prime        (the fixed prime prompt)      OFF      not sent
#
# Three of the six have no env off-switch, so an "everything off" arm is NOT
# clean on them by environment alone; this runner says so out loud, exactly as
# try.sh does for its four.
ENV_LEVERS='spec-verify|DS4_MTP_SPEC_DISABLE|-|1|1|kill switch for the speculative verify path (ds4_server.c:13794; also ds4_agent.c:10499, ds4_cli.c:598)'

ARGV_LEVERS='dspark|--dspark --mtp-model FILE|-|-|1|the drafter, its Markov head and the verify pass (ds4_help.c:190)
streaming|--ssd-streaming|-|-|0|weights stream from disk; omitting it is a resident box, not this branch (ds4_help.c:173)
cold|--ssd-streaming-cold|-|-|1|skip the popularity preload so the opening is genuinely cold (ds4_help.c:174)
prime|(fixed prime prompt)|-|-|1|warm leg: one discarded pass at the fixed context, before measuring'

# dspark is the one argv lever that ALSO has a real env off-switch, because the
# verify path it engages is gated on DS4_MTP_SPEC_DISABLE.  That is what makes
# the same binary and the same loaded weights serve both legs of the pair.

env_lever_names() { printf '%s\n' "$ENV_LEVERS" | cut -d'|' -f1 | tr '\n' ' '; }
argv_lever_names() { printf '%s\n' "$ARGV_LEVERS" | cut -d'|' -f1 | tr '\n' ' '; }
lever_rec() { { printf '%s\n' "$ENV_LEVERS"; printf '%s\n' "$ARGV_LEVERS"; } | awk -F'|' -v L="$1" '$1==L{print; exit}'; }
lever_field() { lever_rec "$1" | cut -d'|' -f"$2"; }
lever_names() { printf '%s ' "$(env_lever_names)$(argv_lever_names)"; }

tog_state() {
	for t in ${TOGGLES:-}; do
		case "$t" in
			"$1:on")  printf 'on'; return 0 ;;
			"$1:off") printf 'off'; return 0 ;;
		esac
	done
	return 1
}

# An arm is a starting state for the four levers; --on/--off then moves one
# lever at a time, so a single binary serves every arm.
#   stack   the branch AS IT SHIPS: streaming, warm, dspark OFF
#   warm    the warm-working-set pair: default preload; --on dspark for the
#           on leg, --off dspark for the off leg.  This is the arm nobody has
#           run.
#   cold    the raw pair: --ssd-streaming-cold, and evict the pages first
#   all-on  every lever forced on, the opt-ins included (dspark on by default)
build_arm() {
	case "$ARM" in
		stack)  DSPARK_WANT=off ; COLD_WANT=off ; STREAM_WANT=on ;;
		warm)   DSPARK_WANT=off ; COLD_WANT=off ; STREAM_WANT=on ;;
		cold)   DSPARK_WANT=off ; COLD_WANT=on  ; STREAM_WANT=on ;;
		all-on) DSPARK_WANT=on  ; COLD_WANT=off ; STREAM_WANT=on ;;
		*)      warn "unknown arm '$ARM' (use stack, warm, cold or all-on)"; exit 2 ;;
	esac

	for t in ${TOGGLES:-}; do
		L=$(printf '%s' "$t" | cut -d: -f1)
		[ -n "$(lever_rec "$L")" ] || { warn "no such lever: $L (have: $(lever_names))"; exit 2; }
	done

	o=$(tog_state dspark) && DSPARK_WANT=$o
	o=$(tog_state cold)   && COLD_WANT=$o
	o=$(tog_state streaming) && STREAM_WANT=$o
	o=$(tog_state spec-verify) && VERIFY_WANT=$o
	o=$(tog_state prime) && PRIME=$o

	# The spec axis, stated once.  Three states, not two:
	#   DSPARK_WANT=on   -> the drafter is loaded and the verify path is LIVE
	#   DSPARK_WANT=off  -> the SAME argv, and DS4_MTP_SPEC_DISABLE=1 kills the
	#                       verify path.  This is the same-vintage pair partner:
	#                       one binary, one loaded weight set, one difference.
	#   (no --on/--off)  -> --arm stack: no --dspark, no drafter at all.  That
	#                       is the pure baseline, and it costs a different
	#                       launch, so it is NOT the pair partner.
	# VERIFY_WANT therefore follows DSPARK_WANT unless the user says otherwise.
	VERIFY_WANT=${VERIFY_WANT:-$DSPARK_WANT}
	[ "$VERIFY_WANT" = on ] && [ "$DSPARK_WANT" = off ] && \
		warn "spec-verify ON while dspark is OFF: the verify path is never entered, so this leg is target-only either way."

	# argv, accumulated in the order the server expects them
	ARGV=''
	[ "$STREAM_WANT" = on ] && ARGV="$ARGV --ssd-streaming"
	[ "$COLD_WANT" = on ] && ARGV="$ARGV --ssd-streaming-cold"
	ARGV="$ARGV --ssd-streaming-cache-experts $SSD_CACHE"

	# The dspark flag is in the argv for BOTH legs of the pair.  That is the
	# whole point: the same binary, the same loaded drafter, and the ONE
	# difference is DS4_MTP_SPEC_DISABLE.  Dropping the flag instead would
	# change the loaded weight set and break the same-vintage rule.
	#   on     -> --dspark --mtp-model <file>, no DS4_MTP_SPEC_DISABLE
	#   off    -> --dspark --mtp-model <file>, DS4_MTP_SPEC_DISABLE=1
	#   absent -> neither; no drafter loaded at all (the pure baseline, and
	#             it costs a different launch, so it is NOT the pair partner)
	case " ${TOGGLES:-} " in
		*" dspark:on "*|*" dspark:off "*) DSPARK_LOADED=yes ;;
		*) DSPARK_LOADED=no ;;
	esac
	DSPARK_STATE=absent
	if [ "$DSPARK_LOADED" = yes ]; then
		ARGV="$ARGV --dspark --mtp-model $DSPARK_MODEL"
		if [ "$DSPARK_WANT" = on ]; then DSPARK_STATE=on; else DSPARK_STATE=off; fi
	fi
	ARGV=$(printf '%s' "$ARGV" | sed -e 's/^ *//')

	# env, accumulated the way try.sh does it
	ENVSTR=''; NOOFF=''
	if [ "$VERIFY_WANT" = off ] && [ "$DSPARK_LOADED" = yes ]; then
		ENVSTR="DS4_MTP_SPEC_DISABLE=1"
	fi
	# NOTE on the control's cleanliness, said out loud in print_arms():
	#   with dspark OFF and spec-verify ON, the DS4_MTP_SPEC_DISABLE variable
	#   is irrelevant (the branch is not entered).  With dspark ON and
	#   spec-verify OFF, the drafter is STILL LOADED and still resident (or
	#   still streaming) - it just does not verify.  That is the cleanest
	#   same-vintage pair this engine offers; it is NOT a no-drafter control.
	ENGAGED=0
	[ "$STREAM_WANT" = on ] && ENGAGED=$((ENGAGED + 1))
	[ "$DSPARK_WANT" = on ] && ENGAGED=$((ENGAGED + 1))
	[ "$COLD_WANT" = on ] && ENGAGED=$((ENGAGED + 1))
	[ "$PRIME" = yes ] && ENGAGED=$((ENGAGED + 1))
	# the three with no env off-switch
	NOOFF='streaming cold prime'
}

print_arms() {
	printf '%s\n' "  lever                switch                                  state" | sed "s/^/$AMBER/;s/$/$RST/"
	for L in $(lever_names); do
		sw=$(lever_field "$L" 2)
		case "$L" in
			spec-verify) st=$VERIFY_WANT ;;
			dspark)      st=$DSPARK_STATE ;;
			streaming)   st=$STREAM_WANT ;;
			cold)        st=$COLD_WANT ;;
			prime)       st=$PRIME ;;
			*)           st=off ;;
		esac
		if [ "$st" = on ]; then scol="$GOLD"; else scol="$EMBER"; fi
		printf '  %-20s %-39s %s%s%s\n' "$L" "$sw" "$scol" "$st" "$RST"
	done
}

# ---- what is out there ----------------------------------------------------
banner

BRANCHES=$(discover_branches)
[ -n "$BRANCHES" ] || die "cannot list branches from fork '$FORK' - is the remote there and the name right?"

WT_ROWS=$(remote_worktrees)
WT_ALL=$(printf '%s\n' "${WT_ROWS:-}" | cut -d'|' -f1 | grep -v '^$' | tr '\n' ' ')

menu() {
	i=0; def=0
	while IFS=' ' read -r sha slug; do
		[ -n "${slug:-}" ] || continue
		i=$((i + 1))
		d=$(pick_dir_auto "$slug" "$sha")
		mark=""
		[ -n "$d" ] || mark="  ${DIM}no build on $SPARK - build it: make cuda-spark -j12${RST}"
		if [ "$slug" = "$DEFAULT_BRANCH" ]; then
			def=$i
			printf '  %s%2d) %-28s%s %s<- default%s%s\n' "$GOLD" "$i" "$slug" "$RST" "$GOLD" "$mark" "$RST"
		else
			printf '  %s%2d)%s %-28s %s%s%s%s\n' "$AMBER" "$i" "$RST" "$slug" "$EMBER" "$(short_sha "$sha")" "$RST" "$mark"
		fi
		[ -z "$d" ] || printf '      %sbuilt: ~/%s/_worktrees/%s%s\n' "$DIM" "$REMOTE_ROOT" "$d" "$RST"
	done <<EOF
$BRANCHES
EOF
	[ "$def" = 0 ] && def=1
	printf '\n  %sbranches come straight from%s %s%s%s %s- nothing here is hardcoded%s\n\n' \
		"$DIM" "$RST" "$AMBER" "git ls-remote --heads $FORK" "$RST" "$DIM" "$RST"
}

if [ "$LIST" = 1 ]; then
	menu
	exit 0
fi

if [ "$BRANCH_SET" = 0 ]; then
	if [ -t 0 ] || { [ -r /dev/tty ] && [ -t 1 ]; }; then
		menu
		printf '  %sbranch [%s]:%s ' "$GOLD" "$DEFAULT_BRANCH" "$RST"
		read -r pick </dev/tty || pick=""
		[ -n "$pick" ] || pick="$DEFAULT_BRANCH"
		case "$pick" in
			*[!0-9]*) BRANCH="$pick" ;;
			*) BRANCH=$(printf '%s\n' "$BRANCHES" | awk -v n="$pick" 'NR==n{print $2; exit}') ;;
		esac
		[ -n "$BRANCH" ] || die "nothing selected"
	else
		amber "  no tty - defaulting to $BRANCH (see --list for every branch)"
	fi
fi

BR_SHA=$(printf '%s\n' "$BRANCHES" | awk -v b="$BRANCH" '$2==b{print $1; exit}')
[ -n "$BR_SHA" ] || die "branch '$BRANCH' is not on $FORK - try --list"

# ---- the arm and its build ------------------------------------------------
build_arm

RMODE=""
if [ -n "$BUILD_DIR" ]; then
	RDIR="$BUILD_DIR"; RMODE=explicit
	if [ "$(wt_field "$RDIR" 4)" != yes ]; then
		warn "--dir $RDIR has no ds4-server on $SPARK"
	fi
else
	RDIR=$(arm_for_slug "$BRANCH")
	[ -n "$RDIR" ] && [ "$(wt_field "$RDIR" 4)" = yes ] && RMODE=map
	if [ -z "$RDIR" ]; then
		RDIR=$(resolve_dir "$BRANCH" "$BR_SHA")
		[ -n "$RDIR" ] && RMODE=auto
	fi
fi
[ -n "$RDIR" ] || die "no built ds4-server on $SPARK for $BRANCH (build one: ssh $SPARK 'cd ~/$REMOTE_ROOT/_worktrees/<arm> && make cuda-spark -j12')"
[ "$RMODE" = auto ] && warn "build dir '$RDIR' chosen by proximity - pass --dir <name> to pin it"

TODAY=$(date +%Y-%m-%d)
RUN_ID=$(date +%Y-%m-%d-%H%M%S)   # time of day: runs of one day sort apart
# The spec axis has three labels, not two: an arm that never passed a dspark
# toggle has NO drafter loaded at all, and calling that leg "dspark-off" would
# hide the difference between "loaded and killed" and "never loaded".
case " ${TOGGLES:-} " in
	*" dspark:on "*|*" dspark:off "*) SPEC_LABEL=$DSPARK_STATE ;;
	*) SPEC_LABEL=absent ;;
esac
# The standard naming rule: date, branch slug, what was ON, hyphen separated.
WHAT="dspark-$SPEC_LABEL-$([ "$COLD_WANT" = on ] && printf cold || printf warm)"
OUTFILE="$OUT_DIR/$TODAY-$BRANCH-$WHAT.txt"

rust "  branch   $GOLD$BRANCH$RST $EMBER$(short_sha "$BR_SHA")$RST"
rust "  build    $AMBER~/$REMOTE_ROOT/_worktrees/$RDIR$RST $(ember "@ $(wt_field "$RDIR" 3 | cut -c1-9)")"
rust "  arm      $AMBER$ARM$RST  ($ENGAGED of 4 levers engaged)"
rust "  leg      $GOLD dspark $DSPARK_STATE $RST / $GOLD $( [ "$COLD_WANT" = on ] && printf cold || printf warm ) $RST  at ctx $CTX"
if [ -n "$ENVSTR" ]; then
	rust "  env      $TEAL$ENVSTR$RST"
else
	rust "  env      $(ember "(none)")"
fi
rust "  argv     $AMBER$ARGV$RST"
printf '\n'
print_arms
printf '\n'

# The control's own caveat, stated rather than hidden.
if [ "$DSPARK_STATE" = absent ]; then
	warn "no --dspark in this argv: NO drafter is loaded.  This is the pure"
	warn "baseline.  It is NOT the pair partner for the ON leg - the pair partner"
	warn "loads the drafter and kills the verify path (--off dspark)."
fi
if [ "$DSPARK_STATE" = off ]; then
	warn "dspark is OFF with $TEAL DS4_MTP_SPEC_DISABLE=1 $RUST and the SAME argv as the ON leg:"
	warn "the drafter is loaded and resident (or streaming), and the verify path"
	warn "is dead (ds4_server.c:13794).  This IS the same-vintage pair partner."
	warn "It is still NOT a no-drafter control - the drafter costs memory and I/O."
fi
# A lever with no env off-switch cannot be switched off, so an "everything
# off" arm is not a clean control on those levers.  Say it, do not hide it.
if [ "$ARM" = all-on ]; then
	warn "this arm wants ON, but these have no env off-switch to sit at OFF for a"
	warn "clean control: $(printf '%s' "$NOOFF" | sed -e 's/^ *//' | tr ' ' ',')."
	warn "Switching them means a different argv, which means a different launch."
fi

# ---- the cold leg's eviction (no sudo: stream an unrelated file) ----------
# WIKI/theory/48-decode-speedup-levers.md:229-234 is the law here: a "cold" leg
# whose pages are still in the cache measures the cache.  ~1.2 GB/s on the
# Spark's SSD, so EVICT_GB 112 is about 105 seconds.
if [ "$EVICT" = 1 ] || [ "$COLD_WANT" = on ]; then
	rust "  evict    $AMBER dd if=\$HOME/$REMOTE_ROOT/$EVICT_FILE of=/dev/null ${EVICT_GB}GiB (~105s at 1.2 GB/s)$RST"
fi

# ---- the server command ---------------------------------------------------
# Same shape as try.sh's SERVER_CMD, plus this branch's argv.  The support GGUF
# is passed with --mtp-model; --dspark is what makes the engine read it as a
# DSpark drafter rather than a legacy MTP one (ds4_help.c:190).
SERVER_CMD="./ds4-server -m \$HOME/$REMOTE_ROOT/$MODEL --port $PORT --ctx $CTX $ARGV"
rust "  server   $AMBER$SERVER_CMD$RST"
rust "  result   $AMBER$ROOT/$OUTFILE$RST"
printf '\n'

if [ "$DRY" = 1 ]; then
	gold "  dry run - nothing loaded, nothing run."
	printf '\n'
	exit 0
fi

# ---- unload safety net ----------------------------------------------------
STAMP=$(mktemp); FLIGHT=$(mktemp)
[ -n "${TMPDIR:-}" ] || TMPDIR=/tmp
for f in "$STAMP" "$FLIGHT"; do : >"$f"; done
LOADED=""; TICK=""; HEART=""; WATCH=""; RESFILE=""; VALS=""
REMOTE_BEAT="\$HOME/.ds4-try.beat"

stop_server() {
	[ -n "${LOADED:-}" ] || return 0
	LOADED=""
	[ -n "${TICK:-}" ] && kill "$TICK" 2>/dev/null
	[ -n "${HEART:-}" ] && kill "$HEART" 2>/dev/null
	[ -n "${WATCH:-}" ] && kill "$WATCH" 2>/dev/null
	say "unloading $BRANCH from $SPARK"
	ssh $SSH_OPTS "$SPARK" "pkill -f '[d]s4-server.*--port $PORT'; rm -f $REMOTE_BEAT \$HOME/.ds4-try.pid" \
		</dev/null >/dev/null 2>&1
}
on_exit() { stop_server; rm -f "$STAMP" "$FLIGHT" ${RESFILE:+"$RESFILE"} ${VALS:+"$VALS"}; }
trap 'on_exit' EXIT
# INT/TERM/HUP must EXIT, not only clean up: POSIX sh RESUMES the script
# after a handled signal, and the tests would keep "running" against a
# server the trap just unloaded.
trap 'exit 1' INT TERM HUP

tick_start() { ( while :; do touch "$1"; sleep 5; done ) & TICK=$!; }
tick_stop() { [ -n "${TICK:-}" ] && kill "$TICK" 2>/dev/null; TICK=""; }

heartbeat() { while :; do ssh $SSH_OPTS "$SPARK" "touch $REMOTE_BEAT" </dev/null >/dev/null 2>&1; sleep 5; done; }

spark_watchdog() {
	ssh $SSH_OPTS "$SPARK" "
		while :; do
			sleep 5
			now=\$(date +%s)
			seen=\$(stat -c %Y $REMOTE_BEAT 2>/dev/null || stat -f %m $REMOTE_BEAT 2>/dev/null)
			[ -n \"\$seen\" ] || continue
			if [ \$((now - seen)) -ge $((IDLE_SECS * 6)) ]; then
				pkill -f '[d]s4-server.*--port $PORT'
				rm -f $REMOTE_BEAT \$HOME/.ds4-try.pid
				exit 0
			fi
		done" </dev/null >/dev/null 2>&1 &
}

idle_watch() {
	while [ -n "${LOADED:-}" ]; do
		sleep 5
		[ -n "${LOADED:-}" ] || break
		now=$(now_epoch); seen=$(mtime "$FLIGHT")
		if [ $((now - seen)) -ge "$IDLE_SECS" ]; then
			say "idle ${IDLE_SECS}s - unloading"
			stop_server
			break
		fi
	done
}

# ---- the cold leg's eviction, actually run --------------------------------
if [ "$COLD_WANT" = on ]; then
	say "evicting the page cache with $EVICT_GB GiB of $EVICT_FILE ..."
	ssh $SSH_OPTS "$SPARK" "dd if=\"\$HOME/$REMOTE_ROOT/$EVICT_FILE\" of=/dev/null bs=1m count=$((EVICT_GB * 1024)) 2>&1 | tail -1" </dev/null
	rust "  evicted - the cold leg is now genuinely cold on the device, not in RAM"
fi

# ---- load -----------------------------------------------------------------
say "loading $BRANCH ($RDIR) on $SPARK ..."
tick_start "$FLIGHT"
heartbeat & HEART=$!
spark_watchdog
touch "$STAMP"

ssh $SSH_OPTS "$SPARK" "
	cd \"\$HOME/$REMOTE_ROOT/_worktrees/$RDIR\" || { echo 'NODIR'; exit 1; }
	if pgrep -f '[d]s4-server.*--port $PORT' >/dev/null 2>&1; then echo 'BUSY'; exit 3; fi
	touch $REMOTE_BEAT
	nohup env $ENVSTR $SERVER_CMD > \"/tmp/try-spec-$RDIR.log\" 2>&1 &
	echo \$! > \$HOME/.ds4-try.pid
	echo 'STARTED'
" </dev/null 2>&1 | tee /tmp/try-spec-load.out >/dev/null

case "$(cat /tmp/try-spec-load.out 2>/dev/null)" in
	*BUSY*)   die "a ds4-server is already on port $PORT on $SPARK - refusing to stack another";;
	*NODIR*)  die "no worktree ~/$REMOTE_ROOT/_worktrees/$RDIR on $SPARK";;
	*STARTED*) LOADED=1 ;;
	*) die "could not start the server on $SPARK (see the output above)";;
esac

i=0
while [ $i -lt 180 ]; do
	if curl -s -m 3 "http://$HTTP_HOST:$PORT/v1/models" >/dev/null 2>&1; then break; fi
	touch "$FLIGHT"
	i=$((i + 1)); sleep 5
done
if [ $i -ge 180 ]; then
	warn "server did not answer in 900s - last log lines:"
	ssh $SSH_OPTS "$SPARK" "tail -20 \"/tmp/try-spec-$RDIR.log\"" </dev/null 2>&1 >&2
	stop_server
	exit 1
fi
touch "$FLIGHT"
tick_stop
idle_watch & WATCH=$!
say "loaded after ~$((i * 5))s.  api: http://$HTTP_HOST:$PORT/v1/models"
printf '\n'

# ---- the fixed test set ---------------------------------------------------
# The same ten prompts on every run, so two legs are always compared on one
# set.  Identical to try.sh's ten - do not edit them here alone, or the legs
# stop being comparable with the rest of the series.
SHORT1="In one sentence, what is a mixture-of-experts layer?"
SHORT2="Name the three primary colours. Answer in three words."
SHORT3="What does SSD streaming mean? One sentence."
SHORT4="Give the single word for the process of caching disk pages in RAM."
SHORT5="In one sentence, why is prefill different from decode?"

LONG_BODY="A large language model serving on a single machine with a small fixed
amount of fast memory and a very large amount of slower storage has to decide
what to keep resident and what to fetch on demand. Every layer of a modern
sparse model routes each token to a small number of experts out of a much larger
set, and which experts are chosen depends on the token itself. The total set of
expert weights is far larger than memory, so the runtime must treat the expert
weights as a cache that is backed by the storage device. The interesting
engineering question is not whether to cache, but what to do at the moment a
miss happens: whether to fetch one expert at a time and wait, whether to batch
misses from the whole layer, whether to prefetch something likely to be needed
next, and how large a reserve to keep for the pages that arrive. Each of those
choices has a cost and a benefit, and they interact, because they all contend
for the same device."
LONG1="$LONG_BODY

Summarise the trade-offs in this passage in three short paragraphs."
LONG2="$LONG_BODY

Which of the described strategies would you expect to help latency most, and why? Four sentences."
LONG3="$LONG_BODY

Rewrite the passage as if explaining it to a colleague in a corridor. Keep it under 120 words."
LONG4="$LONG_BODY

List the failure modes this passage implies. Use a numbered list."
LONG5="$LONG_BODY

In two sentences, what measurement would you take to know whether any of these choices helped?"

# The PRIME prompt: the warm leg sends this ONCE and discards it, so the
# working set is resident before the measured block starts.  It is the same
# passage the long tests use, at the same fixed context, so what it warms is
# what the measurement reads.
PRIME_PROMPT="$LONG_BODY

Explain in one paragraph what a working set is and why it matters here."

run_one() { # $1=test name $2=prompt $3=max_tokens
	tn=$1; tp=$2; mx=$3
	payload=$(printf '{"model":"%s","prompt":"%s","max_tokens":%s,"temperature":0,"stream":false}' \
		"$MODEL_ID" "$(json_str "$tp")" "$mx")
	tick_start "$FLIGHT"
	raw=$(curl -s -m "$CURL_MAX" -X POST "http://$HTTP_HOST:$PORT/v1/completions" \
		-H 'Content-Type: application/json' -d "$payload" \
		-w '\n__T__%{time_total}' 2>/dev/null)
	tick_stop
	secs=$(printf '%s\n' "$raw" | awk -F'__T__' '/__T__/{v=$2} END{ if (v=="") printf "0"; else printf "%s", v }')
	toks=$(printf '%s\n' "$raw" |
		sed -n 's/.*"completion_tokens"[[:space:]]*:[[:space:]]*\([0-9][0-9]*\).*/\1/p' | head -1)
	st=ok
	[ -n "$secs" ] || secs=0
	[ -n "$toks" ] || { toks=0; st=err; }
	[ "$toks" -gt 0 ] 2>/dev/null || st=err
	tps=$(awk -v n="$toks" -v t="$secs" 'BEGIN{ if (t+0 > 0 && n+0 > 0) printf "%.2f", n/t; else printf "0.00" }')
	printf '%s|%s|%s|%s|%s' "$tn" "$st" "$toks" "$secs" "$tps"
}

RESFILE=$(mktemp)
run_pair() {
	line=$(run_one "$1" "$2" "$3")
	IFS='|' read -r n s tok sec tps <<EOF
$line
EOF
	if [ "$s" = ok ]; then
		printf '  %s%-12s%s %s%6s%s %8s  %s%8s%s %s\n' "$AMBER" "$n" "$RST" \
			"$GOLD" "$tok" "$RST" "$sec" "$TEAL" "$tps" "$RST" "$(ember 'tok/s')"
	else
		printf '  %s%-12s%s %6s %8s  %s%8s%s  %s\n' "$AMBER" "$n" "$RST" "$tok" "$sec" \
			"$RUST" "ERR" "$RST" "$(ember 'no completion_tokens - see /tmp/try-spec-'"$RDIR"'.log')"
	fi
	printf '%s|%s|%s|%s|%s\n' "$n" "$s" "$tok" "$sec" "$tps" >>"$RESFILE"
}

# ---- the prime pass (warm leg only) --------------------------------------
PRIME_STATE=no
if [ "$PRIME" = yes ]; then
	amber "  prime - one discarded pass at ctx $CTX, so the working set is already resident"
	pline=$(run_one prime "$PRIME_PROMPT" "$PRIME_TOKENS")
	IFS='|' read -r pn ps ptok psec ptps <<EOF
$pline
EOF
	if [ "$ps" = ok ]; then
		printf '  %s%-12s%s %s%6s%s %8s  %s%8s%s %s\n' "$AMBER" "prime" "$RST" \
			"$GOLD" "$ptok" "$RST" "$psec" "$EMBER" "$ptps" "$RST" "$(ember 'discarded, not counted')"
		PRIME_STATE=yes
	else
		warn "prime pass failed - the warm leg is NOT warm; recording it as such"
	fi
	printf '\n'
fi

printf '  %s%-12s %6s %8s  %8s%s\n' "$AMBER" "test" "tokens" "secs" "tok/s" "$RST"
printf '  %s%-12s %6s %8s  %8s%s\n' "$EMBER" "------------" "------" "--------" "-------" "$RST"
rust "  five short prompts"
run_pair short-1 "$SHORT1" "$MAX_SHORT"
run_pair short-2 "$SHORT2" "$MAX_SHORT"
run_pair short-3 "$SHORT3" "$MAX_SHORT"
run_pair short-4 "$SHORT4" "$MAX_SHORT"
run_pair short-5 "$SHORT5" "$MAX_SHORT"
printf '\n'
rust "  five long prompts"
run_pair long-1 "$LONG1" "$MAX_LONG"
run_pair long-2 "$LONG2" "$MAX_LONG"
run_pair long-3 "$LONG3" "$MAX_LONG"
run_pair long-4 "$LONG4" "$MAX_LONG"
run_pair long-5 "$LONG5" "$MAX_LONG"
printf '\n'

# ---- summary --------------------------------------------------------------
VALS=$(mktemp)
awk -F'|' '$2=="ok" && $5+0 > 0 {print $5}' "$RESFILE" | sort -n > "$VALS"
STAT=$(awk '{v[NR]=$1} END{
	if (NR==0) { print "none none none 0"; exit }
	m = (NR%2) ? v[(NR+1)/2] : (v[NR/2] + v[NR/2+1]) / 2
	printf "%.2f %.2f %.2f %d", v[1], m, v[NR], NR }' "$VALS")
set -- $STAT
MIN=$1; MED=$2; MAX=$3; N=$4

if [ "$STAT" = "none none none 0" ]; then
	warn "no measurements at all - check /tmp/try-spec-$RDIR.log on $SPARK"
	gold "  summary : nothing measured"
else
	printf '  %ssummary%s  min %s%s%s  median %s%s%s  max %s%s%s  %s(%s tests)%s\n' \
		"$GOLD" "$RST" "$GOLD" "$MIN" "$RST" "$TEAL" "$MED" "$RST" "$GOLD" "$MAX" "$RST" \
		"$DIM" "$N" "$RST"
fi
printf '\n'

# ---- the result file ------------------------------------------------------
# Plain text, markdown tables, so a row can be pasted straight into a README.
SUMMARY_LINE="$BRANCH · $ARM · dspark $SPEC_LABEL · $([ "$COLD_WANT" = on ] && printf cold || printf warm) · prime $PRIME_STATE"
{
	printf '# try-spec-offload result - %s - %s - %s\n\n' "$TODAY" "$BRANCH" "$ARM"
	printf '| field | value |\n| --- | --- |\n'
	printf '| date | %s |\n' "$TODAY"
	printf '| branch | %s |\n' "$BRANCH"
	printf '| branch sha | %s |\n' "$BR_SHA"
	printf '| build | ~/%s/_worktrees/%s @ %s |\n' "$REMOTE_ROOT" "$RDIR" "$(wt_field "$RDIR" 3)"
	printf '| arm | %s |\n' "$ARM"
	printf '| dspark | %s |\n' "$SPEC_LABEL"
	printf '| residency | %s |\n' "$( [ "$COLD_WANT" = on ] && printf cold || printf warm )"
	printf '| prime | %s |\n' "$PRIME_STATE"
	printf '| env | %s |\n' "$ENVSTR"
	printf '| argv | %s |\n' "$ARGV"
	printf '| engaged | %s of 4 |\n' "$ENGAGED"
	[ -z "${NOOFF:-}" ] || printf '| not switchable by env | %s |\n' "$(printf '%s' "$NOOFF" | tr ' ' ',')"
	printf '| server | %s |\n' "$SERVER_CMD"
	printf '| spark | %s port %s ctx %s |\n' "$SPARK" "$PORT" "$CTX"
	printf '\n## tokens per second\n\n'
	printf '| test | tokens | secs | tok/s |\n| --- | ---: | ---: | ---: |\n'
	awk -F'|' '$2=="ok"{printf "| %s | %s | %.2f | %s |\n", $1, $3, $4, $5}
	           $2!="ok"{printf "| %s | %s | %.2f | ERR |\n", $1, $3, $4}' "$RESFILE"
	printf '\n## summary\n\n'
	printf '| stat | tok/s |\n| --- | ---: |\n'
	if [ "$STAT" = "none none none 0" ]; then
		printf '| min | - |\n| median | - |\n| max | - |\n'
	else
		printf '| min | %s |\n| median | %s |\n| max | %s |\n' "$MIN" "$MED" "$MAX"
	fi
	printf '| tests counted | %s |\n' "$N"
	printf '\n*%s*\n' "$SUMMARY_LINE"
} >> "$ROOT/$OUTFILE"

# ---- the index ------------------------------------------------------------
IDXF="$ROOT/$OUT_DIR/$INDEX"
if [ ! -f "$IDXF" ]; then
	{
		printf '# try-spec-offload results\n\n'
		printf 'One line per leg.  Date, branch, what was ON, and the file.\n\n'
		printf '| date | branch | arm | dspark | residency | prime | file |\n| --- | --- | --- | --- | --- | --- | --- |\n'
	} >> "$IDXF"
fi
printf '| %s | %s | %s | %s | %s | %s | `%s` |\n' \
	"$TODAY" "$BRANCH" "$ARM" "$SPEC_LABEL" \
	"$( [ "$COLD_WANT" = on ] && printf cold || printf warm )" "$PRIME_STATE" "$OUTFILE" >> "$IDXF"

gold "  wrote  $ROOT/$OUTFILE"
gold "  indexed $IDXF"
printf '\n'

rm -f "$VALS" "$RESFILE"
stop_server
printf '\n'
gold "  done - model unloaded from $SPARK."
printf '\n'