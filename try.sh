#!/bin/sh
# ===========================================================================
#  try.sh - the DwarfStar arm runner.
#
#  Runs on the TESTING MACHINE (the M5) and drives a ds4-server on the Spark
#  over ssh.  It picks a branch, builds an "arm" out of environment switches,
#  loads the model on the Spark, runs TEN fixed tests (five short, five long -
#  the same ten every run), prints tokens/second for each, writes a
#  markdown-table result file, and unloads the model on every exit path.
#
#    ./try.sh                       # menu, defaults to triple-all-fastest
#    ./try.sh --branch triple-pool  # pick a branch from the fork
#    ./try.sh --arm stack           # every lever at the default this branch ships
#    ./try.sh --arm all-off         # the control: every off switch, switched off
#    ./try.sh --arm all-on          # every lever forced on, opt-ins included
#    ./try.sh --arm stack --off hits-first --on draincut
#    ./try.sh --list                # branches and what is built, no run
#    ./try.sh --dry-run             # print the plan, load nothing
#
#  POSIX sh.  No python, no jq.  Needs: ssh, curl, and the ordinary text
#  tools (sed, awk, sort, stat, date) that ship with macOS.
#
#  The Spark is a shared box: a 90 GB model left resident is rude, so the
#  model is unloaded when the program ends (trap), when nothing has run for
#  IDLE_SECS (local watchdog), and by a watchdog ON the Spark itself if this
#  machine dies or the shell is killed hard.
# ===========================================================================

set -u

# ---- knobs ----------------------------------------------------------------
SPARK="${SPARK:-spark}"
REMOTE_ROOT="${REMOTE_ROOT:-dwarfstar}"
MODEL="${MODEL:-gguf/DeepSeek-V4.1-Flash-Q2.gguf}"
MODEL_ID="${MODEL_ID:-ds4}"
PORT="${PORT:-8000}"
CTX="${CTX:-8321}"
SSD_CACHE="${SSD_CACHE:-90GB}"
IDLE_SECS="${IDLE_SECS:-30}"
CURL_MAX="${CURL_MAX:-600}"
DEFAULT_BRANCH="${DEFAULT_BRANCH:-triple-all-fastest}"
BUILD_DIR="${BUILD_DIR:-}"          # override the remote worktree, e.g. tb-afstack
FORK="${FORK:-sparkle}"
OUT_DIR="${OUT_DIR:-try-results}"
INDEX="${INDEX:-INDEX.md}"
MAX_SHORT="${MAX_SHORT:-128}"
MAX_LONG="${MAX_LONG:-256}"

SSH_OPTS="-o ConnectTimeout=15 -o ServerAliveInterval=10 -o ServerAliveCountMax=3"

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
		sed -e 's/\\/\\\\/g' -e 's/"/\\"/g' -e 's/	/\\t/g' |
		sed -e 's/$/\\n/' | tr -d '\n' | sed -e 's/\\n$//'
}
# ---- the banner -----------------------------------------------------------
banner() {
	printf '\n'
	printf '        %s.%s         %s*%s\n' "$EMBER" "$RST" "$GOLD" "$RST"
	printf '          %s\\    /%s\n' "$RUST" "$RST"
	printf '   %s*%s    %s.  \\  /  .%s    %s*%s\n' "$GOLD" "$RST" "$AMBER" "$RST" "$GOLD" "$RST"
	printf '             %s\\/%s\n' "$GOLD" "$RST"
	printf '     %sD W A R F S T A R%s   %s.try.sh.%s\n' "$GOLD" "$RST" "$RUST" "$RST"
	printf '             %s/\\%s      %sten fixed tests, tokens per second%s\n' "$GOLD" "$RST" "$DIM" "$RST"
	printf '   %s*%s    %s.  /  \\  .%s    %s*%s   %sthe Spark unloads itself%s\n' "$GOLD" "$RST" "$AMBER" "$RST" "$GOLD" "$RST" "$DIM" "$RST"
	printf '          %s/    \\%s\n' "$RUST" "$RST"
	printf '        %s*%s         %s.%s\n\n' "$GOLD" "$RST" "$EMBER" "$RST"
}

# ---- arguments ------------------------------------------------------------
BRANCH="$DEFAULT_BRANCH"
BRANCH_SET=0      # 1 once the user or the command line picked a branch
ARM="stack"
TOGGLES=""        # "lever:on lever:off ..." in the order given
DRY=0; QUIET=0; LIST=0

while [ $# -gt 0 ]; do
	case "$1" in
		--branch|--build-branch) BRANCH="${2:-}"; BRANCH_SET=1; shift 2 ;;
		--dir)     BUILD_DIR="${2:-}"; shift 2 ;;
		--arm)     ARM="${2:-}"; shift 2 ;;
		--on)      TOGGLES="$TOGGLES $2:on"; shift 2 ;;
		--off)     TOGGLES="$TOGGLES $2:off"; shift 2 ;;
		--quiet|-q) QUIET=1; shift ;;
		--list)    LIST=1; shift ;;
		--dry-run) DRY=1; shift ;;
		-h|--help) sed -n '2,22p' "$0"; exit 0 ;;
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
# and HEAD, so the menu can say what is already built.
remote_worktrees() {
	ssh $SSH_OPTS "$SPARK" "
		for d in \$HOME/$REMOTE_ROOT/_worktrees/*/; do
			[ -n \"\$d\" ] || continue
			n=\$(basename \"\$d\")
			b=\$(git -C \"\$d\" rev-parse --abbrev-ref HEAD 2>/dev/null || echo none)
			s=\$(git -C \"\$d\" rev-parse HEAD 2>/dev/null || echo none)
			x=no
			[ -x \"\$d/ds4-server\" ] && x=yes
			c=\$(git -C \"\$d\" log -1 --format=%ct 2>/dev/null || echo 0)
			printf '%s|%s|%s|%s|%s\n' \"\$n\" \"\$b\" \"\$s\" \"\$x\" \"\$c\"
		done" </dev/null 2>/dev/null
}

wt_field() { # $1=dir $2=field number
	printf '%s\n' "${WT_ROWS:-}" | awk -F'|' -v d="$1" -v n="$2" '$1==d{print $n; exit}'
}

# Which worktree on the Spark serves this branch?  In order:
#   1. try-arms.txt, a data file of  <branch-slug>  <build-dir>  lines
#   2. a worktree directory named exactly after the branch
#   3. a worktree checked out on the branch itself
#   4. a best guess: among the tb-* arm builds, the one whose HEAD is the
#      fewest commits behind the branch tip (newest build breaks a tie).
# Rule 4 is a GUESS and the runner says so; rules 1-3 are facts.  A future
# branch needs nothing in this script: name its worktree after the branch, or
# add one line to try-arms.txt.
arm_for_slug() { # $1=slug
	awk -v s="$1" '$1==s && $2!="" {print $2; exit}' "$ROOT/try-arms.txt" 2>/dev/null
}

resolve_dir() { # $1=slug $2=branch tip sha
	# a) a worktree named exactly after the branch
	for d in ${WT_ALL:-}; do
		[ "$d" = "$1" ] && [ "$(wt_field "$d" 4)" = yes ] && { printf '%s' "$d"; return 0; }
	done
	# b) a worktree checked out on the branch itself
	for d in ${WT_ALL:-}; do
		[ "$(wt_field "$d" 4)" = yes ] || continue
		[ "$(wt_field "$d" 2)" = "$1" ] && { printf '%s' "$d"; return 0; }
	done
	# c) closest tb-* arm build to the branch tip; newest build breaks a tie
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

pick_dir_auto() { # $1=slug $2=tip sha: map first, then the automatic rules
	d=$(arm_for_slug "$1")
	[ -n "$d" ] && [ "$(wt_field "$d" 4)" = yes ] && { printf '%s' "$d"; return 0; }
	resolve_dir "$1" "$2"
}
# ---- the levers and their switches ----------------------------------------
# Verified against the source in this tree (grep DS4_CUDA / DS4_V41 / DS4_ENGRAM):
#
#   lever                switch                                  default        off
#   -------------------  --------------------------------------  -------------  -----------------------
#   pool                 DS4_CUDA_STREAMING_EXPERT_PREAD_POOL    on             =0
#   hotlist              DS4_CUDA_EXPERT_HOTLIST_WRITE           on             =0
#   hits-first           DS4_CUDA_HITS_FIRST                     OFF            (default)
#   pagecache            DS4_CUDA_KEEP_MODEL_PAGES               new order      =1 restores old
#   engram-lead          DS4_V41_ENGRAM_LEAD_OFF                 on             anything
#   draincut             DS4_CUDA_SELECTED_DRAIN_SYNC            ON             (set 1 restores blocking)
#   margin               DS4_CUDA_EXPERT_CACHE_MARGIN_GB         8 GiB          a knob, no Boolean
#   prefetch-pool        DS4_CUDA_SSD_PREFETCH_CHUNK_MB          8 MB           0 is IGNORED
#   readahead-order      DS4_CUDA_SSD_PREFETCH_STATS             on             stats only
#   engram-read-threads  DS4_ENGRAM_READ_THREADS                 request count  (default)
#
# Six of the ten have a real off switch, which is what lets ONE binary serve
# several arms.  margin, prefetch-pool, readahead-order and engram-read-threads
# have only a knob or no switch at all: nothing can be turned off for them by
# environment, so an "everything off" arm is NOT clean on those four and the
# runner says so out loud.  prefetch-pool's code is not even in this tree.
#
# Fields: lever|switch|env-when-on|env-when-off|has-an-off-switch|note
LEVERS='pool|DS4_CUDA_STREAMING_EXPERT_PREAD_POOL|-|0|1|private read pool for the expert pread
hotlist|DS4_CUDA_EXPERT_HOTLIST_WRITE|-|0|1|seed the expert cache from a learned hot list
hits-first|DS4_CUDA_HITS_FIRST|1|-|1|decode window; measured null, kept for the A/B
pagecache|DS4_CUDA_KEEP_MODEL_PAGES|-|1|1|staged page drop order + read-ahead hint
engram-lead|DS4_V41_ENGRAM_LEAD_OFF|-|1|1|a token of lead for the Engram read
draincut|DS4_CUDA_SELECTED_DRAIN_SYNC|1|-|0|bound the selected-expert decode drain (default on; 1 restores blocking)
margin|DS4_CUDA_EXPERT_CACHE_MARGIN_GB|-|-|0|page-cache reserve after cache slots (knob)
prefetch-pool|DS4_CUDA_SSD_PREFETCH_CHUNK_MB|-|-|0|parallel SSD read-ahead chunk (not in this tree)
readahead-order|DS4_CUDA_SSD_PREFETCH_STATS|-|-|0|prefill read-ahead victim reorder; no off
engram-read-threads|DS4_ENGRAM_READ_THREADS|-|-|0|Engram batch reader width; no off'

lever_names() { printf '%s\n' "$LEVERS" | cut -d'|' -f1 | tr '\n' ' '; }
lever_rec() { printf '%s\n' "$LEVERS" | awk -F'|' -v L="$1" '$1==L{print; exit}'; }
lever_field() { lever_rec "$1" | cut -d'|' -f"$2"; }

tog_state() { # $1=lever -> "on"/"off" if it was toggled on the command line
	for t in ${TOGGLES:-}; do
		case "$t" in
			"$1:on")  printf 'on'; return 0 ;;
			"$1:off") printf 'off'; return 0 ;;
		esac
	done
	return 1
}

# An arm is a starting state for the ten levers; --on/--off then move one
# lever at a time, so a single binary serves several arms.
#   stack    the branch AS IT SHIPS - every lever at its default, no env
#            (here that is 8 of 10: hits-first and draincut are opt-in)
#   all-on   every lever forced on, the opt-ins included
#   all-off  the control: everything that HAS an off switch, switched off
build_arm() {
	case "$ARM" in
		stack)   DEFAULT_WANT=default ;;
		all-on)  DEFAULT_WANT=on ;;
		all-off) DEFAULT_WANT=off ;;
		*)       warn "unknown arm '$ARM' (use stack, all-on or all-off)"; exit 2 ;;
	esac

	# every --on/--off lever must exist
	for t in ${TOGGLES:-}; do
		L=$(printf '%s' "$t" | cut -d: -f1)
		[ -n "$(lever_rec "$L")" ] || { warn "no such lever: $L (have: $(lever_names))"; exit 2; }
	done

	ENVSTR=''; ENGAGED=0; STATE=''; NOOFF=''
	for L in $(lever_names); do
		sw=$(lever_field "$L" 2); von=$(lever_field "$L" 3)
		voff=$(lever_field "$L" 4); hasoff=$(lever_field "$L" 5)
		want=$DEFAULT_WANT
		if [ "$want" = default ]; then
			case "$von" in -) want=on ;; *) want=off ;; esac
		fi
		o=$(tog_state "$L") && want=$o
		if [ "$want" = on ]; then
			ENGAGED=$((ENGAGED + 1))
			[ "$von" = "-" ] || ENVSTR="$ENVSTR $sw=$von"
		else
			if [ "$hasoff" = 1 ]; then
				[ "$voff" = "-" ] || ENVSTR="$ENVSTR $sw=$voff"
			else
				NOOFF="$NOOFF $L"
			fi
		fi
		STATE="$STATE $L=$want"
	done
	ENVSTR=$(printf '%s' "$ENVSTR" | sed -e 's/^ *//')
}

print_levers() {
	printf '%s\n' "  lever                switch                                  state" | sed "s/^/$AMBER/;s/$/$RST/"
	for L in $(lever_names); do
		sw=$(lever_field "$L" 2)
		st=off
		case " $STATE " in *" $L=on "*) st=on ;; esac
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

# Every branch starting with triple-, so a branch pushed tomorrow shows up here
# with no change to this script.  Keyed on the branch slug, end to end.
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
			printf '  %s%2d) %-28s%s %s<- default%s%s\n' "$GOLD" "$i" "$slug" "$RST" \
				"$GOLD" "$mark" "$RST"
		else
			printf '  %s%2d)%s %-28s %s%s%s%s\n' "$AMBER" "$i" "$RST" "$slug" \
				"$EMBER" "$(short_sha "$sha")" "$RST" "$mark"
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

# Which worktree on the Spark runs this branch?  --dir wins; then try-arms.txt
# and the automatic rules; if only a guess is possible and we are interactive,
# ask, with the guess as the default.
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
	if [ -z "$RDIR" ] || [ "$RMODE" = auto ]; then
		# candidates: tb-* arm builds with a binary, newest first
		CANDS=$(printf '%s\n' "${WT_ROWS:-}" | awk -F'|' '$4=="yes" && $1 ~ /^tb-/{print $5" "$1}' |
			sort -rn | cut -d' ' -f2 | tr '\n' ' ')
		if [ -n "$CANDS" ]; then
			guess=$(printf '%s' "$CANDS" | cut -d' ' -f1)
			if [ -t 0 ] || { [ -r /dev/tty ] && [ "$BRANCH_SET" = 0 ]; }; then
				[ -z "$RDIR" ] && RDIR="$guess"
				printf '  %swhich build dir on %s runs %s?%s\n' "$GOLD" "$SPARK" "$BRANCH" "$RST"
				j=0; gd=0
				for c in $CANDS; do
					j=$((j + 1))
					[ "$c" = "$RDIR" ] && gd=$j
					printf '  %s%2d) %s%s%s\n' "$AMBER" "$j" "$c" "$RST" ""
				done
				printf '  %sdir [%s]:%s ' "$GOLD" "$RDIR" "$RST"
				read -r dpick </dev/tty || dpick=""
				case "$dpick" in
					'') : ;;
					*[!0-9]*) RDIR="$dpick" ;;
					*) j=0
					   for c in $CANDS; do
						   j=$((j + 1))
						   [ "$j" = "$dpick" ] && RDIR="$c"
					   done ;;
				esac
				RMODE=picked
			else
				RDIR="$guess"; RMODE=guess
			fi
		fi
	fi
fi
[ -n "$RDIR" ] || die "no built ds4-server on $SPARK for $BRANCH (build one: ssh $SPARK 'cd ~/$REMOTE_ROOT/_worktrees/<arm> && make cuda-spark -j12')"
[ "$RMODE" = guess ] && warn "build dir '$RDIR' is a GUESS - pass --dir <name> to pin it"
[ "$RMODE" = auto ] && warn "build dir '$RDIR' chosen by proximity - pass --dir <name> to pin it"

TODAY=$(date +%Y-%m-%d)
DESC="$ARM-${ENGAGED}on"
for t in ${TOGGLES:-}; do DESC="$DESC-$(printf '%s' "$t" | tr ':' '-')"; done
OUTFILE="$OUT_DIR/$TODAY-$BRANCH-$DESC.txt"

rust "  branch   $GOLD$BRANCH$RST $EMBER$(short_sha "$BR_SHA")$RST"
rust "  build    $AMBER~/$REMOTE_ROOT/_worktrees/$RDIR$RST $(ember "@ $(wt_field "$RDIR" 3 | cut -c1-9)")"
rust "  arm      $AMBER$ARM$RST  ($ENGAGED of $(lever_names | wc -w | tr -d ' ') levers engaged)"
if [ -n "$ENVSTR" ]; then
	rust "  env      $TEAL$ENVSTR$RST"
else
	rust "  env      $(ember "(none - every lever at the default this branch ships)")"
fi
printf '\n'
print_levers
printf '\n'

# A lever with no off switch cannot be switched off, so an "everything off"
# arm is not a clean control on those levers.  Say it, do not hide it.
if [ -n "${NOOFF:-}" ]; then
	warn "this arm wants OFF but cannot get it: $(printf '%s' "$NOOFF" | sed -e 's/^ *//' | tr ' ' ',')"
	warn "those levers have no off switch (a knob, stats only, or no env at all),"
	warn "so this run is NOT a clean control for them - that needs a build without the code."
	printf '\n'
fi

SERVER_CMD="./ds4-server -m \$HOME/$REMOTE_ROOT/$MODEL --port $PORT --ctx $CTX --ssd-streaming --ssd-streaming-cache-experts $SSD_CACHE"
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
LOADED=""; TICK=""; HEART=""; WATCH=""
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
on_exit() { stop_server; rm -f "$STAMP" "$FLIGHT"; }
trap 'on_exit' EXIT INT TERM HUP

# A test can take longer than IDLE_SECS, so activity is asserted with a ticker
# while something is in flight; between tests the clock runs, which is what
# "idle" is supposed to mean.
tick_start() { ( while :; do touch "$1"; sleep 5; done ) & TICK=$!; }
tick_stop() { [ -n "${TICK:-}" ] && kill "$TICK" 2>/dev/null; TICK=""; }

# The heartbeat ssh keeps the beat file on the Spark fresh for as long as this
# process lives.  If this machine dies or the shell is killed hard, that ssh
# dies too and the watchdog ON the Spark unloads the model by itself.
heartbeat() { while :; do ssh $SSH_OPTS "$SPARK" "touch $REMOTE_BEAT" </dev/null >/dev/null 2>&1; sleep 5; done; }

spark_watchdog() {
	ssh $SSH_OPTS "$SPARK" "
		while :; do
			sleep 5
			now=\$(date +%s)
			seen=\$(stat -c %Y $REMOTE_BEAT 2>/dev/null || stat -f %m $REMOTE_BEAT 2>/dev/null || echo 0)
			if [ \$((now - seen)) -ge $IDLE_SECS ]; then
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
# ---- load -----------------------------------------------------------------
say "loading $BRANCH ($RDIR) on $SPARK ..."
tick_start "$FLIGHT"
heartbeat & HEART=$!
spark_watchdog
touch "$STAMP"

ssh $SSH_OPTS "$SPARK" "
	cd \$HOME/$REMOTE_ROOT/_worktrees/$RDIR || { echo 'NODIR'; exit 1; }
	if pgrep -f '[d]s4-server.*--port $PORT' >/dev/null 2>&1; then echo 'BUSY'; exit 3; fi
	touch $REMOTE_BEAT
	nohup env $ENVSTR $SERVER_CMD > /tmp/try-$RDIR.log 2>&1 &
	echo \$! > \$HOME/.ds4-try.pid
	echo 'STARTED'
" </dev/null 2>&1 | tee /tmp/try-load.out >/dev/null

case "$(cat /tmp/try-load.out 2>/dev/null)" in
	*BUSY*)   die "a ds4-server is already on port $PORT on $SPARK - refusing to stack another";;
	*NODIR*)  die "no worktree ~/$REMOTE_ROOT/_worktrees/$RDIR on $SPARK";;
	*STARTED*) : ;;
	*) die "could not start the server on $SPARK (see the output above)";;
esac

i=0
while [ $i -lt 180 ]; do
	if curl -s -m 3 "http://$SPARK:$PORT/v1/models" >/dev/null 2>&1; then break; fi
	touch "$FLIGHT"
	i=$((i + 1)); sleep 5
done
if [ $i -ge 180 ]; then
	warn "server did not answer in 900s - last log lines:"
	ssh $SSH_OPTS "$SPARK" "tail -20 /tmp/try-$RDIR.log" </dev/null 2>&1 >&2
	stop_server
	exit 1
fi
LOADED=1
touch "$FLIGHT"
tick_stop
idle_watch & WATCH=$!
say "loaded after ~$((i * 5))s.  api: http://$SPARK:$PORT/v1/models"
printf '\n'

# ---- the ten tests --------------------------------------------------------
# The same ten prompts on every run, so two arms are always compared on one
# set.  Five short inputs, five long ones built on one shared passage.
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

run_one() { # $1=test name $2=prompt $3=max_tokens
	tn=$1; tp=$2; mx=$3
	payload=$(printf '{"model":"%s","prompt":"%s","max_tokens":%s,"temperature":0,"stream":false}' \
		"$MODEL_ID" "$(json_str "$tp")" "$mx")
	tick_start "$FLIGHT"
	raw=$(curl -s -m "$CURL_MAX" -X POST "http://$SPARK:$PORT/v1/completions" \
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
run_pair() { # $1=name $2=prompt $3=max
	line=$(run_one "$1" "$2" "$3")
	IFS='|' read -r n s tok sec tps <<EOF
$line
EOF
	if [ "$s" = ok ]; then
		printf '  %s%-12s%s %s%6s%s %8s  %s%8s%s %s\n' "$AMBER" "$n" "$RST" \
			"$GOLD" "$tok" "$RST" "$sec" "$TEAL" "$tps" "$RST" "$(ember 'tok/s')"
	else
		printf '  %s%-12s%s %6s %8s  %s%8s%s  %s\n' "$AMBER" "$n" "$RST" "$tok" "$sec" \
			"$RUST" "ERR" "$RST" "$(ember 'no completion_tokens - see /tmp/try-'"$RDIR"'.log')"
	fi
	printf '%s|%s|%s|%s|%s\n' "$n" "$s" "$tok" "$sec" "$tps" >>"$RESFILE"
}

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

ONLIST=$(printf '%s' "$STATE" | tr ' ' '\n' | sed -n 's/=on$//p' | tr '\n' ' ' | sed 's/ *$//')
OFFLIST=$(printf '%s' "$STATE" | tr ' ' '\n' | sed -n 's/=off$//p' | tr '\n' ' ' | sed 's/ *$//')
[ -n "$ENVSTR" ] || ENVSTR="(none - every lever at its default)"

if [ "$STAT" = "none none none 0" ]; then
	warn "no measurements at all - check /tmp/try-$RDIR.log on $SPARK"
	gold "  summary : nothing measured"
else
	printf '  %ssummary%s  min %s%s%s  median %s%s%s  max %s%s%s  %s(%s tests)%s\n' \
		"$GOLD" "$RST" "$GOLD" "$MIN" "$RST" "$TEAL" "$MED" "$RST" "$GOLD" "$MAX" "$RST" \
		"$DIM" "$N" "$RST"
fi
printf '\n'

# ---- the result file ------------------------------------------------------
# Plain text, markdown tables, so a row can be pasted straight into a README.
SUMMARY_LINE="$BRANCH · $ARM · ${ENGAGED} on · hits-first $(tog_state hits-first 2>/dev/null || printf 'default')"
{
	printf '# try.sh result - %s - %s - %s\n\n' "$TODAY" "$BRANCH" "$ARM"
	printf '| field | value |\n| --- | --- |\n'
	printf '| date | %s |\n' "$TODAY"
	printf '| branch | %s |\n' "$BRANCH"
	printf '| branch sha | %s |\n' "$BR_SHA"
	printf '| build | ~/%s/_worktrees/%s @ %s |\n' "$REMOTE_ROOT" "$RDIR" "$(wt_field "$RDIR" 3)"
	printf '| arm | %s |\n' "$ARM"
	printf '| env | %s |\n' "$ENVSTR"
	printf '| levers on | %s |\n' "$ONLIST"
	printf '| levers off | %s |\n' "$OFFLIST"
	printf '| engaged | %s of 10 |\n' "$ENGAGED"
	[ -z "${NOOFF:-}" ] || printf '| not switchable | %s |\n' "$(printf '%s' "$NOOFF" | tr ' ' ',')"
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
		printf '# try.sh results\n\n'
		printf 'One line per run.  Date, branch, what was ON, and the file.\n\n'
		printf '| date | branch | on | file |\n| --- | --- | --- | --- |\n'
	} >> "$IDXF"
fi
printf '| %s | %s | %s · %s on · on: %s · off: %s | `%s` |\n' \
	"$TODAY" "$BRANCH" "$ARM" "$ENGAGED" "$ONLIST" "$OFFLIST" "$OUTFILE" >> "$IDXF"

gold "  wrote  $ROOT/$OUTFILE"
gold "  indexed $IDXF"
printf '\n'

rm -f "$VALS" "$RESFILE"
stop_server
printf '\n'
gold "  done - model unloaded from $SPARK."
printf '\n'
