#!/bin/bash
# kp_matrix.sh - run ds4-bench over a SET of DS4_KP_* flag combinations against the tip,
# sequentially, tip-bracketed, on the standing GB10 sweep. One CSV per run in ds4-bench's
# own --csv format, load and driver stamped, a SIGCONT-only net keyed to this script's pid,
# and the ~/spark_model.lock mkdir protocol: no lock, no run.
#
# <claudes_code_comments>
# ** Function List **
# usage() - the --help tutorial with runnable examples and the NEXT line
# die(msg) - print to stderr and exit 2
# la() ma_gb() gu() driver() - the load, memory, GPU-util and driver stamps
# text_bytes(bin) - .text size of a binary via `size -A`
# gguf_ident(path) - arch string + name read from the GGUF header, never the weights
# arm_env(arm) - the environment an arm runs under (tip/off: none; NAME[+NAME]: DS4_KP_NAME=1; ALL: DS4_KP_ALL=1)
# arm_dir(arm) - the worktree whose binary an arm runs
# build_plan() - warm-up tip, then per cycle: every arm, then a tip bracket (cycle-edge)
# take_lock() release_lock() - mkdir ~/spark_model.lock with an owner line; refuse if held
# start_net() stop_net() - the CONT net: SIGCONT only, this pid, its children, ds4-bench
# run_one(label, dir, envs...) - one ds4-bench sweep with START/END lines in the runlog
# selftest() - plan, arm parsing, lock refusal and a stub bench, no model, no GPU
#
# ** Technical Review **
# - Arms are names: `tip` = the tip worktree's binary with no flag (the control), `off` = the
#   port's binary with no flag (the byte-identity control), `ALL` = DS4_KP_ALL=1, and any
#   `NAME` or `NAME+NAME` = DS4_KP_NAME=1 for each, all on the port's binary. Flag names are
#   the DS4_KP_ suffixes listed in README.md.
# - The plan is KP1's shape: one discarded tip warm-up, then for each cycle every arm in
#   the order given followed by one tip bracket, so every arm run sits between two tip runs
#   (the warm-up or the previous cycle's bracket before it, the cycle's bracket after it).
#   The verdict statistic is the seal's business; this script only produces the CSVs and
#   the runlog the seal's analysis reads.
# - The sweep is the standing one: ctx-start 2048 (warm-up frontier, discarded by the
#   analysis), ctx-max 6144, step 2048, gen 128, --cuda --ssd-streaming
#   --ssd-streaming-cache-experts $CACHE, promessi_sposi.txt.
# - Stamps: driver (nvidia-smi), power=n/a(linux), the two binaries' .text, the model's
#   arch string, name, size and mtime (sha256 of a 365 GB file is not taken here; pass
#   --sha to record one you already hold), and per run load/gpu/memavail before and after.
# - The lock is a directory: mkdir succeeds exactly once. A held lock prints its owner
#   line and exits 2. Ours is removed on exit only if the owner line is still ours.
# - The CONT net matches $$, every descendant of $$, and ds4-benc[h]; SIGCONT only, so a
#   false match is harmless. Killed by pid at exit.
# - Refuses to start unless every named worktree holds a built ds4-bench, and unless the
#   model's arch string is deepseek41 (the port is arch-gated; a V4 file measures nothing).
# </claudes_code_comments>
#
# Docs: flash41-kernelpool-spark/README.md (the flag table) and
#       flash41-kernelpool-spark/2026-09-20-KP1-PREREGISTERED-RULE.txt (the sweep shape).

set -u
exec < /dev/null

NAME="kpm$(date -u +%m%d%H%M)"
ARMS=""
CYCLES=3
TIP_DIR="$HOME/dwarfstar/_worktrees/kp-main"
ARM_DIR="$HOME/dwarfstar/_worktrees/flash41-kernelpool-spark"
MODEL="$HOME/dwarfstar/gguf/DeepSeek-V4.1-Flash-Q2.gguf"
CACHE="90GB"
FLUSH=""
SHA=""
SWEEPS="$HOME/sweeps"
LOCK="$HOME/spark_model.lock"
DRY=0
BENCH_OVERRIDE=""
SLEEP=5

usage() {
cat <<EOF
kp_matrix.sh - bench a set of DS4_KP_* flag arms against the tip, tip-bracketed, under the lock.

USAGE
  kp_matrix.sh --arms <arm,arm,...> [--cycles N] [options]
  kp_matrix.sh --dry-run --arms ...      print the plan and each run's environment, run nothing
  kp_matrix.sh --selftest                gate: plan, parsing, lock refusal, stub bench (no GPU)

ARMS  (comma separated; order is the order within a cycle)
  tip                the tip worktree's ds4-bench, no flag         (the control)
  off                the port's ds4-bench, no flag                 (byte-identity control)
  QUEUE_LAYERS       DS4_KP_QUEUE_LAYERS=1 on the port's binary    (one per flag; see README)
  QUEUE_LAYERS+HEAD_EARLY   several flags on one arm, joined with +
  ALL                DS4_KP_ALL=1
  Flag names: QUEUE_LAYERS SKIP_SELECT ROUND_IN_KERNEL SELECT_BATCH SHORT_ROWS HEAD_EARLY

OPTIONS
  --cycles N         cycles (default 3); each cycle = every arm, then one tip bracket
  --name S           round name; labels are <name>_<arm>_<cycle> (default kpm<mmddHHMM>)
  --tip-dir D        tip worktree (default $TIP_DIR)
  --arm-dir D        port worktree (default $ARM_DIR)
  --model F          gguf (default $MODEL)
  --cache S          --ssd-streaming-cache-experts value (default $CACHE)
  --flush N          sets DS4_CUDA_V41_DECODE_FLUSH_LAYERS=N on every flagged arm (default: unset, upstream's 2)
  --sha HEX          a sha256 prefix of the model you already hold, recorded in the runlog
  --sweeps D         output dir (default $SWEEPS): <label>.csv, <label>.log, RUNLOG_<name>.txt, plan_<name>.txt
  --sleep S          seconds between runs (default $SLEEP)

EXAMPLES
  kp_matrix.sh --arms tip,QUEUE_LAYERS,ROUND_IN_KERNEL,ALL --cycles 3
  kp_matrix.sh --arms off,ALL --cycles 3 --name kpid          # is the port byte-identical off? (G1 is separate)
  kp_matrix.sh --arms QUEUE_LAYERS --flush 0 --cycles 3 --name kpq0
  kp_matrix.sh --dry-run --arms tip,SKIP_SELECT+HEAD_EARLY --cycles 2

OUTPUT
  $SWEEPS/RUNLOG_<name>.txt  header (driver, .text of both binaries, model ident) + START/END per run
  $SWEEPS/<label>.csv         ds4-bench --csv (ctx_tokens,...,gen_steady_tps,kvcache_bytes), one per run
  $SWEEPS/plan_<name>.txt     the plan as fired; a warm-up tip run is labelled *_tip_warm

WHAT IT REFUSES
  a held ~/spark_model.lock (prints the owner line, exit 2) - never removes another owner's lock
  a worktree without a built ds4-bench; a model whose header arch is not deepseek41

NEXT -> when it prints DONE: scp 'spark:$SWEEPS/RUNLOG_<name>.txt' 'spark:$SWEEPS/<name>_*.csv' and
        run the seal's analysis (paired median vs the bracketing tip mean, min-across-frontiers).
EOF
}

die() { echo "kp_matrix: $*" >&2; exit 2; }
la() { cut -d" " -f1 /proc/loadavg 2>/dev/null || echo n/a; }
ma_gb() { local kb; kb=$(grep MemAvailable /proc/meminfo 2>/dev/null | tr -dc 0-9); [ -n "$kb" ] && echo $(( kb / 1000 / 1000 )) || echo n/a; }
gu() { nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 || echo n/a; }
driver() { nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || echo n/a; }
text_bytes() { size -A "$1" 2>/dev/null | awk '$1==".text"{print $2}'; }

gguf_ident() {
    python3 - "$1" <<'PY' 2>/dev/null || echo "arch=unreadable name=unreadable"
import struct, sys
f = open(sys.argv[1], 'rb')
def rd(fmt): return struct.unpack('<' + fmt, f.read(struct.calcsize(fmt)))
if f.read(4) != b'GGUF': print('arch=notgguf name=notgguf'); sys.exit(0)
rd('I'); rd('Q'); nkv, = rd('Q')
def rstr():
    n, = rd('Q'); return f.read(n).decode('utf-8', 'replace')
sz = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
def skip(t):
    if t == 8: rstr()
    elif t == 9:
        et, = rd('I'); n, = rd('Q')
        for _ in range(n): skip(et)
    else: f.read(sz[t])
arch = name = '?'
for _ in range(nkv):
    k = rstr(); t, = rd('I')
    if k == 'general.architecture' and t == 8: arch = rstr()
    elif k == 'general.name' and t == 8: name = rstr()
    else: skip(t)
    if arch != '?' and name != '?': break
print('arch=%s name="%s"' % (arch, name))
PY
}

arm_env() {
    case "$1" in
        tip|off) echo "";;
        ALL) echo "DS4_KP_ALL=1";;
        *) local out="" f; IFS='+' read -ra parts <<<"$1"
           for f in "${parts[@]}"; do
               case "$f" in QUEUE_LAYERS|SKIP_SELECT|ROUND_IN_KERNEL|SELECT_BATCH|SHORT_ROWS|HEAD_EARLY) ;;
                   *) die "unknown flag '$f' in arm '$1' (see --help ARMS)";; esac
               out="$out DS4_KP_$f=1"
           done
           echo "${out# }";;
    esac
}
arm_dir() { case "$1" in tip) echo "$TIP_DIR";; *) echo "$ARM_DIR";; esac; }

build_plan() {
    local c a
    echo "${NAME}_tip_warm tip"
    for ((c = 1; c <= CYCLES; c++)); do
        for a in "${ARM_LIST[@]}"; do [ "$a" = tip ] && continue; echo "${NAME}_${a}_${c} $a"; done
        echo "${NAME}_tip_${c} tip"
    done
}

OWNER_LINE=""
take_lock() {
    if mkdir "$LOCK" 2>/dev/null; then
        OWNER_LINE="FLAGPORT kp_matrix $NAME pid=$$ $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "$OWNER_LINE" > "$LOCK/owner"
    else
        echo "kp_matrix: $LOCK is HELD: $(cat "$LOCK/owner" 2>/dev/null || echo '(no owner line)')" >&2
        exit 2
    fi
}
release_lock() {
    [ -n "$OWNER_LINE" ] || return 0
    [ "$(cat "$LOCK/owner" 2>/dev/null)" = "$OWNER_LINE" ] && rm -rf "$LOCK"
}

NET_PID=""
start_net() {
    local me=$$
    (
        exec < /dev/null
        while :; do
            for p in $me $(pgrep -P "$me" 2>/dev/null) $(pgrep -f "ds4-benc[h]" 2>/dev/null); do
                st=$(awk '{print $3}' "/proc/$p/stat" 2>/dev/null)
                case "$st" in T*) echo "$(date -u +%H:%M:%SZ) CONT pid=$p" >> "$SWEEPS/${NAME}_cont.log"; kill -CONT "$p" 2>/dev/null;; esac
            done
            sleep 3
        done
    ) &
    NET_PID=$!
}
stop_net() { [ -n "$NET_PID" ] && kill "$NET_PID" 2>/dev/null; }

run_one() {
    local label="$1" dir="$2"; shift 2
    local envs=("$@") runlog="$SWEEPS/RUNLOG_$NAME.txt"
    local bench=("${BENCH_OVERRIDE:-./ds4-bench}" -m "$MODEL" --cuda --ssd-streaming --ssd-streaming-cache-experts "$CACHE"
        --prompt-file speed-bench/promessi_sposi.txt --ctx-start 2048 --ctx-max 6144 --step-incr 2048
        --gen-tokens 128 --csv "$SWEEPS/$label.csv")
    echo "START $label dir=$dir load=$(la) gpu=$(gu)% memavail=$(ma_gb)GB switches=${envs[*]:-none} $(date -u +%H:%M:%SZ)" >> "$runlog"
    if [ "$DRY" = 1 ]; then
        echo "  (dry) cd $dir && env ${envs[*]:-} ${bench[*]}" >> "$runlog"
        echo "END   $label rc=dry $(date -u +%H:%M:%SZ)" >> "$runlog"; return 0
    fi
    (cd "$dir" && env ${envs[@]+"${envs[@]}"} "${bench[@]}" > "$SWEEPS/$label.log" 2>&1 < /dev/null)
    local rc=$?
    echo "END   $label rc=$rc load=$(la) gpu=$(gu)% memavail=$(ma_gb)GB $(date -u +%H:%M:%SZ)" >> "$runlog"
    return $rc
}

selftest() {
    local t; t=$(mktemp -d) || exit 1
    local pass=0 fail=0
    ok() { pass=$((pass+1)); echo "  ok   $1"; }
    bad() { fail=$((fail+1)); echo "  FAIL $1"; }
    # arm parsing
    [ "$(arm_env tip)" = "" ] && ok "tip has no env" || bad "tip env"
    [ "$(arm_env off)" = "" ] && ok "off has no env" || bad "off env"
    [ "$(arm_env ALL)" = "DS4_KP_ALL=1" ] && ok "ALL -> DS4_KP_ALL=1" || bad "ALL env"
    [ "$(arm_env QUEUE_LAYERS)" = "DS4_KP_QUEUE_LAYERS=1" ] && ok "one flag" || bad "one flag"
    [ "$(arm_env SKIP_SELECT+HEAD_EARLY)" = "DS4_KP_SKIP_SELECT=1 DS4_KP_HEAD_EARLY=1" ] && ok "two flags joined with +" || bad "two flags"
    ( arm_env BOGUS >/dev/null 2>&1 ) && bad "unknown flag accepted" || ok "unknown flag refused"
    [ "$(arm_dir tip)" = "$TIP_DIR" ] && [ "$(arm_dir ALL)" = "$ARM_DIR" ] && ok "arm -> worktree" || bad "arm dir"
    # plan shape: warm-up + cycles x (arms-without-tip + 1 bracket)
    NAME=st; CYCLES=2; ARM_LIST=(tip A B)
    local n; n=$(build_plan | wc -l | tr -d ' ')
    [ "$n" = 7 ] && ok "plan = 1 warm-up + 2 x (2 arms + 1 bracket) = 7 runs" || bad "plan rows $n"
    [ "$(build_plan | head -1)" = "st_tip_warm tip" ] && ok "first run is the discarded tip warm-up" || bad "warm-up first"
    [ "$(build_plan | tail -1)" = "st_tip_2 tip" ] && ok "last run is a tip bracket" || bad "bracket last"
    build_plan | grep -q "^st_A_1 A$" && ok "arm label <name>_<arm>_<cycle>" || bad "arm label"
    # lock protocol in a private HOME
    LOCK="$t/lock"; SWEEPS="$t/sweeps"; mkdir -p "$SWEEPS"
    mkdir "$LOCK"; echo "SOMEONE ELSE pid=1" > "$LOCK/owner"
    ( take_lock ) >/dev/null 2>"$t/err"; [ $? = 2 ] && grep -q "SOMEONE ELSE" "$t/err" && ok "held lock: owner printed, exit 2" || bad "held lock"
    rm -rf "$LOCK"
    take_lock; [ -f "$LOCK/owner" ] && grep -q "FLAGPORT kp_matrix" "$LOCK/owner" && ok "free lock taken with our owner line" || bad "take lock"
    echo "SOMEONE ELSE pid=1" > "$LOCK/owner"; release_lock; [ -d "$LOCK" ] && ok "a lock re-owned by another is not removed" || bad "foreign lock removed"
    rm -rf "$LOCK"; take_lock; release_lock; [ ! -d "$LOCK" ] && ok "our lock released" || bad "release"
    # gguf ident on a synthetic header
    python3 - "$t/x.gguf" <<'PY'
import struct, sys
f = open(sys.argv[1], 'wb')
def s(x): f.write(struct.pack('<Q', len(x)) + x.encode())
f.write(b'GGUF' + struct.pack('<IQQ', 3, 0, 3))
s('general.alignment'); f.write(struct.pack('<I', 4) + struct.pack('<I', 32))
s('general.architecture'); f.write(struct.pack('<I', 8)); s('deepseek41')
s('general.name'); f.write(struct.pack('<I', 8)); s('Test V4.1')
PY
    [ "$(gguf_ident "$t/x.gguf")" = 'arch=deepseek41 name="Test V4.1"' ] && ok "gguf header ident" || bad "gguf ident: $(gguf_ident "$t/x.gguf")"
    # a stub bench through run_one writes the CSV and the runlog lines
    cat > "$t/bench" <<'EOF'
#!/bin/bash
out=""; while [ $# -gt 0 ]; do [ "$1" = --csv ] && out="$2"; shift; done
printf 'ctx_tokens,prefill_tokens,prefill_tps,gen_tokens,gen_tps,gen_first_ms,gen_steady_tokens,gen_steady_tps,kvcache_bytes\n2048,2048,1,128,1,1,127,1,1\n' > "$out"
echo "stub ${DS4_KP_ALL:-unset}"
EOF
    chmod +x "$t/bench"; BENCH_OVERRIDE="$t/bench"; NAME=st; MODEL="$t/x.gguf"
    run_one st_ALL_1 "$t" DS4_KP_ALL=1
    [ -f "$SWEEPS/st_ALL_1.csv" ] && grep -q "^2048," "$SWEEPS/st_ALL_1.csv" && ok "stub run wrote the CSV" || bad "csv"
    grep -q "stub 1" "$SWEEPS/st_ALL_1.log" && ok "arm env reached the bench" || bad "env"
    grep -q "^START st_ALL_1 .*switches=DS4_KP_ALL=1" "$SWEEPS/RUNLOG_st.txt" && grep -q "^END   st_ALL_1 rc=0" "$SWEEPS/RUNLOG_st.txt" && ok "START/END lines with switches and rc" || bad "runlog"
    rm -rf "$t"
    echo "kp_matrix --selftest: $pass ok, $fail FAIL"
    [ "$fail" = 0 ]
}

while [ $# -gt 0 ]; do
    case "$1" in
        --help|-h) usage; exit 0;;
        --selftest) selftest; exit $?;;
        --dry-run) DRY=1;;
        --arms) ARMS="$2"; shift;;
        --cycles) CYCLES="$2"; shift;;
        --name) NAME="$2"; shift;;
        --tip-dir) TIP_DIR="$2"; shift;;
        --arm-dir) ARM_DIR="$2"; shift;;
        --model) MODEL="$2"; shift;;
        --cache) CACHE="$2"; shift;;
        --flush) FLUSH="$2"; shift;;
        --sha) SHA="$2"; shift;;
        --sweeps) SWEEPS="$2"; shift;;
        --sleep) SLEEP="$2"; shift;;
        *) die "unknown option $1 (try --help)";;
    esac
    shift
done
[ -n "$ARMS" ] || { usage; die "--arms is required"; }
IFS=',' read -ra ARM_LIST <<<"$ARMS"
for a in "${ARM_LIST[@]}"; do arm_env "$a" >/dev/null || exit 2; done
case "$CYCLES" in ''|*[!0-9]*|0) die "--cycles must be a positive integer";; esac
mkdir -p "$SWEEPS"

for d in "$TIP_DIR" "$ARM_DIR"; do
    [ -x "$d/ds4-bench" ] || { [ "$DRY" = 1 ] || die "no built ds4-bench in $d (make cuda-spark there first)"; }
done
[ -f "$MODEL" ] || { [ "$DRY" = 1 ] || die "model not found: $MODEL"; }
IDENT=$(gguf_ident "$MODEL")
case "$IDENT" in arch=deepseek41*) ;; *) [ "$DRY" = 1 ] || die "model is not a V4.1 file ($IDENT); the port is arch-gated on deepseek41";; esac

[ "$DRY" = 1 ] || take_lock
trap 'stop_net; release_lock' EXIT INT TERM
[ "$DRY" = 1 ] || start_net

RUNLOG="$SWEEPS/RUNLOG_$NAME.txt"
{
    echo "# $NAME $(date -u +%Y-%m-%dT%H:%M:%SZ) kp_matrix pid=$$ arms=$ARMS cycles=$CYCLES"
    echo "# driver=$(driver) power=n/a(linux) load=$(la) gpu=$(gu)% memavail=$(ma_gb)GB"
    echo "# model=$MODEL $IDENT bytes=$(stat -c %s "$MODEL" 2>/dev/null || echo n/a) mtime=$(stat -c %y "$MODEL" 2>/dev/null | cut -c1-19 || echo n/a) sha256=${SHA:-not-taken}"
    echo "# tip=$TIP_DIR sha=$(git -C "$TIP_DIR" rev-parse --short=9 HEAD 2>/dev/null || echo n/a) .text=$(text_bytes "$TIP_DIR/ds4-bench")"
    echo "# arm=$ARM_DIR sha=$(git -C "$ARM_DIR" rev-parse --short=9 HEAD 2>/dev/null || echo n/a) .text=$(text_bytes "$ARM_DIR/ds4-bench")"
    echo "# sweep: ctx 2048 (warm-up, discarded) / 4096 / 6144, gen 128, --cuda --ssd-streaming --ssd-streaming-cache-experts $CACHE flush=${FLUSH:-unset}"
} >> "$RUNLOG"
build_plan > "$SWEEPS/plan_$NAME.txt"

while read -r label arm; do
    [ -n "$label" ] || continue
    envs=()
    e=$(arm_env "$arm"); [ -n "$e" ] && read -ra envs <<<"$e"
    [ -n "$FLUSH" ] && [ "$arm" != tip ] && [ "$arm" != off ] && envs+=("DS4_CUDA_V41_DECODE_FLUSH_LAYERS=$FLUSH")
    run_one "$label" "$(arm_dir "$arm")" ${envs[@]+"${envs[@]}"}
    [ "$DRY" = 1 ] || sleep "$SLEEP"
done < "$SWEEPS/plan_$NAME.txt"
echo "PASS DONE plan_$NAME.txt $(date -u +%H:%M:%SZ)" >> "$RUNLOG"
echo "DONE $NAME: $(wc -l < "$SWEEPS/plan_$NAME.txt") runs planned; runlog $RUNLOG"
echo "NEXT -> scp 'spark:$SWEEPS/RUNLOG_$NAME.txt' 'spark:$SWEEPS/${NAME}_*.csv' <local dir> and run the seal's paired analysis"
