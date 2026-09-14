#!/usr/bin/env bash
# <claudes_code_comments>
# ** Function List **
# usage() - prints the mini-tutorial and exits
# spark_stamp() - captures load, GPU util and driver version from the Spark
# gate_correctness() - greedy A/B, expert cache on vs off, byte-identical or fail
# gate_long() - ~588-token prompt twice: identical to each other and to the long baseline
# gate_speed() - prefill/generation t/s, refused on a busy box
# run_lane() - both gates for one lane, in order
#
# ** Technical Review **
# One verification door for every SPARKPORT lane branch.  Each lane builds its
# own tree on the Spark under ~/dwarfstar/_worktrees/<lane>; this script runs
# the SAME two gates against each so lanes are comparable to each other.
#
# - CORRECTNESS runs first and is load-independent: it compares greedy
#   --dump-logprobs output with the resident expert cache on against the same
#   build with DS4_CUDA_EXPERT_CACHE=0.  Byte-identical or the lane fails.  A
#   weights cache that returns wrong bytes is the failure mode that matters,
#   and it is invisible in a t/s number.
# - SPEED runs second and REFUSES a busy box, per the project's load law: a
#   t/s taken against a contested GPU measures the box, not the code.  Every
#   number carries load, GPU util, driver and context.
# - Lanes are run one at a time on purpose.  The model is 340 GiB and the box
#   has 121 GB; two concurrent loads thrash and both numbers are worthless.
# </claudes_code_comments>
set -uo pipefail

SPARK_ROOT="\$HOME/dwarfstar/_worktrees"
MODEL="\$HOME/dwarfstar/gguf/DeepSeek-V4.1-Flash-Q2.gguf"
COMMON="--cuda --ssd-streaming --ssd-streaming-cache-experts 90GB --ctx 32768"
BUSY_LOAD=8        # 20 cores; above this a timing is not comparable
BUSY_GPU=25        # percent
# The greedy logprob dump of the reference build for the correctness prompt.
# On-vs-off proves the cache is transparent; this proves the KERNELS did not
# move either - a kernel change flips both arms together and would pass the
# first check alone (LUT16 did exactly that, 2026-09-13).
BASELINE_SHA="${SPARKPORT_BASELINE_SHA:-e73f763588d4f253}"
# The long-prompt arm: a ~588-token prompt through the BATCHED prefill, run twice.
# Both runs must match each other (the batched tiers were not reproducible until
# 2026-09-13) and the recorded sha.
LONG_BASELINE_SHA="${SPARKPORT_LONG_BASELINE_SHA:-5566a268d8baab75}"

usage() {
    cat <<'EOF'
sparkport-verify.sh - one verification door for every SPARKPORT lane

  sparkport-verify.sh <lane> [<lane>...]   run both gates per lane, serially
  sparkport-verify.sh --correctness <lane> correctness only (safe on a busy box)
  sparkport-verify.sh --speed <lane>       speed only (a lane already gated today)
  sparkport-verify.sh --long <lane>        long-prompt arm: batched prefill, two runs identical + baseline
  sparkport-verify.sh --stamp              print the Spark's current stamp
  sparkport-verify.sh --selftest           check own wiring, run nothing heavy

Lanes are directory names under ~/dwarfstar/_worktrees on the Spark,
e.g. sparkport preadpool hotlist pagecache draincut

THE TWO GATES
  1 CORRECTNESS  greedy --temp 0 logprob dump, cache ON vs cache OFF.
                 Byte-identical or the lane FAILS.  Load-independent, so
                 this is always valid even while the box is contested.
  2 SPEED        prefill/generation t/s, stamped with load + GPU + driver.
                 REFUSED when the box is busy - a contested timing is not
                 a measurement.

Examples
  sparkport-verify.sh --correctness preadpool     # safe right now
  sparkport-verify.sh sparkport preadpool hotlist # full, on a quiet box

NEXT -> run --correctness on every lane first; only then queue speed.
EOF
}

spark_stamp() {
    ssh spark 'load=$(cut -d" " -f1 /proc/loadavg)
        gpu=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)
        drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
        echo "${load:-?} ${gpu:-?} ${drv:-unknown}"'
}

box_is_busy() {
    local load gpu
    read -r load gpu _ <<<"$1"
    awk -v l="$load" -v b="$BUSY_LOAD" 'BEGIN{exit !(l+0 > b)}' && return 0
    [ "${gpu}" != "?" ] && [ "${gpu:-0}" -gt "$BUSY_GPU" ] 2>/dev/null && return 0
    return 1
}

gate_correctness() {
    local lane="$1"
    echo "  [1/2] correctness: greedy A/B, expert cache on vs off"
    ssh spark "cd $SPARK_ROOT/$lane || exit 9
        P='Explain in two sentences why the sky is blue.'
        rm -f /tmp/${lane}_on.json /tmp/${lane}_off.json   # a killed run must not pass on a stale dump
        ./ds4 -m $MODEL $COMMON --temp 0 -n 32 --dump-logprobs /tmp/${lane}_on.json  -p \"\$P\" >/dev/null 2>&1
        DS4_CUDA_EXPERT_CACHE=0 ./ds4 -m $MODEL $COMMON --temp 0 -n 32 --dump-logprobs /tmp/${lane}_off.json -p \"\$P\" >/dev/null 2>&1
        if [ ! -s /tmp/${lane}_on.json ] || [ ! -s /tmp/${lane}_off.json ]; then
            echo '        FAIL - a run produced no logprobs'; exit 1; fi
        if ! cmp -s /tmp/${lane}_on.json /tmp/${lane}_off.json; then
            echo '        FAIL - cache changes the logit distribution'; exit 1; fi
        S=\$(sha256sum /tmp/${lane}_on.json | cut -c1-16)
        if [ \"\$S\" = \"$BASELINE_SHA\" ]; then
            echo \"        PASS - byte-identical, and equal to the baseline (\$S)\"
        else
            echo \"        FAIL - cache-transparent but the output moved: \$S != $BASELINE_SHA (a kernel changed the logits)\"; exit 1; fi"
}

gate_long() {
    local lane="$1"
    echo "  [long] ~588-token prompt through the batched prefill, two runs"
    ssh spark "cd $SPARK_ROOT/$lane || exit 9
        P=\"\$(printf 'Explain, in careful detail and with examples, how mixture-of-experts routing works in a transformer, why load balancing matters, what the auxiliary loss does, how capacity factors are chosen, and how expert parallelism is implemented across devices. %.0s' \$(seq 1 12))\"
        rm -f /tmp/${lane}_long1.json /tmp/${lane}_long2.json   # same: no stale dumps
        ./ds4 -m $MODEL $COMMON --temp 0 -n 16 --dump-logprobs /tmp/${lane}_long1.json -p \"\$P\" >/dev/null 2>&1
        ./ds4 -m $MODEL $COMMON --temp 0 -n 16 --dump-logprobs /tmp/${lane}_long2.json -p \"\$P\" >/dev/null 2>&1
        if [ ! -s /tmp/${lane}_long1.json ] || [ ! -s /tmp/${lane}_long2.json ]; then
            echo '        FAIL - a long-prompt run produced no logprobs'; exit 1; fi
        if ! cmp -s /tmp/${lane}_long1.json /tmp/${lane}_long2.json; then
            echo '        FAIL - two runs of the same prompt differ (the batched prefill is not reproducible)'; exit 1; fi
        S=\$(sha256sum /tmp/${lane}_long1.json | cut -c1-16)
        if [ \"\$S\" = \"$LONG_BASELINE_SHA\" ]; then
            echo \"        PASS - reproducible, and equal to the long baseline (\$S)\"
        else
            echo \"        FAIL - reproducible but moved: \$S != $LONG_BASELINE_SHA\"; exit 1; fi"
}

gate_speed() {
    local lane="$1" stamp="$2"
    read -r load gpu drv <<<"$stamp"
    if box_is_busy "$stamp"; then
        echo "  [2/2] speed: REFUSED - box busy (load $load, gpu ${gpu}%). Not a measurement."
        return 0
    fi
    echo "  [2/2] speed: load=$load gpu=${gpu}% driver=$drv ctx=32768"
    ssh spark "cd $SPARK_ROOT/$lane && ./ds4 -m $MODEL $COMMON -n 256 \
        -p 'Write a short technical explanation of how mixture-of-experts routing works.' 2>&1 \
        | grep -E 't/s' | sed 's/^/        /'"
}

run_lane() {
    local lane="$1" only="$2"
    echo "=== lane $lane ==="
    if ! ssh spark "test -x $SPARK_ROOT/$lane/ds4"; then
        echo "  SKIP - no built ds4 at $SPARK_ROOT/$lane"; return 0
    fi
    if [ "$only" = "long" ]; then
        gate_long "$lane" || { echo "  lane $lane FAILED the long-prompt gate"; return 1; }
        return 0
    fi
    if [ "$only" != "speed" ]; then
        gate_correctness "$lane" || { echo "  lane $lane FAILED the correctness gate"; return 1; }
    fi
    [ "$only" = "correctness" ] && return 0
    sleep 8   # let the correctness arm's own GPU use drain before the stamp is read
    gate_speed "$lane" "$(spark_stamp)"
}

case "${1:-}" in
    ""|-h|--help) usage; exit 0 ;;
    --stamp) read -r l g d <<<"$(spark_stamp)"; echo "load=$l gpu=${g}% driver=$d"; exit 0 ;;
    --selftest)
        rc=0
        for fn in usage spark_stamp box_is_busy gate_correctness gate_long gate_speed run_lane; do
            declare -F "$fn" >/dev/null || { echo "FAIL missing $fn"; rc=1; }
        done
        box_is_busy "15.0 1 580.00"  && echo "ok  busy by load"   || { echo "FAIL busy-by-load"; rc=1; }
        box_is_busy "0.5 90 580.00"  && echo "ok  busy by gpu"    || { echo "FAIL busy-by-gpu"; rc=1; }
        box_is_busy "0.5 1 580.00"   && { echo "FAIL quiet box read as busy"; rc=1; } || echo "ok  quiet box passes"
        [ $rc -eq 0 ] && echo "selftest PASS" || echo "selftest FAIL"
        exit $rc ;;
    --correctness) shift; only=correctness ;;
    --speed) shift; only=speed ;;
    --long) shift; only=long ;;
    *) only=both ;;
esac

fails=0
for lane in "$@"; do run_lane "$lane" "${only:-both}" || fails=$((fails+1)); done
echo
[ $fails -eq 0 ] && echo "all lanes passed their gates" || echo "$fails lane(s) failed"
exit $fails
