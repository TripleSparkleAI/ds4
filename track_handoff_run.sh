#!/bin/sh
# PREFILLHANDOFF: take the probe run, under the model lock, on the Spark.
#
# Measurement only. The probe build differs from the control by lines that are
# all behind getenv("DS4_PROBE_HANDOFF"), so this run is also the evidence that
# the instrument changes nothing: it takes the gate sha with the probe OFF and
# again with it ON, and both must equal the control's.
#
# Usage:  sh track_handoff_run.sh [OUTDIR]
set -e

TREE=${TREE:-$HOME/lane-prefillhandoff}
CTL=${CTL:-$HOME/lane-prefillhandoff-ctl}
MODEL=${MODEL:-$HOME/dwarfstar/gguf/DeepSeek-V4.1-Flash-Q2.gguf}
OUT=${1:-$HOME/handoff-run}
PROMPT=$OUT/long_prompt.txt

COMMON="--cuda --ssd-streaming --ssd-streaming-cache-experts 90GB --ctx 32768 --temp 0"

mkdir -p "$OUT"

stamp() {
    echo "  stamp: load=$(cut -d' ' -f1 /proc/loadavg)"
    echo "         memavail=$(awk '/MemAvailable/{printf "%.0f GB", $2/1048576}' /proc/meminfo)"
    echo "         gpu=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader)"
    echo "         driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader)"
}

# The box must be quiet or the numbers measure the box. Refuse rather than
# record a confounded run.
LOAD=$(cut -d' ' -f1 /proc/loadavg)
MEM=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
if [ "$(echo "$LOAD > 2" | bc)" = "1" ]; then
    echo "REFUSED: load $LOAD is above 2"; exit 1
fi
if [ "$MEM" -lt 100 ]; then
    echo "REFUSED: MemAvailable ${MEM} GB is under 100"; exit 1
fi

# The long prompt is the one that exercises this lane, and it must be the same
# bytes every time, so it is cut deterministically rather than by eye.
head -c 14000 "$TREE/speed-bench/promessi_sposi.txt" > "$PROMPT"
echo "prompt: $(wc -c < "$PROMPT") bytes, sha $(sha256sum "$PROMPT" | cut -c1-16)"

echo
echo "=== 1 of 4 · SHORT GATE, control tree ==========================="
stamp
rm -f "$OUT/gate_ctl.json"
"$CTL/ds4" -m "$MODEL" $COMMON -n 32 --dump-logprobs "$OUT/gate_ctl.json" \
    -p 'Explain in two sentences why the sky is blue.' > "$OUT/gate_ctl.stdout" 2>&1
echo "  sha $(sha256sum "$OUT/gate_ctl.json" | cut -c1-16)  bytes $(wc -c < "$OUT/gate_ctl.json")"

echo
echo "=== 2 of 4 · SHORT GATE, probe tree, probe OFF ==================="
stamp
rm -f "$OUT/gate_probe_off.json"
"$TREE/ds4" -m "$MODEL" $COMMON -n 32 --dump-logprobs "$OUT/gate_probe_off.json" \
    -p 'Explain in two sentences why the sky is blue.' > "$OUT/gate_probe_off.stdout" 2>&1
echo "  sha $(sha256sum "$OUT/gate_probe_off.json" | cut -c1-16)  bytes $(wc -c < "$OUT/gate_probe_off.json")"

echo
echo "=== 3 of 4 · SHORT GATE, probe tree, probe ON ===================="
stamp
rm -f "$OUT/gate_probe_on.json"
DS4_PROBE_HANDOFF=1 "$TREE/ds4" -m "$MODEL" $COMMON -n 32 \
    --dump-logprobs "$OUT/gate_probe_on.json" \
    -p 'Explain in two sentences why the sky is blue.' \
    > "$OUT/gate_probe_on.stdout" 2> "$OUT/short_probe.log"
echo "  sha $(sha256sum "$OUT/gate_probe_on.json" | cut -c1-16)  bytes $(wc -c < "$OUT/gate_probe_on.json")"
echo "  HO lines: $(grep -c '^HO ' "$OUT/short_probe.log" || true)"

echo
echo "=== 4 of 4 · THE MEASUREMENT · long prompt, probe ON ============="
echo "  -n 64 so 32 early decode tokens have a steady state to be compared against"
stamp
DS4_PROBE_HANDOFF=1 "$TREE/ds4" -m "$MODEL" $COMMON -n 64 \
    --prompt-file "$PROMPT" \
    > "$OUT/long_probe.stdout" 2> "$OUT/long_probe.log"
echo "  HO lines: $(grep -c '^HO ' "$OUT/long_probe.log" || true)"
stamp

echo
echo "=== VERDICT ====================================================="
echo "short gate, three trees (all three shas must agree, and equal bb06e711bc498bb9):"
for f in gate_ctl gate_probe_off gate_probe_on; do
    printf '  %-16s %s  %s bytes\n' "$f" \
        "$(sha256sum "$OUT/$f.json" | cut -c1-16)" "$(wc -c < "$OUT/$f.json")"
done
echo
python3 "$TREE/track_handoff_probe.py" "$OUT/long_probe.log" --early 32 \
    --json "$OUT/handoff.json"
