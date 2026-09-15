#!/bin/sh
# PREFILLHANDOFF: gate the protect patch, then measure it where its effect lives.
#
# The effect is concentrated in the first decode tokens after a long prefill, so
# a steady-state sweep dilutes it to nothing. This measures time to first token
# and the rate over the first 32 decode tokens at a fixed long context, arms
# interleaved, a fresh process every run, and reports the MINIMUM over repeats
# rather than the mean.
#
# Usage:  sh track_handoff_measure.sh [OUTDIR] [REPEATS]
set -e

CTL=${CTL:-$HOME/lane-prefillhandoff-ctl}
ARM=${ARM:-$HOME/lane-prefillhandoff-arm}
PROBEARM=${PROBEARM:-$HOME/lane-prefillhandoff-probearm}
PROBE=${PROBE:-$HOME/lane-prefillhandoff}
MODEL=${MODEL:-$HOME/dwarfstar/gguf/DeepSeek-V4.1-Flash-Q2.gguf}
OUT=${1:-$HOME/handoff-measure}
REPEATS=${2:-3}
PROMPT=$OUT/long_prompt.txt

COMMON="--cuda --ssd-streaming --ssd-streaming-cache-experts 90GB --ctx 32768 --temp 0"

mkdir -p "$OUT"

stamp() {
    printf '    load=%s memavail=%sGB gpu=%s driver=%s\n' \
        "$(cut -d' ' -f1 /proc/loadavg)" \
        "$(awk '/MemAvailable/{printf "%.0f", $2/1048576}' /proc/meminfo)" \
        "$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader)" \
        "$(nvidia-smi --query-gpu=driver_version --format=csv,noheader)"
}

refuse_if_busy() {
    L=$(cut -d' ' -f1 /proc/loadavg)
    M=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
    if [ "$(awk -v l="$L" 'BEGIN{print (l > 2) ? 1 : 0}')" = "1" ]; then
        echo "REFUSED: load $L above 2"; exit 1
    fi
    if [ "$M" -lt 100 ]; then
        echo "REFUSED: MemAvailable ${M}GB under 100"; exit 1
    fi
}

refuse_if_busy
head -c 14000 "$PROBE/speed-bench/promessi_sposi.txt" > "$PROMPT"
echo "prompt: $(wc -c < "$PROMPT") bytes, sha $(sha256sum "$PROMPT" | cut -c1-16)"

echo
echo "################ PART 1 · THE GATES ################"
echo
echo "--- short gate, 45 tokens, must read bb06e711bc498bb9 at 73146 bytes ---"
for tree in CTL ARM PROBEARM; do
    eval dir=\$$tree
    rm -f "$OUT/short_$tree.json"
    "$dir/ds4" -m "$MODEL" $COMMON -n 32 --dump-logprobs "$OUT/short_$tree.json" \
        -p 'Explain in two sentences why the sky is blue.' > /dev/null 2>&1
    printf '  %-9s %s  %s bytes\n' "$tree" \
        "$(sha256sum "$OUT/short_$tree.json" | cut -c1-16)" \
        "$(wc -c < "$OUT/short_$tree.json")"
done

echo
echo "--- LONG gate, 4391 tokens: this is the one that exercises this lane ---"
for tree in CTL ARM PROBEARM; do
    eval dir=\$$tree
    rm -f "$OUT/long_$tree.json"
    "$dir/ds4" -m "$MODEL" $COMMON -n 16 --prompt-file "$PROMPT" \
        --dump-logprobs "$OUT/long_$tree.json" > /dev/null 2>&1
    printf '  %-9s %s  %s bytes\n' "$tree" \
        "$(sha256sum "$OUT/long_$tree.json" | cut -c1-16)" \
        "$(wc -c < "$OUT/long_$tree.json")"
done

echo
echo "################ PART 2 · DOES THE PATCH MOVE RESIDENCY? ################"
echo "  same instrument that took the baseline, now on the patched engine"
DS4_PROBE_HANDOFF=1 "$PROBEARM/ds4" -m "$MODEL" $COMMON -n 64 \
    --prompt-file "$PROMPT" > /dev/null 2> "$OUT/long_probearm.log"
python3 "$PROBE/track_handoff_probe.py" "$OUT/long_probearm.log" --early 32 \
    --json "$OUT/handoff_arm.json"

echo
echo "################ PART 3 · THE CLOCK, WHERE THE EFFECT LIVES ################"
echo "  fixed 4096-token context, 32 generated tokens, arms INTERLEAVED,"
echo "  fresh process per run, $REPEATS repeats, minimum reported"
echo
echo "  ENGRAM PAGE-CACHE POSTURE, stated rather than assumed: on Linux the Engram"
echo "  descriptor is ordinary buffered, so a second run of the same prompt inherits"
echo "  the first's page cache and its Engram reads cost ~0.15ms instead of 20-25ms."
echo "  Expert reads are O_DIRECT and are unaffected, and the expert cache is this"
echo "  lane's whole subject - but gen_first_ms and gen_tps CONTAIN the Engram cost,"
echo "  so both arms are deliberately measured WARM on that path: each arm takes a"
echo "  discarded warmup first, so neither arm is credited with the other's cold run."
echo "  ALL PART 3 NUMBERS BELOW DESCRIBE A WARM ENGRAM PATH. A cold long generation"
echo "  is owed and this is not it."
echo
for tree in CTL ARM; do
    eval dir=\$$tree
    refuse_if_busy
    "$dir/ds4-bench" -m "$MODEL" --cuda --ssd-streaming \
        --ssd-streaming-cache-experts 90GB \
        --prompt-file "$PROMPT" --ctx-start 4096 --ctx-max 4096 \
        --step-incr 2048 --gen-tokens 32 --csv "$OUT/warmup_${tree}.csv" \
        > /dev/null 2>&1
    printf '  warmup %-4s (DISCARDED) ' "$tree"; stamp
done
echo
for r in $(seq 1 "$REPEATS"); do
    for tree in CTL ARM; do
        eval dir=\$$tree
        refuse_if_busy
        csv="$OUT/bench_${tree}_r${r}.csv"
        "$dir/ds4-bench" -m "$MODEL" --cuda --ssd-streaming \
            --ssd-streaming-cache-experts 90GB \
            --prompt-file "$PROMPT" --ctx-start 4096 --ctx-max 4096 \
            --step-incr 2048 --gen-tokens 32 --csv "$csv" > /dev/null 2>&1
        printf '  repeat %s %-4s ' "$r" "$tree"; stamp
    done
done

echo
echo "################ VERDICT ################"
echo
echo "gates (every sha in a column must agree):"
printf '  %-9s %-18s %-18s\n' tree short long
for tree in CTL ARM PROBEARM; do
    printf '  %-9s %-18s %-18s\n' "$tree" \
        "$(sha256sum "$OUT/short_$tree.json" | cut -c1-16)" \
        "$(sha256sum "$OUT/long_$tree.json" | cut -c1-16)"
done

echo
echo "first-token latency and first-32-token rate, MINIMUM over $REPEATS repeats:"
python3 - "$OUT" "$REPEATS" <<'PY'
import csv, sys, glob, os
out, reps = sys.argv[1], int(sys.argv[2])
for tree in ("CTL", "ARM"):
    first_ms, gen_tps, prefill_tps = [], [], []
    for path in sorted(glob.glob(os.path.join(out, "bench_%s_r*.csv" % tree))):
        with open(path) as fh:
            for row in csv.DictReader(fh):
                first_ms.append(float(row["gen_first_ms"]))
                gen_tps.append(float(row["gen_tps"]))
                prefill_tps.append(float(row["prefill_tps"]))
    if not first_ms:
        print("  %-4s no rows" % tree); continue
    print("  %-4s ttft_min %8.3f ms   gen_tps_max %6.2f   prefill_tps_max %6.2f   (n=%d)"
          % (tree, min(first_ms), max(gen_tps), max(prefill_tps), len(first_ms)))
PY
