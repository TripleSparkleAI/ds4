#!/bin/sh
# PREFILLHANDOFF: gate the read-ahead victim reorder and measure BOTH sides of it.
#
# The change gives the read-ahead a different eviction order, so it can make the
# opening faster and the prefill slower. A run that reports only the half that
# flatters it is not a measurement, so every arm here carries prefill
# throughput, time to first token, and the decode miss curve together.
#
# It also carries the read-ahead's OWN fire rate in every instrumented arm. The
# reserve loop abandons a layer through a bare `throw 0`, so a change that
# starved it would look like a win everywhere except the one counter that says
# it stopped running. The reorder cannot reach that path by construction
# (victims.size() is unchanged), and this prints the counter anyway, because
# "cannot happen" is a claim and `no_victims=0` is evidence.
#
# Usage:  sh track_handoff_measure2.sh [OUTDIR] [REPEATS]
set -e

CTL=${CTL:-$HOME/lane-prefillhandoff-ctl}                 # origin/main
COUNTER=${COUNTER:-$HOME/lane-prefillhandoff-counter}     # + counters only
ORDER=${ORDER:-$HOME/lane-prefillhandoff-order}           # + counters + reorder
PROBE=${PROBE:-$HOME/lane-prefillhandoff}                 # probe on origin/main
PROBEORDER=${PROBEORDER:-$HOME/lane-prefillhandoff-order-probe}
MODEL=${MODEL:-$HOME/dwarfstar/gguf/DeepSeek-V4.1-Flash-Q2.gguf}
OUT=${1:-$HOME/handoff-measure2}
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
echo "################ PART 1 - THE GATES ################"
echo "short gate must read bb06e711bc498bb9 at 73146 bytes on every tree;"
echo "the LONG gate is the one that exercises this lane."
for tree in CTL COUNTER ORDER PROBEORDER; do
    eval dir=\$$tree
    rm -f "$OUT/short_$tree.json" "$OUT/long_$tree.json"
    "$dir/ds4" -m "$MODEL" $COMMON -n 32 \
        --dump-logprobs "$OUT/short_$tree.json" \
        -p 'Explain in two sentences why the sky is blue.' > /dev/null 2>&1
    "$dir/ds4" -m "$MODEL" $COMMON -n 16 --prompt-file "$PROMPT" \
        --dump-logprobs "$OUT/long_$tree.json" > /dev/null 2>&1
    printf '  %-11s short %s  long %s\n' "$tree" \
        "$(sha256sum "$OUT/short_$tree.json" | cut -c1-16)" \
        "$(sha256sum "$OUT/long_$tree.json" | cut -c1-16)"
done

echo
echo "################ PART 2 - THE READ-AHEAD'S OWN FIRE RATE ################"
echo "  if the reorder ever starved the reader, no_victims rises here and"
echo "  nowhere else. Baseline arm has the counters and the old order."
for tree in COUNTER ORDER; do
    eval dir=\$$tree
    refuse_if_busy
    DS4_CUDA_SSD_PREFETCH_STATS=1 "$dir/ds4" -m "$MODEL" $COMMON -n 16 \
        --prompt-file "$PROMPT" > /dev/null 2> "$OUT/stats_$tree.log"
    printf '  %-9s %s\n' "$tree" \
        "$(grep 'CUDA SSD read-ahead' "$OUT/stats_$tree.log" | tail -1)"
done

echo
echo "################ PART 3 - THE DECODE MISS CURVE ################"
echo "  the figures to beat: 32.22 misses/token at 0.8658 over the first 32,"
echo "  against 9.16 at 0.9618 steady."
for tree in PROBE PROBEORDER; do
    eval dir=\$$tree
    refuse_if_busy
    DS4_PROBE_HANDOFF=1 "$dir/ds4" -m "$MODEL" $COMMON -n 64 \
        --prompt-file "$PROMPT" > /dev/null 2> "$OUT/probe_$tree.log"
    echo "--- $tree ---"
    python3 "$PROBE/track_handoff_probe.py" "$OUT/probe_$tree.log" --early 32 \
        --json "$OUT/handoff_$tree.json" 2>&1 | sed -n '/^Q2/,/^$/p'
done

echo
echo "################ PART 4 - THE CLOCK, BOTH SIDES ################"
echo "  fixed 4096-token context, 32 generated tokens, arms INTERLEAVED,"
echo "  fresh process per run, a discarded warmup per arm, $REPEATS repeats,"
echo "  MINIMUM reported. Both arms carry the counters, so the only difference"
echo "  between them is the victim order."
echo "  ENGRAM PAGE-CACHE POSTURE: both arms measured WARM on that path, each"
echo "  taking its own discarded warmup, so neither is credited with the"
echo "  other's cold run. A cold long generation is owed and this is not it."
echo
for tree in CTL COUNTER ORDER; do
    eval dir=\$$tree
    refuse_if_busy
    "$dir/ds4-bench" -m "$MODEL" --cuda --ssd-streaming \
        --ssd-streaming-cache-experts 90GB --prompt-file "$PROMPT" \
        --ctx-start 4096 --ctx-max 4096 --step-incr 2048 --gen-tokens 32 \
        --csv "$OUT/warmup_${tree}.csv" > /dev/null 2>&1
    printf '  warmup %-8s (DISCARDED) ' "$tree"; stamp
done
echo
for r in $(seq 1 "$REPEATS"); do
    for tree in CTL COUNTER ORDER; do
        eval dir=\$$tree
        refuse_if_busy
        "$dir/ds4-bench" -m "$MODEL" --cuda --ssd-streaming \
            --ssd-streaming-cache-experts 90GB --prompt-file "$PROMPT" \
            --ctx-start 4096 --ctx-max 4096 --step-incr 2048 --gen-tokens 32 \
            --csv "$OUT/bench_${tree}_r${r}.csv" > /dev/null 2>&1
        printf '  repeat %s %-8s ' "$r" "$tree"; stamp
    done
done

echo
echo "################ VERDICT ################"
echo
echo "gates:"
printf '  %-11s %-18s %-18s\n' tree short long
for tree in CTL COUNTER ORDER PROBEORDER; do
    printf '  %-11s %-18s %-18s\n' "$tree" \
        "$(sha256sum "$OUT/short_$tree.json" | cut -c1-16)" \
        "$(sha256sum "$OUT/long_$tree.json" | cut -c1-16)"
done
echo
echo "read-ahead fire rate (a silent disable cannot hide here):"
for tree in COUNTER ORDER; do
    printf '  %-9s %s\n' "$tree" \
        "$(grep 'CUDA SSD read-ahead' "$OUT/stats_$tree.log" | tail -1)"
done
echo
echo "the clock, MINIMUM over $REPEATS repeats - prefill is the OTHER side of the trade:"
python3 - "$OUT" <<'PY'
import csv, sys, glob, os
out = sys.argv[1]
print("  %-9s %12s %12s %12s" % ("arm", "ttft_ms_min", "gen_tps_max", "prefill_max"))
for tree in ("CTL", "COUNTER", "ORDER"):
    f, g, p = [], [], []
    for path in sorted(glob.glob(os.path.join(out, "bench_%s_r*.csv" % tree))):
        with open(path) as fh:
            for row in csv.DictReader(fh):
                f.append(float(row["gen_first_ms"]))
                g.append(float(row["gen_tps"]))
                p.append(float(row["prefill_tps"]))
    if not f:
        print("  %-9s no rows" % tree); continue
    print("  %-9s %12.3f %12.2f %12.2f" % (tree, min(f), max(g), max(p)))
print()
print("  a win on ttft/gen with a loss on prefill is a TRADE, not a win.")
PY
