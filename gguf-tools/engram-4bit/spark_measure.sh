#!/bin/bash
# spark_measure.sh - the GB10 measurement for the Engram 4-bit sidecar, lane ENGRAM4BIT.
#
# Runs FROM THE MAC. It takes ~/spark_model.lock on the spark under this lane's name only
# when the lock is FREE (never over another lane), then, on the spark, in order:
#   1 build   - detached worktree at the branch sha, make cuda-spark
#   2 convert - engram_hlwq_quantize.py over the V4.1 Flash GGUF -> ~/engram4/v41flash.engram4
#               (reads 202.8 GB once, writes 101.4 GB; refuses to leave a partial file)
#   3 check   - 4096 random rows a table, FP8 vs sidecar: rel RMSE + cosine
#   4 G1      - greedy (--temp 0) continuations of the two G1 prompts, sidecar OFF then ON,
#               token stream + top-20 logprobs dumped; the first divergence index and the
#               identical-token fraction are computed on the Mac from the dumps
#   5 t/s     - ds4-bench, ctx 2048/4096/6144, gen 128, warm-up first, then 3 interleaved
#               repeats of OFF/ON, each run stamped: load, gpu util, memavail, driver, ctx,
#               checkpoint (the GGUF's sha256 head + the sidecar's)
# then releases the lock it wrote. `--dry-run` prints every command and runs nothing.
#
# Byte identity is NOT the gate for this lever (a quant change): the gate is the vectors
# still passing, the greedy ON/OFF divergence reported honestly, and the logprob drift
# against the q2 noise floor. This script produces the inputs to that report.
set -u
LANE=ENGRAM4BIT
SHA="${SHA:-$(git rev-parse --short=9 HEAD)}"
WT="dw-engram4bit-$SHA"
MODEL='~/dwarfstar/gguf/DeepSeek-V4.1-Flash-Q2.gguf'
SIDECAR='~/engram4/v41flash.engram4'
OUT='~/sweeps/engram4bit'
DRY=0
[ "${1:-}" = "--dry-run" ] && DRY=1

# the two G1 prompts, as dspark-verify.sh names them: the default battery's first short
# prompt and the long templated prompt
P1='What is the capital of France? Answer in one sentence.'
P2='You are a helpful assistant. User: List three prime numbers and explain what makes them prime. Assistant:'

read -r -d '' SCRIPT <<EOF
set -u
exec < /dev/null
cd ~/dwarfstar || exit 1
mkdir -p ~/sweeps $OUT ~/engram4 _worktrees
# --- the lock: take it only when FREE, write the owner line, refuse otherwise ---------
OWN=\$(cat ~/spark_model.lock/owner 2>/dev/null || true)
if [ -n "\$OWN" ] && ! grep -q "^$LANE " <<<"\$OWN"; then
  echo "REFUSED_NOLOCK held-by: \$OWN"; exit 3
fi
mkdir -p ~/spark_model.lock
echo "$LANE spark_measure \$(date -u +%Y-%m-%dT%H:%M:%SZ) - engram 4-bit sidecar ($SHA)" > ~/spark_model.lock/owner
mine(){ grep -q "^$LANE " ~/spark_model.lock/owner 2>/dev/null; }
release(){ mine && rm -rf ~/spark_model.lock && echo "LOCK RELEASED \$(date -u +%H:%M:%SZ)"; }
trap release EXIT
stamp(){
  echo "STAMP \$1 load=\$(cut -d' ' -f1 /proc/loadavg) gpu=\$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)% memavail=\$(( \$(grep MemAvailable /proc/meminfo | tr -dc 0-9) / 1000000 ))GB driver=\$(nvidia-smi --query-gpu=driver_version --format=csv,noheader) ctx=\$2 checkpoint=\$CKPT sidecar=\$SCK \$(date -u +%H:%M:%SZ)"
}
# --- 1 build ---------------------------------------------------------------------------
if [ ! -d _worktrees/$WT ]; then git worktree add -q --detach _worktrees/$WT $SHA || exit 1; fi
cd _worktrees/$WT || exit 1
( make cuda-spark -j12 > $OUT/build.log 2>&1 ); echo "BUILD rc=\$? \$(size ds4 | tail -1 | cut -f1)"
# --- 2 convert (skips if the sidecar already exists and checks against the model) ------
if [ ! -f $SIDECAR ]; then
  python3 gguf-tools/engram-4bit/engram_hlwq_quantize.py --source $MODEL --output $SIDECAR 2>&1 | tail -3
fi
CKPT=\$(head -c 1048576 $MODEL | sha256sum | cut -c1-12)
SCK=\$(head -c 4096 $SIDECAR | sha256sum | cut -c1-12)
# --- 3 check ---------------------------------------------------------------------------
python3 gguf-tools/engram-4bit/engram_hlwq_quantize.py --source $MODEL --check $SIDECAR --samples 4096 | tee $OUT/check.txt
# --- 4 G1: greedy ON vs OFF, tokens + top-20 logprobs -----------------------------------
i=0
for P in "$P1" "$P2"; do
  i=\$((i+1))
  for arm in off on; do
    if [ \$arm = on ]; then export DS4_ENGRAM_4BIT=$SIDECAR; else unset DS4_ENGRAM_4BIT; fi
    stamp g1_p\${i}_\$arm 4096
    ./ds4 --cuda --ssd-streaming --ssd-streaming-cache-experts 90GB -m $MODEL -c 4096 --temp 0 -n 128 \\
      --dump-logprobs $OUT/g1_p\${i}_\$arm.json --logprobs-top-k 20 -p "\$P" > $OUT/g1_p\${i}_\$arm.txt 2>&1
    echo "G1 p\$i \$arm rc=\$?"
  done
done
unset DS4_ENGRAM_4BIT
# --- 5 t/s: warm-up, then 3 interleaved repeats, ctx 2048/4096/6144 --------------------
bench(){ # \$1 arm, \$2 label
  if [ \$1 = on ]; then export DS4_ENGRAM_4BIT=$SIDECAR; else unset DS4_ENGRAM_4BIT; fi
  stamp \$2 2048-6144
  ./ds4-bench -m $MODEL --cuda --ssd-streaming --ssd-streaming-cache-experts 90GB \\
    --prompt-file speed-bench/promessi_sposi.txt --ctx-start 2048 --ctx-max 6144 --step-incr 2048 \\
    --gen-tokens 128 --csv $OUT/\$2.csv > $OUT/\$2.log 2>&1
  echo "BENCH \$2 rc=\$?"
}
bench off warmup_off
for r in 1 2 3; do bench off off_r\$r; bench on on_r\$r; done
unset DS4_ENGRAM_4BIT
echo "ALL DONE \$(date -u +%H:%M:%SZ)"
EOF

if [ $DRY = 1 ]; then
  echo "--- DRY RUN: would push $SHA to the sparkbox remote (the spark itself, never origin), then run on ssh spark under setsid ---"
  echo "git push -q sparkbox $SHA:refs/heads/engram4bit-measure"
  echo "$SCRIPT"
  echo "--- end ---"
  exit 0
fi
url=$(git remote get-url sparkbox)
case "$url" in spark:*|ssh://spark*) ;; *) echo "refusing to push anywhere but the spark: $url"; exit 2;; esac
git push -q sparkbox "$SHA:refs/heads/engram4bit-measure" || exit 2
ssh spark "cat > ~/engram4bit_measure.sh" <<<"$SCRIPT"
ssh spark "setsid bash ~/engram4bit_measure.sh > ~/sweeps/RUNLOG_engram4bit.txt 2>&1 < /dev/null &"
echo "started on the spark; follow: ssh spark tail -f ~/sweeps/RUNLOG_engram4bit.txt"
echo "NEXT -> python3 gguf-tools/engram-4bit/g1_compare.py <(ssh spark cat ~/sweeps/engram4bit/g1_p1_off.json) <(ssh spark cat ~/sweeps/engram4bit/g1_p1_on.json)"
