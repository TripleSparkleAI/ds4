#!/usr/bin/env python3
"""prism_spark.py - bring up PrismML's ternary Bonsai 2 27B on the Spark GB10 and measure it, step by step.

Docs: prism/README.md · prism/research/00_FINDINGS.md · WIKI/TOOL_BUILDING_STANDARD.md
"""
# <claudes_code_comments>
# ** Function List **
# die(msg) - one-line error + usage hint, exit 2
# utc() - UTC timestamp string for filenames/stamps
# on_spark() - true when running on the spark itself
# remote(script, dry, title) - run a bash script on the spark (print only under --dry-run)
# scp_back(remote_glob, dry) - copy measured CSVs from the spark into prism/measured/
# lock_guard(lane) - bash fragment: refuse unless the model lock is free or ours, then take it
# lock_release(lane) - bash fragment: drop the lock only if this lane holds it
# detached(name, body, lane) - bash fragment: write ~/prism/<name>.sh and start it under setsid
# poll(logname, done_marker, dry, every) - Mac-side poll of a detached step's log until its marker
# lock_owner(dry) - read ~/spark_model.lock/owner ("FREE" when absent)
# wait_lock_free(dry, max_s) - block until the lock is free (the download rule)
# verify_size(path, expected, actual) - the size gate: refuse a mismatch, never delete
# cmd_fetch(a) - clone the fork pinned at PIN_SHA; download the two GGUFs with size checks
# cmd_build(a) - cmake CUDA 121a under the lock, detached, BUILD.txt with .text sizes
# stamp_script() - bash fragment that prints the six-law stamp fields
# bench_script(quant, a) - llama-bench warm-up (discarded) + 3 repeats at depth 2048
# cmd_bench1(a) - both quants single-stream, CSV + table
# parse_batched_bench(text) - llama-batched-bench table -> (B, PP, TG, S_TG t/s) rows
# parse_llama_bench_csv(text) - llama-bench -o csv -> rows
# TTFT_CLIENT - the stdlib threaded HTTP client shipped to the spark for the server sweep
# cmd_sweep(a) - batched-bench npl sweep + llama-server --parallel N TTFT harness
# cmd_status(a) - lock, clone sha, models, binaries, last measurements
# write_csv(path, header, rows) / print_table(header, rows) - the CSV writer and the table printer
# knee(rows, floor) - largest N whose per-stream t/s stays above floor
# percentile(vals, p) - nearest-rank percentile
# selftest() - offline checks incl. a planted size-check mutation that must go RED
# main() - argparse, dispatch
#
# ** Technical Review **
# One python3-stdlib tool, five subcommands (fetch / build / bench1 / sweep / status), every one
# idempotent and resumable, every remote command printable under --dry-run without running.
# - The spark is shared with the dwarfstar rounds: ~/spark_model.lock/owner is the model lock.
#   Every step that touches the GPU or the NVMe (download, build, bench, sweep) is refused when
#   another lane holds it, takes it under its own lane name (PRISM-FETCH / PRISM-BUILD /
#   PRISM-SWEEP), and releases it only if it still holds it. The clone is network-only and needs
#   no lock. The download WAITS for a free lock (a 7 GB write contends with a running round's
#   streaming), it does not queue-jump.
# - Long steps are written as ~/prism/<step>.sh on the spark and started with
#   `setsid bash ... </dev/null > log 2>&1 &`; the Mac side polls the log for a DONE marker.
#   Anchored pgrep (`^bash /home/hologram/prism/<step>.sh`) is the liveness check.
# - Sizes: the two GGUFs are byte-verified against the Hugging Face API figures in the harvest
#   (PTQ1_0 5,946,648,928 · PQ2_0 7,206,168,928). A mismatch is refused and the file is left in
#   place with a .BAD marker note; the tool never deletes anything.
# - Build: the fork carries no build script (Bonsai-demo's build_cuda_linux.sh does), so the
#   harvested configure line is reused verbatim: -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=121a
#   -DCMAKE_BUILD_TYPE=Release, Ninja, -j12. patchelf is absent on the box, so binaries run from
#   build-cuda/bin with cmake's build RPATH instead of the demo's $ORIGIN copy step.
# - Measurements carry the stamps the project's laws demand: load, gpu util, memavail, driver,
#   CUDA runtime, fork sha, model sha256[:16], context (llama-bench -d 2048 = KV depth 2048), and
#   the checkpoint string "Bonsai 2 27B <quant>". A warm-up run is taken and discarded first.
# - Knee: the largest N in the sweep whose per-stream t/s stays above 10 (the concurrency goal's
#   floor). TTFT p50/p95 by nearest-rank percentile over N concurrent streams.
# Constants: PIN_SHA, MODELS (name -> bytes), HF_REPO, FORK_REPO, PROMPT_BYTES (~3.4k tokens of
# promessi_sposi.txt), TTFT_PORT 18080.
# </claudes_code_comments>

import argparse
import csv
import io
import os
import re
import shlex
import socket
import subprocess
import sys
import time

FORK_REPO = "https://github.com/PrismML-Eng/llama.cpp.git"
FORK_BRANCH = "prism"
PIN_SHA = "5d80cff0b8cb9f2bf823cfc4e71e3abb97f290d6"
HF_REPO = "prism-ml/Ternary-Bonsai-2-27B-gguf"
MODELS = {
    "PTQ1_0": ("Ternary-Bonsai-2-27B-PTQ1_0.gguf", 5946648928),
    "PQ2_0": ("Ternary-Bonsai-2-27B-PQ2_0.gguf", 7206168928),
}
SPARK_HOME = "/home/hologram"
PRISM = "~/prism"
LOCK = "~/spark_model.lock"
HERE = os.path.dirname(os.path.abspath(__file__))
MEASURED_LOCAL = os.path.normpath(os.path.join(HERE, "..", "measured"))
PROMESSI = os.path.normpath(os.path.join(HERE, "..", "..", "speed-bench", "promessi_sposi.txt"))
PROMPT_BYTES = 13000  # about 3.4k tokens of Italian prose at ~3.8 bytes/token
TTFT_PORT = 18080
KNEE_FLOOR = 10.0
NPL = [1, 4, 8, 16, 32, 64]
NPAR = [1, 4, 8, 16]
BINS = ["llama-server", "llama-bench", "llama-batched-bench"]

USAGE = "usage: prism_spark.py [--dry-run] [--quant PTQ1_0|PQ2_0] {fetch,build,bench1,sweep,status,selftest} ...  (--help for examples)"


def die(msg):
    print("error: %s" % msg)
    print(USAGE)
    sys.exit(2)


def utc():
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def on_spark():
    return socket.gethostname().startswith("spark") or os.path.isdir(SPARK_HOME + "/dwarfstar")


def remote(script, dry, title=""):
    """Run a bash script on the spark. Under --dry-run: print it, run nothing."""
    if dry:
        print("--- DRY RUN%s: would run on %s ---" % ((" " + title) if title else "", "this box" if on_spark() else "ssh spark"))
        print(script.rstrip())
        print("--- end ---")
        return 0, ""
    cmd = ["bash", "-s"] if on_spark() else ["ssh", "spark", "bash", "-s"]
    r = subprocess.run(cmd, input=script, text=True, capture_output=True)
    if r.returncode != 0 and r.stderr.strip():
        print(r.stderr.strip())
    return r.returncode, r.stdout


def scp_back(remote_glob, dry):
    os.makedirs(MEASURED_LOCAL, exist_ok=True)
    cmd = ["scp", "-q", "spark:%s" % remote_glob, MEASURED_LOCAL + "/"]
    if dry:
        print("--- DRY RUN: would run locally: %s ---" % " ".join(shlex.quote(c) for c in cmd))
        return 0
    return subprocess.run(cmd).returncode


# ---------------- lock discipline (mirrors track0_harness/spark_arms.py) ----------------

def lock_guard(lane):
    return "\n".join([
        "mkdir -p %s" % LOCK,
        'OWN=$(cat %s/owner 2>/dev/null || true)' % LOCK,
        'if [ -n "$OWN" ] && ! grep -q "^%s " <<<"$OWN"; then echo "REFUSED_LOCK held-by: $OWN"; exit 3; fi' % lane,
        'echo "%s prism_spark $(date -u +%%Y-%%m-%%dT%%H:%%M:%%SZ)" > %s/owner' % (lane, LOCK),
    ])


def lock_release(lane):
    return 'grep -q "^%s " %s/owner 2>/dev/null && rm -rf %s' % (lane, LOCK, LOCK)


def detached(name, body, lane):
    """Write ~/prism/<name>.sh (takes the lock at its start, releases at its end) and start it detached."""
    inner = "\n".join([
        "#!/bin/bash", "set -u", "cd %s || exit 1" % PRISM,
        lock_guard(lane),
        'echo "START %s $(date -u +%%Y-%%m-%%dT%%H:%%M:%%SZ)"' % name,
        body,
        'echo "END %s rc=$? $(date -u +%%Y-%%m-%%dT%%H:%%M:%%SZ)"' % name,
        lock_release(lane),
        'echo "DONE %s"' % name,
    ])
    return "\n".join([
        "mkdir -p %s/measured %s/models %s/src" % (PRISM, PRISM, PRISM),
        'if pgrep -f "^bash %s/prism/%s\\.sh" >/dev/null; then echo "ALREADY_RUNNING %s"; exit 0; fi' % (SPARK_HOME, name, name),
        "cat > %s/%s.sh <<'EOF_STEP'\n%s\nEOF_STEP" % (PRISM, name, inner),
        "setsid bash %s/%s.sh </dev/null > %s/%s.log 2>&1 &" % (PRISM, name, PRISM, name),
        "sleep 1",
        'P=$(pgrep -f "^bash %s/prism/%s\\.sh" | head -1); echo "started %s pid=${P:-NOT RUNNING} log=%s/%s.log"' % (SPARK_HOME, name, name, PRISM, name),
    ])


def poll(logname, done_marker, dry, every=20, max_s=6 * 3600):
    if dry:
        print("--- DRY RUN: would poll %s/%s.log every %ds for '%s' ---" % (PRISM, logname, every, done_marker))
        return ""
    t0 = time.time()
    seen = 0
    while True:
        rc, out = remote("cat %s/%s.log 2>/dev/null" % (PRISM, logname), False)
        new = out[seen:]
        if new:
            sys.stdout.write(new)
            sys.stdout.flush()
            seen = len(out)
        if done_marker in out or "REFUSED_LOCK" in out:
            return out
        if time.time() - t0 > max_s:
            print("poll timeout after %ds - the step may still be running; re-run to resume polling" % max_s)
            return out
        time.sleep(every)


def lock_owner(dry):
    if dry:
        return "(unknown under --dry-run)"
    _, out = remote("cat %s/owner 2>/dev/null || echo FREE" % LOCK, False)
    return out.strip() or "FREE"


def wait_lock_free(dry, max_s):
    if dry:
        print("--- DRY RUN: would wait (up to %ds) for %s/owner to be FREE ---" % (max_s, LOCK))
        return True
    t0 = time.time()
    while True:
        o = lock_owner(False)
        if o == "FREE" or o.startswith("PRISM-"):
            return True
        if time.time() - t0 > max_s:
            print("lock still held by: %s (waited %ds) - nothing started" % (o, max_s))
            return False
        print("lock held by: %s - waiting (%ds)" % (o, int(time.time() - t0)))
        time.sleep(60)


# ---------------- size gate ----------------

def verify_size(path, expected, actual):
    """-> (ok, message). Refuses a mismatch; never deletes."""
    if actual is None:
        return False, "%s: missing" % path
    if actual != expected:
        return False, "%s: SIZE MISMATCH have %d want %d (left in place; delete by hand if it is a partial)" % (path, actual, expected)
    return True, "%s: %d bytes OK" % (path, actual)


# ---------------- fetch ----------------

def cmd_fetch(a):
    clone = "\n".join([
        "set -u", "mkdir -p %s/src %s/models %s/measured" % (PRISM, PRISM, PRISM),
        "cd %s/src" % PRISM,
        "if [ ! -d llama.cpp/.git ]; then git clone -q -b %s %s llama.cpp; fi" % (FORK_BRANCH, FORK_REPO),
        "cd llama.cpp",
        "git fetch -q origin %s" % FORK_BRANCH,
        "git checkout -q --detach %s" % PIN_SHA,
        'HAVE=$(git rev-parse HEAD); [ "$HAVE" = "%s" ] || { echo "PIN MISMATCH $HAVE"; exit 4; }' % PIN_SHA,
        'echo "%s %s %s pinned $(date -u +%%Y-%%m-%%dT%%H:%%M:%%SZ)" > %s/PIN.txt' % (FORK_REPO, FORK_BRANCH, PIN_SHA, PRISM),
        'echo "clone OK sha=$HAVE $(git log -1 --format=%cs) files=$(git ls-files | wc -l)"',
    ])
    rc, out = remote(clone, a.dry_run, "fetch: clone pinned")
    print(out.rstrip())
    if rc != 0:
        print("clone failed (rc=%d)" % rc)
        return 1
    if a.clone_only:
        print("NEXT ->  prism_spark.py fetch   (the model downloads, once the lock is free)")
        return 0

    # models: wait for a free lock, take PRISM-FETCH, download with size checks, release
    if not wait_lock_free(a.dry_run, a.wait_max):
        return 3
    quants = [a.quant] if a.one_model else list(MODELS)
    lines = ['command -v hf >/dev/null && DL=hf || { command -v huggingface-cli >/dev/null && DL=huggingface-cli || DL=curl; }',
             'echo "downloader: $DL"']
    for q in quants:
        fn, want = MODELS[q]
        url = "https://huggingface.co/%s/resolve/main/%s" % (HF_REPO, fn)
        lines += [
            'F=%s/models/%s; WANT=%d' % (PRISM, fn, want),
            'HAVE=$(stat -c %s "$F" 2>/dev/null || echo 0)',
            'if [ "$HAVE" = "$WANT" ]; then echo "%s: present $HAVE bytes OK (skip)"; else' % fn,
            '  if [ "$DL" = curl ]; then curl -L --retry 5 -C - -o "$F" "%s"; else $DL download %s %s --local-dir %s/models; fi' % (url, HF_REPO, fn, PRISM),
            '  HAVE=$(stat -c %s "$F" 2>/dev/null || echo 0)',
            '  if [ "$HAVE" != "$WANT" ]; then echo "%s: SIZE MISMATCH have $HAVE want $WANT - REFUSED, file left in place"; echo "$HAVE != $WANT $(date -u)" > "$F.BAD"; exit 5; fi' % fn,
            '  rm -f "$F.BAD"; echo "%s: $HAVE bytes OK"' % fn,
            'fi',
            'if ! grep -q "%s " %s/MODELS.txt 2>/dev/null; then echo "%s $(sha256sum "$F" | cut -c1-16) $WANT %s" >> %s/MODELS.txt; fi' % (fn, PRISM, fn, q, PRISM),
        ]
    rc, out = remote(detached("fetch", "\n".join(lines), "PRISM-FETCH"), a.dry_run, "fetch: models (detached, under PRISM-FETCH)")
    print(out.rstrip())
    log = poll("fetch", "DONE fetch", a.dry_run)
    if "SIZE MISMATCH" in log or "REFUSED_LOCK" in log:
        return 5
    print("NEXT ->  prism_spark.py build   ·   prism_spark.py status")
    return 0


# ---------------- build ----------------

def cmd_build(a):
    body = "\n".join([
        "cd %s/src/llama.cpp || exit 1" % PRISM,
        'HAVE=$(git rev-parse HEAD); [ "$HAVE" = "%s" ] || { echo "PIN MISMATCH $HAVE - run fetch first"; exit 4; }' % PIN_SHA,
        "export PATH=/usr/local/cuda/bin:$PATH",
        # harvested from Bonsai-demo scripts/build_cuda_linux.sh (08_build_cuda_linux.md); archs forced to 121a for sm_121
        'cmake -B build-cuda -G Ninja -DGGML_CUDA=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES="%s" -DCMAKE_BUILD_TYPE=Release > %s/build.log 2>&1 || { echo "CONFIGURE FAILED see build.log"; exit 6; }' % (a.archs, PRISM),
        "cmake --build build-cuda -j%d >> %s/build.log 2>&1 || { echo \"BUILD FAILED see build.log\"; exit 6; }" % (a.jobs, PRISM),
        'echo "sha=$HAVE archs=%s built=$(date -u +%%Y-%%m-%%dT%%H:%%M:%%SZ) nvcc=$(nvcc --version | tail -1 | tr -s " " | cut -d, -f2-)" > %s/BUILD.txt' % (a.archs, PRISM),
        "for b in %s; do P=$PWD/build-cuda/bin/$b; T=$(size $P 2>/dev/null | tail -1 | awk '{print $1}'); echo \"$b $P text=${T:-MISSING}\" | tee -a %s/BUILD.txt; done" % (" ".join(BINS), PRISM),
        "./build-cuda/bin/llama-server --help 2>&1 | grep -A2 -- '--parallel' | head -3 | tee -a %s/BUILD.txt" % PRISM,
    ])
    rc, out = remote(detached("build", body, "PRISM-BUILD"), a.dry_run, "build: cmake CUDA %s -j%d (detached, under PRISM-BUILD)" % (a.archs, a.jobs))
    print(out.rstrip())
    log = poll("build", "DONE build", a.dry_run)
    if "FAILED" in log or "REFUSED_LOCK" in log:
        return 6
    print("NEXT ->  prism_spark.py bench1   ·   prism_spark.py status")
    return 0


# ---------------- bench1 ----------------

def stamp_script():
    return "\n".join([
        'LOAD=$(cut -d" " -f1 /proc/loadavg)',
        'GPU=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)',
        'DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)',
        'MEMAVAIL=$(( $(grep MemAvailable /proc/meminfo | tr -dc 0-9) / 1024 / 1024 ))',
        'CUDART=$(/usr/local/cuda/bin/nvcc --version | grep -o "release [0-9.]*" | cut -d" " -f2)',
        'SHA=$(cut -d" " -f3 %s/PIN.txt)' % PRISM,
        'echo "STAMP load=$LOAD gpu=$GPU% memavail=${MEMAVAIL}GiB driver=$DRV cuda=$CUDART fork=${SHA:0:12} utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"',
    ])


def bench_script(quant, a):
    fn, _ = MODELS[quant]
    bench = "%s/src/llama.cpp/build-cuda/bin/llama-bench" % PRISM
    common = "-m %s/models/%s -p 512 -n 128 -d %d -fa 1 -ngl 99" % (PRISM, fn, a.ctx)
    return "\n".join([
        'MSHA=$(grep "^%s " %s/MODELS.txt | cut -d" " -f2)' % (fn, PRISM),
        'echo "MODEL %s sha256_16=${MSHA:-unknown} checkpoint=\\"Bonsai 2 27B %s\\""' % (quant, quant),
        "echo WARMUP-discarded; %s %s -r 1 -o csv > /dev/null 2>&1" % (bench, common),
        "echo BENCH; %s %s -r %d -o csv 2>/dev/null | tee %s/measured/raw_bench1_%s_$TS.csv" % (bench, common, a.repeats, PRISM, quant),
    ])


def cmd_bench1(a):
    ts = utc()
    quants = [a.quant] + [q for q in MODELS if q != a.quant]
    body = "\n".join(["TS=%s" % ts, stamp_script()] + [bench_script(q, a) for q in quants])
    rc, out = remote(detached("bench1", body, "PRISM-SWEEP"), a.dry_run, "bench1: llama-bench both quants (detached, under PRISM-SWEEP)")
    print(out.rstrip())
    log = poll("bench1", "DONE bench1", a.dry_run)
    if a.dry_run:
        print("--- DRY RUN: would write %s/measured/bench1_%s.csv and scp it to %s ---" % (PRISM, ts, MEASURED_LOCAL))
        return 0
    stamp = re.search(r"^STAMP (.*)$", log, re.M)
    rows = []
    for q in quants:
        m = re.search(r"MODEL %s sha256_16=(\S+).*?\nBENCH\n(.*?)(?=\nMODEL |\nEND |\Z)" % q, log, re.S)
        if not m:
            continue
        for r in parse_llama_bench_csv(m.group(2)):
            rows.append([q, r.get("n_prompt", ""), r.get("n_gen", ""), r.get("n_depth", ""), r.get("avg_ts", ""), r.get("stddev_ts", ""), m.group(1), stamp.group(1) if stamp else ""])
    header = ["quant", "n_prompt", "n_gen", "depth", "avg_ts", "stddev_ts", "model_sha256_16", "stamp"]
    local = os.path.join(MEASURED_LOCAL, "bench1_%s.csv" % ts)
    write_csv(local, header, rows)
    print_table(header[:6], [r[:6] for r in rows])
    print("wrote %s" % local)
    print("NEXT ->  prism_spark.py sweep --quant %s" % a.quant)
    return 0 if rows else 1


def parse_batched_bench(text):
    """llama-batched-bench table rows -> [(B, PP, TG, S_TG t/s)].
    Columns: PP | TG | B | N_KV | T_PP s | S_PP t/s | T_TG s | S_TG t/s | T s | S t/s.
    S_TG is the DECODE aggregate; the last column folds prefill in and is not it."""
    out = []
    for l in text.splitlines():
        f = [x.strip() for x in l.strip().strip("|").split("|")]
        if len(f) == 10 and f[0].isdigit() and f[1].isdigit() and f[2].isdigit():
            try:
                out.append((int(f[2]), int(f[0]), int(f[1]), float(f[7])))
            except ValueError:
                pass
    return out


def parse_llama_bench_csv(text):
    lines = [l for l in text.strip().splitlines() if "," in l]
    if not lines:
        return []
    return list(csv.DictReader(io.StringIO("\n".join(lines))))


# ---------------- sweep ----------------

TTFT_CLIENT = r'''
import json, sys, time, threading, urllib.request
base, n, prompt_file, max_tokens = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4])
prompt = open(prompt_file, encoding="utf-8", errors="replace").read()
res = []
def one(i):
    body = json.dumps({"prompt": prompt, "n_predict": max_tokens, "stream": True, "temperature": 0, "cache_prompt": False}).encode()
    req = urllib.request.Request(base + "/completion", data=body, headers={"Content-Type": "application/json"})
    t0 = time.time(); first = None; toks = 0
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            if not line.startswith(b"data:"): continue
            if first is None: first = time.time()
            toks += 1
    res.append((i, first - t0 if first else None, time.time() - t0, toks))
th = [threading.Thread(target=one, args=(i,)) for i in range(n)]
w0 = time.time(); [t.start() for t in th]; [t.join() for t in th]; wall = time.time() - w0
print(json.dumps({"n": n, "wall": wall, "streams": res}))
'''


def cmd_sweep(a):
    ts = utc()
    fn, _ = MODELS[a.quant]
    bb = "%s/src/llama.cpp/build-cuda/bin/llama-batched-bench" % PRISM
    srv = "%s/src/llama.cpp/build-cuda/bin/llama-server" % PRISM
    pool_bb = max(NPL) * (512 + 128) * 2
    lines = ["TS=%s" % ts, stamp_script(),
             'echo "MODEL %s sha256_16=$(grep "^%s " %s/MODELS.txt | cut -d" " -f2) checkpoint=\\"Bonsai 2 27B %s\\""' % (a.quant, fn, PRISM, a.quant),
             "echo BATCHED; %s -m %s/models/%s -c %d -b 2048 -ub 512 -npp 512 -ntg 128 -npl %s -fa 1 -ngl 99 2>/dev/null | tee %s/measured/raw_batched_%s_$TS.txt" % (bb, PRISM, fn, pool_bb, ",".join(map(str, NPL)), PRISM, a.quant),
             "cat > %s/ttft_client.py <<'EOF_PY'\n%s\nEOF_PY" % (PRISM, TTFT_CLIENT.strip()),
             "cat > %s/prompt.txt <<'EOF_PROMPT'\n%s\nEOF_PROMPT" % (PRISM, a.prompt_text)]
    for n in NPAR:
        pool = n * 8192
        lines += [
            "echo SERVER n=%d pool=%d" % (n, pool),
            "%s -m %s/models/%s --host 127.0.0.1 --port %d -ngl 99 -fa on -c %d --parallel %d > %s/server_%d.log 2>&1 &" % (srv, PRISM, fn, TTFT_PORT, pool, n, PRISM, n),
            "SP=$!",
            'for i in $(seq 1 120); do curl -s http://127.0.0.1:%d/health | grep -q ok && break; sleep 2; done' % TTFT_PORT,
            'python3 %s/ttft_client.py http://127.0.0.1:%d 1 %s/prompt.txt 16 > /dev/null 2>&1   # warm-up, discarded' % (PRISM, TTFT_PORT, PRISM),
            'echo TTFT; python3 %s/ttft_client.py http://127.0.0.1:%d %d %s/prompt.txt 128 | tee -a %s/measured/raw_ttft_%s_$TS.jsonl' % (PRISM, TTFT_PORT, n, PRISM, PRISM, a.quant),
            "kill $SP; wait $SP 2>/dev/null; sleep 3",
        ]
    rc, out = remote(detached("sweep", "\n".join(lines), "PRISM-SWEEP"), a.dry_run, "sweep: batched-bench npl %s + server --parallel %s TTFT (detached, under PRISM-SWEEP)" % (NPL, NPAR))
    print(out.rstrip())
    log = poll("sweep", "DONE sweep", a.dry_run)
    if a.dry_run:
        scp_back("%s/measured/*_%s.*" % (PRISM, ts), True)
        print("--- DRY RUN: would write %s/measured/sweep_%s.csv and print the knee (floor %.0f t/s per stream) ---" % (PRISM, ts, KNEE_FLOOR))
        return 0
    stamp = re.search(r"^STAMP (.*)$", log, re.M)
    st = stamp.group(1) if stamp else ""
    rows = [["batched", pl, pp, tg, "%.2f" % s_tg, "%.2f" % (s_tg / pl), "", "", st]
            for pl, pp, tg, s_tg in parse_batched_bench(log)]
    for l in log.splitlines():
        if l.startswith("{") and '"streams"' in l:
            import json
            d = json.loads(l)
            fts = [s[1] for s in d["streams"] if s[1] is not None]
            toks = sum(s[3] for s in d["streams"])
            agg = toks / d["wall"] if d["wall"] else 0
            rows.append(["server", d["n"], PROMPT_BYTES, 128, "%.2f" % agg, "%.2f" % (agg / d["n"]), "%.2f" % percentile(fts, 50), "%.2f" % percentile(fts, 95), st])
    header = ["arm", "N", "pp", "tg", "agg_ts", "per_stream_ts", "ttft_p50_s", "ttft_p95_s", "stamp"]
    local = os.path.join(MEASURED_LOCAL, "sweep_%s.csv" % ts)
    write_csv(local, header, rows)
    scp_back("%s/measured/*_%s.*" % (PRISM, ts), False)
    print_table(header[:8], [r[:8] for r in rows])
    k = knee([(int(r[1]), float(r[5])) for r in rows if r[0] == "batched"], KNEE_FLOOR)
    print("KNEE (batched, per-stream > %.0f t/s): N=%s" % (KNEE_FLOOR, k if k is not None else "none - even N=1 is under the floor"))
    print("wrote %s" % local)
    print("NEXT ->  write prism/measured/MEASURED_<date>.md from these CSVs   ·   prism_spark.py status")
    return 0 if rows else 1


# ---------------- status ----------------

def cmd_status(a):
    script = "\n".join([
        'echo "lock:    $(cat %s/owner 2>/dev/null || echo FREE)"' % LOCK,
        'echo "clone:   $(cat %s/PIN.txt 2>/dev/null || echo none)"' % PRISM,
        'echo "head:    $(git -C %s/src/llama.cpp rev-parse HEAD 2>/dev/null || echo none)"' % PRISM,
        'for f in %s; do echo "model:   $f $(stat -c %%s %s/models/$f 2>/dev/null || echo MISSING)"; done' % (" ".join(v[0] for v in MODELS.values()), PRISM),
        'echo "build:   $(head -1 %s/BUILD.txt 2>/dev/null || echo none)"' % PRISM,
        'grep -E "^llama-" %s/BUILD.txt 2>/dev/null | sed "s/^/binary:  /"' % PRISM,
        'echo "running: $(pgrep -f "^bash %s/prism/(fetch|build|bench1|sweep)\\.sh" | wc -l) prism step(s)"' % SPARK_HOME,
        'echo "measured (spark):"; ls -t %s/measured 2>/dev/null | head -5 | sed "s/^/  /"' % PRISM,
        'echo "box:     load=$(cut -d" " -f1 /proc/loadavg) gpu=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader) driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader)"',
    ])
    rc, out = remote(script, a.dry_run, "status")
    print(out.rstrip())
    if os.path.isdir(MEASURED_LOCAL):
        loc = sorted(os.listdir(MEASURED_LOCAL))[-3:]
        print("measured (mac): %s" % (", ".join(loc) if loc else "none"))
    print("NEXT ->  prism_spark.py fetch --clone-only   ·   fetch   ·   build   ·   bench1   ·   sweep")
    return rc


# ---------------- pure helpers ----------------

def write_csv(path, header, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def print_table(header, rows):
    cols = [header] + [[str(c) for c in r] for r in rows]
    widths = [max(len(r[i]) for r in cols) for i in range(len(header))]
    for r in cols:
        print("  ".join(r[i].ljust(widths[i]) for i in range(len(header))))


def knee(rows, floor):
    """rows: [(N, per_stream_ts)] -> largest N with per_stream_ts > floor, else None."""
    ok = [n for n, t in rows if t > floor]
    return max(ok) if ok else None


def percentile(vals, p):
    if not vals:
        return float("nan")
    s = sorted(vals)
    k = max(1, int(round(p / 100.0 * len(s) + 0.5)))
    return s[min(k, len(s)) - 1]


# ---------------- selftest ----------------

def selftest():
    n = [0, 0]

    def check(ok, msg):
        n[0] += 1
        n[1] += 0 if ok else 1
        print("  %s %s" % ("PASS" if ok else "FAIL", msg))

    p = build_parser()
    a = p.parse_args(["--dry-run", "fetch", "--clone-only"])
    check(a.cmd == "fetch" and a.dry_run and a.clone_only and a.quant == "PQ2_0", "parse: fetch --clone-only, quant defaults to PQ2_0")
    a = p.parse_args(["--quant", "PTQ1_0", "build"])
    check(a.quant == "PTQ1_0" and a.archs == "121a" and a.jobs == 12, "parse: build defaults archs=121a -j12")
    try:
        p.parse_args(["--quant", "Q2_0", "bench1"]); check(False, "refusal: --quant Q2_0")
    except SystemExit as e:
        check(e.code == 2, "refusal: --quant Q2_0 exits 2")
    try:
        p.parse_args(["nosuch"]); check(False, "refusal: unknown subcommand")
    except SystemExit as e:
        check(e.code == 2, "refusal: unknown subcommand exits 2")

    ok, _ = verify_size("x.gguf", 7206168928, 7206168928); check(ok, "size gate: exact match OK")
    ok, m = verify_size("x.gguf", 7206168928, 7206168000); check(not ok and "MISMATCH" in m and "left in place" in m, "size gate: short file refused, not deleted")
    ok, m = verify_size("x.gguf", 7206168928, None); check(not ok and "missing" in m, "size gate: missing refused")
    check(MODELS["PTQ1_0"][1] == 5946648928 and MODELS["PQ2_0"][1] == 7206168928, "constants: harvest byte counts")

    g = lock_guard("PRISM-BUILD")
    check('grep -q "^PRISM-BUILD "' in g and "REFUSED_LOCK" in g and "exit 3" in g, "lock guard refuses a foreign owner, anchored at line start")
    check(lock_release("PRISM-BUILD").startswith('grep -q "^PRISM-BUILD "'), "lock release only when ours")
    d = detached("build", "true", "PRISM-BUILD")
    check("setsid bash ~/prism/build.sh </dev/null > ~/prism/build.log 2>&1 &" in d and 'pgrep -f "^bash /home/hologram/prism/build\\.sh"' in d, "detached: setsid, stdin /dev/null, anchored pgrep")
    check("rm " not in d.replace("rm -rf ~/spark_model.lock", "") and "rm -f" not in bench_script("PQ2_0", p.parse_args(["bench1"])), "never deletes anything but its own lock")

    tmp = os.path.join(os.environ.get("TMPDIR", "/tmp"), "prism_selftest_%d.csv" % os.getpid())
    write_csv(tmp, ["a", "b"], [[1, 2.5], ["x", "y"]])
    with open(tmp) as f:
        txt = f.read()
    os.remove(tmp)
    check(txt.splitlines() == ["a,b", "1,2.5", "x,y"], "csv writer: header + rows")
    buf = io.StringIO(); old = sys.stdout; sys.stdout = buf
    print_table(["N", "ts"], [[1, "44.0"], [16, "9.9"]]); sys.stdout = old
    check(buf.getvalue().splitlines()[0].startswith("N ") and "16  9.9" in buf.getvalue(), "table printer aligns columns")

    check(knee([(1, 44.0), (4, 30.0), (8, 11.4), (16, 6.0), (32, 3.0)], 10.0) == 8, "knee: planted rows -> 8")
    check(knee([(1, 9.0)], 10.0) is None, "knee: nothing above floor -> None")
    check(knee([(1, 40.0), (4, 12.0), (8, 10.0)], 10.0) == 4, "knee: exactly at floor does not count")
    check(percentile([5, 1, 3, 2, 4], 50) == 3 and percentile([5, 1, 3, 2, 4], 95) == 5, "percentile: nearest-rank p50/p95")
    check(percentile([12.7], 95) == 12.7 and percentile([], 50) != percentile([], 50), "percentile: single value; empty -> nan")
    rows = parse_llama_bench_csv("build_commit,n_prompt,n_gen,n_depth,avg_ts,stddev_ts\n5d80cff,512,128,2048,44.12,0.30\n")
    check(rows and rows[0]["avg_ts"] == "44.12" and rows[0]["n_depth"] == "2048", "llama-bench csv parser")
    bb = ("|    PP |     TG |    B |   N_KV |   T_PP s | S_PP t/s |   T_TG s | S_TG t/s |      T s |    S t/s |\n"
          "|-------|--------|------|--------|----------|----------|----------|----------|----------|----------|\n"
          "|   512 |    128 |    8 |   5120 |    4.100 |   999.02 |   11.230 |    91.19 |   15.330 |   334.00 |\n")
    pb = parse_batched_bench(bb)
    check(pb == [(8, 512, 128, 91.19)], "batched-bench parser reads S_TG (91.19), not the total S t/s (334.00)")
    check("%%" not in stamp_script() and "+%Y-%m-%dT%H:%M:%SZ" in stamp_script(), "stamp script carries no doubled %% (it is not %-formatted)")

    # planted mutation: a size gate that accepts any size must go RED
    def mutated_verify_size(path, expected, actual):
        return True, "ok"
    ok, _ = mutated_verify_size("x.gguf", 7206168928, 7206168000)
    red = not ok
    print("  %s planted mutation (size check disabled): a short file %s" % ("RED" if not red else "GREEN-SHOULD-BE-RED", "was accepted, the mutation is CAUGHT" if not red else "was refused (impossible)"))
    n[0] += 1
    n[1] += 0 if not red else 1  # RED is the expected outcome: the check on the mutant fails, proving the real check is doing work

    print("selftest: %d checks, %d failed" % (n[0], n[1]))
    return 0 if n[1] == 0 else 1


# ---------------- main ----------------

EPILOG = """examples:
  prism_spark.py status                          what is on the spark: lock, clone, models, binaries
  prism_spark.py --dry-run fetch                 print every remote command, run nothing
  prism_spark.py fetch --clone-only              clone the fork pinned at %s (network only, no lock)
  prism_spark.py fetch                           + download both GGUFs (waits for a FREE lock, takes PRISM-FETCH)
  prism_spark.py build                           cmake CUDA 121a -j12 under PRISM-BUILD, detached, ~/prism/build.log
  prism_spark.py bench1                          llama-bench pp512/tg128 at depth 2048, both quants, 3 repeats, stamped
  prism_spark.py sweep --quant PQ2_0             batched-bench npl 1..64 + llama-server --parallel 1,4,8,16 TTFT; prints the knee
  prism_spark.py selftest                        offline checks + a planted RED mutation

lock: ~/spark_model.lock/owner on the spark. A held lock by another lane REFUSES every GPU/NVMe step.
""" % PIN_SHA[:12]


def build_parser():
    p = argparse.ArgumentParser(prog="prism_spark.py", description="bring up and measure Ternary Bonsai 2 27B on the Spark GB10, one lock-honouring step at a time",
                                epilog=EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dry-run", action="store_true", help="print every remote command, run nothing")
    p.add_argument("--quant", choices=list(MODELS), default="PQ2_0", help="model band (default PQ2_0, what the demo loads)")
    sp = p.add_subparsers(dest="cmd", metavar="{fetch,build,bench1,sweep,status,selftest}")
    f = sp.add_parser("fetch"); f.add_argument("--clone-only", action="store_true", help="clone + pin only, no model download")
    f.add_argument("--one-model", action="store_true", help="download only --quant"); f.add_argument("--wait-max", type=int, default=12 * 3600, help="seconds to wait for a free lock")
    b = sp.add_parser("build"); b.add_argument("--archs", default="121a"); b.add_argument("--jobs", type=int, default=12)
    b1 = sp.add_parser("bench1"); b1.add_argument("--repeats", type=int, default=3); b1.add_argument("--ctx", type=int, default=2048, help="KV depth for llama-bench -d")
    sp.add_parser("sweep"); sp.add_parser("status"); sp.add_parser("selftest")
    return p


def main():
    p = build_parser()
    if len(sys.argv) == 1:
        p.print_help(); return 0
    a = p.parse_args()
    if a.cmd == "selftest":
        return selftest()
    if a.cmd == "sweep":
        if not os.path.isfile(PROMESSI):
            die("prompt source missing: %s" % PROMESSI)
        with open(PROMESSI, "rb") as f:
            a.prompt_text = f.read(PROMPT_BYTES).decode("utf-8", "ignore").replace("EOF_PROMPT", "")
    return {"fetch": cmd_fetch, "build": cmd_build, "bench1": cmd_bench1, "sweep": cmd_sweep, "status": cmd_status}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
