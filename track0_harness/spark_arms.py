#!/usr/bin/env python3
"""spark_arms - run one measurement round of triple-* arms against a tip on the DGX Spark.

Docs: track0_harness/SPARK_ARMS.md · the spark rules: OVERNIGHT_2026-09-17_MEMORY.md §2-3
Usage:  spark_arms.py run [ARM ...]       (plan + seal check + build + run, lean defaults)
        spark_arms.py --help              (every subcommand has its own --help)
"""
# <claudes_code_comments>
# ** Function List **
# Usage(Exception) - a bad-param error: one line, then usage, exit 2
# sh(cmd, capture) - run a shell string locally
# git(*args) - run git in the repo root, return stdout
# branch_refs() - {triple-* branch: short sha} from git for-each-ref
# carded_branches(refs) - triple-* branches whose README carries a card box, minus CONTROL/TOOLING
# norm_branch(name) - accept a name with or without the triple- prefix
# parse_arm_spec(spec) - "branch@DS4_X=1,DS4_Y=0;cache=60GB" -> (branch, env, args); bad syntax is a Usage
# parse_bench_args(text, spec) - "cache=60GB,gen=256" -> [(key, value)] from the closed set, or a Usage
# env_suffix(env) - [(VAR, value)] -> the short label suffix (+hits, -hold, margin12)
# args_suffix(args) - [(key, value)] -> the label suffix (~c60, ~g256, ~x8192)
# env_str(env) - [(VAR, value)] -> "DS4_X=1 DS4_Y=0"
# args_str(args) - [(key, value)] -> "cache=60GB gen=256"; plan_tokens(args) -> "ARG:cache=60GB ARG:gen=256"
# short_label(branch, taken) - branch -> unique short label (hitsfirst, hotlist ...)
# parse_frontiers(text) - "4096,6144" -> [4096, 6144], refusing a list that drops 6144
# is_warmup(label) - "r8_tip_warm" -> True, the discarded warm-up run
# make_plan(names, refs, tip, cycles, frontiers, adjacent, name, warmup) - the round design as a dict
# plan_sequence(plan) - the ordered run list [(worktree, label)], warm-up first
# wall_estimate(plan) - runs (warm-up included) x measured seconds, with its provenance
# runs_phrase(plan) - "33 runs + 1 warm-up" for plan, seal and run
# print_plan(plan) - the plan block a human reads
# plan_file_text(plan, seal) - the plan file the driver reads
# seal_text(plan) - the PREREGISTERED-RULE file from a plan
# seal_path(plan, when) - YYYY-MM-DD-<NAME>-PREREGISTERED-RULE.txt in the repo root
# parse_seal(text) - the arms+shas block back out of a seal file
# is_committed(path) - git log -- path is non-empty (a seal is a commit-ordering property)
# check_seal(plan, path) - refuse unless committed AND its arms/shas equal the plan's
# on_spark() - True when ~/dwarfstar/_worktrees exists locally
# remote(script, dry) - run a bash script on the spark (or here), or print it under --dry-run
# sparkbox_ok(url) - the push target must be the spark or the TripleSparkleAI fork
# build_script(plan) - push shas, refresh detached worktrees, make cuda-spark, log rc + .text
# parse_build_log(text) - {worktree: (rc, text_bytes, sha)}
# text_gate(plan, built) - arms whose .text disagrees with the tip's by > 0.2 MB are REFUSED
# bench_argv_lines(mutate) - the one-script's argv block: defaults, ARG: tokens REPLACE a flag, env tokens go to env
# lane_scripts(plan, mutate) - the one/driver/watchdog scripts, lane-named, setsid-safe
# run_script(plan, planfile, header) - take the lock, start watchdog then driver, verify alive
# status_script(lane, name) - END/OK/KILLED counts, last stamps, load, memavail, vitest, CONTs
# read_csv(path, frontiers) - one run's gen/prefill per frontier + the min-across
# read_runlog(path) - START/END stamps per label
# arm_of(label) - "r3_tip_1" -> TIP, "r3_hits_1" -> HITS
# analyze(seq, runs, stamps, frontiers, legacy) - contention + straggler drops, paired deltas, floor, verdicts
# verdict(D, rate, floor_max) - BEATS / TIES / LOSES / PARTIAL by the sealed rule
# print_analysis(res) - the analysis a human reads, no-drop sensitivity beside each verdict
# cmd_plan/cmd_seal/cmd_build/cmd_run/cmd_status/cmd_analyze(args) - the subcommands
# selftest() - offline: plan, seal, refusals, analyzer with a known answer, and a mutation that reds
# build_parser() - argparse with a tutorial --help per subcommand
# main(argv) - dispatch; any Usage error prints one line + usage and exits 2
#
# ** Technical Review **
# One tool for the whole cycle: plan -> seal -> build -> run -> status -> analyze. The design
# defaults are the MEASURED lean regime (2026-09-17, rounds 1-2, see SPARK_ARMS.md): one TIP
# bracket per cycle (33 runs for 7 arms instead of 57), frontiers 4096/6144 with 2048 as the
# discarded warmup (8192 dropped; 6144 may never be), 5 s between runs. `--adjacent` and
# `--frontiers 4096,6144,8192` restore the old design. The spark is driven over `ssh spark`
# with generated bash (printed whole under --dry-run, which touches nothing); on the spark
# itself the same scripts run locally. Every long-lived spark process starts under `setsid`
# with stdin from /dev/null and is checked with an ANCHORED pgrep; the lane name is in every
# script and log name so two rounds never share a process pattern. Refusals, never guesses:
# an uncommitted seal, a seal whose shas differ from the plan, a lock held by another lane, a
# push target that is not ours, a binary whose .text is off the tip's vintage. The analyzer is
# a port of ~/sweeps/analyze_bisect.py (paired delta against the mean of the bracketing TIPs,
# per frontier and min-across, floor from adjacent TIP pairs, straggler rule with the no-drop
# sensitivity), with the verdict rule from the round-3 seal: sign RATE, not count. stdlib only:
# the spark's system python3 has no numpy and this must run there.
# PER-ARM SWITCHES (round 7 on): an arm is `branch@DS4_VAR=1,DS4_VAR2=0`. Each (branch, switches)
# pair is its own arm with its own label (branch label + a suffix from the switches: +hits, -hold),
# its own plan rows and its own verdict; the branch gets ONE worktree and ONE build shared by its
# switched arms. The plan file carries the assignments as trailing columns and the one-script
# hands them to `env` for that run only; the START line and the seal record them.
# THE STRAGGLER RULE (round 7 on, from the round-4 finding): it applies to ARMS only, never to
# TIP; the floor is taken from ALL adjacent live TIP pairs; and an arm with fewer than 6 live
# repeats is not filtered at all (MAD collapses at n=4..5 - round 4 dropped 2 of 5 controls at
# margins of 0.06 and 0.20 t/s against 3*MAD = 0.03 and shrank the floor from 2.71 to 0.21).
# `--legacy-straggler` restores the round-4 rule so rounds 5 and 6 read as they were sealed.
# THE WARM-UP RUN (round 9 on): six `make cuda-spark -j12` builds right before run 1 evict the 81 GB
# model from the page cache, so the round's first run streams experts cold (r8_tip_1 8.61 vs 9.6-9.7
# t/s; r6_tip_1 9.45 vs 9.56-9.88; r4 pool_1 9.24 vs 10.34-10.47) and, under the all-pairs floor, a
# cold tip_1 inflates the floor and wrecks every verdict. So the plan starts with `<name>_tip_warm`:
# the TIP binary, the same argv, CSV written, START stamped `warmup=1`, and the analyzer discards it
# BY LABEL before any bracket or floor is computed, printing its gen-min beside the first real TIP's
# in the RAW header. `--no-warmup` drops it (for --no-build onto a warm box). mutate="warm" is the
# selftest's planted defect: the warm-up admitted as a TIP inflates the floor and the check reds.
# PER-ARM BENCH ARGUMENTS (round 11 on): after the `@`, a `;` separates the env switches from bench
# argument overrides drawn from a CLOSED SET - cache=<N>GB (--ssd-streaming-cache-experts, default
# 90GB), gen=<N> (--gen-tokens, default 128), ctxmax=<N> (--ctx-max, default the plan's, refused below
# the highest frontier). `winners@;cache=60GB` is an args-only arm. The label suffix names them
# (~c60, ~g256, ~x8192); the plan row carries them as `ARG:key=value` tokens after the env columns;
# the one-script sorts its trailing tokens, ARG: tokens REPLACE the default flag value (a flag is
# emitted exactly once, never appended - mutate="append" plants the duplicate and the selftest reds),
# the rest go to env; START stamps `args=...`; the seal lists them and check_seal refuses a change.
# The arm's identity is (branch, env, args): the label carries all three, the analyzer keys on it.
# </claudes_code_comments>
import argparse
import csv
import datetime as dt
import io
import os
import re
import shlex
import statistics as st
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPARK_HOME = "/home/hologram"
DEFAULT_TIP = "triple-tip-2026-09-16"
DEFAULT_CYCLES = 4
DEFAULT_FRONTIERS = "4096,6144"
WARMUP = 2048
SLEEP_S = 5            # between runs; was 15 (15 s x 57 runs = 14 min of dead time)
RUN_S_THREE = 159      # median run, frontiers 4096/6144/8192 (round 1, 41 runs)
RUN_S_TWO = 122        # the same minus the ~37 s the 8192 frontier costs (rounds 1-2, 59 runs)
TEXT_GATE_BYTES = 200_000   # an arm's .text may differ from the tip's by at most 0.2 MB
SIGN_RATE = 0.75            # BEATS needs a positive sign rate of 75 % or better
LOAD_GATE = 2.0
MIN_KB = 100_000_000        # MemAvailable floor for a run, in kB (the harness's own number)
MODEL = "~/dwarfstar/gguf/DeepSeek-V4.1-Flash-Q2.gguf"
DEFAULT_CACHE = "90GB"      # --ssd-streaming-cache-experts; an arm may override it with ;cache=<N>GB
DEFAULT_GEN = 128           # --gen-tokens; ;gen=<N>
# the closed set of bench-argument overrides: key -> (flag, one-letter label code, value check, gloss)
BENCH_ARGS = {"cache": ("--ssd-streaming-cache-experts", "c", r"[1-9]\d*GB", "<N>GB"),
              "gen": ("--gen-tokens", "g", r"[1-9]\d*", "<N>"),
              "ctxmax": ("--ctx-max", "x", r"[1-9]\d*", "<N>, not below the highest frontier")}
_FIXTURE_REFS = None        # the selftest swaps in a branch set here


class Usage(Exception):
    """A bad parameter: printed as one line, then the usage block, exit 2."""


# ─── git and branches ─────────────────────────────────────────────────────────
def sh(cmd, capture=True):
    r = subprocess.run(cmd, shell=True, text=True, capture_output=capture)
    return r.returncode, (r.stdout or ""), (r.stderr or "")


def git(*args):
    r = subprocess.run(["git", "-C", ROOT] + list(args), text=True, capture_output=True)
    if r.returncode != 0:
        raise Usage("git %s failed: %s" % (" ".join(args), r.stderr.strip()[:200]))
    return r.stdout


def branch_refs():
    if _FIXTURE_REFS is not None:
        return dict(_FIXTURE_REFS)
    out = git("for-each-ref", "--format=%(refname:short) %(objectname:short=8)", "refs/heads/")
    refs = {}
    for ln in out.split("\n"):
        if ln.strip():
            b, s = ln.split()
            refs[b] = s
    return refs


def carded_branches(refs, tip):
    """Every triple-* branch whose README at HEAD carries a card box and is not CONTROL/TOOLING."""
    if _FIXTURE_REFS is not None:
        return [b for b in refs if b.startswith("triple-") and b != tip]
    out = []
    for b in sorted(refs):
        if not b.startswith("triple-") or b == tip:
            continue
        r = subprocess.run(["git", "-C", ROOT, "show", b + ":README.md"], text=True, capture_output=True)
        if r.returncode != 0 or "T R I P L E" not in r.stdout:
            continue
        m = re.search(r"BRANCH\s+\S+\s+(?:CORRECT, )?(POSITIVE|WORTH ZERO|NEGATIVE|NOT YET|CONTROL|TOOLING)", r.stdout)
        if m and m.group(1) in ("CONTROL", "TOOLING"):
            continue
        out.append(b)
    return out


def norm_branch(name):
    return name if name.startswith("triple-") else "triple-" + name


ENV_VAR_RE = re.compile(r"^DS4_[A-Z0-9_]+$")
ARM_SPEC_USAGE = ("an arm is BRANCH or BRANCH@DS4_VAR=value[,DS4_VAR2=value][;cache=<N>GB,gen=<N>,ctxmax=<N>] "
                  "(VAR matches ^DS4_[A-Z0-9_]+$, value non-empty, no spaces; after ; only cache, gen, ctxmax; BRANCH@;cache=60GB is args only)")


def parse_bench_args(text, spec):
    """'cache=60GB,gen=256' -> [(key, value)] from the closed set BENCH_ARGS; anything else is a Usage."""
    args, seen = [], set()
    for tok in text.split(","):
        key, eq, val = tok.partition("=")
        if key not in BENCH_ARGS:
            raise Usage("bad bench arg %r in arm %r: after ; only %s; %s" % (tok, spec, ", ".join(sorted(BENCH_ARGS)), ARM_SPEC_USAGE))
        flag, _, pat, gloss = BENCH_ARGS[key]
        if not eq or not val:
            raise Usage("bad bench arg %r in arm %r: want %s=%s" % (tok, spec, key, gloss))
        if not re.fullmatch(pat, val):
            raise Usage("bad bench arg value %r for %s in arm %r: want %s (it becomes %s)" % (val, key, spec, gloss, flag))
        if key in seen:
            raise Usage("bench arg %s repeated in arm %r" % (key, spec))
        seen.add(key)
        args.append((key, val))
    return args


def parse_arm_spec(spec):
    """'all-fastest-newtip@DS4_CUDA_HITS_FIRST=1;cache=60GB' -> (branch, [(VAR, value)], [(key, value)]).
    Before the ; are env switches, after it bench-argument overrides; either part may be empty, not both."""
    if "@" not in spec:
        return norm_branch(spec), [], []
    name, _, sw = spec.partition("@")
    sw, semi, ap = sw.partition(";")
    if not name or (not sw.strip() and not ap.strip()):
        raise Usage("bad arm %r: %s" % (spec, ARM_SPEC_USAGE))
    if semi and not ap.strip():
        raise Usage("bad arm %r: nothing after ; - %s" % (spec, ARM_SPEC_USAGE))
    args = parse_bench_args(ap, spec) if ap.strip() else []
    env, seen = [], set()
    for tok in (sw.split(",") if sw.strip() else []):
        if "=" not in tok:
            raise Usage("bad switch %r in arm %r: want VAR=value; %s" % (tok, spec, ARM_SPEC_USAGE))
        var, _, val = tok.partition("=")
        if not ENV_VAR_RE.match(var):
            raise Usage("bad switch name %r in arm %r: %s" % (var, spec, ARM_SPEC_USAGE))
        if not val or re.search(r"\s", val):
            raise Usage("bad switch value %r for %s in arm %r: %s" % (val, var, spec, ARM_SPEC_USAGE))
        if var in seen:
            raise Usage("switch %s repeated in arm %r" % (var, spec))
        seen.add(var)
        env.append((var, val))
    return norm_branch(name), env, args


_GENERIC_WORDS = {"FIRST", "ORDER", "OFF", "ON", "ENABLE", "ENABLED", "MODE", "GB", "MB", "KB", "WRITE",
                  "SYNC", "THREADS", "STATS", "VALUE", "SIZE", "COUNT", "PAGES"}
_OFF_VALUES = {"0", "no", "false", "off"}


def env_suffix(env):
    """[(VAR, value)] -> '+hits-hold' (a Boolean switch: sign + word) or 'margin12' (a knob: word + value)."""
    out = ""
    for var, val in env:
        words = var[len("DS4_"):].split("_")
        pick = [w for w in words if w not in _GENERIC_WORDS] or words
        word = pick[-1].lower()[:4]
        if val in ("1",) or val.lower() in _OFF_VALUES:
            out += ("-" if val.lower() in _OFF_VALUES else "+") + word
        else:
            out += word + re.sub(r"[^a-z0-9]", "", val.lower())[:6]
    return out


def args_suffix(args):
    """[(key, value)] -> '~c60' / '~g256' / '~x8192': one ~ per override, the code letter and the number."""
    return "".join("~%s%s" % (BENCH_ARGS[k][1], re.sub(r"[^0-9]", "", v)) for k, v in args)


def env_str(env):
    return " ".join("%s=%s" % kv for kv in env)


def args_str(args):
    return " ".join("%s=%s" % kv for kv in args)


def plan_tokens(args):
    """The plan-file form: ARG:cache=60GB, distinguishable from an env column at a glance and in bash."""
    return " ".join("ARG:%s=%s" % kv for kv in args)


def quoted_specs(plan):
    """The arm specs for a NEXT -> line, shell-quoted because ; is a shell separator."""
    return " ".join(shlex.quote(a["spec"]) for a in plan["arms"])


def short_label(branch, taken):
    base = re.sub(r"[^a-z0-9]", "", branch.replace("triple-", "").lower())[:12] or "arm"
    lab, n = base, 2
    while lab in taken or lab == "tip":
        lab, n = "%s%d" % (base, n), n + 1
    taken.add(lab)
    return lab


def parse_frontiers(text):
    try:
        fr = sorted({int(x) for x in text.split(",") if x.strip()})
    except ValueError:
        raise Usage("bad --frontiers %r: want comma-separated context sizes, e.g. 4096,6144" % text)
    if not fr:
        raise Usage("bad --frontiers: empty")
    if 6144 not in fr:
        raise Usage("bad --frontiers %s: 6144 must never be dropped (it carries the min in 42 of 59 runs)" % text)
    steps = [b - a for a, b in zip([WARMUP] + fr, fr)]
    if any(s <= 0 or s != steps[0] for s in steps):
        raise Usage("bad --frontiers %s: must be evenly spaced from the 2048 warmup (e.g. 4096,6144 or 4096,6144,8192)" % text)
    return fr


# ─── the plan ─────────────────────────────────────────────────────────────────
def is_warmup(label):
    """'r8_tip_warm' -> True: the discarded warm-up TIP run that precedes run 1."""
    p = label.split("_")
    return len(p) == 3 and p[1] == "tip" and p[2] == "warm"


def make_plan(names, refs, tip, cycles, frontiers, adjacent, name, warmup=True):
    tip = norm_branch(tip)
    if tip not in refs:
        raise Usage("unknown tip branch %r (git for-each-ref does not list it)" % tip)
    if not names:
        names = carded_branches(refs, tip)
        if not names:
            raise Usage("no triple-* branch with a card found; name the arms explicitly")
    if cycles < 1:
        raise Usage("--cycles must be at least 1")
    if not re.fullmatch(r"[a-z0-9]{1,12}", name):
        raise Usage("bad --name %r: lowercase letters and digits, 1-12 of them (it names files and processes)" % name)
    arms, taken, seen, blab = [], set(), set(), {}
    for n in names:
        b, env, args = parse_arm_spec(n)
        if b not in refs:
            raise Usage("unknown branch %r (try one of: %s)" % (n, ", ".join(sorted(x for x in refs if x.startswith("triple-"))[:6]) + " ..."))
        if b == tip:
            raise Usage("%s is the tip, not an arm" % b)
        for k, v in args:
            if k == "ctxmax" and int(v) < max(frontiers):
                raise Usage("arm %r: ctxmax=%s is below the highest frontier %d - the CSV could never carry that frontier" % (n, v, max(frontiers)))
        key = (b, tuple(env), tuple(args))
        if key in seen:
            raise Usage("duplicate arm %r (same branch, same switches, same bench args) - name each (branch, switches, args) triple once" % n)
        seen.add(key)
        if b not in blab:                       # one label root, one worktree, one build per BRANCH
            blab[b] = short_label(b, taken)
        lab = blab[b] + env_suffix(env) + args_suffix(args)
        if env or args:
            root, k = lab, 2
            while lab in taken:
                lab, k = "%s%d" % (root, k), k + 1
            taken.add(lab)
        arms.append({"branch": b, "sha": refs[b], "label": lab, "wt": "%s-%s" % (name, blab[b]), "env": env, "args": args,
                     "spec": b + (("@" + ",".join("%s=%s" % kv for kv in env) + (";" + ",".join("%s=%s" % kv for kv in args) if args else ""))
                                  if (env or args) else "")})
    return {"name": name, "lane": "ARMS-" + name.upper(), "tip": {"branch": tip, "sha": refs[tip], "wt": name + "-tip"},
            "arms": arms, "cycles": cycles, "frontiers": frontiers, "adjacent": adjacent, "warmup": warmup,
            "ctxmax": max(frontiers), "incr": frontiers[0] - WARMUP}


def plan_sequence(plan, with_env=False):
    """[(worktree, label)] in run order, the warm-up TIP first when plan['warmup'];
    with_env=True appends each row's env [(VAR, value)] and bench args [(key, value)] (both empty for TIP)."""
    n, seq = plan["name"], []
    if plan.get("warmup", True):
        seq.append((plan["tip"]["wt"], "%s_tip_warm" % n, [], []))
    for c in range(1, plan["cycles"] + 1):
        if not plan["adjacent"]:
            seq.append((plan["tip"]["wt"], "%s_tip_%d" % (n, c), [], []))
        for a in plan["arms"]:
            if plan["adjacent"]:
                seq.append((plan["tip"]["wt"], "%s_tip_%d_%s" % (n, c, a["label"]), [], []))
            seq.append((a["wt"], "%s_%s_%d" % (n, a["label"], c), a["env"], a.get("args", [])))
    seq.append((plan["tip"]["wt"], "%s_tip_final" % n, [], []))
    return seq if with_env else [(w, l) for w, l, _, _ in seq]


def plan_worktrees(plan):
    """The distinct worktrees: the tip plus one per BRANCH (switched arms share theirs)."""
    out, seen = [], set()
    for w in [plan["tip"]] + plan["arms"]:
        if w["wt"] not in seen:
            seen.add(w["wt"])
            out.append(w)
    return out


def wall_estimate(plan):
    """-> (runs incl. the warm-up, seconds per run, compute seconds, sleep seconds)."""
    runs = len(plan_sequence(plan))
    nf = len(plan["frontiers"])
    per = RUN_S_TWO + (RUN_S_THREE - RUN_S_TWO) * (nf - 2)   # 122 s at two frontiers, 159 at three
    return runs, per, runs * per, runs * SLEEP_S


def runs_phrase(plan):
    """'33 runs + 1 warm-up' or '33 runs': the measured count, the warm-up named beside it."""
    runs = wall_estimate(plan)[0]
    return "%d runs + 1 warm-up" % (runs - 1) if plan.get("warmup", True) else "%d runs" % runs


def print_plan(plan):
    runs, per, compute, sleep = wall_estimate(plan)
    print("PLAN %s  lane %s  design %s" % (plan["name"], plan["lane"], "adjacent brackets" if plan["adjacent"] else "one TIP bracket per cycle"))
    print("  TIP   %-16s %-34s @%s" % (plan["tip"]["wt"], plan["tip"]["branch"], plan["tip"]["sha"]))
    for a in plan["arms"]:
        print("  ARM   %-16s %-34s @%s  label %s%s%s" % (a["wt"], a["branch"], a["sha"], a["label"],
                                                            ("  env " + env_str(a["env"])) if a["env"] else "",
                                                            ("  args " + args_str(a["args"])) if a.get("args") else ""))
    nb = len({a["wt"] for a in plan["arms"]})
    if nb != len(plan["arms"]):
        print("  (%d arms on %d branches: one worktree and one build per branch, switches per run)" % (len(plan["arms"]), nb))
    print("  frontiers %s  (ctx %d warmup, discarded)  ctx-max %d  step %d" % (
        "/".join(map(str, plan["frontiers"])), WARMUP, plan["ctxmax"], plan["incr"]))
    wu = 1 if plan.get("warmup", True) else 0
    print("  cycles %d  ->  %s  (%d arms x %d + %d TIP%s)" % (
        plan["cycles"], runs_phrase(plan), len(plan["arms"]), plan["cycles"], runs - wu - len(plan["arms"]) * plan["cycles"],
        "; the warm-up TIP run is discarded, it exists because the builds evict the model from the page cache" if wu else ""))
    print("  wall  %d x ~%d s = ~%.0f min compute + %.1f min sleep = ~%.0f min%s" % (
        runs, per, compute / 60, sleep / 60, (compute + sleep) / 60, "  (warm-up included)" if wu else ""))
    print("        (per-run seconds: median 159 s at three frontiers on round 1, 41 runs; ~37 s less at two)")


def plan_file_text(plan, seal):
    lines = ["# spark_arms plan name=%s lane=%s frontiers=%s warmup=%d cycles=%d design=%s sealed=%s" % (
        plan["name"], plan["lane"], ",".join(map(str, plan["frontiers"])), WARMUP, plan["cycles"],
        "adjacent" if plan["adjacent"] else "cycle-edge", seal or "UNSEALED")]
    lines.append("# columns: worktree label ctx-max step-incr [VAR=value ...] [ARG:key=value ...]  (switches exported for that run only;"
                 " ARG: tokens replace that run's default bench flag: cache, gen, ctxmax)")
    if plan.get("warmup", True):
        lines.append("# <name>_tip_warm is the warm-up run: same binary, same argv, CSV written, discarded by the analyzer")
    for wt, label, env, args in plan_sequence(plan, with_env=True):
        lines.append("%s %s %d %d%s%s" % (wt, label, plan["ctxmax"], plan["incr"], (" " + env_str(env)) if env else "",
                                          (" " + plan_tokens(args)) if args else ""))
    return "\n".join(lines) + "\n"


# ─── the seal ─────────────────────────────────────────────────────────────────
def seal_text(plan, when=None):
    when = when or dt.datetime.now(dt.timezone.utc)
    runs, per, compute, sleep = wall_estimate(plan)
    L = ["%s %s - PREREGISTERED RULE - sealed before any measurement" % (when.strftime("%Y-%m-%d"), plan["name"].upper()),
         "=" * 78, "Written by track0_harness/spark_arms.py seal at %sZ. Commit this file, THEN run." % when.strftime("%H:%M:%S"),
         "", "THE CONTROL", "  TIP   %-10s @%s  %s" % (plan["tip"]["wt"], plan["tip"]["sha"], plan["tip"]["branch"]),
         "", "THE ARMS"]
    for a in plan["arms"]:
        L.append("  %-6s %-16s @%s  %s" % (a["label"].upper(), a["wt"], a["sha"], a["branch"]))
        if a["env"]:
            L.append("         switches %s" % env_str(a["env"]))
        if a.get("args"):
            L.append("         args %s" % args_str(a["args"]))
    switched = [a for a in plan["arms"] if a["env"]]
    arged = [a for a in plan["arms"] if a.get("args")]
    L += ["", "THE DESIGN",
          "  %s; %d cycles; %s; frontiers %s with ctx %d as discarded warmup; %d s between runs." % (
              "adjacent TIP brackets" if plan["adjacent"] else "one TIP bracket per cycle (cycle-edge)",
              plan["cycles"], runs_phrase(plan), "/".join(map(str, plan["frontiers"])), WARMUP, SLEEP_S),
          ("  One warm-up TIP run precedes run 1 and is discarded; it exists because the builds evict the\n"
           "  model from the page cache (r8_tip_1 8.61 vs 9.6-9.7 t/s later; r6_tip_1 9.45 vs 9.56-9.88;\n"
           "  r4 pool_1 9.24 vs 10.34-10.47). It is never a bracket and never in the floor."
           if plan.get("warmup", True) else "  No warm-up run (--no-warmup): the box is assumed warm."),
          ("  Every arm's default env, no switches set." if not switched else
           "  %d of %d arms carry switches (listed under THE ARMS), exported for that run only; every other" % (len(switched), len(plan["arms"])) + "\n"
           "  arm and every TIP run the binary's default env. One worktree and one build per branch."),
          "  argv: ds4-bench --cuda --ssd-streaming",
          "  --ssd-streaming-cache-experts %s --prompt-file speed-bench/promessi_sposi.txt" % DEFAULT_CACHE,
          "  --ctx-start 2048 --ctx-max %d --step-incr %d --gen-tokens %d." % (plan["ctxmax"], plan["incr"], DEFAULT_GEN),
          ("  %d of %d arms override a bench argument (args under THE ARMS: cache -> --ssd-streaming-cache-experts," % (len(arged), len(plan["arms"])) + "\n"
           "  gen -> --gen-tokens, ctxmax -> --ctx-max); the flag is REPLACED for that arm's runs only, every other run"
           "\n  and every TIP run the argv above." if arged else "  No arm overrides a bench argument."),
          "  Wall estimate ~%.0f min (%d x ~%d s + sleep%s)." % ((compute + sleep) / 60, runs, per, ", warm-up included" if plan.get("warmup", True) else ""),
          "", "THE BUILD GATE",
          "  Only arms whose build returns rc=0 enter the plan. An arm whose ds4-server .text differs",
          "  from the tip's by more than %.1f MB is REFUSED and named (the JIT-vintage trap)." % (TEXT_GATE_BYTES / 1e6),
          "  Every binary's .text is recorded in the runlog header before the first run.",
          "", "THE STATISTIC",
          "  Per repeat: the arm against the MEAN of its bracketing TIP runs, percent of gen_steady_tps.",
          "  Per frontier and at min-across-frontiers: paired MEDIAN, sign COUNT and RATE, SE, MAD, min, max.",
          "  In-run TIP floor = ALL adjacent live TIP pairs, median AND max; a TIP run is never dropped as",
          "  a straggler. Straggler rule, ARMS ONLY: per arm on gen-min, |G - med| > 3 MAD, single pass,",
          "  applied only when the arm has 6 or more live repeats (MAD collapses below that: round 4",
          "  dropped 2 of 5 controls at 0.06 and 0.20 t/s against 3*MAD = 0.03); the no-drop reading",
          "  is printed beside every verdict.",
          "  A killed run (rc != 0) is void: re-run it, never reweight the survivors.",
          "", "THE VERDICT RULE, per arm, at min-across-frontiers, floor_max = THIS round's own TIP floor max",
          "  BEATS   iff  D >= floor_max  AND  the positive sign RATE is %d %% or better" % int(SIGN_RATE * 100),
          "  TIES    iff  |D| <= floor_max",
          "  LOSES   iff  D <= -floor_max",
          "  otherwise PARTIAL, with its number and no category word.",
          "  floor_max is computed from this round's TIP pairs, never imported, never chosen after D.",
          "  Prefill is reported and is never the verdict. Nothing here ranks two arms against each other.",
          "", "PREDICTIONS, sealed so they can be scored (fill in BEFORE committing)",
          "  P1  <arm> <beats|ties|loses> the tip     confidence 0.x",
          "  P2  the TIP floor max is under <x>       confidence 0.x",
          "  P3  zero runs lost to the OOM killer     confidence 0.x", ""]
    return "\n".join(L)


def seal_path(plan, when=None):
    when = when or dt.datetime.now(dt.timezone.utc)
    return os.path.join(ROOT, "%s-%s-PREREGISTERED-RULE.txt" % (when.strftime("%Y-%m-%d"), plan["name"].upper()))


def parse_seal(text):
    """-> {'tip': (wt, sha), 'arms': [(label, wt, sha, switches-string, args-string)]}"""
    tip, arms, section = None, [], None
    for ln in text.split("\n"):
        if ln.startswith("THE "):
            section = ln.strip()
            continue
        ms = re.match(r"\s+switches\s+(.+?)\s*$", ln)
        if ms and section == "THE ARMS" and arms:
            arms[-1] = arms[-1][:3] + (ms.group(1), arms[-1][4])
            continue
        ma = re.match(r"\s+args\s+(.+?)\s*$", ln)
        if ma and section == "THE ARMS" and arms:
            arms[-1] = arms[-1][:4] + (ma.group(1),)
            continue
        m = re.match(r"\s+(\S+)\s+(\S+)\s+@([0-9a-f]{7,})", ln)
        if not m:
            continue
        if section == "THE CONTROL" and m.group(1) == "TIP":
            tip = (m.group(2), m.group(3))
        elif section == "THE ARMS":
            arms.append((m.group(1).lower(), m.group(2), m.group(3), "", ""))
    return {"tip": tip, "arms": arms}


def plan_arm_rows(plan):
    return [(a["label"], a["wt"], a["sha"], env_str(a["env"]), args_str(a.get("args", []))) for a in plan["arms"]]


def is_committed(path):
    r = subprocess.run(["git", "-C", os.path.dirname(path) or ".", "log", "-1", "--format=%h", "--", os.path.basename(path)],
                       text=True, capture_output=True)
    return r.returncode == 0 and r.stdout.strip() != ""


def check_seal(plan, path):
    if not os.path.exists(path):
        raise Usage("no seal at %s - run: spark_arms.py seal %s --name %s, fill PREDICTIONS, commit it" % (
            os.path.basename(path), quoted_specs(plan), plan["name"]))
    if not is_committed(path):
        raise Usage("seal %s is NOT COMMITTED - a seal is a commit-ordering property: git add + commit it, then run" % os.path.basename(path))
    s = parse_seal(open(path).read())
    if s["tip"] != (plan["tip"]["wt"], plan["tip"]["sha"]) or s["arms"] != plan_arm_rows(plan):
        raise Usage("seal %s names different arms, switches, bench args or shas than this plan - a branch moved or an arm changed after the seal; re-seal or pin" % os.path.basename(path))
    return os.path.basename(path)


# ─── the spark ────────────────────────────────────────────────────────────────
def on_spark():
    return os.path.isdir(os.path.expanduser("~/dwarfstar/_worktrees"))


def remote(script, dry, title=""):
    """Run a bash script on the spark (or locally when on it). Under --dry-run: print it, run nothing."""
    if dry:
        print("--- DRY RUN%s: would run on %s ---" % ((" " + title) if title else "", "this box" if on_spark() else "ssh spark"))
        print(script.rstrip())
        print("--- end ---")
        return 0, ""
    if on_spark():
        r = subprocess.run(["bash", "-s"], input=script, text=True, capture_output=True)
    else:
        r = subprocess.run(["ssh", "spark", "bash", "-s"], input=script, text=True, capture_output=True)
    if r.returncode != 0:
        print(r.stderr.strip())
    return r.returncode, r.stdout


def sparkbox_ok(url):
    return url.startswith("spark:") or url.startswith("ssh://spark") or "github.com/TripleSparkleAI/" in url


def build_script(plan):
    wts = plan_worktrees(plan)
    L = ["set -u", "cd ~/dwarfstar || exit 1", "mkdir -p _worktrees ~/sweeps",
         "LOG=~/sweeps/BUILD_%s.log" % plan["name"],
         'OWN=$(cat ~/spark_model.lock/owner 2>/dev/null || true)',
         'if [ -n "$OWN" ] && ! grep -q "%s" <<<"$OWN"; then echo "REFUSED: model lock held by another lane: $OWN (a build loads the box under a live round)"; exit 3; fi' % plan["lane"],
         # take the lock NOW, before the first build: a chained round that keys on "no driver, lock
         # free" would otherwise jump the queue while this round is still compiling (round 6 did)
         "mkdir -p ~/spark_model.lock",
         'echo "%s spark_arms $(date -u +%%Y-%%m-%%dT%%H:%%M:%%SZ) - round %s (building)" > ~/spark_model.lock/owner' % (plan["lane"], plan["name"])]
    for w in wts:
        L += ['if [ -d _worktrees/%s ]; then git -C _worktrees/%s checkout -q --detach %s; else git worktree add -q --detach _worktrees/%s %s; fi' % (w["wt"], w["wt"], w["sha"], w["wt"], w["sha"]),
              'echo "BUILD START %s %s $(date -u +%%H:%%M:%%SZ)" >> $LOG' % (w["wt"], w["sha"]),
              '( cd _worktrees/%s && make cuda-spark -j12 > ~/sweeps/build_%s.log 2>&1 ); rc=$?' % (w["wt"], w["wt"]),
              'txt=$(size _worktrees/%s/ds4-server 2>/dev/null | tail -1 | cut -f1)' % w["wt"],
              'echo "BUILD END   %s rc=$rc text=${txt:-none} sha=%s $(date -u +%%H:%%M:%%SZ)" >> $LOG' % (w["wt"], w["sha"])]
    L += ['echo "BUILD ALL DONE $(date -u +%H:%M:%SZ)" >> $LOG', "cat $LOG"]
    return "\n".join(L) + "\n"


def parse_build_log(text):
    built = {}
    for m in re.finditer(r"^BUILD END\s+(\S+)\s+rc=(-?\d+)\s+text=(\S+)(?:\s+sha=(\S+))?", text, re.M):
        t = m.group(3)
        built[m.group(1)] = (int(m.group(2)), int(t) if t.isdigit() else None, m.group(4) or "")
    return built


def text_gate(plan, built):
    """-> (ok_arms, refused: [(wt, reason)]). The tip must have built or nothing can run."""
    tw = plan["tip"]["wt"]
    if tw not in built or built[tw][0] != 0 or built[tw][1] is None:
        return [], [(tw, "the TIP did not build (rc=%s) - no plan" % (built.get(tw, ("missing",))[0]))]
    ttext, ok, refused = built[tw][1], [], []
    for a in plan["arms"]:
        b = built.get(a["wt"])
        if b is None:
            refused.append((a["wt"], "no BUILD END line - not built"))
        elif b[0] != 0:
            refused.append((a["wt"], "build rc=%d - UNBUILT, a result about that branch" % b[0]))
        elif b[1] is None:
            refused.append((a["wt"], "no .text size - binary missing"))
        elif abs(b[1] - ttext) > TEXT_GATE_BYTES:
            refused.append((a["wt"], ".text %d vs tip %d: off by %.2f MB > %.1f MB (the JIT-vintage trap)" % (
                b[1], ttext, abs(b[1] - ttext) / 1e6, TEXT_GATE_BYTES / 1e6)))
        elif b[2] and b[2] != a["sha"]:
            refused.append((a["wt"], "built sha %s is not the plan's %s" % (b[2], a["sha"])))
        else:
            ok.append(a)
    return ok, refused


def bench_argv_lines(mutate=None):
    """The one-script's argv block, between two markers so the selftest can run it under bash alone.
    Trailing tokens of the plan row arrive in "$@": ARG:key=value REPLACES that flag's default (one flag,
    one value, never appended), anything else is an env switch. mutate="append" is the planted defect:
    the override is appended after the default, so the argv carries the flag twice."""
    L = ["# --- argv begin: defaults, then the row's ARG: overrides REPLACE them; the rest is env",
         "CACHE=%s; GEN=%d; CMAX=\"$cmax\"; ENVS=(); EXTRA=()" % (DEFAULT_CACHE, DEFAULT_GEN),
         'for t in "$@"; do', '  case "$t" in']
    if mutate == "append":
        L.append('    ARG:cache=*) EXTRA+=(--ssd-streaming-cache-experts "${t#ARG:cache=}");;')
    else:
        L.append('    ARG:cache=*) CACHE="${t#ARG:cache=}";;')
    L += ['    ARG:gen=*) GEN="${t#ARG:gen=}";;', '    ARG:ctxmax=*) CMAX="${t#ARG:ctxmax=}";;',
          '    ARG:*) echo "REFUSED_BADARG $label $t (not in the closed set: cache, gen, ctxmax)"; exit 0;;',
          '    *) ENVS+=("$t");;', "  esac", "done",
          'ARGS="cache=$CACHE,gen=$GEN,ctxmax=$CMAX"',
          'BENCH=(./ds4-bench -m %s --cuda --ssd-streaming --ssd-streaming-cache-experts "$CACHE"' % MODEL,
          "  --prompt-file speed-bench/promessi_sposi.txt --ctx-start %d --ctx-max \"$CMAX\" --step-incr \"$incr\" --gen-tokens \"$GEN\"" % WARMUP,
          '  --csv ~/sweeps/$label.csv "${EXTRA[@]}")',
          "# --- argv end"]
    return L


def lane_scripts(plan, mutate=None):
    """The three lane-named spark scripts: one-run harness, driver, CONT watchdog."""
    n, lane = plan["name"], plan["lane"]
    one = "\n".join([
        "#!/bin/bash", "# spark_arms one-run harness, lane %s. Generated; edit the generator." % lane,
        "exec < /dev/null",
        'w="$1"; label="$2"; cmax="$3"; incr="$4"; shift 4',
        "cd ~/dwarfstar/_worktrees/$w || exit 1",
        "MIN_KB=%d" % MIN_KB,
        'la(){ cut -d" " -f1 /proc/loadavg; }', 'ma_kb(){ grep MemAvailable /proc/meminfo | tr -dc 0-9; }',
        'ma_gb(){ echo $(( $(ma_kb) / 1000 / 1000 )); }',
        'gu(){ nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits; }',
        'mine(){ grep -q "%s" ~/spark_model.lock/owner 2>/dev/null; }' % lane,
        'if ! mine; then echo "REFUSED_NOLOCK $label held-by: $(cat ~/spark_model.lock/owner 2>/dev/null || echo FREE)"; exit 0; fi',
        "waited=0",
        "while : ; do", "  L=$(la); K=$(ma_kb)",
        '  if awk "BEGIN{exit !($L < %s)}" && [ "$K" -gt "$MIN_KB" ]; then break; fi' % LOAD_GATE,
        '  if ! mine; then echo "REFUSED_LOSTLOCK $label $(date -u +%H:%M:%SZ)"; exit 0; fi',
        '  if [ "$waited" -ge 7200 ]; then echo "REFUSED_BUSY $label load=$L memavail=$(ma_gb)GB after ${waited}s $(date -u +%H:%M:%SZ)"; exit 0; fi',
        "  sleep 30; waited=$((waited+30))", "done",
        'case "$label" in *_tip_warm) wu=1;; *) wu=0;; esac'] + bench_argv_lines(mutate) + [
        'echo "START $label load=$(la) gpu=$(gu)% memavail=$(ma_gb)GB waited=${waited}s switches=${ENVS[*]:-none} args=$ARGS warmup=$wu $(date -u +%H:%M:%SZ)"',
        'env "${ENVS[@]}" "${BENCH[@]}" > ~/sweeps/$label.log 2>&1 < /dev/null',
        "rc=$?",
        'echo "END   $label rc=$rc load=$(la) gpu=$(gu)% memavail=$(ma_gb)GB $(date -u +%H:%M:%SZ)"', ""])
    driver = "\n".join([
        "#!/bin/bash", "# spark_arms driver, lane %s. usage: arms_%s_driver.sh <planfile>" % (lane, n),
        "exec < /dev/null", "export SWEEP_LANE=%s" % lane, 'plan="$1"',
        "while read -r line; do", '  [ -z "$line" ] && continue', '  case "$line" in \\#*) continue;; esac',
        "  bash ~/arms_%s_one.sh $line >> ~/sweeps/RUNLOG_%s.txt 2>&1" % (n, n),
        "  sleep %d" % SLEEP_S, 'done < "$plan"',
        'echo "PASS DONE $(basename "$plan") $(date -u +%%H:%%M:%%SZ)" >> ~/sweeps/RUNLOG_%s.txt' % n,
        "# release the box: only a lock this lane wrote, and only this lane's watchdog",
        'grep -q "^%s " ~/spark_model.lock/owner 2>/dev/null && rm -rf ~/spark_model.lock' % lane,
        'pkill -f "^bash %s/arms_%s_watchdog\\.sh" 2>/dev/null' % (SPARK_HOME, n),
        'echo "LOCK RELEASED $(date -u +%%H:%%M:%%SZ)" >> ~/sweeps/RUNLOG_%s.txt' % n, ""])
    watchdog = "\n".join([
        "#!/bin/bash", "# spark_arms CONT watchdog, lane %s: SIGCONT only, ANCHORED pgrep." % lane,
        "exec < /dev/null",
        "while :; do",
        '  for p in $(pgrep -f "^bash %s/arms_%s_(driver|one)\\.sh" 2>/dev/null; pgrep -x ds4-bench 2>/dev/null); do' % (SPARK_HOME, n),
        '    [ "$p" = "$$" ] && continue',
        '    st=$(awk \'{print $3}\' "/proc/$p/stat" 2>/dev/null)',
        '    case "$st" in T*) echo "$(date -u +%%H:%%M:%%SZ) CONT pid=$p state=$st" >> "$HOME/arms_%s_watchdog.log"; kill -CONT "$p" 2>/dev/null;; esac' % n,
        "  done", "  sleep 5", "done", ""])
    return {"one": one, "driver": driver, "watchdog": watchdog}


def run_script(plan, planfile_text, header):
    n, lane = plan["name"], plan["lane"]
    S = lane_scripts(plan)
    L = ["set -u", "cd ~ || exit 1", "mkdir -p ~/sweeps ~/spark_model.lock",
         'OWN=$(cat ~/spark_model.lock/owner 2>/dev/null || true)',
         'if [ -n "$OWN" ] && ! grep -q "%s" <<<"$OWN"; then echo "REFUSED: model lock held by another lane: $OWN"; exit 3; fi' % lane,
         'if pgrep -f "^bash %s/arms_%s_driver\\.sh" >/dev/null; then echo "REFUSED: a %s driver is already running"; exit 3; fi' % (SPARK_HOME, n, lane),
         "cat > ~/arms_%s_one.sh <<'EOF_ONE'\n%sEOF_ONE" % (n, S["one"]),
         "cat > ~/arms_%s_driver.sh <<'EOF_DRV'\n%sEOF_DRV" % (n, S["driver"]),
         "cat > ~/arms_%s_watchdog.sh <<'EOF_WD'\n%sEOF_WD" % (n, S["watchdog"]),
         "cat > ~/sweeps/plan_%s.txt <<'EOF_PLAN'\n%sEOF_PLAN" % (n, planfile_text),
         "cat >> ~/sweeps/RUNLOG_%s.txt <<'EOF_HDR'\n%sEOF_HDR" % (n, header),
         'echo "%s spark_arms $(date -u +%%Y-%%m-%%dT%%H:%%M:%%SZ) - round %s" > ~/spark_model.lock/owner' % (lane, n),
         "setsid bash ~/arms_%s_watchdog.sh </dev/null >/dev/null 2>&1 &" % n,
         "sleep 1",
         "setsid bash ~/arms_%s_driver.sh ~/sweeps/plan_%s.txt </dev/null >/dev/null 2>&1 &" % (n, n),
         "sleep 2",
         'WD=$(pgrep -f "^bash %s/arms_%s_watchdog\\.sh" | head -1); DR=$(pgrep -f "^bash %s/arms_%s_driver\\.sh" | head -1)' % (SPARK_HOME, n, SPARK_HOME, n),
         'echo "watchdog pid=${WD:-NOT RUNNING}  driver pid=${DR:-NOT RUNNING}  lock=$(cat ~/spark_model.lock/owner)"',
         '[ -n "$WD" ] && [ -n "$DR" ] || { echo "NOT ALIVE after start - nothing to trust"; exit 4; }']
    return "\n".join(L) + "\n"


def status_script(lane, n):
    return "\n".join([
        "R=~/sweeps/RUNLOG_%s.txt" % n,
        'echo "lock:      $(cat ~/spark_model.lock/owner 2>/dev/null || echo FREE)"',
        'echo "driver:    $(pgrep -f "^bash %s/arms_%s_driver\\.sh" | head -1 || true) $(pgrep -f "^bash %s/arms_%s_driver\\.sh" >/dev/null && echo alive || echo NOT RUNNING)"' % (SPARK_HOME, n, SPARK_HOME, n),
        'echo "watchdog:  $(pgrep -f "^bash %s/arms_%s_watchdog\\.sh" >/dev/null && echo alive || echo NOT RUNNING)  CONTs=$(grep -c CONT ~/arms_%s_watchdog.log 2>/dev/null || echo 0)"' % (SPARK_HOME, n, n),
        'echo "bench:     $(pgrep -x ds4-bench | head -1 || echo none)"',
        '[ -f "$R" ] && echo "runs:      END=$(grep -c "^END" "$R") OK=$(grep -c "^END.*rc=0 " "$R") KILLED=$(grep -c "^END.*rc=137" "$R") OTHER=$(grep "^END" "$R" | grep -vc "rc=0 \\|rc=137") REFUSED=$(grep -c "^REFUSED" "$R")" || echo "runs:      no runlog yet"',
        '[ -f "$R" ] && { echo "last:"; grep -E "^(START|END|PASS)" "$R" | tail -3 | sed "s/^/  /"; }',
        'echo "box:       load=$(cut -d" " -f1 /proc/loadavg) memavail=$(( $(grep MemAvailable /proc/meminfo | tr -dc 0-9) / 1000 / 1000 ))GB vitest=$(pgrep -fc vitest || echo 0)"', ""])


# ─── the analysis (ported from ~/sweeps/analyze_bisect.py, the sealed statistic) ──
def read_csv(path, frontiers):
    if not os.path.exists(path):
        return None
    rows = {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            rows[int(r["ctx_tokens"])] = {k: float(v) for k, v in r.items() if k != "ctx_tokens"}
    if not all(f in rows for f in [WARMUP] + list(frontiers)):
        return None
    g = {f: rows[f]["gen_steady_tps"] for f in frontiers}
    pf = {f: rows[f]["prefill_tps"] for f in frontiers}
    return {"gen": g, "gen_min": min(g.values()), "pf": pf, "pf_min": min(pf.values())}


def read_runlog(path):
    rec = {}
    if not os.path.exists(path):
        return rec
    pat = re.compile(r"^(START|END)\s+(\S+)\s+(.*)$")

    def num(p, s, default=0.0):
        m = re.search(p, s)
        return float(m.group(1)) if m else default
    for ln in open(path):
        m = pat.match(ln.strip())
        if not m:
            continue
        kind, label, rest = m.groups()
        d = rec.setdefault(label, {})
        if kind == "START":
            d.update(load_s=num(r"load=([\d.]+)", rest), gpu_s=num(r"gpu=([\d.]+)", rest),
                     mem_s=num(r"memavail=([\d.]+)", rest), waited=num(r"waited=(\d+)", rest))
        else:
            d.update(rc=num(r"rc=(-?\d+)", rest, -999), load_e=num(r"load=([\d.]+)", rest),
                     gpu_e=num(r"gpu=([\d.]+)", rest), mem_e=num(r"memavail=([\d.]+)", rest))
    return rec


def arm_of(label):
    p = label.split("_")
    return "TIP" if len(p) > 1 and p[1] == "tip" else (p[1].upper() if len(p) > 1 else label.upper())


def _mad(xs):
    m = st.median(xs)
    return st.median([abs(x - m) for x in xs])


def verdict(D, rate, floor_max, mutate=False):
    if mutate:                      # the selftest's planted defect: the floor rule flipped
        floor_max = -floor_max
    if D >= floor_max and rate >= SIGN_RATE:
        return "BEATS"
    if abs(D) <= floor_max:
        return "TIES"
    if D <= -floor_max:
        return "LOSES"
    return "PARTIAL"


STRAGGLER_MIN_N = 6         # MAD is trusted only from six live repeats (round 4: it collapsed at n=4 and n=5)


def analyze(seq, runs, stamps, frontiers, mutate=False, legacy=False):
    """seq: [(label, arm)] in plan order; runs: {label: read_csv}; stamps: read_runlog.
    legacy=True: the round-4 straggler rule (every arm incl. TIP, from n=3) for rounds sealed under it.
    mutate: 'verdict' flips the floor rule; 'floor' re-plants the round-4 defect (TIP filtered) under the new rule."""
    keys = list(frontiers) + ["min"]
    if mutate is True:
        mutate = "verdict"
    # the warm-up run is discarded BY LABEL before anything is computed: never a bracket, never in the floor
    warm = [l for l, _ in seq if is_warmup(l)]
    if mutate != "warm":                    # the planted defect: the cold warm-up treated as a TIP
        seq = [(l, a) for l, a in seq if not is_warmup(l)]
    order = [l for l, _ in seq if l in runs]
    cont = {}
    for l in order:
        L, why = stamps.get(l, {}), []
        if not L:
            why.append("no stamp in RUNLOG")
        else:
            if L.get("rc") != 0:
                why.append("rc=%d (a killed run is void)" % L.get("rc"))
            if L.get("load_s", 0) >= LOAD_GATE:
                why.append("START load %.2f" % L["load_s"])
            if L.get("load_e", 0) >= 3.0:
                why.append("END load %.2f" % L["load_e"])
            if L.get("mem_s", 999) < 110 or L.get("mem_e", 999) < 110:
                why.append("memavail<110GB")
            if L.get("gpu_s", 0) >= 50 or L.get("gpu_e", 0) >= 50:
                why.append("gpu>=50")
            if L.get("waited", 0) >= 120:
                why.append("waited>=120s")
        if why:
            cont[l] = "; ".join(why)
    live = [l for l in order if l not in cont]
    armname = dict(seq)
    stray, notes = {}, {}
    for arm in sorted({a for _, a in seq}):
        ls = [l for l in live if armname[l] == arm]
        if legacy:
            if len(ls) < 3:
                continue
        elif arm == "TIP" and mutate == "floor":
            pass                            # the planted defect: TIP filtered from n=3, as round 4 did
        else:
            if arm == "TIP":
                notes[arm] = "never filtered (the floor is taken from all live TIP pairs)"
                continue
            if len(ls) < STRAGGLER_MIN_N:
                notes[arm] = "n<%d, not applied" % STRAGGLER_MIN_N
                continue
        G = [runs[l]["gen_min"] for l in ls]
        m, d = st.median(G), _mad(G)
        if d == 0:
            continue
        for l, g in zip(ls, G):
            if abs(g - m) > 3 * d:
                stray[l] = "|G-med|=%.3f > 3*MAD=%.3f" % (abs(g - m), 3 * d)
    kept = [l for l in live if l not in stray]
    rule = ("LEGACY (round-4 seal): every arm including TIP, from n=3; the floor from straggler-filtered TIPs" if legacy
            else "arms only, n>=%d; TIP never filtered; the floor from ALL adjacent live TIP pairs" % STRAGGLER_MIN_N)

    def bracket(i, pool):
        pre = post = None
        for j in range(i - 1, -1, -1):
            if seq[j][1] == "TIP" and seq[j][0] in pool:
                pre = seq[j][0]
                break
        for j in range(i + 1, len(seq)):
            if seq[j][1] == "TIP" and seq[j][0] in pool:
                post = seq[j][0]
                break
        return pre, post

    def val(l, f, key="gen"):
        return runs[l][key + "_min"] if f == "min" else runs[l][key][f]

    def deltas(arm, pool):
        out = {}
        for i, (l, a) in enumerate(seq):
            if a != arm or l not in pool:
                continue
            pre, post = bracket(i, pool)
            if pre is None or post is None:
                continue
            e = {"pre": pre, "post": post, "gen": {}, "pf": {}}
            for key in ("gen", "pf"):
                for f in keys:
                    ref = (val(pre, f, key) + val(post, f, key)) / 2
                    e[key][f] = 100 * (val(l, f, key) - ref) / ref
            out[l] = e
        return out

    def stats(dd, key="gen"):
        out = {}
        for f in keys:
            v = [dd[l][key][f] for l in dd]
            if not v:
                continue
            sd = st.stdev(v) if len(v) > 1 else 0.0
            pos = sum(1 for x in v if x > 0)
            out[f] = {"n": len(v), "median": st.median(v), "se": sd / len(v) ** 0.5, "mad": _mad(v),
                      "min": min(v), "max": max(v), "pos": pos, "neg": sum(1 for x in v if x < 0),
                      "rate": pos / len(v), "vals": v}
        return out

    def floor(pool, arm="TIP"):
        ls = [l for l, a in seq if a == arm and l in pool]
        out = {}
        for f in keys:
            v = [100 * abs(val(a, f) - val(b, f)) / min(val(a, f), val(b, f)) for a, b in zip(ls, ls[1:])]
            out[f] = {"median": st.median(v) if v else None, "max": max(v) if v else None, "n": len(v)}
        return out

    arms = sorted({a for _, a in seq if a != "TIP"})
    res = {"plan_runs": len(seq), "parsed": len(order), "live": len(live), "kept": len(kept), "cont": cont,
           "warm": [l for l in warm if l in runs and mutate != "warm"],
           "stray": stray, "notes": notes, "rule": rule, "legacy": legacy, "order": order, "runs": runs,
           "armname": armname, "keys": keys, "arms": {}, "floor_kept": floor(set(kept)), "floor_live": floor(set(live))}
    for arm in arms:
        A = {}
        for setname, pool in (("kept", set(kept)), ("no-drop", set(live))):
            dd = deltas(arm, pool)
            sb, pb = stats(dd), stats(dd, "pf")
            fl = res["floor_kept" if setname == "kept" else "floor_live"]["min"]["max"]
            v = "NO DATA"
            if "min" in sb and fl is not None:
                v = verdict(sb["min"]["median"], sb["min"]["rate"], fl, mutate == "verdict")
            A[setname] = {"dd": dd, "gen": sb, "pf": pb, "floor_max": fl, "verdict": v}
        res["arms"][arm] = A
    return res


def print_analysis(res, title="ARMS"):
    keys = res["keys"]
    print("=" * 96)
    print("%s  plan %d  parsed %d  live %d  kept %d" % (title, res["plan_runs"], res["parsed"], res["live"], res["kept"]))
    print("  straggler rule: %s" % res["rule"])
    print("=" * 96)
    for k, v in res["cont"].items():
        print("  CONTENTION DROP %s (%s): %s" % (k, res["armname"][k], v))
    for k, v in res["stray"].items():
        print("  STRAGGLER DROP %s (%s): %s" % (k, res["armname"][k], v))
    for k, v in sorted(res["notes"].items()):
        print("  straggler rule %s: %s" % (k, v))
    print("\nRAW gen_steady_tps  [%s]  min   prefill-min" % " / ".join(map(str, keys[:-1])))
    tips = [l for l in res["order"] if res["armname"][l] == "TIP"]
    for l in res["warm"]:
        w = res["runs"][l]["gen_min"]
        beside = ("  vs first TIP %s %.2f (%+.1f %%)" % (tips[0], res["runs"][tips[0]]["gen_min"],
                                                        100 * (w - res["runs"][tips[0]]["gen_min"]) / res["runs"][tips[0]]["gen_min"])) if tips else ""
        print("  WARM-UP %-10s gen-min %.2f%s  (discarded: the builds evict the model from the page cache)" % (l, w, beside))
        print("  %-18s %-6s [%s]  %.2f   %.2f  <WARM-UP, discarded>" % (
            l, "TIP", " / ".join("%.2f" % res["runs"][l]["gen"][f] for f in keys[:-1]), w, res["runs"][l]["pf_min"]))
    for l in res["order"]:
        r, mark = res["runs"][l], ""
        if l in res["cont"]:
            mark = "  <CONTENTION-DROP>"
        elif l in res["stray"]:
            mark = "  <STRAGGLER-DROP>"
        print("  %-18s %-6s [%s]  %.2f   %.2f%s" % (l, res["armname"][l], " / ".join("%.2f" % r["gen"][f] for f in keys[:-1]), r["gen_min"], r["pf_min"], mark))
    fk = res["floor_kept"]
    print("\nTIP FLOOR (adjacent TIP pairs, %s): " % ("straggler-filtered, legacy" if res["legacy"] else "all live") + "  ".join(
        "%s: med %s max %s (n=%d)" % (f, "%.2f" % fk[f]["median"] if fk[f]["median"] is not None else "-",
                                       "%.2f" % fk[f]["max"] if fk[f]["max"] is not None else "-", fk[f]["n"]) for f in keys))
    for arm, A in res["arms"].items():
        k = A["kept"]
        print("\n--- %s vs TIP (kept, n=%d) ---" % (arm, len(k["dd"])))
        for l, e in k["dd"].items():
            print("  %-18s %-30s %s  min %+.2f" % (l, e["pre"] + " / " + e["post"], " ".join("%+8.2f" % e["gen"][f] for f in keys[:-1]), e["gen"]["min"]))
        for f in keys:
            if f in k["gen"]:
                s = k["gen"][f]
                print("  %8s  median %+7.2f  SE %5.2f  MAD %5.2f  min %+7.2f  max %+7.2f  sign %d/%d (%.0f%%)" % (
                    str(f), s["median"], s["se"], s["mad"], s["min"], s["max"], s["pos"], s["n"], 100 * s["rate"]))
        if "min" in k["pf"]:
            p = k["pf"]["min"]
            print("  prefill min-across: median %+.2f  sign %d/%d  (reported, never the verdict)" % (p["median"], p["pos"], p["n"]))
        nd = A["no-drop"]
        if "min" in k["gen"]:
            print("  VERDICT %-7s  D=%+.2f %%  floor_max=%.2f  rate=%.0f%%   |  no-drop sensitivity: %s D=%+.2f floor_max=%.2f" % (
                k["verdict"], k["gen"]["min"]["median"], k["floor_max"] or 0, 100 * k["gen"]["min"]["rate"],
                nd["verdict"], nd["gen"]["min"]["median"] if "min" in nd["gen"] else 0, nd["floor_max"] or 0))
        else:
            print("  VERDICT NO DATA (no bracketed repeat survived)")


# ─── the subcommands ──────────────────────────────────────────────────────────
def _plan_from_args(a):
    return make_plan(a.arms, branch_refs(), a.tip, a.cycles, parse_frontiers(a.frontiers), a.adjacent, a.name,
                     warmup=not getattr(a, "no_warmup", False))


def cmd_plan(a):
    plan = _plan_from_args(a)
    print_plan(plan)
    if a.dry_run:
        print("\n(plan file the driver would read)\n" + plan_file_text(plan, None))
    print("NEXT ->  spark_arms.py seal %s --name %s   (then fill PREDICTIONS and commit it)" % (quoted_specs(plan), plan["name"]))
    return 0


def cmd_seal(a):
    plan = _plan_from_args(a)
    path = seal_path(plan)
    if os.path.exists(path):
        raise Usage("%s exists - a seal is written once; pick another --name or delete it BEFORE it is committed" % os.path.basename(path))
    text = seal_text(plan)
    if a.dry_run:
        print(text)
        print("(dry-run: would write %s)" % path)
    else:
        with open(path, "w") as fh:
            fh.write(text)
        print("wrote %s" % os.path.relpath(path, ROOT))
    print("NEXT ->  edit the PREDICTIONS block · git add %s · git commit · spark_arms.py run %s --name %s" % (
        os.path.basename(path), quoted_specs(plan), plan["name"]))
    return 0


def _push(plan, dry):
    rc, url, _ = sh("git -C %s remote get-url sparkbox" % shlex.quote(ROOT))
    url = url.strip()
    if rc != 0 or not sparkbox_ok(url):
        raise Usage("remote sparkbox is %r - refusing to push anywhere but the spark or the TripleSparkleAI fork" % url)
    refspecs = " ".join("%s:refs/arms/%s" % (w["sha"], w["wt"]) for w in plan_worktrees(plan))
    cmd = "git -C %s push -q --force sparkbox %s" % (shlex.quote(ROOT), refspecs)
    if dry:
        print("--- DRY RUN push (never origin): %s\n%s\n--- end ---" % (url, cmd))
        return
    rc, out, err = sh(cmd)
    if rc != 0:
        raise Usage("push to sparkbox failed: %s" % err.strip()[:300])


def cmd_build(a):
    plan = _plan_from_args(a)
    print_plan(plan)
    _push(plan, a.dry_run)
    rc, out = remote(build_script(plan), a.dry_run, "build (sequential, make cuda-spark -j12)")
    if a.dry_run:
        print("NEXT ->  spark_arms.py build %s --name %s   (without --dry-run)" % (quoted_specs(plan), plan["name"]))
        return 0
    if rc != 0:
        print(out.strip())
        raise Usage("build refused or failed on the spark (rc=%d)" % rc)
    built = parse_build_log(out)
    ok, refused = text_gate(plan, built)
    for wt, why in refused:
        print("  REFUSED %-16s %s" % (wt, why))
    for x in ok:
        print("  OK      %-16s .text %d" % (x["wt"], built[x["wt"]][1]))
    print("built %d of %d arms; tip .text %s" % (len(ok), len(plan["arms"]), built.get(plan["tip"]["wt"], (0, "?"))[1]))
    print("NEXT ->  spark_arms.py run %s --name %s" % (quoted_specs(plan), plan["name"]))
    return 0 if ok else 1


def cmd_run(a):
    plan = _plan_from_args(a)
    print_plan(plan)
    seal = check_seal(plan, seal_path(plan)) if not a.dry_run else (os.path.basename(seal_path(plan)) if os.path.exists(seal_path(plan)) else None)
    if a.dry_run and seal is None:
        print("  (dry-run: no seal file yet - a real run refuses here)")
    if a.dry_run and seal and not is_committed(seal_path(plan)):
        print("  (dry-run: seal %s is NOT committed - a real run refuses here)" % seal)
    if not a.no_build:
        _push(plan, a.dry_run)
        rc, out = remote(build_script(plan), a.dry_run, "build")
        if not a.dry_run and rc != 0:
            print(out.strip())
            raise Usage("build refused or failed on the spark (rc=%d)" % rc)
        built = parse_build_log(out) if not a.dry_run else {}
    else:
        rc, out = remote("cat ~/sweeps/BUILD_%s.log 2>/dev/null || true" % plan["name"], a.dry_run, "read build log")
        built = parse_build_log(out) if not a.dry_run else {}
    if a.dry_run:
        # a dry run cannot read a build log without touching the spark; it shows the plan as if all built
        ok, refused = plan["arms"], []
        header = "# spark_arms runlog header <utc> - .text per binary: (read from BUILD_%s.log at run time)\n" % plan["name"]
    else:
        ok, refused = text_gate(plan, built)
        for wt, why in refused:
            print("  REFUSED %-16s %s" % (wt, why))
        if not ok:
            raise Usage("no arm passed the build gate - nothing to run")
        header = "# spark_arms runlog header %s lane %s - .text per binary:\n" % (dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), plan["lane"])
        for w in plan_worktrees(dict(plan, arms=ok)):
            header += "#   %-16s %s text=%s\n" % (w["wt"], w["sha"], built[w["wt"]][1])
    run_plan = dict(plan, arms=ok)
    ptext = plan_file_text(run_plan, seal)
    rc, out = remote(run_script(run_plan, ptext, header), a.dry_run, "start (lock, watchdog, driver, anchored pgrep)")
    if a.dry_run:
        print("NEXT ->  commit the seal if not yet · spark_arms.py run %s --name %s   (without --dry-run)" % (quoted_specs(plan), plan["name"]))
        return 0
    print(out.strip())
    if rc != 0:
        raise Usage("start refused (rc=%d) - nothing was left running" % rc)
    runs_, per, compute, sleep = wall_estimate(run_plan)
    print("started %s, ~%.0f min. NEXT ->  spark_arms.py status --name %s   ·   spark_arms.py analyze --name %s --fetch" % (
        runs_phrase(run_plan), (compute + sleep) / 60, plan["name"], plan["name"]))
    return 0


def cmd_status(a):
    lane = "ARMS-" + a.name.upper()
    rc, out = remote(status_script(lane, a.name), a.dry_run, "status")
    if not a.dry_run:
        print(out.rstrip())
    print("NEXT ->  spark_arms.py analyze --name %s --fetch   (when runs: END equals the plan count)" % a.name)
    return 0


def cmd_analyze(a):
    d = a.sweeps or os.path.join(ROOT, "sweeps", a.name)
    planfile = a.plan or os.path.join(d, "plan_%s.txt" % a.name)
    if a.fetch:
        if a.dry_run:
            print("--- DRY RUN fetch: ssh spark tar of ~/sweeps/plan_%s.txt RUNLOG_%s.txt %s_*.csv -> %s ---" % (a.name, a.name, a.name, d))
        else:
            os.makedirs(d, exist_ok=True)
            rc, out, err = sh("ssh spark 'cd ~/sweeps && tar cf - plan_%s.txt RUNLOG_%s.txt %s_*.csv' | tar xf - -C %s" % (a.name, a.name, a.name, shlex.quote(d)))
            if rc != 0:
                raise Usage("fetch failed: %s" % err.strip()[:200])
    if not os.path.exists(planfile):
        raise Usage("no plan file at %s (pass --plan, or --fetch to pull it from the spark)" % planfile)
    frontiers, seq = None, []
    for ln in open(planfile):
        ln = ln.strip()
        if ln.startswith("#"):
            m = re.search(r"frontiers=([\d,]+)", ln)
            if m:
                frontiers = parse_frontiers(m.group(1))
            continue
        if ln:
            label = ln.split()[1]
            seq.append((label, arm_of(label)))
    frontiers = frontiers or parse_frontiers(a.frontiers)
    runs = {}
    for label, _ in seq:
        r = read_csv(os.path.join(d, label + ".csv"), frontiers)
        if r is None:
            print("MISSING/INCOMPLETE CSV: %s" % label)
        else:
            runs[label] = r
    if not runs:
        raise Usage("no complete CSV in %s - nothing to analyze" % d)
    res = analyze(seq, runs, read_runlog(os.path.join(d, "RUNLOG_%s.txt" % a.name)), frontiers, legacy=a.legacy_straggler)
    print_analysis(res, "ROUND %s" % a.name.upper())
    print("\nNEXT ->  write the result file from these verdicts · score the sealed PREDICTIONS · commit the CSVs beside it")
    return 0


# ─── the selftest ─────────────────────────────────────────────────────────────
def selftest():
    global _FIXTURE_REFS
    n = [0]
    fails = []

    def check(cond, name):
        n[0] += 1
        (print("  ok   %s" % name) if cond else (fails.append(name), print("  FAIL %s" % name)))

    def refused(argv, needle):
        buf, old = io.StringIO(), sys.stdout
        sys.stdout = buf
        try:
            code = main(argv)
        except SystemExit as e:
            code = e.code
        finally:
            sys.stdout = old
        out = buf.getvalue()
        ok = code == 2 and needle in out and "Traceback" not in out and "usage:" in out
        if not ok:
            print("      got exit %r, output: %s" % (code, out[:200].replace("\n", " | ")))
        return ok

    fixture = {"triple-tip-2026-09-16": "12997e9c", "triple-hitsfirst": "c5eb891a", "triple-hotlist": "ce886cdd",
               "triple-engram-read-threads": "37b2962a", "triple-pool": "1b80adae", "triple-prefill-readahead-order": "f552eb3c",
               "triple-prefetch-pool": "dd72115e", "triple-all-fastest": "b17231fd"}
    _FIXTURE_REFS = fixture
    seven = [b for b in fixture if b != "triple-tip-2026-09-16"]
    print("spark_arms --selftest (offline, no spark)")
    # plan
    p7 = make_plan(seven, fixture, DEFAULT_TIP, 4, [4096, 6144], False, "r3", warmup=False)
    runs, per, compute, sleep = wall_estimate(p7)
    check(runs == 33, "7 arms x 4 cycles, cycle-edge, --no-warmup -> 33 runs")
    check(per == 122 and abs(compute / 60 - 67.1) < 0.1, "wall: 33 x 122 s = 67.1 min compute")
    check(len(plan_sequence(make_plan(seven, fixture, DEFAULT_TIP, 4, [4096, 6144], True, "r3", warmup=False))) == 57, "--adjacent restores 57 runs")
    p3 = make_plan(["hitsfirst", "triple-hotlist", "pool"], fixture, DEFAULT_TIP, 4, [4096, 6144], False, "t", warmup=False)
    check(len(plan_sequence(p3)) == 17 and [a["branch"] for a in p3["arms"]][0] == "triple-hitsfirst", "prefix optional: 3 arms -> 17 runs")
    check(wall_estimate(make_plan(seven, fixture, DEFAULT_TIP, 4, [4096, 6144, 8192], False, "r3"))[1] == 159, "three frontiers -> 159 s per run")
    seq = plan_sequence(p3)
    check(seq[0] == ("t-tip", "t_tip_1") and seq[-1] == ("t-tip", "t_tip_final") and arm_of("t_hitsfirst_2") == "HITSFIRST" and arm_of("t_tip_1") == "TIP", "labels and arm_of")
    # the warm-up run: first in the plan by default, counted in runs and wall, gone under --no-warmup
    p7w = make_plan(seven, fixture, DEFAULT_TIP, 4, [4096, 6144], False, "r3")
    seqw = plan_sequence(p7w)
    check(seqw[0] == ("r3-tip", "r3_tip_warm") and seqw[1] == ("r3-tip", "r3_tip_1") and len(seqw) == 34 and is_warmup("r3_tip_warm")
          and not is_warmup("r3_tip_1") and not is_warmup("r3_tip_final") and arm_of("r3_tip_warm") == "TIP",
          "default plan: r3_tip_warm is row 0, tip_1 is row 1, 34 rows, is_warmup by label only")
    check(wall_estimate(p7w)[0] == 34 and wall_estimate(p7w)[2] == 34 * 122 and runs_phrase(p7w) == "33 runs + 1 warm-up"
          and runs_phrase(p7) == "33 runs", "count and wall include the warm-up: 34 x 122 s; the phrase says 33 runs + 1 warm-up")
    check(plan_file_text(p7w, "x").split("\n")[3].startswith("r3-tip r3_tip_warm 6144 2048") and "tip_warm" not in plan_file_text(p7, "x"),
          "plan file: the warm-up row first (the driver runs it), absent under --no-warmup")
    stw = seal_text(p7w, dt.datetime(2026, 9, 17, 9, 0, tzinfo=dt.timezone.utc))
    check("33 runs + 1 warm-up" in stw and "One warm-up TIP run precedes run 1 and is discarded" in stw and "page cache" in stw
          and "warm-up included" in stw and "No warm-up run" in seal_text(p7), "seal states the warm-up, its reason, and counts it in the wall")
    check("warmup=$wu" in lane_scripts(p7w)["one"] and "*_tip_warm) wu=1" in lane_scripts(p7w)["one"], "the START line carries warmup=1 for the warm-up label")
    check(len(make_plan([], fixture, DEFAULT_TIP, 1, [4096, 6144], False, "all")["arms"]) == 7, "no arms -> every carded triple-* (fixture: 7)")
    taken = set()
    check(short_label("triple-pool", taken) == "pool" and short_label("triple-pool", taken) == "pool2", "labels unique")
    # frontiers
    check(parse_frontiers("4096,6144") == [4096, 6144] and parse_frontiers("4096,6144,8192") == [4096, 6144, 8192], "frontiers parse")
    for bad, why in (("4096,8192", "6144"), ("abc", "comma-separated"), ("4096,6144,10240", "evenly spaced")):
        try:
            parse_frontiers(bad)
            check(False, "frontiers %r refused" % bad)
        except Usage as e:
            check(why in str(e), "frontiers %r refused: %s" % (bad, why))
    # seal
    tmp = tempfile.mkdtemp(prefix="spark_arms_")
    txt = seal_text(p7, dt.datetime(2026, 9, 17, 9, 0, tzinfo=dt.timezone.utc))
    path = os.path.join(tmp, "2026-09-17-R3-PREREGISTERED-RULE.txt")
    open(path, "w").write(txt)
    s = parse_seal(txt)
    check(s["tip"] == ("r3-tip", "12997e9c") and s["arms"] == plan_arm_rows(p7), "seal round-trips arms+shas")
    check("PREDICTIONS" in txt and "RATE is 75 %" in txt and "33 runs" in txt, "seal carries predictions block, rate rule, run count")
    check(not is_committed(path), "seal in a temp dir is not committed")
    try:
        check_seal(p7, path)
        check(False, "uncommitted seal refused")
    except Usage as e:
        check("NOT COMMITTED" in str(e), "uncommitted seal refused")
    moved = make_plan(seven, dict(fixture, **{"triple-pool": "deadbeef"}), DEFAULT_TIP, 4, [4096, 6144], False, "r3")
    check(s["arms"] != plan_arm_rows(moved), "a moved sha no longer matches the seal (check_seal refuses it)")
    # per-arm switches: syntax, labels, one worktree per branch, plan rows, seal, one-script
    fx2 = dict(fixture, **{"triple-all-fastest-newtip": "b8f8a5a0"})
    _FIXTURE_REFS = fx2
    check(parse_arm_spec("all-fastest-newtip@DS4_CUDA_HITS_FIRST=1,DS4_PREFILL_READAHEAD_HOLD=0") == (
        "triple-all-fastest-newtip", [("DS4_CUDA_HITS_FIRST", "1"), ("DS4_PREFILL_READAHEAD_HOLD", "0")], []), "arm spec parses, prefix optional, order kept")
    check(parse_arm_spec("triple-pool") == ("triple-pool", [], []), "a plain branch has no switches and no bench args")
    for bad, why in (("pool@FOO=1", "bad switch name"), ("pool@DS4_x=1", "bad switch name"), ("pool@DS4_X", "want VAR=value"),
                     ("pool@DS4_X=", "bad switch value"), ("pool@", "bad arm"), ("pool@DS4_X=1,DS4_X=0", "repeated")):
        try:
            parse_arm_spec(bad)
            check(False, "arm spec %r refused" % bad)
        except Usage as e:
            check(why in str(e), "arm spec %r refused: %s" % (bad, why))
    check(env_suffix([("DS4_CUDA_HITS_FIRST", "1")]) == "+hits" and env_suffix([("DS4_PREFILL_READAHEAD_HOLD", "0")]) == "-hold"
          and env_suffix([("DS4_CUDA_HITS_FIRST", "1"), ("DS4_PREFILL_READAHEAD_HOLD", "0")]) == "+hits-hold"
          and env_suffix([("DS4_CUDA_EXPERT_CACHE_MARGIN_GB", "12")]) == "marg12", "label suffix: +hits, -hold, +hits-hold, marg12")
    sw = ["all-fastest-newtip@DS4_CUDA_HITS_FIRST=1", "all-fastest-newtip@DS4_PREFILL_READAHEAD_HOLD=0",
          "all-fastest-newtip@DS4_CUDA_HITS_FIRST=1,DS4_PREFILL_READAHEAD_HOLD=0", "hitsfirst", "pool"]
    p7s = make_plan(sw, fx2, DEFAULT_TIP, 4, [4096, 6144], False, "r7", warmup=False)
    labels = [a["label"] for a in p7s["arms"]]
    check(labels == ["allfastestne+hits", "allfastestne-hold", "allfastestne+hits-hold", "hitsfirst", "pool"], "switched labels: %s" % labels)
    check(len({a["wt"] for a in p7s["arms"]}) == 3 and [w["wt"] for w in plan_worktrees(p7s)] == ["r7-tip", "r7-allfastestne", "r7-hitsfirst", "r7-pool"],
          "5 arms on 3 branches -> 3 arm worktrees + tip; switched arms share r7-allfastestne")
    check(len({a for _, a in [(l, arm_of(l)) for _, l in plan_sequence(p7s)]}) == 6, "the analyzer sees 5 distinct arms + TIP")
    pf = plan_file_text(p7s, "x")
    check("r7-allfastestne r7_allfastestne+hits_1 6144 2048 DS4_CUDA_HITS_FIRST=1\n" in pf
          and "r7-allfastestne r7_allfastestne+hits-hold_1 6144 2048 DS4_CUDA_HITS_FIRST=1 DS4_PREFILL_READAHEAD_HOLD=0\n" in pf
          and "r7-hitsfirst r7_hitsfirst_1 6144 2048\n" in pf and "r7-tip r7_tip_1 6144 2048\n" in pf, "plan rows carry the switches as trailing columns, none for TIP or plain arms")
    st7 = seal_text(p7s, dt.datetime(2026, 9, 17, 9, 0, tzinfo=dt.timezone.utc))
    check("switches DS4_CUDA_HITS_FIRST=1 DS4_PREFILL_READAHEAD_HOLD=0" in st7 and "3 of 5 arms carry switches" in st7
          and parse_seal(st7)["arms"] == plan_arm_rows(p7s), "seal lists each arm's switches and round-trips them")
    check(parse_seal(st7)["arms"] != plan_arm_rows(make_plan(sw[:2] + ["all-fastest-newtip@DS4_CUDA_HITS_FIRST=0,DS4_PREFILL_READAHEAD_HOLD=0"] + sw[3:], fx2, DEFAULT_TIP, 4, [4096, 6144], False, "r7")),
          "a changed switch no longer matches the seal (check_seal refuses it)")
    check(build_script(p7s).count("make cuda-spark") == 4 and 'env "${ENVS[@]}" "${BENCH[@]}"' in lane_scripts(p7s)["one"]
          and "switches=${ENVS[*]:-none}" in lane_scripts(p7s)["one"], "one build per branch; the one-script exports the row's switches and stamps them on START")
    check("ALL adjacent live TIP pairs" in st7 and "ARMS ONLY" in st7 and "6 or more live repeats" in st7, "seal states the new straggler rule")
    check(refused(["plan", "pool@FOO=1"], "bad switch name"), "bad switch name -> one line + usage, exit 2")
    check(refused(["plan", "pool", "pool"], "duplicate arm"), "the same arm twice -> exit 2")
    check(refused(["plan", "pool@DS4_X=1", "pool@DS4_X=1"], "duplicate arm"), "the same branch+switches twice -> exit 2")
    _FIXTURE_REFS = fixture
    # build gate
    log = "\n".join("BUILD END   %s rc=%d text=%d sha=%s" % r for r in [
        ("r3-tip", 0, 24862051, "12997e9c"), ("r3-hitsfirst", 0, 24886127, "c5eb891a"), ("r3-hotlist", 0, 24905021, "ce886cdd"),
        ("r3-engramreadth", 0, 24860275, "37b2962a"), ("r3-pool", 1, 0, "1b80adae"), ("r3-prefillreada", 0, 24883939, "f552eb3c"),
        ("r3-prefetchpool", 0, 24873111, "dd72115e"), ("r3-allfastest", 0, 23148552, "b17231fd")])
    ok, ref = text_gate(p7, parse_build_log(log))
    check(len(ok) == 5 and [w for w, _ in ref] == ["r3-pool", "r3-allfastest"], "build gate: rc=1 refused, 1.71 MB .text refused, five pass")
    check(any("1.71 MB" in why for _, why in ref), "the .text refusal names the size")
    check(text_gate(p7, parse_build_log("BUILD END   r3-tip rc=2 text=none"))[0] == [], "tip unbuilt -> no plan")
    check(sparkbox_ok("spark:~/dwarfstar") and sparkbox_ok("https://github.com/TripleSparkleAI/ds4.git") and not sparkbox_ok("https://github.com/antirez/ds4.git"), "push target: spark/fork yes, antirez no")
    # scripts
    S, R = lane_scripts(p7), run_script(p7, plan_file_text(p7, "x"), "# hdr\n")
    check("setsid bash ~/arms_r3_watchdog.sh </dev/null" in R and "setsid bash ~/arms_r3_driver.sh" in R, "run starts watchdog then driver under setsid, stdin /dev/null")
    check('pgrep -f "^bash /home/hologram/arms_r3_driver\\.sh"' in R and "^bash /home/hologram/arms_r3_(driver|one)" in S["watchdog"], "anchored pgrep in run and watchdog")
    check("exec < /dev/null" in S["one"] and "sleep %d" % SLEEP_S in S["driver"] and 'grep -q "ARMS-R3"' in S["one"], "harness: /dev/null stdin, 5 s sleep, lane-named lock check")
    check("REFUSED: model lock held by another lane" in R and "REFUSED: model lock held by another lane" in build_script(p7), "run and build refuse a foreign lock")
    check('CMAX="$cmax"' in S["one"] and '--ctx-max "$CMAX"' in S["one"] and "6144 2048" in plan_file_text(p7, "x"), "plan lines carry ctx-max 6144 step 2048")
    # per-arm bench arguments (round 11 on): syntax, label, plan tokens, seal, and the one-script's argv under real bash
    check(parse_arm_spec("hitsfirst@;cache=60GB") == ("triple-hitsfirst", [], [("cache", "60GB")]), "args only: hitsfirst@;cache=60GB")
    check(parse_arm_spec("pool@DS4_CUDA_HITS_FIRST=1;cache=60GB,gen=256") == ("triple-pool", [("DS4_CUDA_HITS_FIRST", "1")], [("cache", "60GB"), ("gen", "256")]),
          "env and args together, order kept")
    for bad, why in (("pool@;cache=60", "want <N>GB"), ("pool@;cache=", "want cache=<N>GB"), ("pool@;cachee=60GB", "bad bench arg"),
                     ("pool@;gen=abc", "want <N>"), ("pool@;cache=60GB,cache=70GB", "repeated"), ("pool@DS4_X=1;", "nothing after ;"),
                     ("pool@;", "bad arm"), ("pool@;ctxmax=0", "want <N>")):
        try:
            parse_arm_spec(bad)
            check(False, "bench arg spec %r refused" % bad)
        except Usage as e:
            check(why in str(e), "bench arg spec %r refused: %s" % (bad, why))
    check(args_suffix([("cache", "60GB")]) == "~c60" and args_suffix([("gen", "256")]) == "~g256"
          and args_suffix([("cache", "60GB"), ("ctxmax", "8192")]) == "~c60~x8192", "label suffix: ~c60, ~g256, ~c60~x8192")
    pa = make_plan(["hitsfirst", "hitsfirst@;cache=60GB", "pool@DS4_CUDA_HITS_FIRST=1;gen=256"], fixture, DEFAULT_TIP, 4, [4096, 6144], False, "ra", warmup=False)
    check([a["label"] for a in pa["arms"]] == ["hitsfirst", "hitsfirst~c60", "pool+hits~g256"] and len(plan_worktrees(pa)) == 3
          and pa["arms"][1]["spec"] == "triple-hitsfirst@;cache=60GB" and pa["arms"][2]["spec"] == "triple-pool@DS4_CUDA_HITS_FIRST=1;gen=256",
          "args arms: labels hitsfirst~c60 / pool+hits~g256, one worktree per branch, spec round-trips")
    check(len({arm_of(l) for _, l in plan_sequence(pa)}) == 4, "the analyzer keys on (branch, env, args): 3 distinct arms + TIP")
    pfa = plan_file_text(pa, "x")
    check("ra-hitsfirst ra_hitsfirst~c60_1 6144 2048 ARG:cache=60GB\n" in pfa and "ra-hitsfirst ra_hitsfirst_1 6144 2048\n" in pfa
          and "ra-pool ra_pool+hits~g256_1 6144 2048 DS4_CUDA_HITS_FIRST=1 ARG:gen=256\n" in pfa,
          "plan rows carry ARG: tokens after the env columns, none for the plain arm or TIP")
    sta = seal_text(pa, dt.datetime(2026, 9, 17, 9, 0, tzinfo=dt.timezone.utc))
    check("         args cache=60GB" in sta and "2 of 3 arms override a bench argument" in sta and parse_seal(sta)["arms"] == plan_arm_rows(pa),
          "seal lists each arm's bench args and round-trips them")
    pa2 = make_plan(["hitsfirst", "hitsfirst@;cache=70GB", "pool@DS4_CUDA_HITS_FIRST=1;gen=256"], fixture, DEFAULT_TIP, 4, [4096, 6144], False, "ra", warmup=False)
    check(parse_seal(sta)["arms"] != plan_arm_rows(pa2), "a changed bench arg no longer matches the seal (check_seal refuses it)")
    try:
        make_plan(["hitsfirst@;ctxmax=4096"], fixture, DEFAULT_TIP, 4, [4096, 6144], False, "ra")
        check(False, "ctxmax below the highest frontier refused at plan time")
    except Usage as e:
        check("below the highest frontier 6144" in str(e), "ctxmax=4096 under frontiers 4096/6144 refused at plan time")
    check(make_plan(["hitsfirst@;ctxmax=6144"], fixture, DEFAULT_TIP, 4, [4096, 6144], False, "ra")["arms"][0]["label"] == "hitsfirst~x6144",
          "ctxmax equal to the highest frontier is accepted")

    def argv_under_bash(lines, *tokens):
        """Run the one-script's argv block under real bash with a row's trailing tokens; -> (env tokens, bench argv)."""
        body = "\n".join(lines) + "\nprintf '%s\\n' \"${ENVS[@]}\"; echo ----; printf '%s\\n' \"${BENCH[@]}\"\n"
        r = subprocess.run(["bash", "-c", 'label=x; cmax=6144; incr=2048; ' + body, "argv"] + list(tokens), text=True, capture_output=True)
        env_part, _, argv_part = r.stdout.partition("----\n")
        return r.returncode, env_part.split(), argv_part.split()
    rc, e1, v1 = argv_under_bash(bench_argv_lines(), "ARG:cache=60GB")
    rc0, e0, v0 = argv_under_bash(bench_argv_lines())
    rc2, e2, v2 = argv_under_bash(bench_argv_lines(), "DS4_CUDA_HITS_FIRST=1", "ARG:gen=256", "ARG:ctxmax=8192")
    check(rc == 0 and v1.count("--ssd-streaming-cache-experts") == 1 and v1[v1.index("--ssd-streaming-cache-experts") + 1] == "60GB" and "90GB" not in v1 and e1 == [],
          "the one-script applies cache=60GB EXACTLY ONCE: one flag, value 60GB, 90GB nowhere in that argv")
    check(rc0 == 0 and v0.count("--ssd-streaming-cache-experts") == 1 and v0[v0.index("--ssd-streaming-cache-experts") + 1] == "90GB"
          and v0[v0.index("--gen-tokens") + 1] == "128" and v0[v0.index("--ctx-max") + 1] == "6144", "the default arm still gets 90GB, gen 128, ctx-max from the row")
    check(rc2 == 0 and e2 == ["DS4_CUDA_HITS_FIRST=1"] and v2[v2.index("--gen-tokens") + 1] == "256" and v2[v2.index("--ctx-max") + 1] == "8192"
          and v2.count("--gen-tokens") == 1 and v2.count("--ctx-max") == 1 and v2[v2.index("--ssd-streaming-cache-experts") + 1] == "90GB",
          "env tokens go to env, gen/ctxmax replace their flags once, cache stays at the default")
    check("args=$ARGS" in lane_scripts(pa)["one"] and "REFUSED_BADARG" in lane_scripts(pa)["one"], "START stamps args=...; an unknown ARG: token refuses the run by name")
    rcm, em, vm = argv_under_bash(bench_argv_lines(mutate="append"), "ARG:cache=60GB")
    append_mutation_red = not (vm.count("--ssd-streaming-cache-experts") == 1 and "90GB" not in vm)
    check(append_mutation_red, "MUTATION (override appended, not replaced) turns the once-only check RED: %d flags, 90GB present=%s" % (
        vm.count("--ssd-streaming-cache-experts"), "90GB" in vm))
    # analyzer with a known answer: BEATS +8 %, TIES +0.3 %, LOSES -8 %; tip alternates 10.0/10.1
    a3 = make_plan(["hitsfirst", "hotlist", "pool"], fixture, DEFAULT_TIP, 4, [4096, 6144], False, "t", warmup=False)
    sq = [(l, arm_of(l)) for _, l in plan_sequence(a3)]
    gain = {"HITSFIRST": 1.08, "HOTLIST": 1.003, "POOL": 0.92}
    runs_, stamps, ti = {}, {}, 0
    for l, arm in sq:
        if arm == "TIP":
            base = 10.0 + 0.1 * (ti % 2)
            ti += 1
        else:
            base = 10.05 * gain[arm]
        runs_[l] = {"gen": {4096: base * 1.02, 6144: base}, "gen_min": base, "pf": {4096: 90.0, 6144: 89.0}, "pf_min": 89.0}
        stamps[l] = {"rc": 0, "load_s": 1.0, "load_e": 1.2, "mem_s": 117, "mem_e": 117, "gpu_s": 2, "gpu_e": 3, "waited": 0}
    res = analyze(sq, runs_, stamps, [4096, 6144])
    V = {a: res["arms"][a]["kept"]["verdict"] for a in res["arms"]}
    check(V == {"HITSFIRST": "BEATS", "HOTLIST": "TIES", "POOL": "LOSES"}, "verdicts on the planted set: %s" % V)
    check(abs(res["floor_kept"]["min"]["max"] - 1.0) < 1e-9 and res["floor_kept"]["min"]["n"] == 4, "TIP floor max 1.00 % from 4 adjacent pairs")
    check(res["arms"]["HITSFIRST"]["kept"]["gen"]["min"]["rate"] == 1.0 and res["arms"]["HITSFIRST"]["kept"]["gen"]["min"]["n"] == 4, "sign rate 4/4 = 100 %")
    check(res["arms"]["POOL"]["no-drop"]["verdict"] == "LOSES" and not res["stray"] and not res["cont"], "no-drop sensitivity agrees; no drops")
    check(res["notes"].get("TIP", "").startswith("never filtered") and res["notes"].get("POOL") == "n<6, not applied" and "arms only" in res["rule"],
          "new rule: TIP never filtered, an n=4 arm says 'n<6, not applied', the rule is named")
    # the round-4 finding, re-planted: 5 TIPs where 3 agree to the second decimal -> legacy MAD drops two controls
    fl_runs = dict(runs_)
    for l, g in zip([l for l, a in sq if a == "TIP"], [10.00, 10.06, 10.20, 10.00, 10.01]):
        fl_runs[l] = dict(fl_runs[l], gen={4096: g * 1.02, 6144: g}, gen_min=g)
    new, old = analyze(sq, fl_runs, stamps, [4096, 6144]), analyze(sq, fl_runs, stamps, [4096, 6144], legacy=True)
    check(len([k for k in old["stray"] if old["armname"][k] == "TIP"]) == 2 and abs(old["floor_kept"]["min"]["max"] - 0.1) < 1e-6 and old["floor_kept"]["min"]["n"] == 2,
          "--legacy-straggler reproduces round 4: 2 of 5 TIPs dropped at 0.06/0.20 vs 3*MAD=0.03, floor max 0.10 from 2 pairs")
    check(not [k for k in new["stray"] if new["armname"][k] == "TIP"] and abs(new["floor_kept"]["min"]["max"] - 2.0) < 1e-6 and new["floor_kept"]["min"]["n"] == 4,
          "new rule: no TIP dropped, floor max 2.00 from all 4 pairs")
    check(new["floor_kept"]["min"]["max"] != old["floor_kept"]["min"]["max"] and "LEGACY" in old["rule"] and "LEGACY" not in new["rule"],
          "the two rules give different floors and each names itself")
    # an arm at n>=6 is still filtered under the new rule (one wild repeat among six)
    a6 = make_plan(["hitsfirst"], fixture, DEFAULT_TIP, 6, [4096, 6144], False, "s", warmup=False)
    sq6 = [(l, arm_of(l)) for _, l in plan_sequence(a6)]
    r6, s6 = {}, {}
    for i, (l, arm) in enumerate(sq6):
        g = 10.0 if arm == "TIP" else (7.0 if l == "s_hitsfirst_3" else 10.8 + 0.01 * (i % 3))
        r6[l] = {"gen": {4096: g, 6144: g}, "gen_min": g, "pf": {4096: 90.0, 6144: 89.0}, "pf_min": 89.0}
        s6[l] = dict(stamps["t_tip_1"])
    n6 = analyze(sq6, r6, s6, [4096, 6144])
    check(list(n6["stray"]) == ["s_hitsfirst_3"] and "HITSFIRST" not in n6["notes"], "new rule at n=6: the wild arm repeat is dropped, TIP untouched")
    # the second planted mutation: TIP filtered under the new rule (the round-4 defect back) -> the floor check goes RED
    mf = analyze(sq, fl_runs, stamps, [4096, 6144], mutate="floor")
    floor_mutation_red = abs(mf["floor_kept"]["min"]["max"] - 2.0) > 1e-6
    check(floor_mutation_red, "MUTATION (TIP straggler-filtered again) turns the floor check RED: floor max %.2f, not 2.00" % mf["floor_kept"]["min"]["max"])
    # the cold warm-up: 8.6 t/s planted before tips at 9.6/9.65 -> the same floor as the plan without it;
    # the mutation that admits the warm-up as a TIP inflates the floor and goes RED
    aw = make_plan(["hitsfirst", "pool"], fixture, DEFAULT_TIP, 4, [4096, 6144], False, "w")
    an = make_plan(["hitsfirst", "pool"], fixture, DEFAULT_TIP, 4, [4096, 6144], False, "w", warmup=False)
    sqw, sqn = [(l, arm_of(l)) for _, l in plan_sequence(aw)], [(l, arm_of(l)) for _, l in plan_sequence(an)]
    rw, sw_, ti = {}, {}, 0
    for l, arm in sqw:
        if is_warmup(l):
            g = 8.6
        elif arm == "TIP":
            g = 9.6 + 0.05 * (ti % 2)
            ti += 1
        else:
            g = 9.9 if arm == "HITSFIRST" else 9.62
        rw[l] = {"gen": {4096: g * 1.01, 6144: g}, "gen_min": g, "pf": {4096: 90.0, 6144: 89.0}, "pf_min": 89.0}
        sw_[l] = dict(stamps["t_tip_1"])
    rww, rwn = analyze(sqw, rw, sw_, [4096, 6144]), analyze(sqn, rw, sw_, [4096, 6144])
    check(rww["warm"] == ["w_tip_warm"] and "w_tip_warm" not in rww["order"] and rww["plan_runs"] == rwn["plan_runs"] == 13,
          "the warm-up is discarded by label: not in order, 13 real runs either way")
    check(abs(rww["floor_kept"]["min"]["max"] - rwn["floor_kept"]["min"]["max"]) < 1e-12 and rww["floor_kept"]["min"]["n"] == 4
          and abs(rww["floor_kept"]["min"]["max"] - 100 * 0.05 / 9.6) < 1e-9,
          "with the cold warm-up the floor is THE SAME as without it: max %.3f %% from 4 pairs" % rww["floor_kept"]["min"]["max"])
    check({a: rww["arms"][a]["kept"]["verdict"] for a in rww["arms"]} == {a: rwn["arms"][a]["kept"]["verdict"] for a in rwn["arms"]}
          and rww["arms"]["HITSFIRST"]["kept"]["dd"]["w_hitsfirst_1"]["pre"] == "w_tip_1", "verdicts identical; tip_1, not the warm-up, brackets hitsfirst_1")
    mw = analyze(sqw, rw, sw_, [4096, 6144], mutate="warm")
    warm_mutation_red = abs(mw["floor_kept"]["min"]["max"] - rwn["floor_kept"]["min"]["max"]) > 1.0
    check(warm_mutation_red, "MUTATION (warm-up admitted to the floor) turns the floor check RED: floor max %.2f, not %.2f" % (
        mw["floor_kept"]["min"]["max"], rwn["floor_kept"]["min"]["max"]))
    buf, old = io.StringIO(), sys.stdout
    sys.stdout = buf
    try:
        print_analysis(rww, "SELFTEST")
    finally:
        sys.stdout = old
    check("WARM-UP w_tip_warm gen-min 8.60  vs first TIP w_tip_1 9.60 (-10.4 %)" in buf.getvalue() and "<WARM-UP, discarded>" in buf.getvalue(),
          "RAW header prints the warm-up's gen-min beside the first real TIP's, marked discarded")
    stamps["t_hotlist_2"] = dict(stamps["t_hotlist_2"], rc=137)
    r2 = analyze(sq, runs_, stamps, [4096, 6144])
    check("t_hotlist_2" in r2["cont"] and r2["arms"]["HOTLIST"]["kept"]["gen"]["min"]["n"] == 3, "a killed run (rc=137) is void, n drops to 3")
    buf, old = io.StringIO(), sys.stdout
    sys.stdout = buf
    try:
        print_analysis(res, "SELFTEST")
    finally:
        sys.stdout = old
    check("VERDICT BEATS" in buf.getvalue() and "no-drop sensitivity" in buf.getvalue(), "print_analysis prints the verdict and the sensitivity")
    # the mutation: flip the floor rule and the TIE verdict must change -> the check goes RED
    rm = analyze(sq, runs_, {k: dict(v, rc=0) for k, v in stamps.items()}, [4096, 6144], mutate=True)
    Vm = {a: rm["arms"][a]["kept"]["verdict"] for a in rm["arms"]}
    mutation_red = Vm != {"HITSFIRST": "BEATS", "HOTLIST": "TIES", "POOL": "LOSES"}
    check(mutation_red, "MUTATION (floor rule flipped) turns the verdict check RED: %s" % Vm)
    # bad params: one line + usage, exit 2, no traceback
    check(refused(["frobnicate"], "error:"), "unknown subcommand -> one line + usage, exit 2")
    check(refused(["plan", "nosuchbranch"], "unknown branch"), "unknown branch -> exit 2")
    check(refused(["plan", "hitsfirst", "--frontiers", "4096,8192"], "6144"), "frontiers dropping 6144 -> exit 2")
    check(refused(["plan", "hitsfirst", "--cycles", "0"], "cycles"), "cycles 0 -> exit 2")
    check(refused(["plan", "hitsfirst", "--name", "Round 3"], "bad --name"), "bad --name -> exit 2")
    check(refused(["plan", "hitsfirst@;cachee=60GB"], "bad bench arg"), "unknown bench arg key -> one line + usage, exit 2")
    check(refused(["plan", "hitsfirst@;ctxmax=4096"], "below the highest frontier"), "ctxmax below a frontier -> exit 2")
    buf, old = io.StringIO(), sys.stdout
    sys.stdout = buf
    try:
        main(["plan", "--dry-run", "--name", "zz", "hitsfirst", "hitsfirst@;cache=60GB"])
    finally:
        sys.stdout = old
    out = buf.getvalue()
    check("label hitsfirst~c60  args cache=60GB" in out and "zz-hitsfirst zz_hitsfirst~c60_1 6144 2048 ARG:cache=60GB" in out
          and "zz-hitsfirst zz_hitsfirst_1 6144 2048\n" in out and "seal triple-hitsfirst 'triple-hitsfirst@;cache=60GB'" in out,
          "plan --dry-run: the ~c60 arm prints its args, its rows carry ARG:cache=60GB, the NEXT line quotes the ;")
    buf, old = io.StringIO(), sys.stdout
    sys.stdout = buf
    try:
        main(["plan", "hitsfirst", "pool", "--name", "zz", "--dry-run"])
        main(["plan", "hitsfirst", "pool", "--name", "zz", "--dry-run", "--no-warmup"])
    finally:
        sys.stdout = old
    out = buf.getvalue()
    check(out.count("zz-tip zz_tip_warm 6144 2048") == 1 and "13 runs + 1 warm-up" in out and "13 runs  (" in out,
          "plan --dry-run: the warm-up row once; --no-warmup removes it and the count reads 13 runs")
    check(refused(["analyze", "--name", "zz9", "--sweeps", tmp], "no plan file"), "analyze with no plan -> exit 2")
    check(refused(["run", "hitsfirst", "--name", "zz9nope"], "no seal"), "run with no seal -> refused before touching the spark")
    check(refused([], "error:"), "no args -> usage, exit 2")
    _FIXTURE_REFS = None
    print("selftest: %d checks, %d failed" % (n[0], len(fails)))
    for f in fails:
        print("  FAILED: %s" % f)
    return 1 if fails else 0


# ─── the CLI ──────────────────────────────────────────────────────────────────
class _Parser(argparse.ArgumentParser):
    def error(self, message):
        print("error: %s" % message)
        self.print_usage()
        sys.exit(2)


EX = {
    "plan": "  spark_arms.py plan                          # every carded triple-* branch, lean defaults\n"
            "  spark_arms.py plan hitsfirst hotlist pool   # named arms, prefix optional\n"
            "  spark_arms.py plan all-fastest-newtip all-fastest-newtip@DS4_CUDA_HITS_FIRST=1   # one branch, two arms:\n"
            "                                              # the switches are exported for that arm's runs only\n"
            "  spark_arms.py plan winners 'winners@;cache=60GB'   # one branch, two arms: the second runs\n"
            "                                              # --ssd-streaming-cache-experts 60GB instead of 90GB (label winners~c60)\n"
            "  spark_arms.py plan --adjacent --frontiers 4096,6144,8192   # the old 57-run design",
    "seal": "  spark_arms.py seal hitsfirst hotlist --name r4   # writes 2026-MM-DD-R4-PREREGISTERED-RULE.txt\n"
            "  then: fill PREDICTIONS · git add · git commit     # run refuses until the seal is committed",
    "build": "  spark_arms.py build hitsfirst hotlist --name r4 --dry-run   # print the push + build script\n"
             "  spark_arms.py build hitsfirst hotlist --name r4             # push to sparkbox, build, gate .text",
    "run": "  spark_arms.py run hitsfirst hotlist --name r4 --dry-run   # everything printed, nothing touched\n"
           "  spark_arms.py run hitsfirst hotlist --name r4             # seal check, build, lock, watchdog, driver\n"
           "  spark_arms.py run --name r4 --no-build                    # reuse ~/sweeps/BUILD_r4.log",
    "status": "  spark_arms.py status --name r4",
    "analyze": "  spark_arms.py analyze --name r7 --fetch     # pull plan + runlog + CSVs, then judge\n"
               "  spark_arms.py analyze --name r7 --sweeps sweeps/r7\n"
               "  spark_arms.py analyze --name r5 --legacy-straggler   # rounds 5-6 were sealed under the round-4 rule",
}


def build_parser():
    ap = _Parser(prog="spark_arms.py", formatter_class=argparse.RawDescriptionHelpFormatter,
                 description="Run one measurement round of triple-* arms against a tip on the DGX Spark.",
                 epilog="The one line:  spark_arms.py run hitsfirst hotlist --name r4   (plan, seal check, build, run)\n"
                        "Lean defaults, all measured (see track0_harness/SPARK_ARMS.md):\n"
                        "  one TIP bracket per cycle (7 arms: 33 runs, not 57; --adjacent restores)\n"
                        "  frontiers 4096,6144 with 2048 as discarded warmup (8192 dropped, ~37 s/run; --frontiers restores)\n"
                        "  5 s between runs. NOT a lever: model load is ~12 s of 159 and the expert cache is per-process.\n"
                        "  one warm-up TIP run before run 1, discarded (the builds evict the model from the page cache; --no-warmup)\n"
                        "Every subcommand has --dry-run (prints all, touches nothing) and its own --help.")
    ap.add_argument("--selftest", action="store_true", help="offline checks, no spark (wired into verify-all.sh)")
    sub = ap.add_subparsers(dest="cmd", metavar="COMMAND")

    def common(p, arms=True):
        if arms:
            p.add_argument("arms", nargs="*", metavar="ARM", help="triple-* branch names, prefix optional, each optionally BRANCH@DS4_VAR=1,DS4_VAR2=0"
                           " and/or ;cache=<N>GB,gen=<N>,ctxmax=<N> (e.g. 'winners@;cache=60GB' - quote the ;); none = every carded branch")
            p.add_argument("--tip", default=DEFAULT_TIP, help="control branch (default %(default)s)")
            p.add_argument("--cycles", type=int, default=DEFAULT_CYCLES, help="repeats per arm (default %(default)s)")
            p.add_argument("--frontiers", default=DEFAULT_FRONTIERS, help="context frontiers after the 2048 warmup (default %(default)s; 6144 is never dropped)")
            p.add_argument("--adjacent", action="store_true", help="a TIP before every arm run (the old design, 57 runs for 7 arms)")
            p.add_argument("--no-warmup", action="store_true",
                           help="skip the discarded warm-up TIP run before run 1 (only for --no-build onto a box whose page cache is already warm)")
        p.add_argument("--name", default="arms", help="round name: files, worktrees, labels and the lane ARMS-<NAME> (default %(default)s)")
        p.add_argument("--dry-run", action="store_true", help="print everything, touch nothing")

    for name, fn, doc in (("plan", cmd_plan, "resolve shas, print the design, run count and wall estimate"),
                          ("seal", cmd_seal, "write the PREREGISTERED-RULE file; commit it before run"),
                          ("build", cmd_build, "push shas to sparkbox, detached worktrees, make cuda-spark, .text gate"),
                          ("run", cmd_run, "plan + seal check + build + start (lock, watchdog, driver, anchored pgrep)")):
        p = sub.add_parser(name, help=doc, description=doc, epilog="Examples:\n" + EX[name], formatter_class=argparse.RawDescriptionHelpFormatter)
        common(p)
        p.set_defaults(fn=fn)
    sub.choices["run"].add_argument("--no-build", action="store_true", help="skip the build; read ~/sweeps/BUILD_<name>.log")
    p = sub.add_parser("status", help="END/OK/KILLED counts, last stamps, load, memavail, vitest, watchdog CONTs",
                       epilog="Examples:\n" + EX["status"], formatter_class=argparse.RawDescriptionHelpFormatter)
    common(p, arms=False)
    p.set_defaults(fn=cmd_status)
    p = sub.add_parser("analyze", help="the paired statistic and the verdict per arm by the sealed rule",
                       epilog="Examples:\n" + EX["analyze"], formatter_class=argparse.RawDescriptionHelpFormatter)
    common(p, arms=False)
    p.add_argument("--fetch", action="store_true", help="pull plan_<name>.txt, RUNLOG_<name>.txt and <name>_*.csv from ssh spark first")
    p.add_argument("--sweeps", help="local dir holding them (default sweeps/<name>)")
    p.add_argument("--plan", help="plan file (default <sweeps>/plan_<name>.txt)")
    p.add_argument("--frontiers", default=DEFAULT_FRONTIERS, help="used only if the plan header carries none")
    p.add_argument("--legacy-straggler", action="store_true",
                   help="the round-4 straggler rule (TIP filtered too, from n=3) for rounds 5 and 6, sealed under it")
    p.set_defaults(fn=cmd_analyze)
    return ap


def main(argv=None):
    ap = build_parser()
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if not a.cmd:
        ap.error("a COMMAND is needed: plan, seal, build, run, status or analyze (--help for examples)")
    try:
        return a.fn(a)
    except Usage as e:
        print("error: %s" % e)
        ap.print_usage()
        sys.exit(2)


if __name__ == "__main__":
    sys.exit(main())
