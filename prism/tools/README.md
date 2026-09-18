# prism/tools - bring-up and measurement of Ternary Bonsai 2 27B on the Spark GB10

One tool, `prism_spark.py` (python3 stdlib, no deps). Run it from the Mac; it drives the spark
over `ssh spark`. Every subcommand is idempotent and resumable, prints every remote command
under `--dry-run`, and never deletes anything except its own lock and its own `.BAD` marker.

    python3 prism/tools/prism_spark.py --help
    python3 prism/tools/prism_spark.py selftest              # offline checks + a planted RED mutation
    python3 prism/tools/prism_spark.py status                # lock, clone sha, models, binaries, last CSVs

## The steps, in order

| step | what it does on the spark | lock |
|---|---|---|
| `fetch --clone-only` | clones `PrismML-Eng/llama.cpp` branch `prism`, checks out the pin `5d80cff0b8cb...`, writes `~/prism/PIN.txt` | none (network only) |
| `fetch` | the clone, then downloads `Ternary-Bonsai-2-27B-PTQ1_0.gguf` (5,946,648,928 B) and `-PQ2_0.gguf` (7,206,168,928 B) into `~/prism/models/`, byte-verified; a mismatch is refused and the file left in place with a `.BAD` note | WAITS for FREE, then `PRISM-FETCH` |
| `build` | cmake with CUDA for `121a` (sm_121; the demo's default list omits the Spark), Ninja, `-j12`, log `~/prism/build.log`, binaries + `.text` sizes + the `--parallel` default into `~/prism/BUILD.txt` | `PRISM-BUILD` |
| `bench1` | `llama-bench` pp512 / tg128 at KV depth 2048, `-fa 1 -ngl 99`, one discarded warm-up then 3 repeats, BOTH quants, stamped (load, gpu, memavail, driver, CUDA, fork sha, model sha256[:16]); `prism/measured/bench1_<utc>.csv` | `PRISM-SWEEP` |
| `sweep` | `llama-batched-bench` npl 1,4,8,16,32,64; then `llama-server --parallel N -c N*8192` for N in 1,4,8,16 with N concurrent ~3.4k-token prompts from `speed-bench/promessi_sposi.txt` (TTFT p50/p95, aggregate t/s); CSVs scp'd to `prism/measured/`; prints the knee (largest N with per-stream t/s above 10) | `PRISM-SWEEP` |

`--quant PTQ1_0|PQ2_0` defaults to `PQ2_0` (what the demo loads). `bench1` runs both regardless.

## The lock

`~/spark_model.lock/owner` on the spark is shared with the dwarfstar rounds (`track0_harness/spark_arms.py`).
Every step that touches the GPU or the NVMe refuses when another lane holds it (`REFUSED_LOCK held-by: ...`),
takes it under its own lane name, and releases it only if it still holds it. The model download waits for a
FREE lock rather than queue-jumping: a 7 GB write contends with a running round's streaming.

## How a long step runs

The tool writes `~/prism/<step>.sh` on the spark and starts it with `setsid bash ... </dev/null > ~/prism/<step>.log 2>&1 &`.
Liveness is an anchored `pgrep -f "^bash /home/hologram/prism/<step>.sh"`. The Mac side polls the log until
`DONE <step>`; a killed Mac session re-runs the same subcommand to resume polling (an already-running step is
reported as `ALREADY_RUNNING`, not started twice).

## Numbers

Every CSV row carries the stamp string and the checkpoint `Bonsai 2 27B <quant>`. Every public Spark figure
in `prism/research/00_FINDINGS.md` is generation 1 (Ternary-Bonsai-27B); a comparison across that seam is
confounded and must say so.
