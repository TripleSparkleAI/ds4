# CUDA parallel pread pool for the SSD expert cache: DGX Spark measurements

Patch: one commit on top of `a04f46fa4` ("DeepSeek v4.1 Flash support for CUDA"),
`ds4_cuda.cu` only. Machine: NVIDIA DGX Spark, GB10 (sm_121), 128 GB unified
memory, driver 580.159.03, CUDA 13.0, NVMe boot drive. Model:
`DeepSeek-V4.1-Flash-Q2.gguf` (IQ2_XXS/Q2_K mix, 341 GiB, Engram tables disk-only).
Build recipe on both arms: `make -B ds4 ds4-server ds4-bench ds4-eval ds4-agent`
with no `CUDA_ARCH` (the default JIT build).

## Correctness

Greedy dump, same prompt and flags on both builds:

```sh
./ds4 -m DeepSeek-V4.1-Flash-Q2.gguf --cuda --ssd-streaming \
  --ssd-streaming-cache-experts 90GB --ctx 32768 --temp 0 -n 32 \
  --dump-logprobs /tmp/gate.json -p 'Explain in two sentences why the sky is blue.'
sha256sum /tmp/gate.json | cut -c1-16
```

| build | sha256 of the dump (first 16) | bytes |
|---|---|---:|
| a04f46fa4 | bb06e711bc498bb9 | 73146 |
| a04f46fa4 + this patch | bb06e711bc498bb9 | 73146 |

Byte-identical (`cmp` on the two files). The patch changes when bytes arrive
from the drive, never what is computed. The build of the patched tree is warning-free.

`make cuda-regression` does not link on `a04f46fa4` itself on this box
(`undefined reference to ds4_deepseek4_attention_bounds` in
`tests/cuda_long_context_smoke`), which is the subject of PR 1030. It fails the
same way with and without this patch, so it is not a signal here.

## Throughput, 256 generated tokens, interleaved pairs

Same box, same model, same flags, `-n 256` on a short prompt, three pairs
interleaved (serial, pool, serial, pool, ...), each run a fresh process.
`DS4_CUDA_EXPERT_CACHE_STATS=1` for the read seconds. Load and MemAvailable stamped
per run; the box was shared with other processes and the expert cache was squeezed to
about 50 GB, so absolute numbers are below a quiet box.

| | serial (a04f46fa4) | pool |
|---|---|---|
| generation t/s | 5.96 / 5.73 / 6.23 | 6.55 / 6.87 / 6.62 |
| cumulative miss read time per run | 28-31 s | 20-21 s |
| short exact prefill t/s | 4.0-4.3 | 6.2-6.3 |

## ds4-bench sweep

Pending. The box carried another tenant holding 40-72 GB during the day, and
`ds4-bench` at `--ssd-streaming-cache-experts 90GB` was OOM-killed on both builds
before its first row. The sweep below runs on a quiet box and its CSVs land beside
this file:

```sh
./ds4-bench -m DeepSeek-V4.1-Flash-Q2.gguf --cuda --ssd-streaming \
  --ssd-streaming-cache-experts 90GB --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 2048 --ctx-max 6144 --step-incr 2048 --gen-tokens 128 --csv pr-<arm>.csv
```

## Why the pool helps

Threaded O_DIRECT probe on this drive, not `dd`: one 3 MiB tensor read takes about
416 us (7.6 GB/s); 8, 16 and 32 reads in flight deliver 10.1, 9.9 and 9.7 GB/s.
A miss is three tensors and a V4.1 Flash token touches 40 MoE layers with 6 experts
each, so the serial staging ring leaves most of the drive's bandwidth unused.
