# V4.1 Engram batch reader, GB10 record

NVIDIA DGX Spark, GB10 (sm_121), 128 GB unified memory, driver 580.159.03, CUDA 13.0, NVMe, Ubuntu 24.04.
Model `DeepSeek-V4.1-Flash-Q2.gguf`. Base commit `9139e2ae5`.
Build recipe `make -B ds4 ds4-server ds4-bench ds4-eval ds4-agent`, no `CUDA_ARCH`.

## The cost being removed

Measured on the base commit with `DS4_V41_ENGRAM_PROFILE=1`, which times the blocking Engram
table read at the head of each decode step against the step that waits on it. 31 decode steps,
45-token prompt, first run after model load, so the page cache holds none of the rows.

    engram ms   min 0.208  p50 26.581  p90 33.515  max 37.414  mean 23.922
    step ms     min 143.5  p50 189.2   p90 253.8   max 299.6   mean 198.636
    engram share of step: 12.04%

Bytes per step, derived from the GGUF header rather than guessed:

    blk.1.engram_embd.weight    dims [264, 384006168]  abs offset 162,955,640,832
    blk.14.engram_embd.weight   dims [264, 384016682]  abs offset 264,333,279,232
    per step  2 tables x 24 rows x 264 bytes = 12,672 bytes in 48 pread calls
    tables    384,006,168 + 384,016,682 rows x 264 = 202,758,032,400 B = 188.83 GiB

## The drive, with no model and no GPU

The same 48-read pattern replayed through the shipped reader, 1,000 steps, uniformly scattered
row ids, standalone and cold:

    serial   min 9.001  p50 9.600   p90 11.299  max 18.953 ms
    batch    min 8.999  p50 10.054  p90 15.556  max 27.885 ms

About 200 microseconds per read. The `batch` row is `ds4_engram_read_batch` at one token, which
is 24 requests, below the old 256-request gate, so it took the serial path plus a sort and
measured slightly worse. That is the defect this patch fixes, visible as a number.

## Gates

    arm                                     sha256[0:16]        bytes
    short, patched                          bb06e711bc498bb9    73146
    short, patched DS4_ENGRAM_READ_THREADS=1 bb06e711bc498bb9   73146
    short, unpatched 9139e2ae5              bb06e711bc498bb9    73146
    long prompt 4,392 tokens, control       2f2dd7f89d107bbc    33527
    long prompt, patched                    2f2dd7f89d107bbc    33527
    long prompt, control run again          2f2dd7f89d107bbc    33527
    generation -n 256, control              652dcda32c176cab   383825
    generation -n 256, patched              652dcda32c176cab   383825

Window held the model lock 13m48s, 04:06:24Z to 04:20:12Z. Load 0.38 to 1.55, GPU 0 to 4
percent, MemAvailable 112 to 113 GB, driver 580.159.03 on every run. No run was signalled.

## A caution for anyone re-measuring this path

The Engram descriptor gets `F_NOCACHE` and `F_RDAHEAD(0)` on macOS only. On Linux it is an
ordinary cached descriptor, so a repeated n-gram is served from the page cache at about 0.15 ms
while a novel one costs 20 to 25 ms. Two consequences, both measured here the hard way:

- **Two consecutive runs of one prompt are not two samples.** The second inherits the first's
  rows. A second run of the 45-token prompt read 11.4 ms per step against the cold run's 24.1,
  and its first 31 steps were positions the earlier run had just read.
- **Arm order is a variable.** A sweep that runs its arms in sequence over the same row set
  measures the first arm honestly and the rest against the cache that arm filled. One such
  sweep here reported a 96-fold speedup, which is not physical.

A baseline prompt is effectively single-use: once a window has run it, its rows are resident and
a rerun reports a warm number as a cold one.

## Still owed

The end-to-end A/B and the scaled probe row. <!-- PENDING SWEEP -->
