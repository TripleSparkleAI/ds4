## CUDA: parallel pread pool for the SSD expert cache misses

One commit on top of a04f46fa4 ("DeepSeek v4.1 Flash support for CUDA"). Touches `ds4_cuda.cu` only.

### What it changes

The selected-expert load reads every miss through the serial staging ring, so the NVMe sees queue depth one during the miss phase. This patch sends the misses of a layer to a small pool of worker threads (8 by default, `DS4_CUDA_STREAMING_EXPERT_PREAD_THREADS`). Each worker owns a pinned staging buffer and its own upload stream, does the O_DIRECT pread and the H2D copy for one tensor, and the layer waits on the batch. Hits, the LRU slot cache, eviction and the gate-indexed lookup are unchanged. If the pool declines a batch (allocation failure, thread creation failure) the layer falls back to the existing serial path, so the output path is the same code either way.

`DS4_CUDA_EXPERT_CACHE_STATS=1` prints lookups, hit rate, resident count, evictions and cumulative read seconds every 4000 lookups.

### Why

A single 3 MiB tensor read costs about 416 us on the Spark NVMe (7.6 GB/s); with 8 to 16 reads in flight the drive delivers about 10 GB/s. One miss is three tensors, and a V4.1 Flash token has 40 MoE layers with 6 experts each, so the serial ring leaves most of the drive's bandwidth on the table.

### Measured (DGX Spark GB10, driver 580.159.03, V4.1 Flash Q2, ctx 32768, 256 generated tokens, `-n 256`)

| | serial (a04f46fa4) | pool |
|---|---|---|
| generation t/s, three interleaved runs | 5.96 / 5.73 / 6.23 | 6.55 / 6.87 / 6.62 |
| cumulative miss read time per run | 28-31 s | 20-21 s |
| short exact prefill t/s | 4.0-4.3 | 6.2-6.3 |

About +12% generation. The box had other tenants during these runs (the expert cache arena was squeezed to about 50 GB), so absolute numbers are low; the pairs were interleaved and each arm is a minimum-of-three.

### Correctness

Greedy `--temp 0 --dump-logprobs` byte-identical against the pre-patch build on the same model, both for V4 Flash and V4.1 Flash (sha256 of the dump unchanged). The pool only changes when bytes arrive, not what is computed.

### Not included

A hits-first variant (compute the resident pairs while the miss reads stream) measured a null on this tree because the shared-expert overlap already fills that window. It is left out.
