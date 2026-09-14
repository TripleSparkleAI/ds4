This reads the SSD expert-cache misses of a layer in parallel instead of one at a time. On a DGX Spark, DeepSeek V4.1 Flash Q2 generation goes from 5.73-6.23 to 6.55-6.87 tokens/s, about +12%, with the greedy output byte-identical.

### Cause and change

The selected-expert load runs every miss through the serial staging ring, so the NVMe sees queue depth one during the miss phase. One 3 MiB tensor read costs about 416 us on this drive (7.6 GB/s); with 8 to 16 reads in flight it delivers about 10 GB/s. A miss is three tensors and a V4.1 token touches 40 MoE layers with 6 experts each, so most of the drive's bandwidth was unused.

The misses of a layer now go to a small pool of worker threads (8 by default, `DS4_CUDA_STREAMING_EXPERT_PREAD_THREADS`). Each worker owns a pinned staging buffer and its own upload stream, does the O_DIRECT pread and the H2D copy for one tensor, and the layer waits on the batch. Hits, the LRU slot cache, eviction and the gate-indexed lookup are unchanged. If the pool declines a batch (allocation or thread failure) the layer falls back to the existing serial path. `DS4_CUDA_EXPERT_CACHE_STATS=1` prints lookups, hit rate, resident count, evictions and cumulative read seconds every 4000 lookups.

One file, `ds4_cuda.cu`, +488/-14. Metal and ROCm untouched.

### Performance

DGX Spark GB10 (sm_121), 128 GB unified memory, driver 580.159.03, CUDA 13.0, NVMe boot drive. `DeepSeek-V4.1-Flash-Q2.gguf`, `--ssd-streaming --ssd-streaming-cache-experts 90GB --ctx 32768`, 256 generated tokens on a short prompt, each run a fresh process. Base: `a04f46fa4`. Three pairs interleaved, serial then pool.

| Run | Pool | Generation t/s | Miss reads per run |
| --- | --- | ---: | ---: |
| 1 | off | 5.96 | 28-31 s |
| 2 | on | 6.55 | 20-21 s |
| 3 | off | 5.73 | 28-31 s |
| 4 | on | 6.87 | 20-21 s |
| 5 | off | 6.23 | 28-31 s |
| 6 | on | 6.62 | 20-21 s |

Short exact prefill 4.0-4.3 to 6.2-6.3 t/s. The box was shared with other processes during these runs and the expert cache was squeezed to about 50 GB, so absolute numbers are below a quiet box; the pairs were interleaved under the same conditions.

### Tests

- Greedy `--temp 0 --dump-logprobs`, same prompt and flags, base vs patched: sha256 `bb06e711bc498bb9` on both, `cmp` identical.
- Clean build of the patched tree with the default recipe, no warnings.
- `make cuda-regression` does not link on `a04f46fa4` itself on this box (`ds4_deepseek4_attention_bounds` undefined in the smoke test, the subject of PR 1030); it fails the same way with and without this patch.
- Not run: `make test` on the CUDA box. A `ds4-bench` before/after sweep is pending a quiet box; the box carried another tenant's 40-72 GB process during the day and the bench was OOM-killed before its first row on both builds.

[Commands, gate output and the bench sweep](https://github.com/TripleSparkleAI/ds4/blob/triple-pr-parallel-ssd-reads/speed-bench/v41_cuda_pread_pool_gb10.md).
