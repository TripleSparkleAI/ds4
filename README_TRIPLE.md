# triple-cuda-speedups

Upstream `a04f46fa4` ("DeepSeek v4.1 Flash support for CUDA") plus the
parallel pread pool (`triple-pr-parallel-ssd-reads`) plus one commit per
lever from the TripleSparkle backend, each re-derived by hand against
upstream's slot cache and gated on a DGX Spark with the same greedy
logprob dumps (short `bb06e711bc498bb9`, long `d3355c94c70a4bb1`).

1. **Expert cache reserve knob** - `DS4_CUDA_EXPERT_CACHE_MARGIN_GB`, default
   8 GiB unchanged. Gate: byte-identical.
2. **Persistent learned hot list** - the cache writes what it was asked for to
   `~/.cache/ds4/cuda_expert_hotlist.txt` at exit; V4.1 Flash seeds from it
   next run. Gate: byte-identical, seed active.
3. **Event-gated expert-id readback (DRAINCUT)** - the streaming load waits
   on an event recorded after the router, not on a device drain that also
   covers the shared expert. Gate: byte-identical.
4. **Router argmax** - skipped, upstream already runs a warp-wide top-k.
5. **Zero-copy planar arena** - skipped for decode (upstream's slot cache is
   already read in place); prefill still compacts per layer, noted.
6. **Page cache** - madvise before fadvise so mapped pages really drop, and
   readahead of misses on the buffered path. Gate: byte-identical.
7. **Hits-first** - opt-in `DS4_CUDA_HITS_FIRST=1`, measured a null on this
   tree. Gate: byte-identical with the flag off.

`CUDA_SUGGESTIONS.md` has the detail: what each lever changes, why, gate and
speed status, and every knob.

Full backend: `triple-cuda-backend-archive`.
