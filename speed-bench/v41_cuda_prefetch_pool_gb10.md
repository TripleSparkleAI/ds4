# V4.1 Flash, CUDA: the SSD read-ahead in parallel, on a GB10

Raw commands, gate output and the sweep behind branch `triple-prefetch-pool`.

## The box

NVIDIA DGX Spark, GB10 (sm_121), 128 GB unified memory, driver 580.159.03, CUDA 13.0, NVMe boot
drive, Ubuntu 24.04. Base commit `9139e2ae5` plus `cuda: parallel pread pool for the SSD expert
cache misses`. Build on every arm, no `CUDA_ARCH`:

    make -B ds4 ds4-server ds4-bench ds4-eval ds4-agent

## The prompt, and why it is not the standard one

The project's standard greedy gate uses a 45-token prompt. The read-ahead is gated at 2048 tokens,
so that gate never enters `ds4_gpu_stream_expert_cache_prefetch` at all and cannot see this patch.
Every arm below uses a fixed 14,000-byte slice of `speed-bench/promessi_sposi.txt`, which
tokenises to 4,392 prompt tokens:

    head -c 14000 speed-bench/promessi_sposi.txt > /tmp/long_prompt.txt
    ./ds4 -m DeepSeek-V4.1-Flash-Q2.gguf --prompt-file /tmp/long_prompt.txt --dump-tokens | wc -l
    # 4392

## One arm

    env DS4_CUDA_SSD_PREFETCH_THREADS=4 DS4_CUDA_SSD_PREFETCH_PROFILE=1 \
      ./ds4 -m DeepSeek-V4.1-Flash-Q2.gguf --cuda --ssd-streaming \
        --ssd-streaming-cache-experts 90GB --ctx 32768 --temp 0 -n 16 \
        --prompt-file /tmp/long_prompt.txt --dump-logprobs /tmp/out.json

Knobs: `DS4_CUDA_SSD_PREFETCH_POOL=0` restores the single reader ·
`DS4_CUDA_SSD_PREFETCH_THREADS` sets the worker count, default 4 ·
`DS4_CUDA_SSD_PREFETCH_CHUNK_MB` sets the task and per-worker staging size, default 8 ·
`DS4_CUDA_NO_DIRECT_IO=1` forces both read paths buffered.

## Correctness: twelve runs, one output

Every run below produced `/tmp/<arm>.json` with **sha256 `2f2dd7f89d107bbc`, 33527 bytes**, and
reported `published=1` on all **38** read-ahead layers. That value is what the unpatched base tree
gives on the same prompt and flags.

    arm              readers  I/O        sha256[0:16]        bytes   layers published=1
    single_a         1        O_DIRECT   2f2dd7f89d107bbc    33527   38
    t4_a             4        O_DIRECT   2f2dd7f89d107bbc    33527   38
    t8_a             8        O_DIRECT   2f2dd7f89d107bbc    33527   38
    t16_a           16        O_DIRECT   2f2dd7f89d107bbc    33527   38
    t24_a           24        O_DIRECT   2f2dd7f89d107bbc    33527   38
    t8nodir_a        8        buffered   2f2dd7f89d107bbc    33527   38
    singlenodir_a    1        buffered   2f2dd7f89d107bbc    33527   38
    single_b         1        O_DIRECT   2f2dd7f89d107bbc    33527   38
    t4_b             4        O_DIRECT   2f2dd7f89d107bbc    33527   38
    t8_b             8        O_DIRECT   2f2dd7f89d107bbc    33527   38
    t16_b           16        O_DIRECT   2f2dd7f89d107bbc    33527   38
    t24_b           24        O_DIRECT   2f2dd7f89d107bbc    33527   38

Seven distinct configurations, twelve runs, one output.

The pooled arms are demonstrably using the pool rather than quietly declining to. A pooled run
prints its own pool beside the demand pool's:

    ds4: CUDA expert pread pool: 8 threads
    ds4: CUDA ssd prefetch pread pool: 4 threads

A serial arm prints only the first line.

## The profile, verbatim

`DS4_CUDA_SSD_PREFETCH_PROFILE=1`. `read` is issue-to-drain wall time and overlaps the compute;
`wait` is the foreground blocked in `ds4_gpu_stream_expert_cache_prefetch_finish`, which is the
quantity this patch reduces.

Serial reader, first two layers:

    ds4: CUDA SSD prefetch layer=2 experts=384 bytes=3822059520 read=2976.888 wait=336.164 ms published=1
    ds4: CUDA SSD prefetch layer=3 experts=384 bytes=3822059520 read=1561.007 wait=0.143 ms published=1

Four readers, same two layers:

    ds4: CUDA SSD prefetch layer=2 experts=384 bytes=3822059520 read=2681.465 wait=0.058 ms published=1
    ds4: CUDA SSD prefetch layer=3 experts=384 bytes=3822059520 read=1464.514 wait=0.072 ms published=1

Note `bytes=3822059520` on every line of every arm: the read-ahead moves the same 3.82 GB per
layer whatever the reader is. A prefetch changes when bytes arrive, never how many.

## The sweep

Two passes, arms interleaved in the order listed within each pass, pass A then pass B, a fresh
process per run. The runner stamped load, GPU utilisation, MemAvailable and driver before each run
and refused to start one at load above 2 or under 100 GB available, waiting and re-checking four
times rather than recording a confounded number. Stamps across the twelve runs: load 0.07 to 2.59,
GPU 0 to 7 percent, MemAvailable 111 to 113 GB, driver 580.159.03 throughout.

Means over 38 layers, milliseconds:

    readers   wait A   wait B   wait MIN   read A   read B   read MIN   pinned staging
    1         398.0    403.8     398.0     2588.9   2528.6    2528.6    16 MiB (two buffers)
    4         203.0    196.5     196.5     2164.8   2238.7    2164.8    32 MiB
    8         227.1    221.6     221.6     2255.3   2197.0    2197.0    64 MiB
    16        226.3    223.0     223.0     2314.0   2241.9    2241.9   128 MiB
    24        198.2    VOID      198.2*    2199.3   VOID      2199.3   192 MiB

    * pass B's 24-reader arm is WITHDRAWN, so that column is one sample.

Totals over the whole prefill, milliseconds: `wait` 15122.2 serial against 7468.2 at four readers.

### The withdrawn arm

Lane PREFILLHANDOFF found this sweep's process stopped by a signal mid-run, holding the model
lock, and restarted it with `kill -CONT`, verifying the T to R transition. The arm in flight was
`t24_b`. A run whose timing crossed a stop is not offered as timing, so it is withdrawn.

The visible trace is small, and saying so is not a licence to keep the arm: `t24_b` aggregated to
198.1 ms against `t24_a`'s 198.2, while its `read` mean was 2.8 percent higher at 2261.8 against
2199.3. Its OUTPUT is untouched and still counts - a stopped process resumes and finishes the same
arithmetic, which is why all twelve runs remain at one sha.

What survives two passes: four beats eight and sixteen by about 11 percent, with no overlap between
four's worse run at 203.0 and eight's better run at 221.6. What survives one pass: twenty-four near
four, unreplicated. The shape is a step from one to four, then eight and sixteen reproducibly worse
than four. It is NOT a flat band from four upward.

### O_DIRECT, one pass

    readers   I/O        read mean   wait mean
    8         O_DIRECT     2255.3      227.1
    8         buffered     3188.2      828.7
    1         O_DIRECT     2588.9      398.0
    1         buffered     5540.7     3177.8

The read-ahead already uses `O_DIRECT` by default: `ds4_gpu_stream_expert_cache_prefetch` dups
`g_model_direct_fd`, which is opened unconditionally on Linux unless `DS4_CUDA_NO_DIRECT_IO` is
set, and the `posix_fadvise(DONTNEED)` in the serial reader is guarded by `p.direct_fd < 0`, so it
fires only on the buffered fallback. These figures therefore price what the existing design already
collects, not a lever this patch adds. MemAvailable did not move between the direct and buffered
arms at this prompt size.

## The standard gate

Run for completeness. It passes and it does not reach this code:

    rm -f /tmp/gate.json
    ./ds4 -m DeepSeek-V4.1-Flash-Q2.gguf --cuda --ssd-streaming \
      --ssd-streaming-cache-experts 90GB --ctx 32768 --temp 0 -n 32 \
      --dump-logprobs /tmp/gate.json -p 'Explain in two sentences why the sky is blue.'
    sha256sum /tmp/gate.json | cut -c1-16    # bb06e711bc498bb9, 73146 bytes

Its prompt is 45 tokens against a 2048-token threshold, so no prefetch line appears in that run
and only the demand pool's init line is printed.

## The page cache, and whether these arms were cold

**They were WARM.** The flags that would bypass the Engram file descriptor's page cache are
Apple-only, so on this box it stays cached, and every arm ran the same prompt against the same
model back to back. Only the first run of a session met a cold cache. These arms are
prefill-dominated, so the effect bears on them less than it would on a decode lane, but two
consecutive same-prompt passes are less independent than two passes normally imply. Read the pass
A against pass B agreement with that discount.

## Not measured

The chunk-size knob at any value · both pools running concurrently against one drive · sorting a
batch by file offset · what fraction of a layer a prefill chunk actually touches · whether the
2048-token threshold can be lowered · an end-to-end `ds4-bench` context sweep.
