# Prefill expert scheduling

An optimal expert fetch schedule for layer-major prefill, for DeepSeek V4.1 Flash running from
SSD-streamed expert weights.

Files: `ds4_prefill_sched.h`, `ds4_prefill_sched.c`, test `./ds4_test --prefill-expert-schedule`.

## Why prefill is a different problem from decode

Prefill is layer-major. Every prompt token passes through layer 0, then layer 1, and so on, which
is what `prefill_layer_major_cpu` in `ds4.c` says and what the batch matmul depends on. That
ordering gives prefill a property decode does not have.

When layer L-1 has finished for every prompt token, the router for layer L can be run for every
token at once. The set of experts layer L will consult is then **exactly known**, before a single
expert byte is read. The router is a small per-layer F32 projection, so learning the demand costs
almost nothing next to the expert weights it decides.

During decode the same trick gives one layer of warning. At roughly 9.7 tokens per second and 40
layers, one layer is about 2.6 ms, and one 9,953,280-byte expert slot takes about 2.0 ms to read
from a single NVMe. So decode buys about one slot of lead, which is not a schedule. Prefill runs a
whole prompt's compute per layer, so there is real time to use.

## What this adds

With the demand known, choosing what to read and what to drop stops being a guess:

- **Evict by Belady MIN.** Drop the resident expert whose next use is furthest away, or never.
- **Fetch just in time.** Issue each read in the order its layer consumes it, no earlier than the
  lead window allows.

Belady MIN is optimal for a fixed capacity under a known request sequence, so the fetch count is a
floor. Nothing online can do better on the same demand. That makes the result useful twice: as a
schedule to run, and as a ruler that prices how much any cheaper policy leaves behind.

`ds4_prefill_schedule_lru` runs the identical walk with LRU eviction, so a difference in the
statistics is a difference in the eviction rule and in nothing else.

## What it does not do

It is a pure function over a demand table. It reads no model, starts no thread, and is not wired
into the streaming fetch path. Nothing on any inference backend changes, so there is no correctness
or speed regression surface to check. Wiring it to the router and to `ds4_ssd`'s reads is separate
work, and should be measured before it is believed.

The existing prefill layer read-ahead (`DS4_METAL_ENABLE_STREAMING_PREFILL_LAYER_READAHEAD` and its
ROCm twin) is a byte-range `F_RDADVISE` advisory. It hints ranges and the kernel decides. It makes
no victim choice and imposes no order, which is the gap this schedule fills.

## The worked case in the test

Two slots, four layers, demand `0, 1, 2, 0`:

| policy | what it does | reads |
|---|---|---|
| LRU | evicts 0 at layer 2 because it is oldest, so layer 3 must read 0 again | **4** |
| Belady | evicts 1 at layer 2 because 1 is never needed again and 0 is needed at layer 3, so layer 3 hits | **3** |

One read of 9.49 MiB is the difference, on a case small enough to check by hand.

The test also asserts that the accounting closes (`hits + fetches == needed`), that a fetch is
issued no later than the layer consuming it, that the schedule is byte-identical across runs, that
a short output buffer reports the length required instead of overflowing, that a repeated expert
within one layer is one read, and that a layer needing more distinct experts than the cache holds
is refused rather than thrashed.

## Running it

```sh
make clean
make ds4_test
./ds4_test --prefill-expert-schedule
```

Measured on an Apple M5 Max, macOS 26.5, Metal backend, `cc -O3 -std=c99`: builds with zero
warnings and the test passes. The scheduler compiles and links standalone with no `ds4` objects and
with `-DDS4_NO_GPU`, since it depends on neither.

The eviction rule was red-proven rather than assumed. Flipping the Belady comparison to pick the
nearest next use instead of the furthest turns three assertions red, including the read count and
the hit count, and restoring it returns the test to green.
