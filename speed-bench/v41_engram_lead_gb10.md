# V4.1 Engram decode read, a token of lead: GB10 record

NVIDIA DGX Spark, GB10 (sm_121), 128 GB unified memory, driver 580.159.03, CUDA 13.0, NVMe, Ubuntu 24.04.
Model `DeepSeek-V4.1-Flash-Q2.gguf`. Base commit `9139e2ae5`.
Build recipe `make -B ds4 ds4-server ds4-bench ds4-eval ds4-agent`, no `CUDA_ARCH`.

## The cost being hidden

`DS4_V41_ENGRAM_PROFILE=1` times the blocking two-table Engram read at the head of each decode step
against the step that waits on it. 31 steps, 45-token prompt, first run after model load:

    engram ms   min 0.208  p50 26.581  p90 33.515  max 37.414  mean 23.922
    step ms     min 143.5  p50 189.2   p90 253.8   max 299.6   mean 198.636
    engram share of step: 12.04%

The read completes before `ds4_gpu_begin_commands()` is called, so none of it is overlapped.

## Why a token of lead rather than the prefill shape

Upstream's prefill path already starts table 0's read before layer 0 and table 1's at layer 2,
joining each before the layer that consumes it. Copying that shape into decode is
speculation-free and needs no cross-step state, and it was measured and rejected:

    lead available          covers a 20-25 ms cold read?
    table 0, at layer 1      1/40 of 199 ms  =  ~5 ms    NO
    table 1, at layer 14    14/40 of 199 ms  =  ~70 ms   yes
    a full token             199 ms                      yes, both

A public entry point the caller invokes after sampling was rejected on a second reading: the
generation loops call `ds4_session_argmax` and then `ds4_session_eval` with only bookkeeping
between, so the lead available that way is microseconds, and it would need changes at every call
site in the CLI, the server and the agent.

## The invalidation contract

The row ids are a pure function of the token and the rolling history. `ds4_engram_hash` reads the
token id and a three-entry tail, no activation and no layer output. The reader therefore carries
the exact `(token, history, pos)` it was hashed under, and the consumer publishes nothing unless
all three match what the next step asks for.

    match = l->token == token && l->pos == pos &&
            memcmp(&l->history, history, sizeof(*history)) == 0;
    if (!join(l) || !match) return false;      /* join ALWAYS, publish only on match */

A rejected speculative token changes the token. A session rewind changes both: `ds41_load_payload`
rebuilds `history.tail` from the restored tokens and sets `pos`. A fork changes pos. Each misses,
joins, and falls back to a demand read. Because the rows depend on nothing but the token and the
history, even an exact collision of all three would publish correct rows.

## Build

    tree                       warnings  errors
    lead (this branch)                1       0
    threads (sibling branch)          1       0
    control, unpatched 9139e2ae5      1       0

The one warning is `bits/unistd.h:38 __pread_alias ... -Wstringop-overflow`, from glibc inside
upstream's own code. This patch adds none of its own.

## A caution for anyone re-measuring this path

The Engram descriptor gets `F_NOCACHE` and `F_RDAHEAD(0)` on macOS only. On Linux it is an
ordinary cached descriptor, so a repeated n-gram is served from the page cache at about 0.15 ms
while a novel one costs 20 to 25 ms. Consequences learned here by getting them wrong first:

- **Two consecutive runs of one prompt are not two samples.** A second run of the 45-token prompt
  read 11.4 ms per step against the cold run's 24.1.
- **Arm order is a variable.** A probe that ran its arms in sequence over one fixed row set
  reported a 96-fold speedup, which is not physical: every arm after the first was reading the
  rows the first had just faulted in.
- **A baseline prompt is single-use.** Once a window has run it, its rows are resident.
- **The lead's value depends on how novel the generated text is**, so a repetitive prompt
  understates it and a long prefill may warm the rows its own continuation needs.

## Still owed

Both gates, the lead-on against lead-off comparison, the end-to-end A/B, and a genuinely cold
long-prompt measurement taken as the first model run in a window. <!-- PENDING SWEEP -->
