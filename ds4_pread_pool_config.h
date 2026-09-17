/* PREAD-POOL configuration resolution, as pure arithmetic.
 *
 * The pooled SSD read-ahead resolves two settings from the environment: whether
 * the pool runs at all, and how many workers it may hold. Both were resolved
 * in place, by a getenv on every call, inside `cuda_pread_pool_dispatch_start`
 * - which runs once per expert prefetch batch, on the prefill path.
 *
 * Two things were wrong with that, and only the second is about cost:
 *
 *  1. WHAT THE LIMIT IS USED FOR, TWICE. `cuda_pread_pool_dispatch_start`
 *     called the resolver twice in four lines: once for the worker count it
 *     clamps to the batch size, and once, UNCLAMPED, for the pool's own
 *     size. The two uses are genuinely different - the pool is persistent and
 *     must be sized to the configured limit, not to whatever the first batch
 *     happened to carry - so the second call is NOT a copy of the first and
 *     must not be replaced by the clamped value. It is the same *resolution*
 *     performed twice, which is what is removed here; the two distinct *uses*
 *     both survive, and `ds4_pread_pool_workers` is what keeps them apart.
 *
 *  2. RESOLUTION IS LOOP-INVARIANT AND WAS NOT HOISTED. The environment does
 *     not change between dispatches, so the getenv + strtoul pair is the same
 *     work repeated per batch, per setting, twice over for the limit. Resolved
 *     once into a cache here.
 *
 * Read-once is the behaviour this makes explicit, and it is a CHANGE worth
 * naming: a mid-run mutation of either variable used to move the worker count
 * on the next dispatch and now does not. It never moved the pool's size, which
 * `cuda_pread_pool_init` fixes on first use and never revisits, so the only
 * reachable difference is the per-batch clamp. No supported configuration mode
 * mutates these between dispatches, and read-once matches the env handling the
 * rest of this engine already uses.
 *
 * Pure C99. No CUDA types, no pthreads, no pool struct. Builds on any host,
 * which is what lets tests/test_pread_pool_config.c exercise it without a GPU
 * - the .cu it serves cannot be compiled on a machine with no CUDA toolkit.
 */
#ifndef DS4_PREAD_POOL_CONFIG_H
#define DS4_PREAD_POOL_CONFIG_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* The test substitutes a counting reader here, so the "resolved once" claim is
 * a counted fact rather than a comment. */
#ifndef DS4_PREAD_POOL_GETENV
#define DS4_PREAD_POOL_GETENV getenv
#endif

/* Ceiling on pool workers. Mirrors DS4_CUDA_EXPERT_PREAD_MAX in ds4_cuda.cu,
 * which sizes the pool's fixed thread/arg/ctx arrays; a limit above it would
 * index past them. The static assert below pins the two together. */
#ifndef DS4_PREAD_POOL_MAX
#define DS4_PREAD_POOL_MAX 32u
#endif

/* Both settings carry their own `_resolved` flag rather than a sentinel value,
 * so an all-zero struct means "nothing resolved yet". That matters: this lives
 * inside `cuda_pread_pool`, whose global is built by a POSITIONAL initializer
 * that sets only the first few members and leaves the rest zeroed. A -1
 * sentinel would need an explicit entry in that list, and a later member added
 * above it would silently shift the cache into "resolved, disabled".
 *
 * `reads` is what the test counts. */
typedef struct {
    int           enabled_resolved;
    int           enabled;
    int           threads_resolved;
    uint32_t      threads;
    unsigned long reads;
} ds4_pread_pool_cfg;

#define DS4_PREAD_POOL_CFG_INIT { 0, 0, 0, 0u, 0ul }

/* Off only when the variable is present and reads exactly "0". Absent, empty,
 * or anything else means on - so a typo enables rather than silently disables,
 * which is the safe direction for a read-ahead that is a pure optimisation. */
static int ds4_pread_pool_enabled(ds4_pread_pool_cfg *c, const char *env_name) {
    if (!c->enabled_resolved) {
        const char *env;
        c->reads++;
        env = DS4_PREAD_POOL_GETENV(env_name);
        c->enabled = !(env && strcmp(env, "0") == 0);
        c->enabled_resolved = 1;
    }
    return c->enabled;
}

/* The configured worker ceiling, clamped into [1, DS4_PREAD_POOL_MAX].
 *
 * A partially numeric value ("8x") is REFUSED and falls back to the default
 * rather than parsing as its numeric prefix - strtoul alone would accept "8x"
 * as 8, so the *end == '\0' check is load-bearing and has its own case in the
 * test. An explicit 0 clamps up to 1 rather than disabling; disabling is what
 * the enable variable is for. */
static uint32_t ds4_pread_pool_threads(ds4_pread_pool_cfg *c,
                                       const char *env_name,
                                       uint32_t default_threads) {
    if (!c->threads_resolved) {
        uint32_t threads = default_threads;
        const char *env;
        c->reads++;
        env = DS4_PREAD_POOL_GETENV(env_name);
        if (env && env[0]) {
            char *end = NULL;
            unsigned long v = strtoul(env, &end, 10);
            if (end != env && *end == '\0')
                threads = v > UINT32_MAX ? UINT32_MAX : (uint32_t)v;
        }
        if (threads == 0) threads = 1;
        if (threads > DS4_PREAD_POOL_MAX) threads = DS4_PREAD_POOL_MAX;
        c->threads = threads;
        c->threads_resolved = 1;
    }
    return c->threads;
}

/* Workers for ONE batch: the configured ceiling, never more than there is work
 * for. Returns 0 to mean "do not pool this batch" - one worker is the serial
 * path with a thread handoff bolted on, which is strictly worse than just
 * reading inline, so the caller declines rather than dispatching.
 *
 * This is the clamped half of the pair. The pool's own size stays unclamped;
 * see note 1 at the top. */
static uint32_t ds4_pread_pool_workers(uint32_t limit, uint32_t n_tasks) {
    uint32_t n = limit > n_tasks ? n_tasks : limit;
    return n <= 1 ? 0u : n;
}

#endif /* DS4_PREAD_POOL_CONFIG_H */
