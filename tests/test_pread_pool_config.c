/* tests/test_pread_pool_config.c - the pread pool's config resolution.
 *
 * The logic under test serves ds4_cuda.cu, which does not compile on a host
 * with no CUDA toolkit. It lives in a pure-C99 header for exactly that reason,
 * so these run anywhere.
 *
 * The read-count cases are the point of the exercise: the resolver used to run
 * a getenv per call, twice per dispatch for the thread limit. A counting reader
 * is substituted for getenv below, so "resolved once" is a counted fact rather
 * than a claim. Against the unhoisted resolver these cases fail - the count
 * rises with the call count instead of stopping at one.
 */
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>

/* ---- the counting reader, substituted for getenv ------------------------ */

static unsigned long g_getenv_calls = 0;
static const char   *g_env_value    = NULL;   /* what the fake environment holds */
static const char   *g_last_name    = NULL;   /* which name was asked for */

static const char *test_getenv(const char *name) {
    g_getenv_calls++;
    g_last_name = name;
    return g_env_value;
}

#define DS4_PREAD_POOL_GETENV test_getenv
#include "../ds4_pread_pool_config.h"

/* ---- harness ------------------------------------------------------------ */

static int g_checks = 0, g_failed = 0;

static void check(int cond, const char *what) {
    g_checks++;
    if (!cond) { g_failed++; printf("  FAIL: %s\n", what); }
}

static void check_u32(uint32_t got, uint32_t want, const char *what) {
    g_checks++;
    if (got != want) {
        g_failed++;
        printf("  FAIL: %s (got %u, want %u)\n", what, got, want);
    }
}

static void check_ul(unsigned long got, unsigned long want, const char *what) {
    g_checks++;
    if (got != want) {
        g_failed++;
        printf("  FAIL: %s (got %lu, want %lu)\n", what, got, want);
    }
}

#define RUN(fn) do { printf("RUN: %s\n", #fn); fn(); printf("  ok\n"); } while (0)

/* Each case starts from a clean environment and a clean cache. */
static void reset(ds4_pread_pool_cfg *c, const char *value) {
    ds4_pread_pool_cfg fresh = DS4_PREAD_POOL_CFG_INIT;
    *c = fresh;
    g_env_value    = value;
    g_getenv_calls = 0;
    g_last_name    = NULL;
}

/* ---- 1. the enable switch ----------------------------------------------- */

static void test_enabled_when_unset(void) {
    ds4_pread_pool_cfg c;
    reset(&c, NULL);
    check(ds4_pread_pool_enabled(&c, "POOL_ON") == 1, "absent means on");
    check(strcmp(g_last_name, "POOL_ON") == 0, "asked for the name it was given");
}

static void test_disabled_only_by_exact_zero(void) {
    ds4_pread_pool_cfg c;
    reset(&c, "0");
    check(ds4_pread_pool_enabled(&c, "POOL_ON") == 0, "\"0\" disables");

    reset(&c, "00");
    check(ds4_pread_pool_enabled(&c, "POOL_ON") == 1, "\"00\" is not \"0\", stays on");

    reset(&c, "0 ");
    check(ds4_pread_pool_enabled(&c, "POOL_ON") == 1, "\"0 \" is not \"0\", stays on");

    reset(&c, "");
    check(ds4_pread_pool_enabled(&c, "POOL_ON") == 1, "empty stays on");

    reset(&c, "1");
    check(ds4_pread_pool_enabled(&c, "POOL_ON") == 1, "\"1\" stays on");

    reset(&c, "no");
    check(ds4_pread_pool_enabled(&c, "POOL_ON") == 1, "a typo enables, never silently disables");
}

/* ---- 2. the thread limit ------------------------------------------------ */

static void test_threads_default_when_unset(void) {
    ds4_pread_pool_cfg c;
    reset(&c, NULL);
    check_u32(ds4_pread_pool_threads(&c, "POOL_N", 8u), 8u, "absent gives the default");
}

static void test_threads_parses_a_clean_number(void) {
    ds4_pread_pool_cfg c;
    reset(&c, "4");
    check_u32(ds4_pread_pool_threads(&c, "POOL_N", 8u), 4u, "\"4\" parses");

    reset(&c, "12");
    check_u32(ds4_pread_pool_threads(&c, "POOL_N", 8u), 12u, "\"12\" parses");
}

static void test_threads_refuses_a_partial_number(void) {
    /* strtoul alone would take "8x" as 8. The *end == '\0' check is what
     * refuses it, so a mistyped value falls back to the default rather than
     * silently meaning something close to what was typed. */
    ds4_pread_pool_cfg c;
    reset(&c, "8x");
    check_u32(ds4_pread_pool_threads(&c, "POOL_N", 6u), 6u, "\"8x\" is refused, default kept");

    reset(&c, "4 ");
    check_u32(ds4_pread_pool_threads(&c, "POOL_N", 6u), 6u, "trailing space refused");

    reset(&c, "x");
    check_u32(ds4_pread_pool_threads(&c, "POOL_N", 6u), 6u, "non-numeric refused");
}

static void test_threads_clamps_both_ends(void) {
    ds4_pread_pool_cfg c;
    reset(&c, "0");
    check_u32(ds4_pread_pool_threads(&c, "POOL_N", 8u), 1u, "0 clamps up to 1");

    reset(&c, "9999");
    check_u32(ds4_pread_pool_threads(&c, "POOL_N", 8u), DS4_PREAD_POOL_MAX, "clamps to the max");

    reset(&c, "32");
    check_u32(ds4_pread_pool_threads(&c, "POOL_N", 8u), 32u, "exactly the max is kept");
}

static void test_threads_default_is_itself_clamped(void) {
    /* A caller's default is not trusted past the array bound either. */
    ds4_pread_pool_cfg c;
    reset(&c, NULL);
    check_u32(ds4_pread_pool_threads(&c, "POOL_N", 9999u), DS4_PREAD_POOL_MAX,
              "an over-large default is clamped too");

    reset(&c, NULL);
    check_u32(ds4_pread_pool_threads(&c, "POOL_N", 0u), 1u, "a zero default clamps up");
}

static void test_threads_never_exceeds_the_array_bound(void) {
    /* The pool indexes fixed arrays of DS4_PREAD_POOL_MAX. Sweep the values a
     * person might actually set and assert the bound holds for every one. */
    const char *vals[] = { NULL, "", "1", "2", "7", "8", "16", "31", "32", "33",
                           "64", "1000", "4294967295", "4294967296", "0", "x", "8x" };
    size_t i;
    for (i = 0; i < sizeof(vals) / sizeof(vals[0]); i++) {
        ds4_pread_pool_cfg c;
        uint32_t n;
        reset(&c, vals[i]);
        n = ds4_pread_pool_threads(&c, "POOL_N", 8u);
        check(n >= 1u && n <= DS4_PREAD_POOL_MAX, "limit stays inside [1, MAX]");
    }
    printf("  swept %zu values, all inside [1, %u]\n",
           sizeof(vals) / sizeof(vals[0]), DS4_PREAD_POOL_MAX);
}

/* ---- 3. resolved once --------------------------------------------------- */

static void test_threads_resolved_once(void) {
    /* This is the case that fails against the unhoisted resolver. */
    ds4_pread_pool_cfg c;
    int i;
    reset(&c, "4");
    for (i = 0; i < 4000; i++) {
        check_u32(ds4_pread_pool_threads(&c, "POOL_N", 8u), 4u, "same answer every call");
        g_checks--;              /* one assertion for the loop, not 4000 */
    }
    g_checks++;
    check_ul(g_getenv_calls, 1ul, "one env read over 4000 calls");
    printf("  env reads: %lu over 4000 calls\n", g_getenv_calls);
}

static void test_enabled_resolved_once(void) {
    ds4_pread_pool_cfg c;
    int i;
    reset(&c, "0");
    for (i = 0; i < 4000; i++) {
        if (ds4_pread_pool_enabled(&c, "POOL_ON") != 0) { check(0, "stable answer"); break; }
    }
    check(1, "stable answer");
    check_ul(g_getenv_calls, 1ul, "one env read over 4000 calls");
}

static void test_zero_init_means_unresolved(void) {
    /* The cache lives inside cuda_pread_pool, whose global is built by a
     * POSITIONAL initializer that names only the first few members and leaves
     * the rest zeroed. So an all-zero cache must mean "nothing resolved yet" -
     * if zero meant "resolved, disabled", the pool would silently never run.
     *
     * Checked as raw zero bytes, not via the INIT macro, because the struct
     * that matters is the one the compiler zero-fills. */
    ds4_pread_pool_cfg c;
    memset(&c, 0, sizeof c);
    g_env_value = NULL; g_getenv_calls = 0;
    check(ds4_pread_pool_enabled(&c, "POOL_ON") == 1, "zero-filled cache still resolves: on");
    check_ul(g_getenv_calls, 1ul, "zero-filled cache did read the environment");

    memset(&c, 0, sizeof c);
    g_env_value = "4"; g_getenv_calls = 0;
    check_u32(ds4_pread_pool_threads(&c, "POOL_N", 8u), 4u, "zero-filled cache resolves threads");
    check_ul(g_getenv_calls, 1ul, "zero-filled cache did read the environment");
}

static void test_the_two_settings_are_independent(void) {
    /* One cache, two settings: resolving one must not mark the other done. */
    ds4_pread_pool_cfg c;
    reset(&c, "0");
    check(ds4_pread_pool_enabled(&c, "POOL_ON") == 0, "enable resolved");
    check_ul(g_getenv_calls, 1ul, "one read so far");
    /* "0" for the thread count clamps to 1 - a different question, its own read. */
    check_u32(ds4_pread_pool_threads(&c, "POOL_N", 8u), 1u, "threads resolved separately");
    check_ul(g_getenv_calls, 2ul, "the second setting read its own variable");
}

static void test_dispatch_shape_reads_once_not_twice(void) {
    /* The shape of cuda_pread_pool_dispatch_start: the clamped worker count
     * and the unclamped pool size come from ONE resolution.
     *
     * Against the unhoisted resolver this reads 2 per dispatch and 20 over ten
     * dispatches; the assertion below is what pins it to 1. */
    ds4_pread_pool_cfg c;
    int d;
    reset(&c, "8");
    for (d = 0; d < 10; d++) {
        uint32_t limit   = ds4_pread_pool_threads(&c, "POOL_N", 8u);  /* pool size, unclamped */
        uint32_t workers = ds4_pread_pool_workers(limit, 3u);         /* this batch, clamped  */
        if (limit != 8u)   { check(0, "pool size is the configured limit"); break; }
        if (workers != 3u) { check(0, "workers clamped to the batch"); break; }
    }
    check(1, "ten dispatches agree");
    check_ul(g_getenv_calls, 1ul, "one env read over ten dispatches");
    printf("  env reads: %lu over 10 dispatches (was 2 per dispatch)\n", g_getenv_calls);
}

/* ---- 4. the per-batch clamp --------------------------------------------- */

static void test_workers_clamp_to_the_batch(void) {
    check_u32(ds4_pread_pool_workers(8u, 3u), 3u, "fewer tasks than workers");
    check_u32(ds4_pread_pool_workers(8u, 8u), 8u, "exactly as many");
    check_u32(ds4_pread_pool_workers(8u, 99u), 8u, "more tasks than workers");
}

static void test_workers_decline_a_pointless_pool(void) {
    /* One worker is the serial read with a thread handoff added; the caller
     * must decline so it reads inline instead. */
    check_u32(ds4_pread_pool_workers(8u, 1u), 0u, "a single task declines");
    check_u32(ds4_pread_pool_workers(1u, 8u), 0u, "a single configured worker declines");
    check_u32(ds4_pread_pool_workers(8u, 0u), 0u, "an empty batch declines");
    check_u32(ds4_pread_pool_workers(1u, 1u), 0u, "both one declines");
    check_u32(ds4_pread_pool_workers(2u, 2u), 2u, "two is the smallest pool that runs");
}

static void test_workers_never_exceed_either_bound(void) {
    uint32_t limit, tasks;
    for (limit = 0; limit <= 40u; limit++) {
        for (tasks = 0; tasks <= 40u; tasks++) {
            uint32_t n = ds4_pread_pool_workers(limit, tasks);
            if (n != 0 && (n > limit || n > tasks || n < 2u)) {
                check(0, "workers within both bounds");
                return;
            }
        }
    }
    check(1, "workers within both bounds over a 41x41 sweep");
}

int main(void) {
    RUN(test_enabled_when_unset);
    RUN(test_disabled_only_by_exact_zero);
    RUN(test_threads_default_when_unset);
    RUN(test_threads_parses_a_clean_number);
    RUN(test_threads_refuses_a_partial_number);
    RUN(test_threads_clamps_both_ends);
    RUN(test_threads_default_is_itself_clamped);
    RUN(test_threads_never_exceeds_the_array_bound);
    RUN(test_threads_resolved_once);
    RUN(test_enabled_resolved_once);
    RUN(test_zero_init_means_unresolved);
    RUN(test_the_two_settings_are_independent);
    RUN(test_dispatch_shape_reads_once_not_twice);
    RUN(test_workers_clamp_to_the_batch);
    RUN(test_workers_decline_a_pointless_pool);
    RUN(test_workers_never_exceed_either_bound);

    printf("\n%d checks, %d failed\n", g_checks, g_failed);
    return g_failed != 0;
}
