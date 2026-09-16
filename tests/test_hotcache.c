/*
 * tests/test_hotcache.c - the CUDA expert cache's decisions, proved without a GPU.
 *
 * ds4_hotcache.h is pure, so the two things the cache DECIDES can be checked
 * on a laptop: how much history a run inherits (the decay rule) and which
 * slots an eviction takes (the victim scan).  ds4_cuda.cu is an nvcc
 * translation unit and cannot be built here at all, which is exactly why these
 * two decisions were moved out of it.
 *
 * The victim tests carry their own transcription of the PRE-FIX scan - one
 * full pass per miss - and assert two things against it: that the new scan
 * picks the IDENTICAL victims in the identical order (the policy did not
 * change; a changed eviction policy has already measured worse once, see
 * WIKI/theory/177), and that it does so in one pass rather than M.
 *
 * Build and run: make test-hotcache    (no CUDA, no model, no GPU)
 */
#include "../ds4_hotcache.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_failures;
static int g_checks;

#define CHECK(cond, ...)                                                       \
    do {                                                                       \
        g_checks++;                                                            \
        if (!(cond)) {                                                         \
            g_failures++;                                                      \
            fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__);               \
            fprintf(stderr, __VA_ARGS__);                                      \
            fprintf(stderr, "\n");                                             \
        }                                                                      \
    } while (0)

/* --- the slot, exactly as ds4_cuda.cu lays it out ------------------------ */
typedef struct { uint64_t gate, up, down, used; } slot;

/* ========================================================================= */
/* 1. THE DECAY RULE                                                         */
/* ========================================================================= */

/* --- 1a. the rules that are words --------------------------------------- */
static void test_decay_named_rules(void) {
    unsigned s = 99;
    CHECK(ds4_hotlist_decay_shift_from_env(NULL, &s) && s == 1,
          "an unset DS4_CUDA_EXPERT_HOTLIST_DECAY must be halve");
    s = 99;
    CHECK(ds4_hotlist_decay_shift_from_env("", &s) && s == 1,
          "an empty DS4_CUDA_EXPERT_HOTLIST_DECAY must be halve");
    s = 99;
    CHECK(ds4_hotlist_decay_shift_from_env("none", &s) && s == 0, "none is shift 0");
    s = 99;
    CHECK(ds4_hotlist_decay_shift_from_env("halve", &s) && s == 1, "halve is shift 1");
    s = 99;
    CHECK(ds4_hotlist_decay_shift_from_env("quarter", &s) && s == 2, "quarter is shift 2");
    s = 99;
    CHECK(ds4_hotlist_decay_shift_from_env("eighth", &s) && s == 3, "eighth is shift 3");
}

/* --- 1b. shift:<n>, the values that ARE readable ------------------------- */
static void test_decay_shift_accepts_whole_numbers(void) {
    unsigned s = 99;
    CHECK(ds4_hotlist_decay_shift_from_env("shift:0", &s) && s == 0, "shift:0 is 0");
    s = 99;
    CHECK(ds4_hotlist_decay_shift_from_env("shift:5", &s) && s == 5, "shift:5 is 5");
    s = 99;
    CHECK(ds4_hotlist_decay_shift_from_env("shift:32", &s) && s == 32,
          "shift:32 is the top of the range and must be accepted");
}

/* --- 1c. THE REGRESSION GUARD: a value that cannot be read WHOLE is refused,
 *         and refusal keeps today's behaviour rather than picking another one.
 *
 * This is the case the fix exists for.  strtol(env + 6, NULL, 10) on "shift:x"
 * returns 0, and 0 is the VALID shift "none" - carry the whole history - which
 * is the other end of the axis from the default.  The run then looks fine and
 * inherits the wrong amount of history.  A silently-different policy is worse
 * than a refused one, so every unreadable value must report refusal AND leave
 * the shift at the default.                                                 */
static void test_decay_refuses_what_it_cannot_read_whole(void) {
    static const char *junk[] = {
        "shift:x",      /* no digits at all: the strtol-0 case, reads as none  */
        "shift:3abc",   /* a prefix it can read and a tail it cannot          */
        "shift: 3",     /* leading space: strtol would take it, the rule does not */
        "shift:-1",     /* negative                                            */
        "shift:33",     /* out of range                                        */
        "shift:99999",  /* out of range and long                               */
        "shift:3.5",    /* a prefix again                                      */
        "shift:+2",     /* a sign strtol accepts and the rule does not         */
        "halved",       /* a near miss on a named rule                         */
        "0",            /* the OFF switch of a different variable              */
    };
    size_t i;
    for (i = 0; i < sizeof junk / sizeof junk[0]; i++) {
        unsigned s = 99;
        const int ok = ds4_hotlist_decay_shift_from_env(junk[i], &s);
        CHECK(ok == 0, "\"%s\" must be REFUSED, not read as a shift", junk[i]);
        CHECK(s == DS4_HOTLIST_DECAY_DEFAULT_SHIFT,
              "\"%s\" was refused, so the shift must stay at the default halve, not %u",
              junk[i], s);
    }
}

/* --- 1d. the two parsers of one syntax agree on the boundary ------------- */
static void test_decay_shift_boundary(void) {
    unsigned s = 99;
    /* "shift:" with nothing after it is not a rule */
    CHECK(ds4_hotlist_decay_shift_from_env("shift:", &s) == 0 && s == 1,
          "a bare shift: is refused and keeps halve");
    s = 99;
    CHECK(ds4_hotlist_decay_shift_from_env("shift:00", &s) && s == 0,
          "shift:00 is a whole number and reads as 0");
}

/* ========================================================================= */
/* 2. THE VICTIM SCAN                                                        */
/* ========================================================================= */

/*
 * THE PRE-FIX SCAN, transcribed from cuda_stream_selected_cache_begin_load as
 * it stood before this change: one full pass per miss, the victim stamped with
 * `stamp` before the next pass runs.  It is the reference for BOTH assertions
 * - the victims it picks, and the cost it pays.
 */
static uint32_t reference_scan_per_miss(slot *slots, uint32_t n_slots,
                                        const unsigned char *skip,
                                        uint32_t want, uint64_t stamp,
                                        uint32_t *out, uint64_t *examined) {
    uint32_t got = 0, i;
    uint64_t seen = 0;
    for (i = 0; i < want; i++) {
        uint32_t victim = UINT32_MAX, j;
        uint64_t oldest = stamp;
        for (j = 0; j < n_slots; j++) {
            seen++;
            if ((!skip || !skip[j]) && slots[j].used < oldest) {
                oldest = slots[j].used;
                victim = j;
                if (!oldest) break;
            }
        }
        if (victim == UINT32_MAX) break;
        slots[victim].used = stamp;     /* the old loop filled it and stamped it */
        out[got++] = victim;
    }
    if (examined) *examined = seen;
    return got;
}

static void fill(slot *s, uint32_t n, const uint64_t *used) {
    uint32_t i;
    for (i = 0; i < n; i++) {
        s[i].gate = 1000 + i;
        s[i].up = 2000 + i;
        s[i].down = 3000 + i;
        s[i].used = used[i];
    }
}

/* --- 2a. a cold cache: empties are taken in index order, and the pass stops */
static void test_victim_cold_cache_stops_early(void) {
    enum { N = 64, WANT = 4 };
    uint64_t used[N];
    slot s[N];
    uint32_t out[WANT];
    uint64_t examined = 0, n;
    uint32_t i;
    for (i = 0; i < N; i++) used[i] = 0;       /* every slot empty */
    fill(s, N, used);
    n = ds4_expert_victim_scan(s, sizeof s[0], offsetof(slot, used), N, NULL,
                               WANT, 100, out, &examined);
    CHECK(n == WANT, "a cold cache must yield every victim asked for, got %llu",
          (unsigned long long)n);
    CHECK(out[0] == 0 && out[1] == 1 && out[2] == 2 && out[3] == 3,
          "empties are taken in index order: got %u %u %u %u",
          out[0], out[1], out[2], out[3]);
    CHECK(examined == WANT,
          "the early exit must stop the pass at the %uth empty slot, examined %llu of %u",
          (unsigned)WANT, (unsigned long long)examined, (unsigned)N);
}

/* --- 2b. THE REGRESSION GUARD: a FULL cache costs ONE pass, not one per miss.
 *
 * This is the cost the fix exists to bound.  Seeding fills every free slot, so
 * from the first token there is no empty slot and the early exit can never
 * fire; the pre-fix scan then re-read the whole slot vector for every missed
 * expert, on every routed layer, on every token.  One pass must answer the
 * whole batch.                                                              */
static void test_victim_full_cache_is_one_pass(void) {
    enum { N = 4096, WANT = 8 };
    uint64_t used[N];
    slot s[N];
    uint32_t out[WANT];
    uint64_t examined = 0, ref_examined = 0, n;
    uint32_t i;
    uint32_t ref_out[WANT];
    slot ref[N];

    for (i = 0; i < N; i++) used[i] = 1 + (uint64_t)((i * 7919u) % 500u);  /* no empties */
    fill(s, N, used);
    fill(ref, N, used);

    n = ds4_expert_victim_scan(s, sizeof s[0], offsetof(slot, used), N, NULL,
                               WANT, 100000, out, &examined);
    CHECK(n == WANT, "a full cache still yields victims, got %llu", (unsigned long long)n);
    CHECK(examined == N,
          "a full cache must cost ONE pass over %u slots, cost %llu",
          (unsigned)N, (unsigned long long)examined);

    (void)reference_scan_per_miss(ref, N, NULL, WANT, 100000, ref_out, &ref_examined);
    CHECK(ref_examined == (uint64_t)N * WANT,
          "the pre-fix scan is %u passes by construction; if this ever stops being "
          "true the reference has drifted (%llu)",
          (unsigned)WANT, (unsigned long long)ref_examined);
    CHECK(examined * WANT == ref_examined,
          "the whole point: %llu against %llu slot reads for the same batch",
          (unsigned long long)examined, (unsigned long long)ref_examined);
}

/* --- 2c. THE POLICY DID NOT CHANGE: identical victims, identical order ---- */
static void test_victim_matches_the_old_policy(void) {
    enum { N = 257 };
    uint32_t trial;
    unsigned seed = 12345u;
    int mismatches = 0;

    for (trial = 0; trial < 400; trial++) {
        uint64_t used[N];
        unsigned char skip[N];
        slot a[N], b[N];
        uint32_t out_a[32], out_b[32];
        uint32_t i, n_slots, want, na, nb;
        int any_skip;

        seed = seed * 1103515245u + 12345u;
        n_slots = 1 + (seed >> 16) % N;
        seed = seed * 1103515245u + 12345u;
        want = 1 + (seed >> 16) % 32u;
        seed = seed * 1103515245u + 12345u;
        any_skip = (int)((seed >> 16) % 3u) == 0;

        for (i = 0; i < n_slots; i++) {
            seed = seed * 1103515245u + 12345u;
            /* a mix of empty slots, cold slots and slots this batch stamped */
            used[i] = (seed >> 16) % 7u == 0 ? 0 : (uint64_t)((seed >> 8) % 120u);
            seed = seed * 1103515245u + 12345u;
            skip[i] = (unsigned char)(any_skip && (seed >> 16) % 5u == 0);
        }
        fill(a, n_slots, used);
        fill(b, n_slots, used);

        na = ds4_expert_victim_scan(a, sizeof a[0], offsetof(slot, used), n_slots,
                                    any_skip ? skip : NULL, want, 100, out_a, NULL);
        nb = reference_scan_per_miss(b, n_slots, any_skip ? skip : NULL, want, 100,
                                     out_b, NULL);
        if (na != nb) { mismatches++; continue; }
        for (i = 0; i < na; i++) if (out_a[i] != out_b[i]) { mismatches++; break; }
    }
    CHECK(mismatches == 0,
          "one pass must pick exactly the victims the per-miss scan picked, in the "
          "same order, over 400 random caches: %d disagreed", mismatches);
}

/* --- 2d. a protected slot is never taken, even when it is the oldest ------ */
static void test_victim_never_takes_a_protected_slot(void) {
    enum { N = 8, WANT = 2 };
    uint64_t used[N] = { 50, 0, 40, 30, 60, 70, 80, 90 };
    unsigned char skip[N] = { 0, 1, 1, 0, 0, 0, 0, 0 };
    slot s[N];
    uint32_t out[WANT], i;
    uint64_t n;
    fill(s, N, used);
    n = ds4_expert_victim_scan(s, sizeof s[0], offsetof(slot, used), N, skip,
                               WANT, 100, out, NULL);
    CHECK(n == WANT, "two victims exist outside the protected pair");
    CHECK(out[0] == 3 && out[1] == 0,
          "the oldest UNPROTECTED slots are 3 (30) then 0 (50), got %u %u",
          out[0], out[1]);
    for (i = 0; i < n; i++)
        CHECK(skip[out[i]] == 0, "slot %u is held by the prefetch reader and was taken",
              out[i]);
}

/* --- 2e. a slot this batch already stamped is not a candidate ------------- */
static void test_victim_never_evicts_this_batchs_hit(void) {
    enum { N = 5, WANT = 3 };
    const uint64_t stamp = 42;
    uint64_t used[N] = { 42, 42, 10, 42, 42 };   /* one cold slot, four hits */
    slot s[N];
    uint32_t out[WANT];
    uint64_t n;
    fill(s, N, used);
    n = ds4_expert_victim_scan(s, sizeof s[0], offsetof(slot, used), N, NULL,
                               WANT, stamp, out, NULL);
    CHECK(n == 1, "only the one slot older than this batch may be evicted, got %llu",
          (unsigned long long)n);
    CHECK(n >= 1 && out[0] == 2, "the cold slot is slot 2");
}

/* --- 2f. fewer evictable slots than the batch needs is reported, not faked  */
static void test_victim_reports_a_short_cache(void) {
    enum { N = 4, WANT = 4 };
    uint64_t used[N] = { 100, 100, 100, 100 };   /* all stamped by this batch */
    slot s[N];
    uint32_t out[WANT];
    uint64_t n;
    fill(s, N, used);
    n = ds4_expert_victim_scan(s, sizeof s[0], offsetof(slot, used), N, NULL,
                               WANT, 100, out, NULL);
    CHECK(n == 0, "no slot is evictable, so no victim may be reported, got %llu",
          (unsigned long long)n);
}

int main(void) {
    test_decay_named_rules();
    test_decay_shift_accepts_whole_numbers();
    test_decay_refuses_what_it_cannot_read_whole();
    test_decay_shift_boundary();
    test_victim_cold_cache_stops_early();
    test_victim_full_cache_is_one_pass();
    test_victim_matches_the_old_policy();
    test_victim_never_takes_a_protected_slot();
    test_victim_never_evicts_this_batchs_hit();
    test_victim_reports_a_short_cache();

    if (g_failures) {
        fprintf(stderr, "test_hotcache: %d of %d checks FAILED\n", g_failures, g_checks);
        return 1;
    }
    printf("test_hotcache: %d checks passed (no GPU, no model, no CUDA)\n", g_checks);
    return 0;
}
