/* Unit tests for the scout decision logic (ds4_scout_logic.h).
 *
 * Pure C99: no CUDA, no GPU, no model. Builds and runs on any host.
 *
 * Three subjects, one per rule in the header:
 *   - REMEMBER: a token's ids are recorded per layer; a prefill batch is not;
 *   - SELECT: remembered minus resident, deduplicated, range-checked, capped by
 *     the victims the ledger offers, and a NO-OP when it offers none;
 *   - SCORE: a remembered id the next layer routes to is a hit, counted once.
 *
 * Red-proven 2026-09-18 on a Mac by a planted mutation: making
 * ds4_scout_select ignore victims_available (the `!victims_available` guard
 * dropped AND the `cap = victims_available` clamp replaced by a (void) cast)
 * turned 3 of 40 checks red - both checks of test_full_cache_is_a_no_op and
 * the two-victims check of test_select_is_capped_by_victims_then_cap - while
 * every other case stayed green; restoring the two lines made 40/40 green.
 * Dropping the guard alone stays green: the clamp already carries the rule,
 * and the guard is the belt to its braces.
 */
#include "../ds4_scout_logic.h"

#include <stdio.h>
#include <string.h>

static int g_failed = 0;
static int g_total  = 0;

#define CHECK(cond, msg) do {                                                  \
    g_total++;                                                                 \
    if (!(cond)) {                                                             \
        fprintf(stderr, "  FAIL: %s (line %d)\n", (msg), __LINE__);            \
        g_failed++;                                                            \
    }                                                                          \
} while (0)

#define RUN(fn) do {                                                           \
    fprintf(stderr, "RUN: %s\n", #fn);                                         \
    int _before = g_failed;                                                    \
    (fn)();                                                                    \
    fprintf(stderr, "  %s\n", (_before == g_failed) ? "ok" : "FAIL");          \
} while (0)

#define N_TOTAL 256u

/* ---- 1. remember -------------------------------------------------------- */

static void test_remember_records_a_token(void) {
    ds4_scout_layer_memory mem;
    memset(&mem, 0, sizeof mem);
    const int32_t ids[6] = {3, 77, 200, 5, 9, 255};
    CHECK(ds4_scout_remember(&mem, ids, 6) == 1, "six ids are a token and are recorded");
    CHECK(mem.n == 6, "the count is six");
    for (uint32_t i = 0; i < 6; i++) CHECK(mem.id[i] == ids[i], "each id is kept in order");
}

static void test_remember_refuses_a_prefill_batch(void) {
    ds4_scout_layer_memory mem;
    memset(&mem, 0, sizeof mem);
    const int32_t old[2] = {1, 2};
    (void)ds4_scout_remember(&mem, old, 2);
    int32_t wide[48];
    for (uint32_t i = 0; i < 48; i++) wide[i] = (int32_t)i;
    CHECK(ds4_scout_remember(&mem, wide, 48) == 0, "48 selected ids is a batch, not a token");
    CHECK(mem.n == 2 && mem.id[0] == 1 && mem.id[1] == 2, "the old memory stands untouched");
    CHECK(ds4_scout_remember(&mem, wide, 0) == 0, "zero ids records nothing");
    CHECK(ds4_scout_remember(NULL, old, 2) == 0, "a null memory records nothing");
}

static void test_remember_cap_is_hits_first_cap(void) {
    ds4_scout_layer_memory mem;
    memset(&mem, 0, sizeof mem);
    int32_t eight[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    CHECK(ds4_scout_remember(&mem, eight, 8) == 1, "eight fits: hits-first's own cap");
    int32_t nine[9] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    CHECK(ds4_scout_remember(&mem, nine, 9) == 0, "nine does not");
    CHECK(DS4_SCOUT_MAX_IDS == 8u, "the cap is 8, the same number as begin_load's slot_count <= 8");
}

/* ---- 2. select ---------------------------------------------------------- */

static void test_select_is_remembered_minus_resident(void) {
    const int32_t remembered[6] = {10, 20, 30, 40, 50, 60};
    const uint8_t resident[6]   = { 0,  1,  0,  1,  0,  0};
    int32_t out[8];
    const uint32_t n = ds4_scout_select(remembered, 6, resident, N_TOTAL, 100, 8, out);
    CHECK(n == 4, "four of six are not resident");
    CHECK(out[0] == 10 && out[1] == 30 && out[2] == 50 && out[3] == 60,
          "the non-resident ids, in remembered order");
}

static void test_select_all_resident_reads_nothing(void) {
    const int32_t remembered[3] = {1, 2, 3};
    const uint8_t resident[3]   = {1, 1, 1};
    int32_t out[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
    CHECK(ds4_scout_select(remembered, 3, resident, N_TOTAL, 100, 8, out) == 0,
          "every remembered id is resident: nothing to read");
    CHECK(out[0] == -1, "out is untouched");
}

static void test_full_cache_is_a_no_op(void) {
    /* THE INVARIANT. Zero victims available means every slot is claimed by
     * this layer's hits and misses or protected; the scout must never win a
     * slot over a demand read, so it does nothing at all. */
    const int32_t remembered[4] = {10, 20, 30, 40};
    const uint8_t resident[4]   = { 0,  0,  0,  0};
    int32_t out[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
    CHECK(ds4_scout_select(remembered, 4, resident, N_TOTAL, 0, 8, out) == 0,
          "no victims: the scout issues no read");
    CHECK(out[0] == -1, "no victims: out is untouched");
}

static void test_select_is_capped_by_victims_then_cap(void) {
    const int32_t remembered[6] = {10, 20, 30, 40, 50, 60};
    const uint8_t resident[6]   = { 0,  0,  0,  0,  0,  0};
    int32_t out[8];
    CHECK(ds4_scout_select(remembered, 6, resident, N_TOTAL, 2, 8, out) == 2,
          "two victims available: two reads, not six");
    CHECK(out[0] == 10 && out[1] == 20, "the first two remembered, in order");
    CHECK(ds4_scout_select(remembered, 6, resident, N_TOTAL, 100, 3, out) == 3,
          "a cap of three below the victims: three reads");
    CHECK(ds4_scout_select(remembered, 6, resident, N_TOTAL, 100, 8, out) == 6,
          "room for all: six reads");
}

static void test_select_dedups_and_range_checks(void) {
    const int32_t remembered[7] = {5, 5, -1, 256, 7, 5, 7};
    const uint8_t resident[7]   = {0, 0,  0,   0, 0, 0, 0};
    int32_t out[8];
    const uint32_t n = ds4_scout_select(remembered, 7, resident, N_TOTAL, 100, 8, out);
    CHECK(n == 2, "5 twice and 7 twice are two reads; -1 and 256 are outside 0..255");
    CHECK(out[0] == 5 && out[1] == 7, "the distinct in-range ids");
}

static void test_select_refuses_null_arguments(void) {
    const int32_t remembered[1] = {1};
    const uint8_t resident[1]   = {0};
    int32_t out[8];
    CHECK(ds4_scout_select(NULL, 1, resident, N_TOTAL, 100, 8, out) == 0, "null remembered");
    CHECK(ds4_scout_select(remembered, 1, NULL, N_TOTAL, 100, 8, out) == 0, "null resident");
    CHECK(ds4_scout_select(remembered, 1, resident, N_TOTAL, 100, 8, NULL) == 0, "null out");
}

/* ---- 3. score ----------------------------------------------------------- */

static void test_score_counts_hits_once(void) {
    const int32_t guess[6]    = {10, 20, 30, 40, 50, 60};
    const int32_t selected[6] = {60, 11, 20, 20, 99, 10};
    uint64_t hits = 0, guessed = 0;
    ds4_scout_score(guess, 6, selected, 6, &hits, &guessed);
    CHECK(guessed == 6, "six distinct guesses");
    CHECK(hits == 3, "10, 20 and 60 were routed to; 20 twice counts once");
}

static void test_score_dedups_the_guess(void) {
    const int32_t guess[4]    = {7, 7, 8, 7};
    const int32_t selected[2] = {7, 1};
    uint64_t hits = 0, guessed = 0;
    ds4_scout_score(guess, 4, selected, 2, &hits, &guessed);
    CHECK(guessed == 2, "7 and 8 are two guesses however often 7 repeats");
    CHECK(hits == 1, "7 is one hit");
}

static void test_score_accumulates(void) {
    const int32_t guess[2]    = {1, 2};
    const int32_t selected[2] = {2, 3};
    uint64_t hits = 5, guessed = 9;
    ds4_scout_score(guess, 2, selected, 2, &hits, &guessed);
    CHECK(guessed == 11 && hits == 6, "the counters accumulate across layers");
    ds4_scout_score(guess, 2, selected, 2, NULL, &guessed);
    CHECK(guessed == 11, "a null counter is a no-op, nothing moves");
}

/* ---- the whole token-to-token cycle ------------------------------------- */

static void test_sticky_routing_end_to_end(void) {
    /* token t-1 at layer L+1 routed to A; token t at layer L remembers it and
     * scouts the non-resident part; token t at layer L+1 then routes to B. */
    ds4_scout_layer_memory next;
    memset(&next, 0, sizeof next);
    const int32_t a[6] = {3, 17, 42, 100, 128, 250};
    CHECK(ds4_scout_remember(&next, a, 6) == 1, "t-1 at L+1 is remembered");
    const uint8_t resident[6] = {1, 0, 1, 0, 0, 1};
    int32_t reads[8];
    const uint32_t n = ds4_scout_select(next.id, next.n, resident, N_TOTAL, 1000, 8, reads);
    CHECK(n == 3 && reads[0] == 17 && reads[1] == 100 && reads[2] == 128,
          "three reads go ahead of the token");
    const int32_t b[6] = {3, 17, 42, 100, 128, 9};   /* five of six sticky */
    uint64_t hits = 0, guessed = 0;
    ds4_scout_score(next.id, next.n, b, 6, &hits, &guessed);
    CHECK(guessed == 6 && hits == 5, "five of six remembered ids were routed to again");
    CHECK(ds4_scout_remember(&next, b, 6) == 1 && next.id[5] == 9,
          "the memory now holds token t for the next token");
}

int main(void) {
    RUN(test_remember_records_a_token);
    RUN(test_remember_refuses_a_prefill_batch);
    RUN(test_remember_cap_is_hits_first_cap);
    RUN(test_select_is_remembered_minus_resident);
    RUN(test_select_all_resident_reads_nothing);
    RUN(test_full_cache_is_a_no_op);
    RUN(test_select_is_capped_by_victims_then_cap);
    RUN(test_select_dedups_and_range_checks);
    RUN(test_select_refuses_null_arguments);
    RUN(test_score_counts_hits_once);
    RUN(test_score_dedups_the_guess);
    RUN(test_score_accumulates);
    RUN(test_sticky_routing_end_to_end);
    fprintf(stderr, "\n%d checks, %d failed\n", g_total, g_failed);
    return g_failed ? 1 : 0;
}
