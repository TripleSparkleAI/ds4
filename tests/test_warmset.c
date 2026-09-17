/*
 * tests/test_warmset.c - the warm-set rules, proved without a GPU.
 *
 * ds4_warmset.h is pure: no I/O, no CUDA, no allocation.  Everything the warm
 * set decides - what is kept across a session, what is kept across a day, what
 * consolidates, what is forgotten, and how two sessions that disagree about one
 * pair are merged - is decided by that header and is exercised here.
 *
 * The last test asserts the SAFETY PROPERTY at the only level a host test can
 * reach: a warm set is a set of (layer, expert) identities with evidence
 * counts, so an entry that is WRONG can only ever differ by BEING PRESENT.  It
 * cannot carry a magnitude, it cannot change another entry's evidence, and it
 * cannot name anything but an expert.
 *
 * Build and run: make test-warmset    (no CUDA, no model, no GPU)
 */
#include "../ds4_warmset.h"

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
            fprintf(stderr, "\n");                                            \
        }                                                                      \
    } while (0)

static ds4_warmset_entry ent(uint32_t layer, uint32_t expert) {
    ds4_warmset_entry e;
    memset(&e, 0, sizeof(e));
    e.layer = layer;
    e.expert = expert;
    return e;
}

/* --- 1. the switch defaults to today's behaviour ------------------------- */
static void test_tier_defaults(void) {
    CHECK(ds4_warmset_tier_from_env(NULL) == DS4_WARMSET_TIER_SESSION,
          "an unset DS4_WARMSET_TIER must be the session tier");
    CHECK(ds4_warmset_tier_from_env("") == DS4_WARMSET_TIER_SESSION,
          "an empty DS4_WARMSET_TIER must be the session tier");
    CHECK(ds4_warmset_tier_from_env("session") == DS4_WARMSET_TIER_SESSION,
          "session must be the session tier");
    CHECK(ds4_warmset_tier_from_env("day") == DS4_WARMSET_TIER_DAY,
          "day must be the day tier");
    CHECK(ds4_warmset_tier_from_env("week") == DS4_WARMSET_TIER_SESSION,
          "an unknown tier must degrade to today's behaviour, not to a guess");
}

/* --- 2. recording counts distinct days, not calls ------------------------ */
static void test_record_counts_distinct_days(void) {
    const uint32_t d1 = ds4_warmset_day_key(2026, 9, 16);
    const uint32_t d2 = ds4_warmset_day_key(2026, 9, 17);
    CHECK(d1 == 20260916u, "day key for 2026-09-16, got %u", d1);
    CHECK(d2 == 20260917u, "day key for 2026-09-17, got %u", d2);
    CHECK(ds4_warmset_day_key(0, 1, 1) == 0, "a nonsense year must key to 0");
    CHECK(ds4_warmset_day_key(2026, 13, 1) == 0, "a nonsense month must key to 0");

    ds4_warmset_entry e = ent(3, 91);
    for (int i = 0; i < 500; i++) ds4_warmset_record(&e, d1);
    CHECK(e.hits == 500, "500 demands must be 500 hits, got %llu",
          (unsigned long long)e.hits);
    CHECK(e.days == 1, "500 demands in ONE day must be ONE day of evidence, got %u",
          e.days);
    CHECK(e.last_day == d1, "last_day must be the day of the demand");

    ds4_warmset_record(&e, d1);
    CHECK(e.days == 1, "a repeat demand on the same day must not add a day");

    ds4_warmset_record(&e, d2);
    CHECK(e.days == 2, "a demand on a new day must add exactly one day, got %u",
          e.days);
    CHECK(e.hits == 502, "the second day's demand must also add a hit");
    CHECK(e.last_day == d2, "last_day must advance to the new day");
}

/* --- 3. the merge rule --------------------------------------------------- */
static void test_merge_rule(void) {
    const uint32_t d1 = ds4_warmset_day_key(2026, 9, 16);
    const uint32_t d2 = ds4_warmset_day_key(2026, 9, 17);
    const uint32_t d3 = ds4_warmset_day_key(2026, 9, 18);

    /* two sessions, same day, same expert: demand adds, the day counts once */
    ds4_warmset_entry a = ent(7, 12), b = ent(7, 12);
    ds4_warmset_record(&a, d1); ds4_warmset_record(&a, d1); /* a: 2 hits, 1 day */
    ds4_warmset_record(&b, d1);                             /* b: 1 hit,  1 day */
    ds4_warmset_merge(&a, &b);
    CHECK(a.hits == 3, "two sessions that both asked must add demand, got %llu",
          (unsigned long long)a.hits);
    CHECK(a.days == 1, "the same day seen twice must be ONE day, got %u", a.days);

    /* projecting the SAME history twice must be inert (that is what makes
     * day.txt and durable.txt safe to union) */
    ds4_warmset_entry a2 = a;
    ds4_warmset_merge_view(&a2, &b);
    ds4_warmset_merge_view(&a2, &b);
    CHECK(a2.hits == a.hits && a2.days == a.days,
          "a view folded in twice must not double count, got hits=%llu days=%u",
          (unsigned long long)a2.hits, a2.days);

    /* a disagreement in days: the union is the LATER day, the count is the
     * larger, so day.txt and durable.txt can be unioned without double count */
    ds4_warmset_entry c = ent(7, 12); ds4_warmset_record(&c, d1);
    ds4_warmset_entry d = ent(7, 12);
    ds4_warmset_record(&d, d1); ds4_warmset_record(&d, d2); ds4_warmset_record(&d, d3);
    ds4_warmset_entry m = c;
    ds4_warmset_merge(&m, &d);
    CHECK(m.days == 3, "days must be the maximal recurrence of the two views, got %u",
          m.days);
    CHECK(m.last_day == d3, "last_day must be the later of the two views");

    /* a merge that would empty the entry must be inert */
    ds4_warmset_entry z = ent(1, 1);
    ds4_warmset_entry empty; memset(&empty, 0, sizeof(empty));
    ds4_warmset_merge(&z, &empty);
    CHECK(z.days == 0 && z.hits == 0, "merging an empty view must be inert");
}

/* --- 4. the session boundary: today's behaviour -------------------------- */
static void test_session_boundary(void) {
    ds4_warmset_entry e = ent(0, 5);
    e.hits = 9;
    CHECK(ds4_warmset_session_boundary(&e) == 1, "a nonzero count survives");
    CHECK(e.hits == 4, "the session boundary halves, got %llu",
          (unsigned long long)e.hits);
    e.hits = 1;
    CHECK(ds4_warmset_session_boundary(&e) == 0,
          "a count that halves to zero is gone");
}

/* --- 5. the day boundary: what is kept, what is forgotten ---------------- */
static void test_day_boundary(void) {
    const uint32_t d1 = ds4_warmset_day_key(2026, 9, 16);
    const uint32_t d2 = ds4_warmset_day_key(2026, 9, 17);

    /* demanded earlier TODAY: carried free, no decay at all */
    ds4_warmset_entry today = ent(2, 3);
    ds4_warmset_record(&today, d1);
    today.hits = 40;
    CHECK(ds4_warmset_day_boundary(&today, d1) == 1,
          "an entry demanded earlier today must survive");
    CHECK(today.hits == 40,
          "an entry demanded earlier today must NOT decay at a session "
          "boundary inside the day, got %llu", (unsigned long long)today.hits);

    /* one day only, and that day was not today: FORGOTTEN */
    ds4_warmset_entry once = ent(2, 4);
    ds4_warmset_record(&once, d1);
    CHECK(ds4_warmset_day_boundary(&once, d2) == 0,
          "a pair demanded on exactly one day and never again must be forgotten");

    /* two days of evidence: decays, does not die */
    ds4_warmset_entry twice = ent(2, 5);
    ds4_warmset_record(&twice, d1);
    ds4_warmset_record(&twice, d2);
    twice.last_day = d1;
    const uint64_t before = twice.hits;
    CHECK(ds4_warmset_day_boundary(&twice, d2) == 1,
          "a pair seen on two days must survive a missed day");
    CHECK(twice.hits == before >> 1,
          "a pair seen on two days must halve, got %llu",
          (unsigned long long)twice.hits);
    CHECK(twice.days == 2, "decay must not erase the recurrence count");

    /* a survivor whose evidence is exhausted is forgotten */
    ds4_warmset_entry thin = ent(2, 6);
    ds4_warmset_record(&thin, d1);
    ds4_warmset_record(&thin, d2);
    thin.last_day = d1;
    thin.hits = 1;
    CHECK(ds4_warmset_day_boundary(&thin, d2) == 0,
          "an entry whose decayed count reaches zero must be forgotten");
}

/* --- 5b. the boundary is a DAY's, not a SESSION's ------------------------
 *
 * ds4_warmset_day_boundary says what one boundary does.  This says whether a
 * load is crossing one.  A file carries the day it was written and its entries
 * have already been carried through that day's boundary, so a file written
 * today owes nothing.                                                       */
static void test_boundary_is_due_once_per_day(void) {
    const uint32_t d1 = ds4_warmset_day_key(2026, 9, 16);
    const uint32_t d2 = ds4_warmset_day_key(2026, 9, 17);

    CHECK(ds4_warmset_boundary_due(d1, d1) == 0,
          "a file written today has already paid today's boundary");
    CHECK(ds4_warmset_boundary_due(d1, d2) == 1,
          "a file written yesterday owes today's boundary");
    CHECK(ds4_warmset_boundary_due(0, d1) == 1,
          "a file with no day key has no provenance, so the boundary is due");
    CHECK(ds4_warmset_boundary_due(d1, 0) == 0,
          "with no clock nothing is decided, so nothing is decayed");
}

/* --- 5c. THE REGRESSION GUARD: N idle sessions in one day are ONE boundary.
 *
 * The defect this replaces: the boundary ran on EVERY load and the writer put
 * last_day back unchanged, so nothing recorded that today's boundary had been
 * paid.  An entry last demanded yesterday was halved by every session started
 * today that did not demand it - hits >> N after N sessions, which is death by
 * log2(hits) inside the single day the day tier exists to protect.
 *
 * The loop below is the load/write round trip in the small: read a file, apply
 * the boundary if it is due, write it back stamped with the day it was carried
 * through.                                                                  */
static ds4_warmset_entry load_write_cycle(ds4_warmset_entry e, uint32_t file_day,
                                          uint32_t today, uint32_t *out_file_day,
                                          int *survived) {
    if (ds4_warmset_boundary_due(file_day, today))
        *survived = ds4_warmset_day_boundary(&e, today);
    else
        *survived = 1;
    *out_file_day = today;     /* the file is written stamped with this day */
    return e;
}

static void test_idle_sessions_do_not_decay_the_day(void) {
    const uint32_t d1 = ds4_warmset_day_key(2026, 9, 16);
    const uint32_t d2 = ds4_warmset_day_key(2026, 9, 17);
    const uint32_t d3 = ds4_warmset_day_key(2026, 9, 18);
    uint32_t file_day = d1;
    int alive = 1;
    int i;

    /* demanded on two days, last on d1, carrying 64 hits */
    ds4_warmset_entry e = ent(4, 9);
    ds4_warmset_record(&e, d1);
    ds4_warmset_record(&e, d2);
    e.last_day = d1;
    e.hits = 64;

    /* FIVE sessions on d2, none of which demands the pair */
    for (i = 0; i < 5; i++)
        e = load_write_cycle(e, file_day, d2, &file_day, &alive);

    CHECK(alive == 1, "the pair must survive five idle sessions in one day");
    CHECK(e.hits == 32,
          "five idle sessions inside ONE day are ONE boundary: 64 must halve "
          "once to 32, got %llu", (unsigned long long)e.hits);
    CHECK(file_day == d2, "the file records the day it was carried through");

    /* a real second day boundary halves again, once, however many sessions */
    for (i = 0; i < 3; i++)
        e = load_write_cycle(e, file_day, d3, &file_day, &alive);
    CHECK(alive == 1, "the pair still has evidence and must survive");
    CHECK(e.hits == 16,
          "a second day is a second boundary and no more: 32 must halve once "
          "to 16, got %llu", (unsigned long long)e.hits);

    /* and the pair demanded TODAY is carried free however many sessions run */
    ds4_warmset_entry hot = ent(4, 10);
    ds4_warmset_record(&hot, d2);
    hot.hits = 100;
    file_day = d2;
    for (i = 0; i < 4; i++)
        hot = load_write_cycle(hot, file_day, d2, &file_day, &alive);
    CHECK(hot.hits == 100,
          "a pair demanded today must not decay at all inside the day, got %llu",
          (unsigned long long)hot.hits);
}

/* --- 6. consolidation ---------------------------------------------------- */
static void test_consolidation(void) {
    const uint32_t days[3] = {
        ds4_warmset_day_key(2026, 9, 16),
        ds4_warmset_day_key(2026, 9, 17),
        ds4_warmset_day_key(2026, 9, 18),
    };
    ds4_warmset_entry e = ent(4, 77);
    CHECK(ds4_warmset_consolidated(&e, DS4_WARMSET_DAYS_DEFAULT) == 0,
          "an unseen pair must not be consolidated");
    for (int i = 0; i < 3; i++) {
        ds4_warmset_record(&e, days[i]);
        if (i < 2)
            CHECK(ds4_warmset_consolidated(&e, DS4_WARMSET_DAYS_DEFAULT) == 0,
                  "a pair seen on %d days must not be durable yet", i + 1);
    }
    CHECK(e.days == 3, "three distinct days, got %u", e.days);
    CHECK(ds4_warmset_consolidated(&e, DS4_WARMSET_DAYS_DEFAULT) == 1,
          "three distinct days must consolidate");
    CHECK(ds4_warmset_consolidated(&e, 0) == 0,
          "a zero threshold must consolidate nothing rather than everything");
}

/* --- 7. the order the slot budget truncates on --------------------------- */
static int cmp_before(const void *x, const void *y) {
    const ds4_warmset_entry *a = x, *b = y;
    if (ds4_warmset_before(a, b)) return -1;
    if (ds4_warmset_before(b, a)) return 1;
    return 0;
}
static void test_total_order_is_deterministic(void) {
    const uint32_t d1 = ds4_warmset_day_key(2026, 9, 16);
    const uint32_t d2 = ds4_warmset_day_key(2026, 9, 17);
    ds4_warmset_entry base[5];
    memset(base, 0, sizeof(base));
    base[0] = ent(1, 9); base[0].hits = 10; base[0].days = 2; base[0].last_day = d2;
    base[1] = ent(1, 4); base[1].hits = 10; base[1].days = 2; base[1].last_day = d2;
    base[2] = ent(2, 1); base[2].hits = 99; base[2].days = 1; base[2].last_day = d1;
    base[3] = ent(1, 4); base[3].hits = 11; base[3].days = 2; base[3].last_day = d2;
    base[4] = ent(0, 2); base[4].hits = 10; base[4].days = 3; base[4].last_day = d2;

    for (int i = 0; i < 5; i++)
        for (int j = 0; j < 5; j++)
            if (i != j) {
                const int ab = ds4_warmset_before(&base[i], &base[j]);
                const int ba = ds4_warmset_before(&base[j], &base[i]);
                CHECK(!(ab && ba), "the order must be asymmetric");
                if (!(base[i].days == base[j].days &&
                      base[i].hits == base[j].hits &&
                      base[i].layer == base[j].layer &&
                      base[i].expert == base[j].expert))
                    CHECK(ab != ba, "two distinct entries must never tie");
            }

    /* the same set read in a different order sorts to the same answer */
    ds4_warmset_entry p[5], q[5];
    memcpy(p, base, sizeof(p));
    memcpy(q, base, sizeof(q));
    ds4_warmset_entry t = q[0]; q[0] = q[3]; q[3] = t;
    t = q[1]; q[1] = q[4]; q[4] = t;
    qsort(p, 5, sizeof(p[0]), cmp_before);
    qsort(q, 5, sizeof(q[0]), cmp_before);
    for (int i = 0; i < 5; i++)
        CHECK(p[i].layer == q[i].layer && p[i].expert == q[i].expert,
              "the truncated set must not depend on the order the sources were read");
    CHECK(base[4].days == 3 && ds4_warmset_before(&base[4], &base[0]) == 1,
          "the most recurrent pair must sort first");
}

/* --- 8. THE SAFETY PROPERTY --------------------------------------------- */
/*
 * A warm set is a set of IDENTITIES plus EVIDENCE.  The only thing a wrong
 * entry can do is BE PRESENT.  Proved at the level a host test can reach:
 *
 *   (a) the vocabulary is five integers, so there is no magnitude to carry;
 *   (b) identity is (layer, expert) and nothing else, so a wrong entry cannot
 *       collide with a right one;
 *   (c) a wrong entry cannot change a right entry's evidence, so the right
 *       entries' fate under the retention policy is untouched by it.
 */
static void test_safety_property(void) {
    CHECK(sizeof(ds4_warmset_entry) == 24,
          "the entry must be exactly five integers, got %zu bytes",
          sizeof(ds4_warmset_entry));
    CHECK(ds4_warmset_key(3, 9) == ds4_warmset_key(3, 9),
          "identity must be stable");
    CHECK(ds4_warmset_key(3, 9) != ds4_warmset_key(9, 3),
          "identity must distinguish layer from expert");

    /* (b) evidence is NOT part of identity: two entries that differ only in
     * hits/days are the same pair and merge rather than coexist */
    ds4_warmset_entry r1 = ent(5, 5), r2 = ent(5, 5);
    r1.hits = 1; r2.hits = 1u << 30; r2.days = 40; r2.last_day = 20260916u;
    CHECK(ds4_warmset_key(r1.layer, r1.expert) ==
              ds4_warmset_key(r2.layer, r2.expert),
          "evidence must not enter identity");

    /* (c) a wrong entry leaves every right entry's evidence untouched */
    const uint32_t d1 = ds4_warmset_day_key(2026, 9, 16);
    ds4_warmset_entry right = ent(6, 8), wrong = ent(6, 9);
    for (int i = 0; i < 7; i++) ds4_warmset_record(&right, d1);
    ds4_warmset_record(&wrong, d1);
    const uint64_t right_hits = right.hits;
    const uint32_t right_days = right.days;
    /* merging the wrong entry in cannot move a right entry: they are different
     * keys, so nothing merges at all */
    CHECK(ds4_warmset_key(right.layer, right.expert) !=
              ds4_warmset_key(wrong.layer, wrong.expert),
          "a wrong entry must not collide with a right one");
    CHECK(right.hits == right_hits && right.days == right_days &&
              right.last_day == d1,
          "a wrong entry must not alter a right entry's evidence");
    CHECK(wrong.hits == 1 && wrong.days == 1,
          "a wrong entry is carried as one wasted slot of evidence, nothing more");

    /* and a wrong entry cannot survive consolidation on its own */
    CHECK(ds4_warmset_consolidated(&wrong, DS4_WARMSET_DAYS_DEFAULT) == 0,
          "a wrong entry seen on one day must never consolidate");
}

int main(void) {
    test_tier_defaults();
    test_record_counts_distinct_days();
    test_merge_rule();
    test_session_boundary();
    test_day_boundary();
    test_boundary_is_due_once_per_day();
    test_idle_sessions_do_not_decay_the_day();
    test_consolidation();
    test_total_order_is_deterministic();
    test_safety_property();

    if (g_failures) {
        fprintf(stderr, "test_warmset: %d of %d checks FAILED\n", g_failures, g_checks);
        return 1;
    }
    printf("test_warmset: %d checks passed (no GPU, no model, no CUDA)\n", g_checks);
    return 0;
}