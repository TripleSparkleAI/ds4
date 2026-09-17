/*
 * ds4_warmset.h - THE HIPPOCAMPAL WARM SET.
 *
 * The tier above the per-session expert hot list.  The hot list (see
 * cuda_expert_hotlist_* in ds4_cuda.cu, from branch triple-hotlist) already
 * persists a learned set between sessions: it writes the (layer, expert) pairs
 * a run asked for, halves them on load, and seeds the next session from them.
 * This header holds the RULES that let the same set persist across DAYS:
 *
 *   session  one process lifetime.  carried and halved at every load.
 *   day      every session of the local calendar day.  carried FREE inside
 *            the day, halved once per day boundary, provenance kept.
 *   durable  what RECURS: pairs demanded on DS4_WARMSET_DAYS distinct days.
 *
 * DELIBERATELY FORGOTTEN: a pair demanded on exactly one day and never again
 * (dropped at the first day boundary it misses); a pair whose decayed hits
 * reach 0; a set recorded against another model size; every seeded slot
 * (seeds are never recorded, so a guess can never entrench itself).
 *
 * ---------------------------------------------------------------------------
 * THE SAFETY PROPERTY, AND IT IS STRUCTURAL, NOT A POLICY
 *
 * Every field of ds4_warmset_entry is an integer and every field is either an
 * IDENTITY (layer, expert) or an EVIDENCE COUNT (hits, days, last_day).
 * There is no float, no value vector, no weight, no scale and no gate anywhere
 * in this vocabulary, so there is nothing here that could be added to a
 * residual stream even if someone tried.  The only consumer of a warm set is
 * the SSD expert cache slot seeder, whose sole effect is which expert's bytes
 * are resident.  The forward pass never reads this file.
 *
 * A wrong entry therefore costs ONE WASTED SLOT and nothing else: it occupies
 * a slot a correct entry could have used, it cannot evict a correct entry (the
 * seeder fills empty slots only), and it cannot change a token.
 * ---------------------------------------------------------------------------
 *
 * No I/O, no CUDA, no allocation: this header is pure so it can be tested
 * without a GPU.  See tests/test_warmset.c and `make test-warmset`.
 */
#ifndef DS4_WARMSET_H
#define DS4_WARMSET_H

#include <stdint.h>
#include <string.h>

#define DS4_WARMSET_TIER_SESSION 0
#define DS4_WARMSET_TIER_DAY     1

/* The default recurrence threshold: three distinct days. */
#define DS4_WARMSET_DAYS_DEFAULT 3u

typedef struct {
    uint32_t layer;
    uint32_t expert;
    uint64_t hits;      /* demand observations: one per layer batch per pair  */
    uint32_t days;      /* DISTINCT days on which the pair was demanded      */
    uint32_t last_day;  /* YYYYMMDD of the last day it was demanded          */
} ds4_warmset_entry;

/* The tier, from an environment string.  An absent or unknown value is the
 * session tier, which is today's behaviour: the switch defaults OFF. */
static inline int ds4_warmset_tier_from_env(const char *v) {
    if (!v || !v[0]) return DS4_WARMSET_TIER_SESSION;
    if (strcmp(v, "day") == 0) return DS4_WARMSET_TIER_DAY;
    return DS4_WARMSET_TIER_SESSION;
}

/* A pair's identity is (layer, expert).  Nothing else in the entry takes part
 * in identity, so two views of the same pair can never be two entries. */
static inline uint64_t ds4_warmset_key(uint32_t layer, uint32_t expert) {
    return ((uint64_t)layer << 32) | (uint64_t)expert;
}

/* A YYYYMMDD key.  Only equality with "today" and equality with another entry
 * are ever needed, so no calendar arithmetic is required. */
static inline uint32_t ds4_warmset_day_key(int year, int month, int day) {
    if (year < 1970 || month < 1 || month > 12 || day < 1 || day > 31) return 0;
    return (uint32_t)year * 10000u + (uint32_t)month * 100u + (uint32_t)day;
}

/*
 * RECORD: the router demanded this pair on day `today`, once for this layer
 * batch.  Demand adds.  Recurrence counts DISTINCT DAYS, so N demands inside
 * one day are one day of evidence and a second day is +1 exactly once.
 */
static inline void ds4_warmset_record(ds4_warmset_entry *e, uint32_t today) {
    if (!e || !today) return;
    e->hits++;
    if (e->days == 0) {
        e->days = 1;
        e->last_day = today;
    } else if (e->last_day != today) {
        e->days++;
        e->last_day = today;
    }
}

/*
 * THE MERGE RULE.
 *
 * Two sessions disagree about one pair when they rank it differently, or when
 * one saw it and the other did not.  Four parts, and the order matters:
 *
 *   (1) DEMAND IS ADDITIVE.  Two sessions that both asked for the pair both
 *       count: hits add.  Demand is evidence and evidence accumulates.
 *
 *   (2) ABSENCE IS NOT A NEGATIVE VOTE.  A session that did not ask for the
 *       pair is not saying the pair is cold, it simply did not exercise it.
 *       The pair is therefore KEPT, and it pays one halving at the next day
 *       boundary (ds4_warmset_day_boundary) rather than being deleted.
 *
 *   (3) RECURRENCE IS MAXIMAL, NEVER SUMMED.  A view carries one last_day, not
 *       the set of days it saw, so two views cannot prove their day sets are
 *       disjoint.  Summing days would therefore invent recurrence and would
 *       promote entries into the durable set on evidence we do not have.  The
 *       merge takes the maximum, which can only ever understate recurrence.
 *
 *   (4) IDENTITY IS (layer, expert) AND NOTHING ELSE.  Evidence never enters
 *       identity, so two views of one pair merge instead of coexisting.
 *
 * Ordering disagreement (which pair wins when the list is longer than the slot
 * budget) is settled by ds4_warmset_before, a TOTAL order, so the merged set
 * does not depend on the order the sources were read in.
 *
 * ds4_warmset_merge_view() is the second, narrower rule: two PROJECTIONS of the
 * same history (day.txt and durable.txt, where durable is derived from day).
 * Everything is maxed, including hits, which makes it idempotent so the same
 * view folded in twice cannot double count.
 */
static inline void ds4_warmset_merge(ds4_warmset_entry *dst,
                                     const ds4_warmset_entry *src) {
    if (!dst || !src) return;
    if (src->days == 0 && src->hits == 0) return;
    dst->hits += src->hits;                                   /* (1) additive   */
    if (src->days > dst->days) dst->days = src->days;         /* (3) maximal    */
    if (src->last_day > dst->last_day) dst->last_day = src->last_day;
    if (dst->days == 0 && dst->hits) dst->days = 1;
}

/* Two projections of one history.  Idempotent in every field. */
static inline void ds4_warmset_merge_view(ds4_warmset_entry *dst,
                                          const ds4_warmset_entry *src) {
    if (!dst || !src) return;
    if (src->hits > dst->hits) dst->hits = src->hits;
    if (src->days > dst->days) dst->days = src->days;
    if (src->last_day > dst->last_day) dst->last_day = src->last_day;
    if (dst->days == 0 && dst->hits) dst->days = 1;
}

/*
 * THE DAY BOUNDARY.  Returns 1 when the entry survives into today, 0 when it
 * is forgotten.  `today` is the day key of this load.
 *
 *   demanded earlier today            -> carried FREE, no decay.  This is the
 *                                        whole point of the tier: a day is a
 *                                        longer horizon than a session, so the
 *                                        session-boundary halving is skipped.
 *   seen on one day only, now missed  -> FORGOTTEN.  This is the pair that was
 *                                        only ever used once.
 *   seen on two or more days          -> halved, not killed, and kept.
 */
static inline int ds4_warmset_day_boundary(ds4_warmset_entry *e, uint32_t today) {
    if (!e) return 0;
    if (e->days == 0) return 1;          /* no provenance: not ours to decide */
    if (e->last_day == today) return 1;  /* carried free inside the day        */
    if (e->days < 2) return 0;           /* used once, never recurred, gone    */
    e->hits >>= 1;                       /* decay, not death                   */
    return e->hits != 0;                 /* dead only when the evidence is out */
}

/*
 * IS A DAY BOUNDARY DUE ON THIS LOAD?
 *
 * ds4_warmset_day_boundary decides what ONE boundary does to an entry.  This
 * decides whether a load is crossing one at all, and it is the difference
 * between "halved once per day" and "halved once per session".
 *
 * A warm-set file carries the day it was written (`# day_key`).  Every entry
 * in it has already been carried through that day's boundary, because it was
 * loaded, bounded and folded before it was written back.  So a file written
 * TODAY owes nothing: applying the boundary again is a second halving for one
 * calendar day, and a third for the session after that.  N idle sessions in
 * one day would shift an entry's hits by N, and the header's "decay, not
 * death" becomes death after log2(hits) sessions - inside a single day, which
 * is the one horizon the day tier exists to protect.
 *
 * `file_day` is 0 for a file that carries no day key at all (an older file, or
 * the session-tier hot list).  Then nothing is known about what it has already
 * paid, so the boundary IS due: the unknown case keeps today's behaviour.
 */
static inline int ds4_warmset_boundary_due(uint32_t file_day, uint32_t today) {
    if (!today) return 0;              /* no clock: decide nothing            */
    if (!file_day) return 1;           /* no provenance: pay it               */
    return file_day != today;
}

/* THE SESSION BOUNDARY, the tier below.  Unconditional halving: today's
 * behaviour, kept so the session tier is byte-identical to the hot list. */
static inline int ds4_warmset_session_boundary(ds4_warmset_entry *e) {
    if (!e) return 0;
    e->hits >>= 1;
    return e->hits != 0;
}

/* CONSOLIDATION: pairs demanded on this many distinct days are promoted into
 * the durable set.  Everything else stays a candidate. */
static inline int ds4_warmset_consolidated(const ds4_warmset_entry *e,
                                           uint32_t threshold_days) {
    if (!e || threshold_days == 0) return 0;
    return e->days >= threshold_days;
}

/* THE TOTAL ORDER the slot budget truncates on.  Deterministic and total: no
 * two distinct entries compare equal, so the surviving set is unique. */
static inline int ds4_warmset_before(const ds4_warmset_entry *a,
                                     const ds4_warmset_entry *b) {
    if (a->days != b->days) return a->days > b->days;
    if (a->hits != b->hits) return a->hits > b->hits;
    if (a->layer != b->layer) return a->layer < b->layer;
    return a->expert < b->expert;
}

/* The struct is five integers and nothing else.  If a future change ever adds
 * a float to this vocabulary, this stops the build: a warm set that could
 * carry a magnitude would no longer be a set of identities.
 *
 * It has to hold in the CONSUMER, not only in the test.  ds4_cuda.cu is a C++
 * translation unit, where __STDC_VERSION__ is not defined, so a guard written
 * for C11 alone was inactive in the one place a change would be made - the
 * build it was meant to stop was the only build it could not stop. */
#if defined(__cplusplus)
static_assert(sizeof(ds4_warmset_entry) == 24,
              "ds4_warmset_entry must be exactly five integers: a warm-set "
              "entry carries identity and evidence, never a magnitude");
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
_Static_assert(sizeof(ds4_warmset_entry) == 24,
               "ds4_warmset_entry must be exactly five integers: a warm-set "
               "entry carries identity and evidence, never a magnitude");
#endif

#endif /* DS4_WARMSET_H */