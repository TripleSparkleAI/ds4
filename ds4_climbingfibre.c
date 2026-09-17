/*
 * ds4_climbingfibre.c - the cerebellar front cache.
 *
 * Reads the design document's own safety argument (NEW_IDEA_THE_CEREBELLAR_
 * FRONT_CACHE_2026-09-01.md, section 5): the loop closes through the full model,
 * which is the teacher.  This file is the small, dumb, cheap end of that loop.
 *
 * It is deliberately ignorant.  It knows about token ids, counts, a clock and a
 * horizon.  It knows nothing about sessions, engines, weight tensors, KV state,
 * attention or a vocabulary table, and it is given no way to reach any of them.
 * That ignorance is not an accident of how it was written: it is the mechanism
 * by which CACHE LAW 1 (never modify the deep path, only skip it) is guaranteed
 * rather than promised.  A shortcut that cannot name the model cannot corrupt
 * it.  tests/test_climbingfibre.c pins it with a source guard.
 *
 * The cost model the design leans on (same document, section 3) is why a
 * shortcut in FRONT is the only placement that does not damage the deep path:
 * injecting mid-stack cost z +7.14 in our layer sweep, injecting at layer 1 was
 * z +1.11 and not significant.  Front is not a stylistic choice here; it is the
 * one place an intervention does not damage the deep path.
 */

#include "ds4_climbingfibre.h"

#include <stdlib.h>
#include <string.h>

#define CF_MAX_NGRAM 8
#define CF_MIN_NGRAM 2
#define CF_DEFAULT_NGRAM 4
/* The table must hold the session's key population or it thrashes and serves
 * nothing.  16384 keys costs about 0.9 MB and sits far above the number of
 * distinct 4-grams a single session visits; the point at which it stops being
 * enough is a measurement we have not run, which is why it is a knob.
 * tests/test_climbingfibre.c drives the cliff deliberately with a tiny cap. */
#define CF_DEFAULT_CAP 16384u
#define CF_MIN_CAP 64u
#define CF_MAX_CAP (1u << 20)
#define CF_STEPS_PER_DAY_DEFAULT 1048576ull
#define CF_STEPS_PER_SESSION_DEFAULT 65536ull
/* How many proposals may be OUTSTANDING when a rejection lands.  One is the
 * sequential world.  More is required as soon as the drafter is allowed to run
 * asynchronously beside verification (Saguaro, 2026): if drafting and verifying
 * overlap, a rejection can arrive while later drafts are still in flight, and a
 * single "the proposal I just made" slot would mis-attribute every one of them.
 * The WRITE-BACK never depended on this: the key comes from the verify site's
 * own prefix, so a late correction still lands on the right context.  The ring
 * exists so that ATTRIBUTION - how often it was OUR guess that got refused -
 * also survives drafts in flight. */
#define CF_INFLIGHT 8

/* Slot states.  A dropped entry becomes a TOMBSTONE rather than EMPTY so the
 * linear probe chain survives it; EMPTY is the only state that ends a probe. */
enum { CF_EMPTY = 0, CF_LIVE = 1, CF_TOMB = 2 };

typedef struct {
    int32_t  key[CF_MAX_NGRAM];
    int32_t  token;
    uint32_t hits;
    uint64_t last_step;
    uint8_t  state;
} cf_entry;

typedef struct {
    int32_t  key[CF_MAX_NGRAM];
    int32_t  token;
    uint64_t step;
    bool     used;
} cf_inflight;

struct ds4_climbingfibre {
    cf_entry *slots;
    uint32_t  cap;
    uint32_t  live;
    /* Slots that are not EMPTY: LIVE plus TOMB.  Only EMPTY terminates a probe
     * chain, so THIS is the number that decides what a lookup costs, and
     * without a sweep it only ever rises.  `used - live` is the tombstone
     * count the sweep triggers on. */
    uint32_t  used;
    int       ngram;

    /* The switch.  When it is false nothing is allocated and every entry point
     * returns before touching memory, so the OFF path and the "never called"
     * path are the same path. */
    bool enabled;

    /* The retention policy.  A HORIZON IN LOGICAL DECODE STEPS, not in seconds,
     * so that "how stale may it be" is a number we can sweep offline and
     * reproduce.  The named presets are a placeholder scale, not a measurement:
     * see the note in the header. */
    uint64_t                horizon;
    ds4_climbingfibre_decay decay;

    uint64_t step;

    /* Proposals handed out and not yet accounted for.  In a sequential engine
     * only the last entry is ever live; the ring is what makes the accounting
     * correct if drafting and verification overlap (Saguaro, 2026).  A refused
     * guess of OURS is a fact about this table; a refused guess of the engine's
     * is not, and the ring is how the two are told apart without assuming the
     * rejection belongs to the most recent proposal. */
    cf_inflight inflight[CF_INFLIGHT];
    uint32_t    inflight_next;

    uint64_t learns;
    uint64_t rejections;
    uint64_t corrections;    /* demotions that reached zero and dropped an entry */
    uint64_t demotions;      /* demotions that landed on a live entry */
    uint64_t evictions;
    uint64_t reclaims;
    uint64_t rehashes;
    /* Probe steps walked by the two chain walks (`cf_find` and `propose`).
     * This is the number the tombstone sweep exists to bound: without it the
     * count per miss climbs to `cap` and stays there for the rest of a long
     * session.  Counted so a test can assert the bound instead of asserting a
     * wall-clock time. */
    uint64_t probes;
    uint64_t proposals;
    uint64_t served;
    uint64_t missed;
    uint64_t own_rejected;
    uint64_t foreign_rejected;
};

/* ------------------------------------------------------------------ env --- */

static bool cf_env_flag(const char *name, bool dflt) {
    const char *v = getenv(name);
    if (!v || !v[0]) return dflt;
    return v[0] != '0';
}

static uint64_t cf_env_u64(const char *name, uint64_t dflt,
                           uint64_t lo, uint64_t hi) {
    const char *v = getenv(name);
    if (!v || !v[0]) return dflt;
    char *end = NULL;
    unsigned long long parsed = strtoull(v, &end, 10);
    if (end == v) return dflt;
    uint64_t out = (uint64_t)parsed;
    if (out < lo) out = lo;
    if (out > hi) out = hi;
    return out;
}

static int cf_env_int(const char *name, int dflt, int lo, int hi) {
    int64_t v = (int64_t)cf_env_u64(name, (uint64_t)dflt, 0, (uint64_t)hi);
    if (v < (int64_t)lo) v = lo;
    if (v > (int64_t)hi) v = hi;
    return (int)v;
}

static void cf_env_word(const char *name, char *out, size_t cap,
                        const char *dflt) {
    const char *v = getenv(name);
    if (!v || !v[0]) v = dflt;
    size_t n = strlen(v);
    if (n >= cap) n = cap - 1;
    memcpy(out, v, n);
    out[n] = '\0';
}

static ds4_climbingfibre_decay cf_parse_decay(const char *w) {
    if (strcmp(w, "hard") == 0) return DS4_CLIMBINGFIBRE_DECAY_HARD;
    if (strcmp(w, "none") == 0) return DS4_CLIMBINGFIBRE_DECAY_NONE;
    if (strcmp(w, "forever") == 0) return DS4_CLIMBINGFIBRE_DECAY_NONE;
    return DS4_CLIMBINGFIBRE_DECAY_HALF;
}

/* The horizon is what makes "session -> day -> week" a PARAMETER.  Each named
 * preset is a step budget; a bare integer is taken as the budget itself. */
static uint64_t cf_parse_horizon(const char *w) {
    uint64_t day = cf_env_u64("DS4_CLIMBINGFIBRE_STEPS_PER_DAY",
                              CF_STEPS_PER_DAY_DEFAULT, 1, UINT64_MAX / 8ull);
    uint64_t session = cf_env_u64("DS4_CLIMBINGFIBRE_STEPS_PER_SESSION",
                                  CF_STEPS_PER_SESSION_DEFAULT, 1,
                                  UINT64_MAX / 8ull);
    if (strcmp(w, "forever") == 0) return UINT64_MAX;
    if (strcmp(w, "session") == 0) return session;
    if (strcmp(w, "day") == 0)     return day;
    if (strcmp(w, "week") == 0)    return day * 7ull;
    char *end = NULL;
    unsigned long long parsed = strtoull(w, &end, 10);
    if (end != w && parsed != 0) return (uint64_t)parsed;
    return session;
}

static uint32_t cf_round_pow2(uint32_t v) {
    uint32_t out = CF_MIN_CAP;
    while (out < v && out < CF_MAX_CAP) out <<= 1;
    return out;
}

/* ------------------------------------------------------------- in flight --- */

/* Record a proposal so a later rejection can be attributed to it.  Old entries
 * are overwritten rather than cleared on a tick: under an asynchronous drafter
 * a rejection can land several cycles after the proposal it belongs to, so the
 * ring must survive ticks and age out by displacement instead. */
static void cf_inflight_record(ds4_climbingfibre *c, const int32_t *key,
                               int32_t token) {
    cf_inflight *slot = &c->inflight[c->inflight_next % CF_INFLIGHT];
    for (int i = 0; i < CF_MAX_NGRAM; i++) {
        slot->key[i] = i < c->ngram ? key[i] : 0;
    }
    slot->token = token;
    slot->step = c->step;
    slot->used = true;
    c->inflight_next++;
}

static bool cf_inflight_consume(ds4_climbingfibre *c, const int32_t *key,
                                int32_t token) {
    for (uint32_t i = 0; i < CF_INFLIGHT; i++) {
        cf_inflight *slot = &c->inflight[i];
        if (!slot->used || slot->token != token) continue;
        bool same = true;
        for (int j = 0; j < c->ngram; j++) {
            if (slot->key[j] != key[j]) { same = false; break; }
        }
        if (!same) continue;
        slot->used = false;
        return true;
    }
    return false;
}

/* ------------------------------------------------------------ table --- */

static uint64_t cf_hash(const ds4_climbingfibre *c, const int32_t *key) {
    /* FNV-1a over the token ids, then the key width folded in so that a 4-gram
     * and its own 2-gram suffix cannot collide on the same hash. */
    uint64_t h = 1469598103934665603ull;
    for (int i = 0; i < c->ngram; i++) {
        uint64_t v = (uint64_t)(uint32_t)key[i];
        for (int b = 0; b < 4; b++) {
            h ^= (v >> (8 * b)) & 0xffull;
            h *= 1099511628211ull;
        }
    }
    h ^= (uint64_t)c->ngram;
    h *= 1099511628211ull;
    return h;
}

static bool cf_key_eq(const ds4_climbingfibre *c, const cf_entry *e,
                      const int32_t *key) {
    for (int i = 0; i < c->ngram; i++) {
        if (e->key[i] != key[i]) return false;
    }
    return true;
}

/* Is this entry allowed to be served at the current step?  Applying the policy
 * here, on the read path, means retention cannot be bypassed by a caller that
 * forgets to sweep.  HALF and NONE both WRITE on this path; that is the only
 * mutation a read performs, and it touches this module's own array and nothing
 * else. */
static bool cf_serve(ds4_climbingfibre *c, cf_entry *e) {
    if (c->horizon == UINT64_MAX ||
        c->decay == DS4_CLIMBINGFIBRE_DECAY_NONE) {
        return true;
    }
    uint64_t age = c->step >= e->last_step ? c->step - e->last_step : 0;
    if (age <= c->horizon) return true;
    if (c->decay == DS4_CLIMBINGFIBRE_DECAY_HARD) return false;
    /* HALF: the entry is not deleted, it is demoted - the same shape the design
     * document measured as the lever that works (MEASURED_DECAYVSDEMOTE: plain
     * proportional forgetting is worth +22.2 percent on its own). */
    e->hits = e->hits / 2u;
    e->last_step = c->step;
    if (e->hits == 0u) {
        /* AND A DEMOTION THAT REACHES ZERO IS A RETIREMENT.  Before this, the
         * entry stayed LIVE with its clock reset, so the very next probe found
         * it inside the horizon again and served it with a weight of zero: the
         * halving was a one-step skip and nothing ever aged out.  An entry
         * whose weight decay has consumed is dropped, exactly as the HARD path
         * drops a past-horizon entry.  This is the same verb `learn` already
         * applies when a demotion reaches zero (see the demotion below). */
        e->state = CF_TOMB;
        if (c->live) c->live--;
        c->reclaims++;
        return false;
    }
    return true;
}

/* Walk the probe chain for `key`.  Returns the live entry whose candidate is
 * `token`, or NULL.  *insert_at receives the first reusable slot (a tombstone,
 * or the terminator) so the caller can insert without walking twice. */
static cf_entry *cf_find(ds4_climbingfibre *c, const int32_t *key,
                         int32_t token, cf_entry **insert_at) {
    uint64_t h = cf_hash(c, key);
    uint32_t mask = c->cap - 1u;
    cf_entry *tomb = NULL;
    for (uint32_t probe = 0; probe < c->cap; probe++) {
        c->probes++;
        cf_entry *e = &c->slots[(uint32_t)(h + probe) & mask];
        if (e->state == CF_EMPTY) {
            if (insert_at) *insert_at = tomb ? tomb : e;
            return NULL;
        }
        if (e->state == CF_TOMB) {
            if (!tomb) tomb = e;
            continue;
        }
        if (e->token == token && cf_key_eq(c, e, key)) {
            if (insert_at) *insert_at = NULL;
            return e;
        }
    }
    if (insert_at) *insert_at = tomb;
    return NULL;
}

/* Bounded reclamation, run only when the table is crowded.  It drops entries
 * the retention policy has already written off, and only if that frees nothing
 * does it evict the weakest entry in a bounded window.  Never touches anything
 * outside this module's array. */
static void cf_reclaim(ds4_climbingfibre *c) {
    uint32_t window = c->cap / 8u;
    if (window < 8u) window = 8u;
    uint32_t mask = c->cap - 1u;
    uint32_t start = (uint32_t)(c->step * 2654435761ull) & mask;
    cf_entry *weakest = NULL;
    for (uint32_t i = 0; i < window; i++) {
        cf_entry *e = &c->slots[(start + i) & mask];
        if (e->state != CF_LIVE) continue;
        if (c->horizon != UINT64_MAX &&
            c->decay == DS4_CLIMBINGFIBRE_DECAY_HARD) {
            uint64_t age =
                c->step >= e->last_step ? c->step - e->last_step : 0;
            if (age > c->horizon) {
                e->state = CF_TOMB;
                c->live--;
                c->reclaims++;
                continue;
            }
        }
        if (!weakest ||
            e->hits < weakest->hits ||
            (e->hits == weakest->hits && e->last_step < weakest->last_step)) {
            weakest = e;
        }
    }
    if (c->live * 4u >= c->cap * 3u && weakest) {
        weakest->state = CF_TOMB;
        c->live--;
        c->evictions++;
    }
}

/* THE TOMBSTONE SWEEP.  A dropped entry becomes a TOMB so the probe chain that
 * ran through it survives, and only EMPTY ends a chain - so without this, the
 * number of non-EMPTY slots is MONOTONE.  Past `cap` distinct inserts the table
 * is all LIVE and TOMB, and every miss in `cf_find` and every `propose` walks
 * all `cap` slots, on every speculative cycle, for the rest of the session.
 *
 * Rebuilding the table from its LIVE entries alone turns every tombstone back
 * into EMPTY.  The trigger is the NON-EMPTY load, not the tombstone count: what
 * a lookup costs is decided by how far it is from the nearest EMPTY slot, so
 * bounding the tombstones without bounding non-EMPTY still lets the table fill
 * completely between sweeps.  Above seven eighths non-EMPTY the table is swept
 * back to its live population, which the reclaimer holds at three quarters - so
 * a sweep is paid for by at least cap/8 inserts and costs about eight slot
 * visits per insert, amortised, against an unbounded walk per lookup.
 *
 * On allocation failure it does nothing and returns; the table keeps its
 * tombstones and behaves exactly as it did before, which is slow and correct. */
static void cf_rehash(ds4_climbingfibre *c) {
    cf_entry *old = c->slots;
    cf_entry *fresh = calloc((size_t)c->cap, sizeof(cf_entry));
    if (!fresh) return;
    uint32_t mask = c->cap - 1u;
    uint32_t live = 0;
    for (uint32_t i = 0; i < c->cap; i++) {
        if (old[i].state != CF_LIVE) continue;
        uint64_t h = cf_hash(c, old[i].key);
        for (uint32_t probe = 0; probe < c->cap; probe++) {
            cf_entry *e = &fresh[(uint32_t)(h + probe) & mask];
            if (e->state != CF_EMPTY) continue;
            *e = old[i];
            live++;
            break;
        }
    }
    c->slots = fresh;
    c->live = live;
    c->used = live;      /* every tombstone is EMPTY again */
    c->rehashes++;
    free(old);
}

static cf_entry *cf_touch(ds4_climbingfibre *c, const int32_t *key,
                          int32_t token, bool insert) {
    cf_entry *insert_at = NULL;
    cf_entry *e = cf_find(c, key, token, &insert_at);
    if (e || !insert) return e;
    /* Sweep the tombstones before deciding the table is crowded: a table that
     * is three-quarters TOMB is not crowded, it is stale. */
    if (c->used > c->live && c->used * 8u >= c->cap * 7u) {
        cf_rehash(c);
        insert_at = NULL;
        e = cf_find(c, key, token, &insert_at);
        if (e) return e;
    }
    if (!insert_at) return NULL;
    if (c->live * 4u >= c->cap * 3u) {
        cf_reclaim(c);
        insert_at = NULL;
        e = cf_find(c, key, token, &insert_at);
        if (e) return e;
        if (!insert_at) return NULL;
    }
    /* Re-anchor after a possible reclaim by walking the chain again for a free
     * slot; the table is small and this happens only on a miss. */
    if (insert_at->state == CF_LIVE) {
        uint64_t h = cf_hash(c, key);
        uint32_t mask = c->cap - 1u;
        cf_entry *slot = NULL;
        for (uint32_t probe = 0; probe < c->cap; probe++) {
            cf_entry *s = &c->slots[(uint32_t)(h + probe) & mask];
            if (s->state != CF_LIVE) { slot = s; break; }
        }
        if (!slot) return NULL;
        insert_at = slot;
    }
    if (insert_at->state == CF_EMPTY) c->used++;
    insert_at->state = CF_LIVE;
    insert_at->token = token;
    insert_at->hits = 0;
    insert_at->last_step = c->step;
    for (int i = 0; i < CF_MAX_NGRAM; i++) {
        insert_at->key[i] = i < c->ngram ? key[i] : 0;
    }
    c->live++;
    return insert_at;
}

/* --------------------------------------------------------------- public --- */

ds4_climbingfibre *ds4_climbingfibre_create(void) {
    ds4_climbingfibre *c = calloc(1u, sizeof(*c));
    if (!c) return NULL;
    c->enabled = cf_env_flag("DS4_CLIMBINGFIBRE", false);
    if (!c->enabled) {
        /* OFF allocates nothing at all: the memory footprint of this feature in
         * a default run is exactly zero. */
        return c;
    }
    c->ngram = cf_env_int("DS4_CLIMBINGFIBRE_NGRAM", CF_DEFAULT_NGRAM,
                          CF_MIN_NGRAM, CF_MAX_NGRAM);
    char word[32];
    cf_env_word("DS4_CLIMBINGFIBRE_DECAY", word, sizeof(word), "half");
    c->decay = cf_parse_decay(word);
    cf_env_word("DS4_CLIMBINGFIBRE_HORIZON", word, sizeof(word), "session");
    c->horizon = cf_parse_horizon(word);
    c->cap = cf_round_pow2((uint32_t)cf_env_u64("DS4_CLIMBINGFIBRE_CAP",
                                                CF_DEFAULT_CAP,
                                                CF_MIN_CAP, CF_MAX_CAP));
    c->slots = calloc((size_t)c->cap, sizeof(cf_entry));
    if (!c->slots) {
        free(c);
        return NULL;
    }
    return c;
}

void ds4_climbingfibre_free(ds4_climbingfibre *c) {
    if (!c) return;
    if (cf_env_flag("DS4_CLIMBINGFIBRE_STATS", false)) {
        ds4_climbingfibre_census(c, stderr);
    }
    free(c->slots);
    free(c);
}

bool ds4_climbingfibre_enabled(const ds4_climbingfibre *c) {
    return c && c->enabled;
}

int ds4_climbingfibre_ngram(const ds4_climbingfibre *c) {
    if (!c || !c->enabled || !c->slots) return 0;
    return c->ngram;
}

void ds4_climbingfibre_tick(ds4_climbingfibre *c) {
    if (!c || !c->enabled) return;
    c->step++;
    /* In-flight proposals are NOT cleared here.  Under a sequential drafter they
     * are resolved within the cycle that made them, so clearing would be a
     * no-op; under an overlapping drafter clearing would throw away exactly the
     * attribution this ring exists to keep. */
}

uint64_t ds4_climbingfibre_step(const ds4_climbingfibre *c) {
    return c ? c->step : 0;
}

void ds4_climbingfibre_learn(ds4_climbingfibre *c,
                             const int32_t *key, int key_len,
                             int32_t drafted, int32_t actual) {
    if (!c || !c->enabled || !c->slots) return;
    if (!key || key_len < c->ngram) return;
    if (actual < 0) return;
    c->learns++;
    if (drafted == actual) {
        /* A confirmation, not a correction.  The rejection sites do not call
         * this way; if one ever does, the honest treatment is to reinforce. */
        cf_entry *e = cf_touch(c, key, actual, true);
        if (e) { e->hits++; e->last_step = c->step; }
        return;
    }
    c->rejections++;

    /* THE WRITE-BACK.  The token the target produced is stored as a candidate
     * against this exact context.  This is the whole of the novel claim. */
    cf_entry *good = cf_touch(c, key, actual, true);
    if (good) { good->hits++; good->last_step = c->step; }

    /* AND THE DEPRESSION.  The token that was refused, if this table was the
     * one that offered it, is demoted.  Albus's climbing fibre drives
     * depression, not reward; a correction signal that could only ever add
     * would turn the table into a list of every mistake ever made. */
    cf_entry *bad = cf_find(c, key, drafted, NULL);
    if (bad) {
        c->demotions++;
        if (bad->hits > 0) bad->hits--;
        if (bad->hits == 0) {
            bad->state = CF_TOMB;
            c->live--;
            c->corrections++;
        }
    }

    /* Attribution, so the census can say whether OUR guesses were the ones
     * being refused.  MEASURED_DECAYVSDEMOTE is the reason this is recorded
     * rather than assumed: most of what looked like a rejection signal turned
     * out to be plain forgetting, and a table that cannot say how often it was
     * wrong cannot tell those apart either.  The match is against the ring, so
     * a rejection that lands while later drafts are in flight still finds its
     * own proposal rather than the newest one. */
    if (cf_inflight_consume(c, key, drafted)) c->own_rejected++;
    else c->foreign_rejected++;
}

bool ds4_climbingfibre_propose(ds4_climbingfibre *c,
                               const int32_t *key, int key_len,
                               int32_t *out, uint32_t out_cap,
                               uint32_t *out_len) {
    if (out_len) *out_len = 0;
    if (!c || !c->enabled || !c->slots) return false;
    if (!key || key_len < c->ngram) return false;
    if (!out || out_cap == 0) return false;

    c->proposals++;
    uint64_t h = cf_hash(c, key);
    uint32_t mask = c->cap - 1u;
    cf_entry *best = NULL;
    for (uint32_t probe = 0; probe < c->cap; probe++) {
        c->probes++;
        cf_entry *e = &c->slots[(uint32_t)(h + probe) & mask];
        if (e->state == CF_EMPTY) break;
        if (e->state != CF_LIVE) continue;
        if (!cf_key_eq(c, e, key)) continue;
        if (!cf_serve(c, e)) continue;
        if (!best || e->hits > best->hits) best = e;
    }
    if (!best) {
        /* A MISS.  Nothing is written, nothing is counted into the caller, and
         * the caller leaves its own state exactly as it was.  This is the point
         * of the design, not an error path. */
        c->missed++;
        return false;
    }
    out[0] = best->token;
    if (out_len) *out_len = 1;
    c->served++;
    cf_inflight_record(c, key, best->token);
    return true;
}

void ds4_climbingfibre_census(const ds4_climbingfibre *c, FILE *out) {
    if (!out) return;
    if (!c) {
        fprintf(out, "ds4: CLIMBINGFIBRE census absent=1\n");
        return;
    }
    fprintf(out,
            "ds4: CLIMBINGFIBRE census enabled=%d ngram=%d cap=%u live=%u "
            "used=%u horizon=%llu decay=%d step=%llu learns=%llu "
            "rejections=%llu "
            "demotions=%llu corrections=%llu evictions=%llu reclaims=%llu "
            "rehashes=%llu probes=%llu "
            "proposals=%llu served=%llu missed=%llu own_rejected=%llu "
            "foreign_rejected=%llu\n",
            c->enabled ? 1 : 0,
            c->ngram,
            c->cap,
            c->live,
            c->used,
            (unsigned long long)c->horizon,
            (int)c->decay,
            (unsigned long long)c->step,
            (unsigned long long)c->learns,
            (unsigned long long)c->rejections,
            (unsigned long long)c->demotions,
            (unsigned long long)c->corrections,
            (unsigned long long)c->evictions,
            (unsigned long long)c->reclaims,
            (unsigned long long)c->rehashes,
            (unsigned long long)c->probes,
            (unsigned long long)c->proposals,
            (unsigned long long)c->served,
            (unsigned long long)c->missed,
            (unsigned long long)c->own_rejected,
            (unsigned long long)c->foreign_rejected);
}