/*
 * ds4_hotcache.h - the pure rules of the CUDA SSD expert cache.
 *
 * Two things live here, and they are here for the same reason: they are
 * DECISIONS, not I/O, so they can be proved on a laptop with no CUDA, no
 * model and no GPU.  ds4_cuda.cu is a nvcc translation unit and cannot be
 * built or tested without a device; anything in it that decides something is
 * therefore untestable in place.  See tests/test_hotcache.c and
 * `make test-hotcache`.  (Same shape as ds4_warmset.h on triple-hippocampal-
 * warmset: pure header, C test, no device.)
 *
 *   1. THE DECAY RULE - how DS4_CUDA_EXPERT_HOTLIST_DECAY is read.  The rule
 *      decides how much of the previous run's history this run inherits, so a
 *      value the parser misreads silently changes what the cache is seeded
 *      with.  The parser therefore REFUSES a value it cannot read whole and
 *      says so, rather than taking the number it happened to get.
 *
 *   2. THE VICTIM SCAN - which cache slots an eviction takes.  The policy is
 *      unchanged and is exactly the policy the scan in
 *      cuda_stream_selected_cache_begin_load has always had: least-recently
 *      used first, ties to the lowest slot index, never a slot the prefetch
 *      reader is holding, never a slot this batch has already stamped.  What
 *      changes is that ONE pass now answers a whole batch of misses instead of
 *      one pass per miss.
 *
 * No I/O, no CUDA, no allocation.
 */
#ifndef DS4_HOTCACHE_H
#define DS4_HOTCACHE_H

#include <stddef.h>
#include <stdint.h>
#include <string.h>

/* ---------------------------------------------------------------------------
 * 1. THE DECAY RULE
 *
 * "none" | "halve" | "quarter" | "eighth" | "shift:<n>", n in 0..32.
 * An absent or empty value is "halve", which is the default the hot list has
 * always had.
 *
 * WHY THE END POINTER MATTERS.  strtol(env + 6, NULL, 10) on "shift:x" returns
 * 0, and 0 is a VALID shift meaning "none" - carry the whole history forward,
 * which is the opposite end of the axis from the default.  So a typo does not
 * fail, it silently selects the other extreme and the run still looks fine.
 * Reading the end pointer is the difference between a refused value and a
 * wrong one.  (The rolling-rules parser in ds4_cuda.cu already did this; the
 * two parsers read the same syntax and now agree.)
 * ------------------------------------------------------------------------- */

#define DS4_HOTLIST_DECAY_DEFAULT_SHIFT 1u   /* "halve" */
#define DS4_HOTLIST_DECAY_MAX_SHIFT     32u

/* Returns 1 when the value was read whole, 0 when it was refused.  *shift is
 * always written: on a refusal it is the default, so a caller that ignores the
 * return value still gets today's behaviour rather than a guess. */
static inline int ds4_hotlist_decay_shift_from_env(const char *env, unsigned *shift) {
    unsigned out = DS4_HOTLIST_DECAY_DEFAULT_SHIFT;
    int ok = 1;
    if (env && env[0]) {
        if (strcmp(env, "none") == 0) out = 0;
        else if (strcmp(env, "halve") == 0) out = 1;
        else if (strcmp(env, "quarter") == 0) out = 2;
        else if (strcmp(env, "eighth") == 0) out = 3;
        else if (strncmp(env, "shift:", 6) == 0 && env[6]) {
            const char *p = env + 6;
            unsigned v = 0;
            size_t digits = 0;
            while (*p >= '0' && *p <= '9') {
                if (digits < 3u) v = v * 10u + (unsigned)(*p - '0');
                digits++;
                p++;
            }
            /* whole value, at least one digit, nothing trailing, in range */
            if (digits && digits <= 3u && *p == '\0' && v <= DS4_HOTLIST_DECAY_MAX_SHIFT) out = v;
            else ok = 0;
        } else {
            ok = 0;
        }
    }
    if (shift) *shift = out;
    return ok;
}

static inline const char *ds4_hotlist_decay_name(unsigned shift) {
    switch (shift) {
    case 0: return "none";
    case 1: return "halve";
    case 2: return "quarter";
    case 3: return "eighth";
    default: return "shift";
    }
}

/* ---------------------------------------------------------------------------
 * 2. THE VICTIM SCAN
 *
 * THE POLICY IS UNCHANGED.  The old scan, per miss:
 *
 *     victim = none; oldest = stamp;
 *     for (j = 0; j < n_slots; j++)
 *         if (!protected(j) && used[j] < oldest) {
 *             oldest = used[j]; victim = j;
 *             if (!oldest) break;          <- the early exit
 *         }
 *
 * i.e. the least-recently-used unprotected slot, ties to the lowest index,
 * with an early exit the moment an EMPTY slot (used == 0) is found, because
 * nothing can be older than empty.  A batch of M misses ran that loop M times,
 * and each run re-read every slot.
 *
 * THE EARLY EXIT LOST ITS PRECONDITION.  It fires only when an empty slot
 * exists.  Seeding the cache at startup fills every free slot, so from the
 * first token there are no empty slots and every miss walks the whole slot
 * vector - tens of thousands of entries on a large cache, per missed expert,
 * per routed layer, per token.  Seeding is worth keeping (the PRESENCE of an
 * entry is 27x to 45x the effect of its content, WIKI/theory/167), so the scan
 * is what changes, not the seed.
 *
 * ONE PASS ANSWERS THE WHOLE BATCH.  Selecting the M most-evictable slots in
 * one pass gives exactly the victims M sequential passes gave, in the same
 * order, because the old loop's only mutation between passes was stamping the
 * victim it had just taken with the current stamp, which the next pass would
 * have skipped anyway (used < stamp is the candidate test).  The early exit
 * survives whole: when M empty slots have been collected the pass stops, and
 * on a cold cache it stops at the first M slots exactly as before.
 *
 * Addressing: the caller's slots are a struct array, so the scan reads
 * `used` through a byte stride rather than owning the slot type.
 * ------------------------------------------------------------------------- */

static inline uint64_t ds4_victim_used_at(const void *base, size_t stride,
                                          size_t used_off, uint32_t i) {
    uint64_t v;
    memcpy(&v, (const unsigned char *)base + (size_t)i * stride + used_off, sizeof v);
    return v;
}

/* Is a "worse" (less evictable) than b?  The heap below keeps the worst of the
 * chosen set at its root so a better candidate can displace it. */
static inline int ds4_victim_worse(uint64_t used_a, uint32_t idx_a,
                                   uint64_t used_b, uint32_t idx_b) {
    if (used_a != used_b) return used_a > used_b;
    return idx_a > idx_b;
}

/*
 * Writes up to `want` slot indices into out[], ascending by (used, index):
 * out[0] is the victim the old loop would have taken first, out[1] the second,
 * and so on.  Returns how many were found, which is less than `want` only when
 * the cache has fewer evictable slots than the batch needs - the same
 * condition the old loop reported by finding no victim.
 *
 *   skip     - skip[j] != 0 means slot j may not be evicted (the prefetch
 *              reader is holding it).  NULL means nothing is protected, which
 *              is the state whenever no prefetch is in flight.
 *   stamp    - this batch's clock value.  A slot already stamped by this batch
 *              is not a candidate, which is what stops a miss evicting a hit.
 *   examined - optional: how many slots the pass visited.  This is the cost
 *              the fix exists to bound; tests/test_hotcache.c asserts it.
 */
static inline uint32_t ds4_expert_victim_scan(const void *base, size_t stride, size_t used_off,
                                              uint32_t n_slots, const unsigned char *skip,
                                              uint32_t want, uint64_t stamp,
                                              uint32_t *out, uint64_t *examined) {
    uint32_t n = 0, j;
    uint64_t seen = 0;
    if (!base || !out || !want) { if (examined) *examined = 0; return 0; }
    for (j = 0; j < n_slots; j++) {
        uint64_t used;
        seen++;
        if (skip && skip[j]) continue;
        used = ds4_victim_used_at(base, stride, used_off, j);
        if (used >= stamp) continue;
        if (n < want) {
            /* push, then sift up against the worst-at-root order */
            uint32_t c = n++;
            out[c] = j;
            while (c) {
                const uint32_t p = (c - 1) / 2;
                if (!ds4_victim_worse(ds4_victim_used_at(base, stride, used_off, out[c]), out[c],
                                      ds4_victim_used_at(base, stride, used_off, out[p]), out[p]))
                    break;
                { const uint32_t t = out[c]; out[c] = out[p]; out[p] = t; }
                c = p;
            }
        } else if (ds4_victim_worse(ds4_victim_used_at(base, stride, used_off, out[0]), out[0],
                                    used, j)) {
            /* j is better than the worst we hold: replace the root, sift down */
            uint32_t c = 0;
            out[0] = j;
            for (;;) {
                const uint32_t l = 2 * c + 1, r = l + 1;
                uint32_t w = c;
                if (l < n && ds4_victim_worse(ds4_victim_used_at(base, stride, used_off, out[l]), out[l],
                                              ds4_victim_used_at(base, stride, used_off, out[w]), out[w]))
                    w = l;
                if (r < n && ds4_victim_worse(ds4_victim_used_at(base, stride, used_off, out[r]), out[r],
                                              ds4_victim_used_at(base, stride, used_off, out[w]), out[w]))
                    w = r;
                if (w == c) break;
                { const uint32_t t = out[c]; out[c] = out[w]; out[w] = t; }
                c = w;
            }
        }
        /* THE EARLY EXIT, whole: the worst of the chosen set is empty, so all
         * of them are, and nothing later in the vector can be older. */
        if (n == want && ds4_victim_used_at(base, stride, used_off, out[0]) == 0) break;
    }
    if (examined) *examined = seen;
    /* ascending by (used, index): insertion sort, `want` is a batch of misses */
    for (j = 1; j < n; j++) {
        const uint32_t v = out[j];
        const uint64_t uv = ds4_victim_used_at(base, stride, used_off, v);
        uint32_t k = j;
        while (k && ds4_victim_worse(ds4_victim_used_at(base, stride, used_off, out[k - 1]), out[k - 1],
                                     uv, v)) {
            out[k] = out[k - 1];
            k--;
        }
        out[k] = v;
    }
    return n;
}

#endif /* DS4_HOTCACHE_H */
