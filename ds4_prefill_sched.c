/* ds4_prefill_sched.c - the optimal expert fetch schedule for layer-major prefill.
 * Contract, rationale and scope: ds4_prefill_sched.h. */
#include "ds4_prefill_sched.h"

#include <stdlib.h>
#include <string.h>

#define SCHED_NEVER UINT32_MAX

/* Residency, one byte per expert id, plus the bookkeeping each eviction rule needs.  n_expert is
 * 384 on V4.1 Flash, so a flat table is both the simplest and the fastest thing here. */
typedef struct {
    uint8_t  *resident;
    uint32_t *stamp;      /* LRU only: the layer at which the expert was last used */
    uint32_t  live;       /* how many experts are resident */
} sched_cache;

static int sched_demand_ok(const ds4_prefill_demand *d) {
    if (!d || !d->experts || !d->layer_off) return 0;
    if (d->n_layer == 0 || d->n_expert == 0 || d->slots == 0) return 0;
    if (d->layer_off[0] != 0) return 0;
    for (uint32_t L = 0; L < d->n_layer; L++) {
        if (d->layer_off[L + 1] < d->layer_off[L]) return 0;
    }
    const uint32_t n = d->layer_off[d->n_layer];
    for (uint32_t i = 0; i < n; i++) {
        if (d->experts[i] >= d->n_expert) return 0;
    }
    /* A layer that needs more distinct experts than the cache can hold cannot be scheduled at all:
     * something resident would have to be evicted to make room for something this same layer still
     * needs.  Refuse rather than silently thrash, because a thrashing schedule is not a floor. */
    uint8_t *seen = calloc(d->n_expert, 1);
    if (!seen) return 0;
    int ok = 1;
    for (uint32_t L = 0; L < d->n_layer && ok; L++) {
        uint32_t distinct = 0;
        for (uint32_t i = d->layer_off[L]; i < d->layer_off[L + 1]; i++) {
            if (!seen[d->experts[i]]) { seen[d->experts[i]] = 1; distinct++; }
        }
        for (uint32_t i = d->layer_off[L]; i < d->layer_off[L + 1]; i++) seen[d->experts[i]] = 0;
        if (distinct > d->slots) ok = 0;
    }
    free(seen);
    return ok;
}

/* The next layer at or after `from` that consults `expert`, or SCHED_NEVER.  This is the whole of
 * Belady: the victim is the resident expert whose answer here is largest. */
static uint32_t sched_next_use(const ds4_prefill_demand *d, uint16_t expert, uint32_t from) {
    for (uint32_t L = from; L < d->n_layer; L++) {
        for (uint32_t i = d->layer_off[L]; i < d->layer_off[L + 1]; i++) {
            if (d->experts[i] == expert) return L;
        }
    }
    return SCHED_NEVER;
}

/* Belady MIN: furthest next use wins, SCHED_NEVER beats every finite distance.  Ties break on the
 * lower expert id so two runs of one demand produce one schedule, byte for byte. */
static int sched_victim_belady(const ds4_prefill_demand *d,
                               const sched_cache        *c,
                               uint32_t                  layer,
                               const uint8_t            *pinned,
                               uint16_t                 *out) {
    uint32_t best_dist = 0;
    int      found = 0;
    for (uint32_t e = 0; e < d->n_expert; e++) {
        if (!c->resident[e] || pinned[e]) continue;
        const uint32_t dist = sched_next_use(d, (uint16_t)e, layer);
        if (!found || dist > best_dist) {
            best_dist = dist;
            *out = (uint16_t)e;
            found = 1;
            if (dist == SCHED_NEVER) break;   /* nothing can beat never */
        }
    }
    return found;
}

/* LRU: the least recently used unpinned resident expert.  Ties break on the lower id, as above. */
static int sched_victim_lru(const ds4_prefill_demand *d,
                            const sched_cache        *c,
                            const uint8_t            *pinned,
                            uint16_t                 *out) {
    uint32_t best_stamp = 0;
    int      found = 0;
    for (uint32_t e = 0; e < d->n_expert; e++) {
        if (!c->resident[e] || pinned[e]) continue;
        if (!found || c->stamp[e] < best_stamp) {
            best_stamp = c->stamp[e];
            *out = (uint16_t)e;
            found = 1;
        }
    }
    return found;
}

static void sched_emit(ds4_prefill_act *acts,
                       uint32_t         cap,
                       uint32_t        *len,
                       uint32_t         at_layer,
                       uint32_t         for_layer,
                       uint16_t         expert,
                       uint8_t          act) {
    if (acts && *len < cap) {
        acts[*len].at_layer  = at_layer;
        acts[*len].for_layer = for_layer;
        acts[*len].expert    = expert;
        acts[*len].act       = act;
    }
    (*len)++;                      /* counted even when it does not fit, so the caller can size */
}

/* One walk serves both rules.  Everything except the victim choice is shared, which is the point:
 * a difference in the statistics is a difference in the eviction rule and in nothing else. */
static int sched_run(const ds4_prefill_demand *d,
                     int                       belady,
                     ds4_prefill_act          *acts,
                     uint32_t                  acts_cap,
                     uint32_t                 *acts_len,
                     ds4_prefill_sched_stats  *st) {
    uint32_t len = 0;
    if (acts_len) *acts_len = 0;
    if (st) memset(st, 0, sizeof(*st));
    if (!sched_demand_ok(d)) return -1;

    sched_cache c = {0};
    c.resident = calloc(d->n_expert, 1);
    c.stamp    = calloc(d->n_expert, sizeof(uint32_t));
    uint8_t *pinned = calloc(d->n_expert, 1);   /* what THIS layer still needs, never a victim */
    uint8_t *seen   = calloc(d->n_expert, 1);
    if (!c.resident || !c.stamp || !pinned || !seen) {
        free(c.resident); free(c.stamp); free(pinned); free(seen);
        return -1;
    }

    for (uint32_t L = 0; L < d->n_layer; L++) {
        const uint32_t beg = d->layer_off[L], end = d->layer_off[L + 1];

        /* Pin this layer's whole demand first.  Without this, a layer needing k experts can evict
         * one it has not consumed yet and fetch it again inside the same layer, which is thrash
         * dressed up as a schedule. */
        for (uint32_t i = beg; i < end; i++) pinned[d->experts[i]] = 1;

        for (uint32_t i = beg; i < end; i++) {
            const uint16_t e = d->experts[i];
            if (seen[e]) continue;              /* a repeat within the layer is one residency */
            seen[e] = 1;
            if (st) st->needed++;

            if (c.resident[e]) {
                if (st) st->hits++;
                c.stamp[e] = L;
                continue;
            }

            while (c.live >= d->slots) {
                uint16_t victim = 0;
                const int got = belady ? sched_victim_belady(d, &c, L, pinned, &victim)
                                       : sched_victim_lru(d, &c, pinned, &victim);
                if (!got) break;                /* every resident expert is pinned; refused above */
                c.resident[victim] = 0;
                c.live--;
                if (st) st->evictions++;
                sched_emit(acts, acts_cap, &len, L, L, victim, DS4_PREFILL_ACT_EVICT);
            }

            /* The lead window.  Layer L's reads are issued during layer L - lead_layers, so the
             * first lead_layers layers cannot be covered and their reads are LATE by construction,
             * exactly as the real prefill's first layer is. */
            const uint32_t at = (L >= d->lead_layers) ? L - d->lead_layers : 0;
            if (L < d->lead_layers && st) st->late++;

            c.resident[e] = 1;
            c.live++;
            c.stamp[e] = L;
            if (st) st->fetches++;
            sched_emit(acts, acts_cap, &len, at, L, e, DS4_PREFILL_ACT_FETCH);
        }

        for (uint32_t i = beg; i < end; i++) { pinned[d->experts[i]] = 0; seen[d->experts[i]] = 0; }
    }

    free(c.resident); free(c.stamp); free(pinned); free(seen);
    if (acts_len) *acts_len = len;
    if (acts && len > acts_cap) return -2;
    return 0;
}

int ds4_prefill_schedule(const ds4_prefill_demand *d,
                         ds4_prefill_act          *acts,
                         uint32_t                  acts_cap,
                         uint32_t                 *acts_len,
                         ds4_prefill_sched_stats  *st) {
    return sched_run(d, 1, acts, acts_cap, acts_len, st);
}

int ds4_prefill_schedule_lru(const ds4_prefill_demand *d,
                             ds4_prefill_act          *acts,
                             uint32_t                  acts_cap,
                             uint32_t                 *acts_len,
                             ds4_prefill_sched_stats  *st) {
    return sched_run(d, 0, acts, acts_cap, acts_len, st);
}
