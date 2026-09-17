#ifndef DS4_HOST_RANGE_CACHE_H
#define DS4_HOST_RANGE_CACHE_H

/* =========================================================================
 * Host expert read cache, keyed by (MODEL EPOCH, FILE OFFSET, BYTES).
 *
 * The Linux path reads experts with O_DIRECT, so the kernel page cache never
 * retains them and every re-fetch of the same routed expert pays full disk
 * latency. This cache keeps recently read small ranges (routed experts are
 * MiB-sized) in host RAM, keyed by the file range itself, so it is
 * model-agnostic: it caches BYTE RANGES, not expert IDs.
 *
 * The mechanism is upstream NeutronStar's, adapted to our sizes (see
 * WIKI/research/neutronstar/NEUTRONSTAR_02_expert-streaming.md section 1):
 *   - 16-way set-associative open-addressed table, bucket = murmur-style hash
 *     of the file offset, so nothing here knows what an expert is;
 *   - about one slot per MiB of budget, same 262,144-slot cap;
 *   - PAGEABLE malloc buffers, not pinned, on upstream's reasoning that the
 *     upload is slower from pageable but allocation is cheap and it is still
 *     far faster than the disk;
 *   - a per-slot `uses` hit counter and an `age`; a hit bumps both;
 *   - aging on an INSERT COUNT, never on a clock: every 4096 inserts ALL
 *     `uses` are halved in place, so once-hot entries cannot squat forever;
 *   - the victim is the least-`uses` entry in the 16-slot probe window, `age`
 *     breaks ties, and under budget pressure that victim is taken even when
 *     an empty slot exists in the window.
 *
 * WHICH POLICY IS THE DEFAULT IS NOT SETTLED AND IS NOT DECIDED HERE.
 * WIKI/theory/177 measured hotness-with-decay eviction WORSE than plain LRU
 * on our box (0.851 against 0.868 hit rate, 4.95 against 5.28 t/s) and it was
 * reverted; upstream's study argues the opposite for their media and keying.
 * That disagreement is what this arm exists to measure, so both policies are
 * carried and DS4_EXPERT_CACHE_MODE picks one - the fixes here are defect
 * fixes and they do not move the default.
 *
 * ---------------------------------------------------------------------
 * THE MODEL EPOCH IS PART OF THE KEY (2026-09-16).
 * An earlier revision keyed on (offset, bytes) alone and never dropped its
 * contents: the cache deliberately survives an expert-arena release, and
 * nothing else cleared it. So closing model A and opening model B in one
 * process - two quants of one architecture, or two engines in one test run -
 * let any expert whose (offset, bytes) collided upload A's BYTES into B's
 * slot, with no error anywhere. Wrong weights, silently.
 * The epoch is bumped in ds4_gpu_set_model_fd, the one place the model file
 * behind these offsets changes. It is in the key, so a stale entry can never
 * match; and a change purges the table, so the RAM those entries hold comes
 * back rather than sitting unreachable for the life of the process.
 * ---------------------------------------------------------------------
 *
 * Single threaded like the rest of the streaming globals: only the foreground
 * demand path reserves, commits or probes here, never the look-ahead reader
 * thread. Header-only and free of any CUDA call, so it is a unit under test.
 * ========================================================================= */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DS4_HRC_WAYS 16u
/* Upstream caps an entry at 64 MiB. Our routed expert tensor is ~3 MiB, so
 * one expert's three tensors fit one entry with room to spare. */
#define DS4_HRC_MAX_ENTRY (16ull << 20)
#define DS4_HRC_AGING_INSERTS 4096ull
#define DS4_HRC_SLOT_BYTES (1ull << 20)
#define DS4_HRC_MAX_SLOTS 262144ull

enum {
    DS4_HRC_OFF = 0,
    DS4_HRC_LFU = 1,   /* upstream: least uses, age tiebreak */
    DS4_HRC_LRU = 2    /* oldest age in the probe window */
};

typedef struct {
    uint64_t epoch;
    uint64_t offset;
    uint64_t bytes;
    uint64_t age;
    uint64_t uses; /* hit count; eviction is least-frequently-used, age breaks ties */
    void *buf;
    int valid;
} ds4_hrc_slot;

typedef struct {
    ds4_hrc_slot *slots;
    uint32_t nslots;
    uint64_t budget;
    uint64_t used;
    uint64_t tick;
    uint64_t inserts;
    uint64_t hits;
    uint64_t misses;
    uint64_t evictions;
    uint64_t epoch;   /* the model generation these entries belong to */
    uint64_t purges;
    uint64_t allocs;  /* reservations that had to malloc */
    uint64_t reuses;  /* reservations that kept a same-size victim's buffer */
    int mode;
} ds4_host_range_cache;

/* Reserve hands back this when the range will not be cached at all, so the
 * caller skips the per-chunk copy as well as the allocation. */
#define DS4_HRC_NO_SLOT UINT32_MAX

static inline int ds4_hrc_parse_mode(const char *env) {
    if (!env || !env[0]) return DS4_HRC_OFF;
    if (!strcmp(env, "hotlist")) return DS4_HRC_OFF;
    if (!strcmp(env, "offsetkey")) return DS4_HRC_LFU;
    if (!strcmp(env, "offsetkey-lru")) return DS4_HRC_LRU;
    fprintf(stderr, "ds4: unknown DS4_EXPERT_CACHE_MODE '%s'; "
            "using hotlist (host expert cache off)\n", env);
    return DS4_HRC_OFF;
}

static inline const char *ds4_hrc_mode_name(int mode) {
    return mode == DS4_HRC_LFU ? "offsetkey" :
           mode == DS4_HRC_LRU ? "offsetkey-lru" : "hotlist";
}

/* Returns 0 and leaves the cache off when the table cannot be built. */
static inline int ds4_hrc_open(ds4_host_range_cache *c, int mode, uint64_t budget) {
    memset(c, 0, sizeof(*c));
    c->mode = DS4_HRC_OFF;
    if (mode == DS4_HRC_OFF || budget == 0) return 0;
    uint64_t n = budget / DS4_HRC_SLOT_BYTES;
    if (n < DS4_HRC_WAYS) n = DS4_HRC_WAYS;
    if (n > DS4_HRC_MAX_SLOTS) n = DS4_HRC_MAX_SLOTS;
    c->slots = (ds4_hrc_slot *)calloc((size_t)n, sizeof(*c->slots));
    if (!c->slots) return 0;
    c->nslots = (uint32_t)n;
    c->budget = budget;
    c->mode = mode;
    return 1;
}

static inline void ds4_hrc_close(ds4_host_range_cache *c) {
    for (uint32_t i = 0; i < c->nslots; i++) {
        free(c->slots[i].buf);
        c->slots[i].buf = NULL;
        c->slots[i].valid = 0;
    }
    free(c->slots);
    c->slots = NULL;
    c->nslots = 0;
    c->used = 0;
}

static inline uint32_t ds4_hrc_bucket(const ds4_host_range_cache *c, uint64_t offset) {
    offset ^= offset >> 33;
    offset *= 0xff51afd7ed558ccdull;
    offset ^= offset >> 33;
    return (uint32_t)(offset % c->nslots);
}

/*
 * Tell the cache which model its offsets now refer to. A change drops every
 * entry: no entry from an older epoch can ever be hit again, so holding them
 * would only hold RAM. The epoch is also stored per slot, so correctness does
 * not depend on this call happening - an entry from another epoch cannot
 * match a lookup even if the purge never ran.
 */
static inline void ds4_hrc_set_epoch(ds4_host_range_cache *c, uint64_t epoch) {
    if (c->epoch == epoch) return;
    c->epoch = epoch;
    if (!c->nslots) return;
    uint64_t dropped = 0;
    for (uint32_t i = 0; i < c->nslots; i++) {
        if (!c->slots[i].valid) continue;
        free(c->slots[i].buf);
        c->slots[i].buf = NULL;
        c->slots[i].valid = 0;
        dropped++;
    }
    c->used = 0;
    /* Count the changes that actually dropped something, not every call: the
     * first set_epoch of a run moves 0 -> 1 over an empty table and is not a
     * model swap. */
    if (dropped) c->purges++;
}

/* The resident host buffer for exactly this (epoch, offset, bytes), or NULL. */
static inline const void *ds4_hrc_get(ds4_host_range_cache *c,
                                      uint64_t offset, uint64_t bytes) {
    if (!c->nslots) return NULL;
    const uint32_t h = ds4_hrc_bucket(c, offset);
    for (uint32_t i = 0; i < DS4_HRC_WAYS; i++) {
        ds4_hrc_slot *s = &c->slots[(h + i) % c->nslots];
        if (s->valid && s->epoch == c->epoch &&
            s->offset == offset && s->bytes == bytes) {
            s->age = ++c->tick;
            s->uses++;
            c->hits++;
            return s->buf;
        }
    }
    c->misses++;
    return NULL;
}

/*
 * Pick the slot this range will live in and hand back a buffer to fill.
 *
 * This replaces an insert-at-the-end that allocated a full host copy for
 * EVERY miss and then freed it again whenever the budget refused the range -
 * the allocation, the free and every per-chunk memcpy into it were spent on a
 * range that was never cached. Deciding first means a refusal costs one
 * 16-slot probe and nothing else. When the evicted victim's buffer is already
 * the right size - the common case, since a layer's expert tensors are all
 * one size - it is reused instead of freed and re-malloc'd.
 *
 * Returns NULL (and *out_slot = DS4_HRC_NO_SLOT) when the range will not be
 * cached. The slot is left INVALID until commit, so a half-filled buffer can
 * never be returned by a lookup.
 *
 * INVARIANT: AT MOST ONE RESERVATION IS OUTSTANDING AT A TIME. The demand
 * path reserves, fills and then commits or abandons inside one call to
 * cuda_model_copy_to_device_streamed, and only the foreground thread is here,
 * so a second reserve can never run while the first slot is sitting invalid.
 * Two outstanding reservations could be handed the same slot, because an
 * uncommitted slot reads as empty - do not introduce a second one.
 */
static inline void *ds4_hrc_reserve(ds4_host_range_cache *c,
                                    uint64_t offset, uint64_t bytes,
                                    uint32_t *out_slot) {
    *out_slot = DS4_HRC_NO_SLOT;
    if (!c->nslots || bytes == 0 || bytes > DS4_HRC_MAX_ENTRY) return NULL;

    /* Periodic decay so once-hot entries cannot squat forever. */
    if ((++c->inserts % DS4_HRC_AGING_INSERTS) == 0) {
        for (uint32_t i = 0; i < c->nslots; i++) c->slots[i].uses >>= 1;
    }

    const uint32_t h = ds4_hrc_bucket(c, offset);
    uint32_t empty = DS4_HRC_NO_SLOT;
    uint32_t cand = DS4_HRC_NO_SLOT;
    for (uint32_t i = 0; i < DS4_HRC_WAYS; i++) {
        const uint32_t idx = (h + i) % c->nslots;
        ds4_hrc_slot *s = &c->slots[idx];
        if (!s->valid) {
            if (empty == DS4_HRC_NO_SLOT) empty = idx;
            continue;
        }
        if (s->epoch == c->epoch && s->offset == offset && s->bytes == bytes) {
            return NULL; /* the same range is already resident */
        }
        if (cand == DS4_HRC_NO_SLOT) { cand = idx; continue; }
        const ds4_hrc_slot *b = &c->slots[cand];
        if (c->mode == DS4_HRC_LFU) {
            if (s->uses < b->uses || (s->uses == b->uses && s->age < b->age))
                cand = idx;
        } else if (s->age < b->age) {
            cand = idx;
        }
    }

    /* Under budget pressure, replace the window's least-used entry even when
     * an empty slot exists: with large entries the budget fills long before
     * the table does, and refusing inserts would freeze the cache contents.
     * Upstream's rationale, their words: "Frequency-weighted eviction protects
     * repeatedly hit experts from being flushed by one-shot streams (pure LRU
     * thrashes at these cache sizes)." */
    uint32_t victim = empty;
    if (c->used + bytes > c->budget && cand != DS4_HRC_NO_SLOT) victim = cand;
    else if (victim == DS4_HRC_NO_SLOT) victim = cand;
    if (victim == DS4_HRC_NO_SLOT) return NULL;

    void *reuse = NULL;
    ds4_hrc_slot *v = &c->slots[victim];
    if (v->valid) {
        c->used -= v->bytes;
        if (v->bytes == bytes) {
            reuse = v->buf; /* right size already: keep the allocation */
        } else {
            free(v->buf);
        }
        v->buf = NULL;
        v->valid = 0;
        c->evictions++;
    }
    if (c->used + bytes > c->budget) {
        free(reuse); /* the window had nothing to reclaim */
        return NULL;
    }
    void *buf = reuse;
    if (buf) {
        c->reuses++;
    } else {
        buf = malloc((size_t)bytes);
        if (!buf) return NULL;
        c->allocs++;
    }
    *out_slot = victim;
    return buf;
}

/* Publish a filled reservation. Takes ownership of buf. */
static inline void ds4_hrc_commit(ds4_host_range_cache *c, uint32_t slot,
                                  uint64_t offset, uint64_t bytes, void *buf) {
    if (slot == DS4_HRC_NO_SLOT || !buf) return;
    ds4_hrc_slot *s = &c->slots[slot];
    s->epoch = c->epoch;
    s->offset = offset;
    s->bytes = bytes;
    s->age = ++c->tick;
    s->uses = 1;
    s->buf = buf;
    s->valid = 1;
    c->used += bytes;
}

/* Drop a reservation that was never filled (a read or upload failed). The
 * slot was already emptied by reserve, so only the buffer is returned. */
static inline void ds4_hrc_abandon(ds4_host_range_cache *c, uint32_t slot, void *buf) {
    (void)c;
    (void)slot;
    free(buf);
}

#endif /* DS4_HOST_RANGE_CACHE_H */
