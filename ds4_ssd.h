#ifndef DS4_SSD_H
#define DS4_SSD_H

#include <stdbool.h>
#include <stdint.h>

typedef struct {
    void *ptr;
    uint64_t bytes;
} ds4_ssd_memory_lock;

typedef struct {
    uint64_t model_target_bytes;   /* engine (expert) cache tier share */
    uint64_t cache_bytes;
    uint64_t effective_cache_bytes;
    uint32_t cache_experts;

    /* The split. Three consumers draw on the same pool of RAM: the engine
     * tier, the pinned host tier, and the kernel page cache that backs the
     * memory-mapped Engram tables. The budget is a division of one pool, not
     * a maximum to push: past a point a bigger engine cache is SLOWER because
     * it steals page cache from those tables (the knee: WIKI/theory/
     * 48-decode-speedup-levers.md section 8; the measured negative result on
     * another rig: research/v41-flash-landscape/08-rtx-cards/CODE.md:279-285). */
    uint64_t pool_bytes;             /* the recommended working set divided */
    uint32_t engine_tier_pct;        /* DS4_SSD_AUTO_CACHE_PCT in force */
    uint32_t pagecache_floor_pct;    /* DS4_SSD_PAGECACHE_FLOOR_PCT in force */
    uint64_t pagecache_floor_bytes;  /* held apart for the kernel page cache */
    uint64_t host_tier_bytes;        /* remainder: pinned host tier + transient */
    bool     clamped_by_pagecache_floor;
} ds4_ssd_cache_plan;

bool ds4_parse_gib_arg(const char *s, uint64_t *bytes);
bool ds4_parse_streaming_cache_experts_arg(const char *s,
                                           uint32_t   *experts,
                                           uint64_t   *bytes);

uint32_t ds4_ssd_cache_experts_for_byte_budget(uint64_t bytes,
                                               uint64_t per_expert_bytes);
bool ds4_ssd_auto_cache_plan(uint64_t            recommended_bytes,
                             uint32_t            default_percent,
                             uint64_t            model_limit_bytes,
                             uint64_t            non_routed_bytes,
                             uint64_t            per_expert_bytes,
                             uint64_t            max_model_experts,
                             ds4_ssd_cache_plan *out);

bool ds4_ssd_memory_lock_acquire(ds4_ssd_memory_lock *lock,
                                 uint64_t             bytes);
void ds4_ssd_memory_lock_release(ds4_ssd_memory_lock *lock);

#endif
