#define _POSIX_C_SOURCE 200809L
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "ds4_ssd.h"

int main(void) {
    const uint64_t gib = 1ull << 30;
    ds4_ssd_cache_plan p;
    unsetenv("DS4_SSD_AUTO_CACHE_PCT");
    unsetenv("DS4_SSD_PAGECACHE_FLOOR_PCT");
    assert(ds4_ssd_auto_cache_plan(100 * gib, 86, 0, 10 * gib, gib, 1000, &p));
    assert(p.model_target_bytes == 86 * gib && p.cache_experts == 76);
    assert(ds4_ssd_auto_cache_plan(100 * gib, 80, 0, 10 * gib, gib, 1000, &p));
    assert(p.model_target_bytes == 80 * gib && p.cache_experts == 70);
    /* A large context reduces the combined static/cache budget. */
    assert(ds4_ssd_auto_cache_plan(100 * gib, 86, 65 * gib, 10 * gib, gib, 1000, &p));
    assert(p.model_target_bytes == 65 * gib && p.cache_experts == 55);
    assert(p.effective_cache_bytes == 55 * gib);
    setenv("DS4_SSD_AUTO_CACHE_PCT", "95", 1);
    assert(ds4_ssd_auto_cache_plan(100 * gib, 86, 65 * gib, 10 * gib, gib, 1000, &p));
    assert(p.model_target_bytes == 65 * gib && p.cache_experts == 55);
    assert(ds4_ssd_auto_cache_plan(100 * gib, 86, 65 * gib, 10 * gib, gib, 8, &p));
    assert(p.cache_experts == 8 && p.effective_cache_bytes == 8 * gib);
    assert(ds4_ssd_auto_cache_plan(100 * gib, 86, 5 * gib, 10 * gib, gib, 1000, &p));
    assert(p.cache_experts == 1);
    assert(!ds4_ssd_auto_cache_plan(0, 86, 0, 0, gib, 0, &p));
    assert(!ds4_ssd_auto_cache_plan(gib, 86, 0, 0, 0, 0, &p));
    assert(!ds4_ssd_auto_cache_plan(gib, 100, 0, 0, gib, 0, &p));

    /* The budget is a split of one pool. Default floor 10 percent is a no-op
     * beside the default 80 percent engine share. */
    unsetenv("DS4_SSD_AUTO_CACHE_PCT");
    assert(ds4_ssd_auto_cache_plan(100 * gib, 80, 0, 10 * gib, gib, 1000, &p));
    assert(!p.clamped_by_pagecache_floor);
    assert(p.pool_bytes == 100 * gib);
    assert(p.engine_tier_pct == 80 && p.pagecache_floor_pct == 10);
    assert(p.pagecache_floor_bytes == 10 * gib);
    assert(p.model_target_bytes == 80 * gib);
    assert(p.host_tier_bytes == 10 * gib);

    /* Raising the engine tier past 100 minus the floor is refused, because
     * that is the direction the measurements call slower. */
    setenv("DS4_SSD_AUTO_CACHE_PCT", "95", 1);
    assert(ds4_ssd_auto_cache_plan(100 * gib, 86, 0, 10 * gib, gib, 1000, &p));
    assert(p.clamped_by_pagecache_floor);
    assert(p.model_target_bytes == 90 * gib);
    assert(p.pagecache_floor_bytes == 10 * gib);
    assert(p.host_tier_bytes == 0);
    assert(p.cache_experts == 80);
    unsetenv("DS4_SSD_AUTO_CACHE_PCT");

    /* A larger explicit floor takes more of the pool and the engine tier
     * gives way again. */
    setenv("DS4_SSD_PAGECACHE_FLOOR_PCT", "30", 1);
    assert(ds4_ssd_auto_cache_plan(100 * gib, 86, 0, 10 * gib, gib, 1000, &p));
    assert(p.clamped_by_pagecache_floor);
    assert(p.pagecache_floor_bytes == 30 * gib);
    assert(p.model_target_bytes == 70 * gib);
    assert(p.host_tier_bytes == 0);

    /* A lower floor opens room for the pinned host tier without moving the
     * engine tier the caller asked for. */
    setenv("DS4_SSD_PAGECACHE_FLOOR_PCT", "5", 1);
    assert(ds4_ssd_auto_cache_plan(100 * gib, 80, 0, 10 * gib, gib, 1000, &p));
    assert(!p.clamped_by_pagecache_floor);
    assert(p.pagecache_floor_bytes == 5 * gib);
    assert(p.model_target_bytes == 80 * gib);
    assert(p.host_tier_bytes == 15 * gib);

    /* Out-of-range floors fall back to the default rather than to nothing. */
    setenv("DS4_SSD_PAGECACHE_FLOOR_PCT", "0", 1);
    assert(ds4_ssd_auto_cache_plan(100 * gib, 80, 0, 10 * gib, gib, 1000, &p));
    assert(p.pagecache_floor_pct == 10 && p.pagecache_floor_bytes == 10 * gib);
    setenv("DS4_SSD_PAGECACHE_FLOOR_PCT", "50", 1);
    assert(ds4_ssd_auto_cache_plan(100 * gib, 80, 0, 10 * gib, gib, 1000, &p));
    assert(p.pagecache_floor_pct == 10 && p.pagecache_floor_bytes == 10 * gib);
    setenv("DS4_SSD_PAGECACHE_FLOOR_PCT", "junk", 1);
    assert(ds4_ssd_auto_cache_plan(100 * gib, 80, 0, 10 * gib, gib, 1000, &p));
    assert(p.pagecache_floor_pct == 10 && p.pagecache_floor_bytes == 10 * gib);

    /* Saturating arithmetic: a pool that cannot supply the floor must not
     * wrap the subtraction. */
    setenv("DS4_SSD_PAGECACHE_FLOOR_PCT", "10", 1);
    assert(ds4_ssd_auto_cache_plan(UINT64_MAX, 80, 0, 10 * gib, gib, 1000, &p));
    assert(p.model_target_bytes <= UINT64_MAX - p.pagecache_floor_bytes);

    unsetenv("DS4_SSD_AUTO_CACHE_PCT");
    unsetenv("DS4_SSD_PAGECACHE_FLOOR_PCT");
    puts("SSD cache sizing: PASS");
    return 0;
}
