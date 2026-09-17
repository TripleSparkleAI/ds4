#include "ds4_ssd.h"

#include <ctype.h>
#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#ifndef MAP_ANONYMOUS
#define MAP_ANONYMOUS MAP_ANON
#endif

static const uint64_t DS4_GIB = 1024ull * 1024ull * 1024ull;

bool ds4_parse_gib_arg(const char *s, uint64_t *bytes) {
    if (bytes) *bytes = 0;
    if (!s || !s[0] || !bytes) return false;

    size_t len = strlen(s);
    if (len > 2 &&
        (s[len - 2] == 'g' || s[len - 2] == 'G') &&
        (s[len - 1] == 'b' || s[len - 1] == 'B')) {
        len -= 2;
    }
    if (len == 0) return false;
    for (size_t i = 0; i < len; i++) {
        if (!isdigit((unsigned char)s[i])) return false;
    }

    char numbuf[32];
    if (len >= sizeof(numbuf)) return false;
    memcpy(numbuf, s, len);
    numbuf[len] = '\0';

    errno = 0;
    unsigned long long v = strtoull(numbuf, NULL, 10);
    if (errno != 0 || v == 0 || v > UINT64_MAX / DS4_GIB) return false;

    *bytes = (uint64_t)v * DS4_GIB;
    return true;
}

bool ds4_parse_streaming_cache_experts_arg(const char *s,
                                           uint32_t   *experts,
                                           uint64_t   *bytes) {
    if (experts) *experts = 0;
    if (bytes) *bytes = 0;
    if (!s || !s[0] || !experts || !bytes) return false;

    const size_t len = strlen(s);
    if (len > 2 &&
        (s[len - 2] == 'g' || s[len - 2] == 'G') &&
        (s[len - 1] == 'b' || s[len - 1] == 'B')) {
        return ds4_parse_gib_arg(s, bytes);
    }

    for (size_t i = 0; i < len; i++) {
        if (!isdigit((unsigned char)s[i])) return false;
    }

    errno = 0;
    unsigned long v = strtoul(s, NULL, 10);
    if (errno != 0 || v == 0 || v > UINT32_MAX) return false;

    *experts = (uint32_t)v;
    return true;
}

uint32_t ds4_ssd_cache_experts_for_byte_budget(uint64_t bytes,
                                               uint64_t per_expert_bytes) {
    if (bytes == 0 || per_expert_bytes == 0) return 0;
    const uint64_t experts = bytes / per_expert_bytes;
    if (experts == 0 || experts > UINT32_MAX) return 0;
    return (uint32_t)experts;
}

/*
 * The cache budget is a SPLIT of one pool of RAM across three consumers, not
 * a maximum to push.
 *
 * Three consumers draw on the same physical RAM: the engine (resident expert)
 * tier, the pinned host tier, and the kernel page cache that backs the
 * memory-mapped Engram tables. A bigger engine cache past a point is SLOWER
 * because it steals page cache from those tables:
 *   - our own knee result, WIKI/theory/48-decode-speedup-levers.md section 8:
 *     on the Spark the lever is hot-expert prediction, "NOT BIGGER CACHE (THE
 *     KNEE PROVED SIZE ISN'T IT)";
 *   - a separately measured negative result on another rig: raising its
 *     pinned host tier above 72 GiB on a 125.7 GiB box is slower for exactly
 *     that reason (research/v41-flash-landscape/08-rtx-cards/CODE.md:279-285).
 * So DS4_SSD_AUTO_CACHE_PCT is the engine tier's SHARE, the page cache gets an
 * explicit floor of its own (DS4_SSD_PAGECACHE_FLOOR_PCT), and whatever is
 * left is the pinned host tier plus the transient prefill budget. Raising the
 * engine share past 100 minus the floor is refused rather than granted,
 * because that is the direction the measurements say is slower.
 */
static uint64_t ds4_ssd_auto_cache_percent(uint32_t default_percent) {
    const char *env = getenv("DS4_SSD_AUTO_CACHE_PCT");
    if (env && env[0]) {
        errno = 0;
        char *end = NULL;
        unsigned long v = strtoul(env, &end, 10);
        if (end != env && *end == '\0' && errno == 0 &&
            v >= 50 && v <= 95) {
            return (uint64_t)v;
        }
        fprintf(stderr,
                "ds4: invalid DS4_SSD_AUTO_CACHE_PCT=%s (want 50..95); "
                "using default\n",
                env);
    }
    return default_percent;
}

/* The kernel page cache's own share of the pool. Default 10 percent, which is
 * a no-op beside the 80 percent engine share in use today and only binds when
 * someone raises DS4_SSD_AUTO_CACHE_PCT past 100 minus this. */
#define DS4_SSD_PAGECACHE_FLOOR_DEFAULT_PCT 10u
#define DS4_SSD_PAGECACHE_FLOOR_MIN_PCT 1u
#define DS4_SSD_PAGECACHE_FLOOR_MAX_PCT 40u

static uint64_t ds4_ssd_pagecache_floor_percent(void) {
    const char *env = getenv("DS4_SSD_PAGECACHE_FLOOR_PCT");
    if (env && env[0]) {
        errno = 0;
        char *end = NULL;
        unsigned long v = strtoul(env, &end, 10);
        if (end != env && *end == '\0' && errno == 0 &&
            v >= DS4_SSD_PAGECACHE_FLOOR_MIN_PCT &&
            v <= DS4_SSD_PAGECACHE_FLOOR_MAX_PCT) {
            return (uint64_t)v;
        }
        fprintf(stderr,
                "ds4: invalid DS4_SSD_PAGECACHE_FLOOR_PCT=%s (want %u..%u); "
                "using default %u\n",
                env, DS4_SSD_PAGECACHE_FLOOR_MIN_PCT,
                DS4_SSD_PAGECACHE_FLOOR_MAX_PCT,
                DS4_SSD_PAGECACHE_FLOOR_DEFAULT_PCT);
    }
    return DS4_SSD_PAGECACHE_FLOOR_DEFAULT_PCT;
}

bool ds4_ssd_auto_cache_plan(uint64_t            recommended_bytes,
                             uint32_t            default_percent,
                             uint64_t            model_limit_bytes,
                             uint64_t            non_routed_bytes,
                             uint64_t            per_expert_bytes,
                             uint64_t            max_model_experts,
                             ds4_ssd_cache_plan *out) {
    if (!out) return false;
    memset(out, 0, sizeof(*out));
    if (recommended_bytes == 0 || per_expert_bytes == 0 ||
        default_percent < 50 || default_percent > 95) return false;

    const uint64_t pct = ds4_ssd_auto_cache_percent(default_percent);
    const uint64_t floor_pct = ds4_ssd_pagecache_floor_percent();

    /* The engine tier's share of the pool... */
    uint64_t target_bytes =
        recommended_bytes > UINT64_MAX / pct ?
            UINT64_MAX : (recommended_bytes * pct) / 100ull;

    /* ...beside the page-cache floor, which is held apart rather than
     * competed for. When the two would oversubscribe the pool, the engine tier
     * gives way: this is the direction both measurements call slower. */
    const uint64_t floor_bytes =
        recommended_bytes > UINT64_MAX / floor_pct ?
            UINT64_MAX : (recommended_bytes * floor_pct) / 100ull;
    const uint64_t engine_cap = recommended_bytes > floor_bytes ?
        recommended_bytes - floor_bytes : 0;
    if (target_bytes > engine_cap) {
        out->clamped_by_pagecache_floor = true;
        fprintf(stderr,
                "ds4: SSD streaming cache split: engine tier %llu%% plus "
                "page-cache floor %llu%% would oversubscribe the pool; "
                "engine tier capped to %llu%% (%.2f GiB of %.2f GiB)\n",
                (unsigned long long)pct,
                (unsigned long long)floor_pct,
                (unsigned long long)(100ull > floor_pct ? 100ull - floor_pct : 0ull),
                (double)engine_cap / (1024.0 * 1024.0 * 1024.0),
                (double)recommended_bytes / (1024.0 * 1024.0 * 1024.0));
        target_bytes = engine_cap;
    }
    out->pool_bytes = recommended_bytes;
    out->engine_tier_pct = (uint32_t)pct;
    out->pagecache_floor_pct = (uint32_t)floor_pct;
    out->pagecache_floor_bytes = floor_bytes;
    out->host_tier_bytes = target_bytes + floor_bytes < recommended_bytes ?
        recommended_bytes - target_bytes - floor_bytes : 0;
    if (out->host_tier_bytes == 0) {
        fprintf(stderr,
                "ds4: SSD streaming cache split leaves the pinned host tier "
                "no room (engine tier %llu%% + page-cache floor %llu%% of the "
                "pool); the next streaming spike has nowhere to go\n",
                (unsigned long long)pct,
                (unsigned long long)floor_pct);
    }

    out->model_target_bytes = target_bytes;
    if (model_limit_bytes != 0 && out->model_target_bytes > model_limit_bytes)
        out->model_target_bytes = model_limit_bytes;
    if (out->model_target_bytes > non_routed_bytes) {
        out->cache_bytes = out->model_target_bytes - non_routed_bytes;
    }

    uint64_t cache_experts = out->cache_bytes / per_expert_bytes;
    if (cache_experts == 0) cache_experts = 1;
    if (max_model_experts != 0 && cache_experts > max_model_experts) {
        cache_experts = max_model_experts;
    }
    if (cache_experts > UINT32_MAX) cache_experts = UINT32_MAX;

    out->cache_experts = (uint32_t)cache_experts;
    out->effective_cache_bytes = cache_experts * per_expert_bytes;
    return out->cache_experts != 0;
}

bool ds4_ssd_memory_lock_acquire(ds4_ssd_memory_lock *lock,
                                 uint64_t             bytes) {
    if (!lock) return false;
    lock->ptr = NULL;
    lock->bytes = 0;
    if (bytes == 0) return true;
    if (bytes > (uint64_t)SIZE_MAX) {
        fprintf(stderr,
                "ds4: --simulate-used-memory is too large for this process\n");
        return false;
    }

    void *ptr = mmap(NULL,
                     (size_t)bytes,
                     PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS,
                     -1,
                     0);
    if (ptr == MAP_FAILED) {
        fprintf(stderr,
                "ds4: --simulate-used-memory mmap %.2f GiB failed: %s\n",
                (double)bytes / (double)DS4_GIB,
                strerror(errno));
        return false;
    }

    const long page_long = sysconf(_SC_PAGESIZE);
    const uint64_t page = page_long > 0 ? (uint64_t)page_long : 4096ull;
    const uint64_t chunk_bytes = 256ull * 1024ull * 1024ull;
    volatile unsigned char *p = (volatile unsigned char *)ptr;

    /*
     * Touch and lock in bounded chunks.  A single very large mlock() is harder
     * to diagnose when it fails and can create long uninterruptible VM work on
     * macOS; chunking mirrors the standalone diagnostic utility.
     */
    uint64_t locked = 0;
    for (uint64_t off = 0; off < bytes; off += chunk_bytes) {
        uint64_t len = bytes - off;
        if (len > chunk_bytes) len = chunk_bytes;

        for (uint64_t pos = off; pos < off + len; pos += page) {
            p[pos] = (unsigned char)(pos / page);
        }
        if (len != 0) p[off + len - 1u] = 1;

        if (mlock((void *)(p + off), (size_t)len) != 0) {
            fprintf(stderr,
                    "ds4: --simulate-used-memory mlock failed after %.2f/%.2f GiB: %s\n",
                    (double)locked / (double)DS4_GIB,
                    (double)bytes / (double)DS4_GIB,
                    strerror(errno));
            if (locked != 0) munlock(ptr, (size_t)locked);
            munmap(ptr, (size_t)bytes);
            return false;
        }
        locked += len;
    }

    lock->ptr = ptr;
    lock->bytes = bytes;
    fprintf(stderr,
            "ds4: simulated used memory: locked %.2f GiB before model load\n",
            (double)bytes / (double)DS4_GIB);
    return true;
}

void ds4_ssd_memory_lock_release(ds4_ssd_memory_lock *lock) {
    if (!lock || !lock->ptr || lock->bytes == 0) return;
    munlock(lock->ptr, (size_t)lock->bytes);
    munmap(lock->ptr, (size_t)lock->bytes);
    lock->ptr = NULL;
    lock->bytes = 0;
}
