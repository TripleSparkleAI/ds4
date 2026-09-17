/*
 * The host range cache, as a unit.
 *
 * The defect this file exists for: the table was keyed on (offset, bytes)
 * alone and was never dropped, so closing model A and opening model B in one
 * process let any expert whose file range collided read A's BYTES out of B.
 * Two quants of one architecture have the same expert byte layout, and two
 * engines in one test run are the common case, so the collision is not
 * exotic - and nothing reported it.
 *
 * check() does not abort, so every claim below reports independently:
 * assert() would show the first failure and hide the rest.
 */
#include <stdio.h>
#include <string.h>

#include "ds4_host_range_cache.h"

#define MIB (1024ull * 1024ull)

static int g_failures = 0;
static void check(int ok, const char *name) {
    if (!ok) {
        g_failures++;
        fprintf(stderr, "FAIL: %s\n", name);
    }
}

/* Reserve, fill with one repeated byte, commit. Mirrors what the demand path
 * does with the chunks it reads off the disk. */
static int put(ds4_host_range_cache *c, uint64_t off, uint64_t bytes, unsigned char fill) {
    uint32_t slot = DS4_HRC_NO_SLOT;
    void *buf = ds4_hrc_reserve(c, off, bytes, &slot);
    if (!buf) return 0;
    memset(buf, fill, (size_t)bytes);
    ds4_hrc_commit(c, slot, off, bytes, buf);
    return 1;
}

int main(void) {
    ds4_host_range_cache c;

    /* ------------------------------------------------------------------
     * THE MODEL SWAP. One process, two models, one colliding file range.
     * ------------------------------------------------------------------ */
    check(ds4_hrc_open(&c, DS4_HRC_LFU, 64 * MIB), "open: the table is built");

    /* Model A is open. Cache the expert living at this file range. */
    ds4_hrc_set_epoch(&c, 1);
    check(put(&c, 4 * MIB, 1 * MIB, 0xAA), "model A: the range is cached");

    /* Control, so that a miss below cannot be a cache that never worked:
     * within one model the same range IS returned. */
    {
        const unsigned char *hit = (const unsigned char *)ds4_hrc_get(&c, 4 * MIB, 1 * MIB);
        check(hit != NULL, "model A: the same range hits");
        check(hit && hit[0] == 0xAA, "model A: the hit carries A's bytes");
    }

    /* Model A is closed and model B is opened: ds4_gpu_set_model_fd bumps the
     * epoch. The SAME file range in B is a different expert.
     *
     * THE EPOCH IS IN THE KEY, and that is tested here WITHOUT the purge:
     * the epoch is moved by hand so no entry is dropped, which is the only
     * way to tell the two halves of the fix apart. With the purge running as
     * well, a table keyed on (offset, bytes) alone also misses - so a test
     * that only called set_epoch would pass over the original defect and
     * prove nothing about the key. */
    c.epoch = 2;
    {
        const void *hit = ds4_hrc_get(&c, 4 * MIB, 1 * MIB);
        check(hit == NULL,
              "model B CANNOT read model A's bytes at the same file range");
    }
    check(c.purges == 0, "that hand-moved epoch dropped nothing, by design");

    /* Now the real call, which also returns the RAM. */
    c.epoch = 1;
    ds4_hrc_set_epoch(&c, 2);

    /* And the RAM those entries held comes back rather than sitting
     * unreachable for the life of the process. */
    check(c.used == 0, "the model change returns the bytes it was holding");
    check(c.purges == 1, "the model change is counted");

    /* B may now cache its own bytes at that range, and gets its own. */
    check(put(&c, 4 * MIB, 1 * MIB, 0xBB), "model B: the range is cached");
    {
        const unsigned char *hit = (const unsigned char *)ds4_hrc_get(&c, 4 * MIB, 1 * MIB);
        check(hit && hit[0] == 0xBB, "model B: the hit carries B's bytes");
    }

    /* Going back to A's epoch number must not resurrect A's entries either -
     * the epoch is a counter precisely so a number cannot come round again,
     * but the purge makes the point independent of that. */
    ds4_hrc_set_epoch(&c, 1);
    check(ds4_hrc_get(&c, 4 * MIB, 1 * MIB) == NULL,
          "an epoch that changes at all drops what it left behind");
    ds4_hrc_close(&c);

    /* ------------------------------------------------------------------
     * RESERVE DECIDES FIRST, so a range that will not be cached costs no
     * allocation and no per-chunk copy.
     * ------------------------------------------------------------------ */

    /* Case: a range already resident is refused up front, so the caller
     * never fills a buffer it would have to throw away. */
    check(ds4_hrc_open(&c, DS4_HRC_LFU, 64 * MIB), "open: second table");
    ds4_hrc_set_epoch(&c, 7);
    check(put(&c, 0, 1 * MIB, 0x11), "resident: first insert lands");
    {
        uint32_t slot = DS4_HRC_NO_SLOT;
        void *buf = ds4_hrc_reserve(&c, 0, 1 * MIB, &slot);
        check(buf == NULL && slot == DS4_HRC_NO_SLOT,
              "an already-resident range reserves nothing");
    }

    /* Case: an entry larger than the cap is refused up front. */
    {
        uint32_t slot = DS4_HRC_NO_SLOT;
        void *buf = ds4_hrc_reserve(&c, 99 * MIB, DS4_HRC_MAX_ENTRY + 1, &slot);
        check(buf == NULL && slot == DS4_HRC_NO_SLOT,
              "an oversized range reserves nothing");
    }

    ds4_hrc_close(&c);

    /* Case: a reservation that is never committed is invisible. A
     * half-filled buffer must never be returned to a caller as weights.
     *
     * The observable is the HIT COUNTER, not the returned pointer: a slot
     * left valid with its buffer already freed returns NULL from a lookup
     * too, so a NULL check cannot tell the two apart and would be a control
     * that can never fail. The evicted range must MISS.
     *
     * Its own table, sized so ONE entry fills the budget: with room to spare
     * a reservation takes an empty slot and evicts nothing, and there is then
     * no victim to ask about. */
    {
        ds4_host_range_cache t;
        check(ds4_hrc_open(&t, DS4_HRC_LFU, 1 * MIB), "open: pressured table");
        ds4_hrc_set_epoch(&t, 5);
        check(put(&t, 0, 1 * MIB, 0x22), "pressured: the first range lands");
        const uint64_t hits_before = t.hits;
        const uint64_t misses_before = t.misses;
        uint32_t slot = DS4_HRC_NO_SLOT;
        void *buf = ds4_hrc_reserve(&t, 32 * MIB, 1 * MIB, &slot);
        check(buf != NULL, "pressured: the second range reserves a slot");
        check(t.evictions == 1, "that reservation evicted the resident range");
        (void)ds4_hrc_get(&t, 0, 1 * MIB);
        check(t.hits == hits_before,
              "the evicted range does not hit while its slot is reserved");
        check(t.misses == misses_before + 1,
              "the evicted range MISSES while its slot is reserved");
        check(ds4_hrc_get(&t, 32 * MIB, 1 * MIB) == NULL,
              "a reserved-but-uncommitted range does not hit");
        ds4_hrc_abandon(&t, slot, buf);
        check(ds4_hrc_get(&t, 32 * MIB, 1 * MIB) == NULL,
              "an abandoned reservation does not hit either");
        ds4_hrc_close(&t);
    }

    /* Case: evicting a same-size victim keeps its allocation instead of
     * freeing and re-mallocing one per miss. A tiny budget forces eviction
     * on every insert, so the buffer must come back. */
    {
        ds4_host_range_cache t;
        check(ds4_hrc_open(&t, DS4_HRC_LFU, 1 * MIB), "open: one-entry table");
        ds4_hrc_set_epoch(&t, 3);
        uint32_t s1 = DS4_HRC_NO_SLOT, s2 = DS4_HRC_NO_SLOT;
        void *b1 = ds4_hrc_reserve(&t, 0, 1 * MIB, &s1);
        check(b1 != NULL, "reuse: the first reservation lands");
        if (b1) { memset(b1, 1, (size_t)(1 * MIB)); ds4_hrc_commit(&t, s1, 0, 1 * MIB, b1); }
        void *b2 = ds4_hrc_reserve(&t, 64 * MIB, 1 * MIB, &s2);
        check(b2 != NULL, "reuse: the second reservation lands");
        check(t.evictions == 1, "reuse: the second reservation evicted one entry");
        /* Counted, not inferred from the pointer: free() followed by a
         * same-size malloc() very often hands back the same address, so a
         * pointer comparison here is a control that cannot fail. */
        check(t.allocs == 1, "a same-size victim costs NO second malloc");
        check(t.reuses == 1, "a same-size victim's buffer is reused");
        check(b1 == b2, "and it is the same buffer");
        if (b2) { memset(b2, 2, (size_t)(1 * MIB)); ds4_hrc_commit(&t, s2, 64 * MIB, 1 * MIB, b2); }
        check(t.used == 1 * MIB, "reuse: the accounting still adds up");
        ds4_hrc_close(&t);
    }

    /* ------------------------------------------------------------------
     * THE POLICY DEFAULT IS NOT MOVED BY ANY OF THE ABOVE.
     * WIKI/theory/177 measured hotness-with-decay WORSE than plain LRU on
     * our box and upstream's study says the opposite for theirs; that
     * disagreement is what this arm exists to measure, so both policies
     * stay and the mode string keeps choosing between them.
     * ------------------------------------------------------------------ */
    check(ds4_hrc_parse_mode(NULL) == DS4_HRC_OFF, "unset: the cache is off");
    check(ds4_hrc_parse_mode("hotlist") == DS4_HRC_OFF, "hotlist: the cache is off");
    check(ds4_hrc_parse_mode("offsetkey") == DS4_HRC_LFU, "offsetkey: LFU");
    check(ds4_hrc_parse_mode("offsetkey-lru") == DS4_HRC_LRU, "offsetkey-lru: LRU");

    if (g_failures != 0) {
        fprintf(stderr, "Host range cache: %d FAILED\n", g_failures);
        return 1;
    }
    puts("Host range cache: PASS");
    return 0;
}
