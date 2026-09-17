/* SCOUT decision logic, as pure arithmetic.
 *
 * The scout goes ahead of the token. At token t, layer L, when the router's
 * expert ids are known and the misses are issued, the scout ALSO issues reads
 * for the experts token t-1 used at layer L+1 that are not resident. MoE
 * routing is sticky token to token, so the next layer's misses are guessable
 * from the last token. A wrong guess costs one read the cache may keep; a
 * right guess means layer L+1's bytes are already in flight, on the same pool
 * batch as layer L's misses, when L+1's router runs. It moves READS only,
 * never math: greedy output stays byte-identical.
 *
 * Three rules live here because they are easy to get wrong in place and
 * impossible to test from a .cu on a host with no CUDA toolkit:
 *
 *  1. REMEMBER. One small record per layer: the previous token's routed ids
 *     (top-6 of 256, capped at DS4_SCOUT_MAX_IDS). A wider batch (prefill)
 *     is not a token and is not remembered.
 *
 *  2. SELECT. The guess for L+1 is the remembered set minus the resident set,
 *     deduplicated, range-checked, capped by the number of victim slots the
 *     ledger can give away. Zero victims means the cache is full and the
 *     scout does nothing.
 *
 *  3. SCORE. A remembered id that the next layer actually routes to is a hit;
 *     hits over guesses is the rate a round reads off the stats line.
 *
 * Pure C99. No CUDA types. Builds on any host, which is what lets
 * tests/test_scout_logic.c exercise it without a GPU.
 */
#ifndef DS4_SCOUT_LOGIC_H
#define DS4_SCOUT_LOGIC_H

#include <stdint.h>
#include <stddef.h>

#define DS4_SCOUT_MAX_LAYERS 128u
#define DS4_SCOUT_MAX_IDS    8u     /* hits-first's own cap: slot_count <= 8 */

typedef struct {
    uint32_t n;                        /* 0 = nothing remembered for this layer */
    int32_t  id[DS4_SCOUT_MAX_IDS];
} ds4_scout_layer_memory;

/* ---- 1. remember -------------------------------------------------------- */

/* Returns 1 when the ids were recorded, 0 when the batch is too wide to be a
 * single token (nothing changes: the old memory stands). */
static inline int ds4_scout_remember(ds4_scout_layer_memory *mem,
                                     const int32_t *ids, uint32_t n) {
    if (!mem || !ids || n == 0 || n > DS4_SCOUT_MAX_IDS) return 0;
    mem->n = n;
    for (uint32_t i = 0; i < n; i++) mem->id[i] = ids[i];
    return 1;
}

/* ---- 2. select ---------------------------------------------------------- */

/* remembered[0..n_remembered) with resident[i] != 0 marking the ids already in
 * the cache. Writes the ids to read into out[0..cap) and returns how many.
 * No-op (returns 0) when victims_available == 0: the cache has no slot the
 * scout may take, and a scout must never win a slot over a demand read. */
static inline uint32_t ds4_scout_select(const int32_t *remembered,
                                        uint32_t n_remembered,
                                        const uint8_t *resident,
                                        uint32_t n_total_expert,
                                        uint32_t victims_available,
                                        uint32_t cap,
                                        int32_t *out) {
    uint32_t n = 0;
    if (!remembered || !resident || !out || !victims_available) return 0;
    if (cap > victims_available) cap = victims_available;
    for (uint32_t i = 0; i < n_remembered && n < cap; i++) {
        const int32_t id = remembered[i];
        if (id < 0 || (uint32_t)id >= n_total_expert) continue;
        if (resident[i]) continue;
        int dup = 0;
        for (uint32_t j = 0; j < n; j++) if (out[j] == id) { dup = 1; break; }
        if (dup) continue;
        out[n++] = id;
    }
    return n;
}

/* ---- 3. score ----------------------------------------------------------- */

/* hits += distinct remembered ids that appear in selected; guessed += the
 * distinct remembered count. Duplicates in either list count once. */
static inline void ds4_scout_score(const int32_t *guess, uint32_t n_guess,
                                   const int32_t *selected, uint32_t n_selected,
                                   uint64_t *hits, uint64_t *guessed) {
    if (!guess || !selected || !hits || !guessed) return;
    for (uint32_t i = 0; i < n_guess; i++) {
        int seen = 0;
        for (uint32_t j = 0; j < i; j++) if (guess[j] == guess[i]) { seen = 1; break; }
        if (seen) continue;
        (*guessed)++;
        for (uint32_t k = 0; k < n_selected; k++)
            if (selected[k] == guess[i]) { (*hits)++; break; }
    }
}

#endif /* DS4_SCOUT_LOGIC_H */
