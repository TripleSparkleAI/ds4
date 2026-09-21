/* ds4_prefill_sched.h - the optimal expert fetch schedule for layer-major prefill.
 *
 * Prefill is layer-major: every prompt token passes through layer 0, then layer 1, and so on
 * (see prefill_layer_major_cpu in ds4.c).  That ordering has a property decode does not have:
 * when layer L-1 has finished for every token, the router for layer L can be run for every token
 * at once, and the set of experts layer L will consult is then EXACTLY KNOWN before a single
 * expert byte is read.  The router is a small per-layer F32 projection, so learning the demand
 * costs almost nothing next to the expert weights it saves.
 *
 * With the demand known, choosing what to read and what to drop stops being a guess and becomes
 * arithmetic.  This file computes the optimal schedule directly:
 *
 *   EVICT  by Belady MIN - drop the resident expert whose next use is furthest away, or never.
 *   FETCH  just in time  - issue each read in the order its layer will consume it, no earlier
 *                          than the lead window allows.
 *
 * Belady MIN is optimal for a fixed capacity under a known request sequence, so the fetch count
 * this returns is a floor: no online policy can do better on the same demand.  That makes it
 * useful two ways - as a schedule to run, and as a ruler that says how much of the gap any
 * cheaper policy leaves on the table.
 *
 * Scope, deliberately narrow: DeepSeek V4.1 Flash, prefill only, expert weights only.  Nothing
 * here reads a model, allocates, or touches the decode path.  It is a pure function over a demand
 * table, which is why it is testable without a checkpoint.
 */
#ifndef DS4_PREFILL_SCHED_H
#define DS4_PREFILL_SCHED_H

#include <stdint.h>

#define DS4_PREFILL_ACT_FETCH 0
#define DS4_PREFILL_ACT_EVICT 1

/* One scheduled act.  `at_layer` is the layer during whose compute the act is issued; `for_layer`
 * is the layer that consumes it.  For a fetch with a lead window of one, at_layer = for_layer - 1. */
typedef struct {
    uint32_t at_layer;
    uint32_t for_layer;
    uint16_t expert;
    uint8_t  act;
} ds4_prefill_act;

/* The demand, as layer-major prefill can read it off the router.  `experts` holds every layer's
 * expert list concatenated; `layer_off` has n_layer + 1 entries, so layer L occupies
 * experts[layer_off[L] .. layer_off[L+1]).  A layer may repeat an expert; duplicates are demand,
 * not an error, and are counted once for residency. */
typedef struct {
    const uint16_t *experts;
    const uint32_t *layer_off;
    uint32_t        n_layer;
    uint32_t        n_expert;
    uint32_t        slots;        /* capacity, in experts */
    uint32_t        lead_layers;  /* how many layers ahead a fetch may be issued; 1 is the
                                   * layer-major window, 0 means issue at the consuming layer */
} ds4_prefill_demand;

typedef struct {
    uint64_t needed;     /* distinct (layer, expert) demands */
    uint64_t hits;       /* demands already resident when the layer ran */
    uint64_t fetches;    /* reads issued */
    uint64_t evictions;
    uint64_t late;       /* reads that could not be issued inside their lead window */
} ds4_prefill_sched_stats;

/* Compute the optimal schedule.  Returns 0 on success, -1 on a malformed demand, and -2 when
 * `acts_cap` is too small (in which case *acts_len holds the number of acts required).  `acts` may
 * be NULL with acts_cap 0 to size the buffer or to collect statistics only. */
int ds4_prefill_schedule(const ds4_prefill_demand *d,
                         ds4_prefill_act          *acts,
                         uint32_t                  acts_cap,
                         uint32_t                 *acts_len,
                         ds4_prefill_sched_stats  *st);

/* The same walk with LRU eviction instead of Belady, for the comparison the schedule exists to
 * win.  Identical signature and identical demand handling, so a difference in `fetches` is a
 * difference in the eviction rule and nothing else. */
int ds4_prefill_schedule_lru(const ds4_prefill_demand *d,
                             ds4_prefill_act          *acts,
                             uint32_t                  acts_cap,
                             uint32_t                 *acts_len,
                             ds4_prefill_sched_stats  *st);

#endif /* DS4_PREFILL_SCHED_H */
