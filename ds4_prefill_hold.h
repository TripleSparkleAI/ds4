/* The prefill read-ahead HELD BAND, as pure arithmetic.
 *
 * `cuda_stream_selected_cache_begin_load` and
 * `ds4_gpu_stream_expert_cache_prefetch_begin` each scan every slot in
 * `g_stream_expert_slots` and ask, per slot, whether that slot is inside the
 * band the prefill read-ahead holds back. The band itself depends only on the
 * table and the cache size, never on the slot, so it is derived ONCE per scan
 * here and the per-slot question is a compare.
 *
 * `ds4_prefill_hold_divisions` counts the variable-divisor 64-bit divisions the
 * derivation performs, so a test can assert that a scan of N slots pays them
 * once rather than N times. It is a work counter, not a timer.
 *
 * Pure C99 plus nothing: no CUDA, no Metal. Builds on any host, which is what
 * lets tests/test_prefill_hold.c exercise it without a GPU.
 */
#ifndef DS4_PREFILL_HOLD_H
#define DS4_PREFILL_HOLD_H

#include <stdint.h>

/* Counts genuine 64-bit divisions by a runtime value. Read by the tests. */
static uint64_t ds4_prefill_hold_divisions = 0;

typedef struct {
    int      on;        /* the band applies at all */
    uint64_t low_gate;  /* the lowest gate offset the read-ahead has seen */
    uint64_t band;      /* width of the held band, in GATE-offset bytes */
} ds4_prefill_hold_band;

/* Derive the band for one whole scan. Every division in the hold rule lives
 * here and nowhere else.
 *
 * `band` is `hold_layers` GATE-block lengths. Consecutive layers' gate blocks
 * are separated in the file by that layer's up, down and attention tensors, so
 * a band of `hold_layers` gate-block lengths spans FEWER than `hold_layers`
 * layers - at least layer 0's, whose gate block starts at the low gate. The
 * band is a quarter of the cache expressed in gate bytes, not a quarter of the
 * cache expressed in layers. A chosen constant, not a tuned one. */
static ds4_prefill_hold_band ds4_prefill_hold_band_make(
        int hold_on, int low_gate_set, uint64_t low_gate,
        uint64_t n_slots, uint64_t n_total_expert, uint64_t gate_expert_bytes) {
    ds4_prefill_hold_band b;
    b.on = 0;
    b.low_gate = low_gate;
    b.band = 0;
    if (!hold_on || !low_gate_set) return b;
    if (!gate_expert_bytes || !n_total_expert) return b;
    ds4_prefill_hold_divisions++;
    if (n_total_expert > UINT64_MAX / gate_expert_bytes) return b;
    const uint64_t layer_bytes = n_total_expert * gate_expert_bytes;
    ds4_prefill_hold_divisions++;
    uint64_t hold_layers = n_slots / 4u / n_total_expert;
    if (!hold_layers) hold_layers = 1;
    ds4_prefill_hold_divisions++;
    if (hold_layers > UINT64_MAX / layer_bytes) return b;
    const uint64_t band = hold_layers * layer_bytes;
    if (low_gate > UINT64_MAX - band) return b;
    b.on = 1;
    b.band = band;
    return b;
}

/* The per-slot question: one compare, no arithmetic on the table. */
static int ds4_prefill_hold_slot_held(const ds4_prefill_hold_band *b,
                                      uint64_t slot_gate, uint64_t slot_used) {
    if (!b->on || !slot_used) return 0;
    return slot_gate < b->low_gate + b->band;
}

#endif /* DS4_PREFILL_HOLD_H */
