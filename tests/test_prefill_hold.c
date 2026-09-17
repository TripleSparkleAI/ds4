/* Unit tests for the prefill read-ahead held band (ds4_prefill_hold.h).
 *
 * Pure C99: no CUDA, no GPU, no model. Builds and runs on any host.
 *
 * The load-bearing case is HOISTED: a scan of many slots must derive the band
 * ONCE. That is asserted on the division counter, not on a clock, so it is a
 * test rather than a performance claim. */

#include "../ds4_prefill_hold.h"

#include <stdio.h>

static int g_failed = 0;
static int g_total  = 0;

#define CHECK(cond, msg) do {                                                  \
    g_total++;                                                                 \
    if (!(cond)) {                                                             \
        fprintf(stderr, "  FAIL: %s (line %d)\n", (msg), __LINE__);            \
        g_failed++;                                                            \
    }                                                                          \
} while (0)

#define RUN(fn) do {                                                           \
    fprintf(stderr, "RUN: %s\n", #fn);                                         \
    int _before = g_failed;                                                    \
    (fn)();                                                                    \
    fprintf(stderr, "  %s\n", (_before == g_failed) ? "ok" : "FAIL");          \
} while (0)

/* The shape the two victim scans have: 40 layers, 256 experts, a cache of a
 * few thousand slots. */
#define N_TOTAL_EXPERT 256u
#define GATE_BYTES     (uint64_t)196608
#define LOW_GATE       (uint64_t)1000000
#define N_SLOTS        (uint64_t)4096

static ds4_prefill_hold_band live_band(void) {
    return ds4_prefill_hold_band_make(1, 1, LOW_GATE, N_SLOTS,
                                      N_TOTAL_EXPERT, GATE_BYTES);
}

/* ---- the work counter: one derivation per scan, not one per slot ---- */

static void test_band_is_derived_once_per_scan(void) {
    ds4_prefill_hold_divisions = 0;
    const ds4_prefill_hold_band b = live_band();
    const uint64_t after_make = ds4_prefill_hold_divisions;
    uint64_t held = 0;
    for (uint64_t j = 0; j < N_SLOTS; j++)
        held += (uint64_t)ds4_prefill_hold_slot_held(&b, LOW_GATE + j * 4096u, 7u);
    fprintf(stderr, "  divisions: make=%llu total=%llu over %llu slots (held=%llu)\n",
            (unsigned long long)after_make,
            (unsigned long long)ds4_prefill_hold_divisions,
            (unsigned long long)N_SLOTS, (unsigned long long)held);
    CHECK(after_make == 3, "the derivation pays exactly three variable divisions");
    CHECK(ds4_prefill_hold_divisions == after_make,
          "a slot scan of N slots adds no division at all");
}

static void test_two_scans_pay_two_derivations(void) {
    ds4_prefill_hold_divisions = 0;
    for (int scan = 0; scan < 2; scan++) {
        const ds4_prefill_hold_band b = live_band();
        for (uint64_t j = 0; j < N_SLOTS; j++)
            (void)ds4_prefill_hold_slot_held(&b, LOW_GATE + j, 1u);
    }
    CHECK(ds4_prefill_hold_divisions == 6,
          "two scans pay six divisions, not six times N_SLOTS");
}

/* ---- the OFF form pays nothing ---- */

static void test_hold_off_is_free_and_holds_nothing(void) {
    ds4_prefill_hold_divisions = 0;
    const ds4_prefill_hold_band b =
        ds4_prefill_hold_band_make(0, 1, LOW_GATE, N_SLOTS, N_TOTAL_EXPERT, GATE_BYTES);
    CHECK(b.on == 0, "hold off leaves the band off");
    CHECK(ds4_prefill_hold_divisions == 0, "hold off pays no division");
    CHECK(!ds4_prefill_hold_slot_held(&b, LOW_GATE, 1u),
          "hold off holds no slot, so every slot is an ordinary LRU victim");
}

static void test_low_gate_unset_holds_nothing(void) {
    ds4_prefill_hold_divisions = 0;
    const ds4_prefill_hold_band b =
        ds4_prefill_hold_band_make(1, 0, 0, N_SLOTS, N_TOTAL_EXPERT, GATE_BYTES);
    CHECK(b.on == 0, "no low gate noted yet leaves the band off");
    CHECK(ds4_prefill_hold_divisions == 0, "an unset low gate pays no division");
}

/* ---- membership ---- */

static void test_slot_inside_the_band_is_held(void) {
    const ds4_prefill_hold_band b = live_band();
    CHECK(b.on == 1, "the band is on for a live table");
    CHECK(ds4_prefill_hold_slot_held(&b, LOW_GATE, 1u),
          "the lowest gate offset is inside the band");
    CHECK(ds4_prefill_hold_slot_held(&b, b.low_gate + b.band - 1u, 1u),
          "the last byte of the band is inside it");
}

static void test_slot_outside_the_band_is_not_held(void) {
    const ds4_prefill_hold_band b = live_band();
    CHECK(!ds4_prefill_hold_slot_held(&b, b.low_gate + b.band, 1u),
          "the first byte past the band is not held");
    CHECK(!ds4_prefill_hold_slot_held(&b, b.low_gate + b.band + 4096u, 1u),
          "a gate well past the band is not held");
    /* The rule is ONE-SIDED: `gate < low_gate + band`, with no lower bound.
     * The low gate is the minimum the read-ahead has seen, so nothing is
     * expected below it; this pins the rule as written rather than as
     * imagined, so a later lower bound cannot be added silently. */
    CHECK(ds4_prefill_hold_slot_held(&b, LOW_GATE - 1u, 1u),
          "the band has no lower bound: a gate below the low gate is held too");
}

static void test_an_unused_slot_is_never_held(void) {
    const ds4_prefill_hold_band b = live_band();
    CHECK(!ds4_prefill_hold_slot_held(&b, LOW_GATE, 0u),
          "a free slot is a victim whatever the band says");
}

/* ---- the band's own shape ---- */

static void test_hold_layers_floors_at_one(void) {
    /* A cache far smaller than one layer of experts: hold_layers would be 0. */
    const ds4_prefill_hold_band b =
        ds4_prefill_hold_band_make(1, 1, LOW_GATE, 8u, N_TOTAL_EXPERT, GATE_BYTES);
    CHECK(b.on == 1, "a tiny cache still has a band");
    CHECK(b.band == (uint64_t)N_TOTAL_EXPERT * GATE_BYTES,
          "hold_layers floors at one layer's gate bytes");
}

static void test_band_is_in_gate_bytes_not_layers(void) {
    /* 4096 slots / 4 / 256 experts = 4 gate-block lengths. The comment in the
     * header says so; this pins it. */
    const ds4_prefill_hold_band b = live_band();
    CHECK(b.band == 4u * (uint64_t)N_TOTAL_EXPERT * GATE_BYTES,
          "band = hold_layers * n_total_expert * gate_expert_bytes");
}

/* ---- the refusals ---- */

static void test_degenerate_table_holds_nothing(void) {
    ds4_prefill_hold_divisions = 0;
    const ds4_prefill_hold_band z1 =
        ds4_prefill_hold_band_make(1, 1, LOW_GATE, N_SLOTS, 0u, GATE_BYTES);
    const ds4_prefill_hold_band z2 =
        ds4_prefill_hold_band_make(1, 1, LOW_GATE, N_SLOTS, N_TOTAL_EXPERT, 0u);
    CHECK(z1.on == 0, "n_total_expert == 0 leaves the band off");
    CHECK(z2.on == 0, "gate_expert_bytes == 0 leaves the band off");
    CHECK(ds4_prefill_hold_divisions == 0, "a degenerate table divides by nothing");
}

static void test_overflow_guards_refuse(void) {
    const ds4_prefill_hold_band o1 =
        ds4_prefill_hold_band_make(1, 1, LOW_GATE, N_SLOTS, UINT64_MAX, 2u);
    CHECK(o1.on == 0, "n_total_expert * gate_expert_bytes overflow leaves it off");
    const ds4_prefill_hold_band o2 =
        ds4_prefill_hold_band_make(1, 1, UINT64_MAX - 1u, N_SLOTS,
                                   N_TOTAL_EXPERT, GATE_BYTES);
    CHECK(o2.on == 0, "low_gate + band overflow leaves it off");
}

int main(void) {
    RUN(test_band_is_derived_once_per_scan);
    RUN(test_two_scans_pay_two_derivations);
    RUN(test_hold_off_is_free_and_holds_nothing);
    RUN(test_low_gate_unset_holds_nothing);
    RUN(test_slot_inside_the_band_is_held);
    RUN(test_slot_outside_the_band_is_not_held);
    RUN(test_an_unused_slot_is_never_held);
    RUN(test_hold_layers_floors_at_one);
    RUN(test_band_is_in_gate_bytes_not_layers);
    RUN(test_degenerate_table_holds_nothing);
    RUN(test_overflow_guards_refuse);
    fprintf(stderr, "\n%d checks, %d failed\n", g_total, g_failed);
    return g_failed ? 1 : 0;
}
