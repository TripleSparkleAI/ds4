/* Unit tests for the hits-first decision logic (ds4_hitsfirst_logic.h).
 *
 * Pure C99: no CUDA, no GPU, no model. Builds and runs on any host.
 *
 * Three subjects:
 *   - who drains a pending hits-first batch, including the Q4_K and MXFP4
 *     paths, which the older in-place predicate did not ask about;
 *   - the per-layer miss set as a mask instead of a heap allocation;
 *   - the negative-slot rule the split down kernel must share with the fused
 *     one, checked by running both reductions over the same selection.
 */

/* Substitute a counting reader before the header picks up getenv. */
static unsigned long g_env_reads = 0;
static const char *counting_getenv(const char *name) {
    (void)name;
    g_env_reads++;
    return (const char *)0;
}
#define DS4_ENV_GATE_GETENV counting_getenv

#include "../ds4_hitsfirst_logic.h"

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

/* Path shorthand: (q4k, mxfp4). IQ2 is neither. */
#define IQ2   0, 0
#define Q4K   1, 0
#define MXFP4 0, 1

/* ---- 1. who drains ------------------------------------------------------ */

/* THE BUG. A Q4_K-expert model takes `if (q4k_path)` in the one-token launch
 * chain and never reaches the hits-first split, so nothing consumes the batch;
 * the drain must happen at the top of the launch instead. */
static void test_q4k_one_token_must_drain(void) {
    CHECK(ds4_hitsfirst_must_wait(1, 1u, 1, Q4K),
          "Q4_K one-token LUT-gate launch must drain the pending batch");
    CHECK(!ds4_hitsfirst_split_consumes(1u, 1, Q4K),
          "the hits-first split does not run on the Q4_K path");
}

/* The MXFP4 block returns from routed_moe_launch before the old wait line was
 * reached at all, so it needs its own drain and the predicate must say so. */
static void test_mxfp4_one_token_must_drain(void) {
    CHECK(ds4_hitsfirst_must_wait(1, 1u, 1, MXFP4),
          "MXFP4 one-token launch must drain the pending batch");
    CHECK(!ds4_hitsfirst_split_consumes(1u, 1, MXFP4),
          "the hits-first split does not run on the MXFP4 path");
}

/* The one path that genuinely consumes its own load: keeping the overlap is
 * the whole point of the branch, so this case is the control that the fix did
 * not simply drain everything. */
static void test_iq2_one_token_lut_gate_does_not_drain(void) {
    CHECK(!ds4_hitsfirst_must_wait(1, 1u, 1, IQ2),
          "the IQ2 one-token decode-LUT split consumes its own batch");
    CHECK(ds4_hitsfirst_split_consumes(1u, 1, IQ2),
          "the split runs on the IQ2 one-token decode-LUT path");
}

static void test_decode_lut_gate_off_must_drain(void) {
    /* DS4_CUDA_MOE_NO_DECODE_LUT_GATE set: the split's launch site is skipped. */
    CHECK(ds4_hitsfirst_must_wait(1, 1u, 0, IQ2),
          "with the decode LUT gate off, nothing consumes the batch");
}

static void test_multi_token_must_drain(void) {
    CHECK(ds4_hitsfirst_must_wait(1, 2u, 1, IQ2),
          "a multi-token launch reads all six experts at once and must drain");
    CHECK(ds4_hitsfirst_must_wait(1, 128u, 1, IQ2),
          "a prefill-width launch must drain");
}

static void test_no_batch_means_no_wait(void) {
    CHECK(!ds4_hitsfirst_must_wait(0, 2u, 1, Q4K),
          "with no hits-first batch active there is nothing to wait for");
    CHECK(!ds4_hitsfirst_must_wait(0, 1u, 1, IQ2),
          "inactive is inactive whatever the path");
}

/* The whole space, so a later edit cannot quietly reopen one corner: every
 * (n_tokens, lut_gate, quant) combination that is not the IQ2 one-token split
 * must drain. */
static void test_every_reading_path_drains(void) {
    int missed = 0, kept = 0;
    const uint32_t tok[] = { 1u, 2u, 8u, 512u };
    for (int t = 0; t < 4; t++)
        for (int lut = 0; lut < 2; lut++)
            for (int q = 0; q < 2; q++)
                for (int m = 0; m < 2; m++) {
                    if (q && m) continue;   /* a table is not both quants */
                    const int wait = ds4_hitsfirst_must_wait(1, tok[t], lut, q, m);
                    const int split = ds4_hitsfirst_split_consumes(tok[t], lut, q, m);
                    if (split) kept++;
                    else if (!wait) missed++;
                }
    fprintf(stderr, "  paths: %d consume their own batch, %d read without draining\n",
            kept, missed);
    CHECK(missed == 0, "no launch path reads a slot without draining or consuming");
    CHECK(kept == 1, "exactly one path consumes its own batch");
}

/* ---- 2. what missed ----------------------------------------------------- */

static void test_miss_mask_carries_the_batch(void) {
    uint64_t mask = 0;
    ds4_hitsfirst_miss_set(&mask, 0);
    ds4_hitsfirst_miss_set(&mask, 3);
    ds4_hitsfirst_miss_set(&mask, 7);
    CHECK(ds4_hitsfirst_miss_get(mask, 0), "slot 0 recorded");
    CHECK(ds4_hitsfirst_miss_get(mask, 3), "slot 3 recorded");
    CHECK(ds4_hitsfirst_miss_get(mask, 7), "slot 7 recorded");
    CHECK(!ds4_hitsfirst_miss_get(mask, 1), "slot 1 was a hit");
    CHECK(!ds4_hitsfirst_miss_get(mask, 6), "slot 6 was a hit");
    CHECK(mask == ((uint64_t)1 | ((uint64_t)1 << 3) | ((uint64_t)1 << 7)),
          "the mask holds those three bits and nothing else");
}

static void test_miss_mask_refuses_out_of_range(void) {
    uint64_t mask = 0;
    ds4_hitsfirst_miss_set(&mask, DS4_HITSFIRST_MISS_MAX);
    ds4_hitsfirst_miss_set(&mask, 1000);
    CHECK(mask == 0, "an index past the mask width writes nothing, never UB");
    CHECK(!ds4_hitsfirst_miss_get((uint64_t)-1, DS4_HITSFIRST_MISS_MAX),
          "an index past the mask width reads false");
}

static void test_miss_mask_covers_the_hits_first_cap(void) {
    /* begin_load starts hits-first only for slot_count <= 8, and the unique
     * count is at most the slot count, so 8 bits is the live requirement. */
    uint64_t mask = 0;
    for (size_t i = 0; i < 8; i++) ds4_hitsfirst_miss_set(&mask, i);
    for (size_t i = 0; i < 8; i++)
        CHECK(ds4_hitsfirst_miss_get(mask, i), "every slot in a full 8-slot batch");
}

/* ---- 3. which slots contribute ------------------------------------------ */

/* Host models of the two reductions, differing only in how they read a slot,
 * so the negative-slot rule is exercised rather than asserted. */
#define SLOTS   6
#define OUT_DIM 4

static float fake_dot(int expert_i, int row) {
    return (float)((expert_i + 1) * 100 + row);
}

static void fused_down(const int32_t *selected, float *out) {
    for (int row = 0; row < OUT_DIM; row++) {
        float total = 0.0f;
        for (int slot = 0; slot < SLOTS; slot++) {
            const int32_t expert_i = selected[slot];
            if (!ds4_hitsfirst_slot_contributes(expert_i)) continue;
            total += fake_dot(expert_i, row);
        }
        out[row] = total;
    }
}

static void split_down(const int32_t *selected, float *out) {
    float partial[SLOTS][OUT_DIM];
    for (int slot = 0; slot < SLOTS; slot++) {
        int32_t expert_i = selected[slot];
        for (int row = 0; row < OUT_DIM; row++) {
            if (!ds4_hitsfirst_slot_contributes(expert_i)) { partial[slot][row] = 0.0f; continue; }
            partial[slot][row] = fake_dot(expert_i, row);
        }
    }
    for (int row = 0; row < OUT_DIM; row++) {
        float total = 0.0f;
        for (int slot = 0; slot < SLOTS; slot++) total += partial[slot][row];
        out[row] = total;
    }
}

static void test_split_matches_fused_with_a_negative_slot(void) {
    const int32_t selected[SLOTS] = { 5, -1, 17, 3, -1, 200 };
    float a[OUT_DIM], b[OUT_DIM];
    fused_down(selected, a);
    split_down(selected, b);
    for (int row = 0; row < OUT_DIM; row++)
        CHECK(a[row] == b[row],
              "the split down sum equals the fused one when a slot is unowned");
    fprintf(stderr, "  fused[0]=%.1f split[0]=%.1f\n", (double)a[0], (double)b[0]);
}

static void test_split_matches_fused_with_every_slot_owned(void) {
    const int32_t selected[SLOTS] = { 0, 1, 2, 3, 4, 5 };
    float a[OUT_DIM], b[OUT_DIM];
    fused_down(selected, a);
    split_down(selected, b);
    for (int row = 0; row < OUT_DIM; row++)
        CHECK(a[row] == b[row], "the two agree on the ordinary all-owned case");
}

static void test_expert_zero_is_not_unowned(void) {
    /* The control for the old rule, which mapped a negative slot onto expert
     * 0: expert 0 is a real expert and must contribute. */
    CHECK(ds4_hitsfirst_slot_contributes(0), "expert 0 contributes");
    CHECK(!ds4_hitsfirst_slot_contributes(-1), "an unowned slot does not");
    CHECK(!ds4_hitsfirst_slot_contributes(-2147483647 - 1), "INT32_MIN does not");
}

/* ---- the env switch ----------------------------------------------------- */

static void test_env_switch_is_read_once(void) {
    ds4_env_gate gate = DS4_ENV_GATE_INIT;
    g_env_reads = 0;
    for (int i = 0; i < 4000; i++) (void)ds4_env_gate_on(&gate, "DS4_CUDA_EXPERT_CACHE_STATS");
    fprintf(stderr, "  env reads: %lu over 4000 calls\n", g_env_reads);
    CHECK(g_env_reads == 1, "the switch is read once, not once per MoE layer");
    CHECK(gate.reads == 1, "the gate records the single read");
}

static void test_env_switch_answers_consistently(void) {
    ds4_env_gate gate = DS4_ENV_GATE_INIT;
    const int first = ds4_env_gate_on(&gate, "DS4_CUDA_EXPERT_CACHE_STATS");
    for (int i = 0; i < 100; i++)
        CHECK(ds4_env_gate_on(&gate, "DS4_CUDA_EXPERT_CACHE_STATS") == first,
              "every later call answers what the first one did");
}

int main(void) {
    RUN(test_q4k_one_token_must_drain);
    RUN(test_mxfp4_one_token_must_drain);
    RUN(test_iq2_one_token_lut_gate_does_not_drain);
    RUN(test_decode_lut_gate_off_must_drain);
    RUN(test_multi_token_must_drain);
    RUN(test_no_batch_means_no_wait);
    RUN(test_every_reading_path_drains);
    RUN(test_miss_mask_carries_the_batch);
    RUN(test_miss_mask_refuses_out_of_range);
    RUN(test_miss_mask_covers_the_hits_first_cap);
    RUN(test_split_matches_fused_with_a_negative_slot);
    RUN(test_split_matches_fused_with_every_slot_owned);
    RUN(test_expert_zero_is_not_unowned);
    RUN(test_env_switch_is_read_once);
    RUN(test_env_switch_answers_consistently);
    fprintf(stderr, "\n%d checks, %d failed\n", g_total, g_failed);
    return g_failed ? 1 : 0;
}
