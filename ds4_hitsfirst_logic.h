/* HITS-FIRST decision logic, as pure arithmetic.
 *
 * Three rules live here because all three are easy to get wrong in place and
 * impossible to test from a .cu on a host with no CUDA toolkit:
 *
 *  1. WHO DRAINS. `cuda_stream_selected_cache_begin_load` may leave a hits-first
 *     batch in flight on the pool's own non-blocking streams. Exactly one
 *     launch path consumes that batch itself (the one-token decode-LUT gate/up
 *     split); every other path must drain it before any kernel reads a slot.
 *     The predicate below says which is which, and it depends on the expert
 *     QUANT, which the older in-place form did not.
 *
 *  2. WHAT MISSED. The per-layer miss set used to be a heap-allocated vector
 *     sized by the unique-expert count, constructed on every call including the
 *     all-hit path. It is only ever read inside the hits-first block, which
 *     requires slot_count <= 8, so a 64-bit mask carries it with no allocation.
 *
 *  3. WHICH SLOTS CONTRIBUTE. The fused down kernel skips a negative slot; the
 *     split partial kernel must agree. One rule, called by both, so they cannot
 *     drift apart again.
 *
 * Pure C99, plus a __host__ __device__ marker for rule 3 so the kernels can use
 * it. No CUDA types. Builds on any host, which is what lets
 * tests/test_hitsfirst_logic.c exercise it without a GPU.
 */
#ifndef DS4_HITSFIRST_LOGIC_H
#define DS4_HITSFIRST_LOGIC_H

#include <stdint.h>
#include <stddef.h>

#ifdef __CUDACC__
#define DS4_HF_FN static __host__ __device__ inline
#else
#define DS4_HF_FN static inline
#endif

/* ---- 1. who drains ------------------------------------------------------ */

/* The hits-first split consumes its own pending load. It runs only on the
 * one-token decode-LUT gate/up path, which is reached in routed_moe_launch's
 * `else if (ok)` chain AFTER `if (q4k_path)` and only for the IQ2 experts - the
 * MXFP4 block returns from the function long before the wait line is reached at
 * all, and the Q4_K branch launches its own gate/up kernels instead.
 *
 * So the quant is part of the question. The older form asked only
 * `n_tokens == 1 && use_decode_lut_gate`, which is true for a Q4_K-expert model
 * whose gate/up kernels then read victim slots the pool is still uploading. */
DS4_HF_FN int ds4_hitsfirst_split_consumes(uint32_t n_tokens,
                                           int use_decode_lut_gate,
                                           int q4k_path,
                                           int mxfp4_path) {
    return n_tokens == 1u && use_decode_lut_gate && !q4k_path && !mxfp4_path;
}

/* Must the caller drain a pending hits-first batch before launching? */
DS4_HF_FN int ds4_hitsfirst_must_wait(int hits_first_active,
                                      uint32_t n_tokens,
                                      int use_decode_lut_gate,
                                      int q4k_path,
                                      int mxfp4_path) {
    if (!hits_first_active) return 0;
    return !ds4_hitsfirst_split_consumes(n_tokens, use_decode_lut_gate,
                                         q4k_path, mxfp4_path);
}

/* ---- 2. what missed ----------------------------------------------------- */

/* The mask only has to carry the hits-first batch, which is capped at 8 slots
 * by `slot_count <= 8u` in begin_load. 64 is the width of the word. */
#define DS4_HITSFIRST_MISS_MAX 64u

DS4_HF_FN void ds4_hitsfirst_miss_set(uint64_t *mask, size_t i) {
    if (i < DS4_HITSFIRST_MISS_MAX) *mask |= (uint64_t)1 << i;
}

DS4_HF_FN int ds4_hitsfirst_miss_get(uint64_t mask, size_t i) {
    if (i >= DS4_HITSFIRST_MISS_MAX) return 0;
    return (int)((mask >> i) & (uint64_t)1);
}

/* ---- 3. which slots contribute ------------------------------------------ */

/* A slot whose selected id is negative is unowned: its quantized intermediate
 * may never have been written. The fused kernel `continue`s past it; the split
 * partial kernel must write zero and return. Same rule, one place. */
DS4_HF_FN int ds4_hitsfirst_slot_contributes(int32_t expert_i) {
    return expert_i >= 0;
}

/* ---- a cached env switch, with a read counter --------------------------- */

/* getenv is a linear scan of environ. A switch read once per process instead of
 * once per MoE layer per token costs one branch. The test substitutes a
 * counting reader for DS4_ENV_GATE_GETENV and asserts the read count, so the
 * saving is a counted fact rather than a claim. */
#ifndef DS4_ENV_GATE_GETENV
#include <stdlib.h>
#define DS4_ENV_GATE_GETENV getenv
#endif

typedef struct { int cached; unsigned long reads; } ds4_env_gate;
#define DS4_ENV_GATE_INIT { -1, 0 }

static int ds4_env_gate_on(ds4_env_gate *g, const char *name) {
    if (g->cached < 0) {
        g->reads++;
        g->cached = DS4_ENV_GATE_GETENV(name) != NULL;
    }
    return g->cached;
}

#endif /* DS4_HITSFIRST_LOGIC_H */
