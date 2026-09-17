/*
 * ds4_draft_gamma.h -- adaptive DSpark draft-length (gamma) controller.
 *
 * Today the DSpark draft length is chosen once, by hand, and never revisits
 * that choice: it is the support model's `block_size` GGUF metadata field
 * (ds4.c:76274-76276), clamped by the compile-time DS4_DSPARK_MAX_BLOCK_SIZE
 * (ds4.c:2906) and only overridable by a static environment value. The branch
 * adds a runtime controller that picks gamma per active-lane bucket from an
 * EMA of the observed accept length, with gamma == 0 as a first-class state
 * that disables drafting entirely.
 *
 * The mechanism is SGLang's AdaptiveStepSlot (engine-harvest 03, our own
 * WIKI/research/engine-harvests/03-adaptive-gamma-under-concurrency.md;
 * upstream python/sglang/srt/speculative/adaptive_spec_params.py):
 * one slot per batch size, an EMA of accept length, a candidate set that
 * includes zero, hysteresis bands, a warmup, an update interval, and an EMA
 * ceiling that caps gamma downward only.
 */
#ifndef DS4_DRAFT_GAMMA_H
#define DS4_DRAFT_GAMMA_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* Candidate gammas are the consecutive steps 0..DS4_DRAFT_GAMMA_MAX_STEPS.
 * ZERO IS A FIRST-CLASS CANDIDATE: selecting it disables drafting entirely,
 * which is the honest "spec-decode is not paying for its verify bandwidth"
 * state. DS4_DRAFT_GAMMA_MAX_STEPS mirrors DS4_DSPARK_MAX_BLOCK_SIZE. */
#define DS4_DRAFT_GAMMA_MAX_STEPS 16u

/* One slot per active-lane bucket: 1, 2, 4, 8, 16, 32-or-more lanes. */
#define DS4_DRAFT_GAMMA_MAX_SLOTS 6u

typedef enum {
    DS4_DRAFT_GAMMA_MODE_FIXED = 0,    /* today's behaviour, the default */
    DS4_DRAFT_GAMMA_MODE_ADAPTIVE = 1  /* the controller decides gamma */
} ds4_draft_gamma_mode;

typedef struct {
    float    ema_alpha;         /* EMA weight on the newest observation (0.2) */
    uint32_t update_interval;   /* re-decide only every N batches (5) */
    uint32_t warmup_batches;    /* observe-only batches before deciding (10) */
    float    down_hysteresis;   /* added to the step-down edge (-0.25) */
    float    up_hysteresis;     /* added to the step-up edge (0.0) */
    float    ceiling_coeff;     /* EMA ceiling on gamma, 0 disables (1.5) */
    uint32_t max_steps;         /* candidate ceiling (16) */
} ds4_draft_gamma_config;

typedef struct {
    ds4_draft_gamma_config cfg;
    float    ema_accept_len;    /* EMA of observed accept length tau */
    uint32_t current_step;      /* gamma in force for this bucket */
    uint32_t batches_seen;      /* observations fed */
    uint32_t batches_since_decide;
    uint32_t decisions;
    uint32_t zero_intervals;    /* times the last decision chose gamma 0 */
    uint32_t zero_probes;       /* times a zero interval was re-probed */
    uint32_t zero_hold_batches; /* batches rested at gamma 0 since the probe */
    uint32_t hard_cap;          /* engine ceiling seen at the last decision */
    bool     warmed;            /* warmup_batches observations are in */
    bool     at_zero;           /* current_step == 0 */
    bool     zero_hold_pending; /* a zero interval has not been re-probed yet */
    bool     ever_observed;
} ds4_draft_gamma_slot;

typedef struct {
    ds4_draft_gamma_mode   mode;
    ds4_draft_gamma_slot   slot[DS4_DRAFT_GAMMA_MAX_SLOTS];
    uint32_t               init_step;   /* gamma each fresh slot starts from */
    bool                   config_ready;
} ds4_draft_gamma_controller;

/* Mode from DS4_DRAFT_GAMMA_MODE: "adaptive" enables the controller, anything
 * else (including unset) is DS4_DRAFT_GAMMA_MODE_FIXED. */
const char *ds4_draft_gamma_mode_name(ds4_draft_gamma_mode mode);
ds4_draft_gamma_mode ds4_draft_gamma_mode_from_env(void);
const char *ds4_draft_gamma_mode_env_string(void);

/* The SGLang defaults, each overridable by DS4_DRAFT_GAMMA_*. */
void ds4_draft_gamma_default_config(ds4_draft_gamma_config *cfg);

/* mode is read from the environment when mode_env is NULL. */
void ds4_draft_gamma_controller_init(ds4_draft_gamma_controller *c,
                                     const char *mode_env,
                                     const ds4_draft_gamma_config *cfg);

/* Reset per-request state. A no-op in fixed mode. */
void ds4_draft_gamma_controller_reset(ds4_draft_gamma_controller *c,
                                      uint32_t fixed_gamma);

/* Gamma for this lane bucket. fixed_gamma is today's value, passed straight
 * through in fixed mode and used as the starting point in adaptive mode. */
uint32_t ds4_draft_gamma_controller_gamma(const ds4_draft_gamma_controller *c,
                                          uint32_t lanes,
                                          uint32_t fixed_gamma);

/* Feed one observed accept length tau. hard_cap is the engine's ceiling on
 * gamma this cycle (the support model's block size). A no-op in fixed mode. */
void ds4_draft_gamma_controller_observe(ds4_draft_gamma_controller *c,
                                        uint32_t lanes,
                                        uint32_t accepted_len,
                                        uint32_t hard_cap);

/*
 * Feed one cycle in which NOTHING WAS DRAFTED.  The caller reports every such
 * cycle; this decides whether it was the controller's own doing.
 *
 * Without it, gamma 0 is ABSORBING: at gamma 0 the engine drafts nothing, the
 * verifier reports a no-draft cycle, `observe` is not called, the decision
 * clock never advances and the re-probe out of zero never runs.  A no-draft
 * cycle carries NO ACCEPTANCE EVIDENCE, so this does not touch the EMA; it
 * advances the decision clock only, and only for a slot that is itself at
 * gamma 0.  A no-op in fixed mode, for a slot that is drafting, and when the
 * engine cannot draft at all (hard_cap 0).
 */
void ds4_draft_gamma_controller_note_no_draft(ds4_draft_gamma_controller *c,
                                              uint32_t lanes,
                                              uint32_t hard_cap);

/* True when this lane bucket's gamma is 0, i.e. the controller is the reason
 * the engine is not drafting. Adaptive mode only. */
bool ds4_draft_gamma_controller_at_zero(const ds4_draft_gamma_controller *c,
                                        uint32_t lanes);

/* One compact line of per-slot state for DS4_DRAFT_GAMMA_LOG. */
int ds4_draft_gamma_controller_dump(const ds4_draft_gamma_controller *c,
                                    char *buf,
                                    size_t buflen);

/* The bucket a lane count maps to. Exposed for tests and for the log. */
uint32_t ds4_draft_gamma_bucket_for_lanes(uint32_t lanes);

#endif /* DS4_DRAFT_GAMMA_H */