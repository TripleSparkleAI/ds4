/*
 * ds4_draft_gamma.c -- adaptive DSpark draft-length (gamma) controller.
 *
 * See ds4_draft_gamma.h for the mechanism and its source. The short version:
 * the DSpark draft length in this tree is a hand-chosen constant taken from the
 * support model's GGUF metadata and never revisited under load. This controller
 * picks it at runtime, per active-lane bucket, from an EMA of the accept length
 * that the B2b loop already produces, and it can pick ZERO.
 *
 * Source discipline implemented here, one for one:
 *   probe one step beyond observed acceptance
 *       target = clamp(round(ema_accept_len) + 1, 0, max_steps)
 *   step down while ema_accept_len <= prev_step - 0.5 + down_hysteresis
 *   step up   while ema_accept_len >  current_step - 0.5 + up_hysteresis
 *   hysteresis bands: between the two edges the step is held
 *   warmup_batches: observe only, decide nothing
 *   update_interval: only re-decide every N batches
 *   EMA ceiling: caps gamma DOWNWARD only, so step-ups stay explorable
 *   after a zero-step interval, probe the smallest positive step
 *
 * In DS4_DRAFT_GAMMA_MODE_FIXED (the default) every entry point here is a
 * no-op or a pass-through, so today's behaviour is preserved exactly.
 */
#include "ds4_draft_gamma.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static float env_float(const char *name, float fallback) {
    const char *env = getenv(name);
    if (!env || !env[0]) return fallback;
    char *end = NULL;
    float v = strtof(env, &end);
    if (end == env || v < 0.0f) return fallback;
    return v;
}

static uint32_t env_u32(const char *name, uint32_t fallback) {
    const char *env = getenv(name);
    if (!env || !env[0]) return fallback;
    char *end = NULL;
    unsigned long v = strtoul(env, &end, 10);
    if (end == env || v > 1000000ul) return fallback;
    return (uint32_t)v;
}

const char *ds4_draft_gamma_mode_name(ds4_draft_gamma_mode mode) {
    return mode == DS4_DRAFT_GAMMA_MODE_ADAPTIVE ? "adaptive" : "fixed";
}

const char *ds4_draft_gamma_mode_env_string(void) {
    const char *env = getenv("DS4_DRAFT_GAMMA_MODE");
    return (env && env[0]) ? env : "fixed";
}

ds4_draft_gamma_mode ds4_draft_gamma_mode_from_env(void) {
    const char *env = ds4_draft_gamma_mode_env_string();
    if (!strcmp(env, "adaptive")) return DS4_DRAFT_GAMMA_MODE_ADAPTIVE;
    /* Anything else, including unset and any typo, stays on today's path. */
    return DS4_DRAFT_GAMMA_MODE_FIXED;
}

void ds4_draft_gamma_default_config(ds4_draft_gamma_config *cfg) {
    if (!cfg) return;
    cfg->ema_alpha       = env_float("DS4_DRAFT_GAMMA_ALPHA", 0.2f);
    cfg->update_interval = env_u32("DS4_DRAFT_GAMMA_UPDATE_INTERVAL", 5u);
    cfg->warmup_batches  = env_u32("DS4_DRAFT_GAMMA_WARMUP", 10u);
    cfg->down_hysteresis = env_float("DS4_DRAFT_GAMMA_DOWN_HYST", -0.25f);
    cfg->up_hysteresis   = env_float("DS4_DRAFT_GAMMA_UP_HYST", 0.0f);
    cfg->ceiling_coeff   = env_float("DS4_DRAFT_GAMMA_CEILING", 1.5f);
    cfg->max_steps       = env_u32("DS4_DRAFT_GAMMA_MAX_STEPS",
                                   DS4_DRAFT_GAMMA_MAX_STEPS);
    if (cfg->max_steps > DS4_DRAFT_GAMMA_MAX_STEPS) {
        cfg->max_steps = DS4_DRAFT_GAMMA_MAX_STEPS;
    }
    if (cfg->ema_alpha > 1.0f) cfg->ema_alpha = 1.0f;
    if (cfg->update_interval == 0u) cfg->update_interval = 1u;
    if (cfg->down_hysteresis > 0.0f) cfg->down_hysteresis = 0.0f;
    if (cfg->up_hysteresis < 0.0f) cfg->up_hysteresis = 0.0f;
}

uint32_t ds4_draft_gamma_bucket_for_lanes(uint32_t lanes) {
    uint32_t bucket = 0u;
    while (lanes > 1u && bucket + 1u < DS4_DRAFT_GAMMA_MAX_SLOTS) {
        lanes >>= 1u;
        bucket++;
    }
    return bucket;
}

static void slot_init(ds4_draft_gamma_slot *sl,
                      const ds4_draft_gamma_config *cfg,
                      uint32_t init_step) {
    memset(sl, 0, sizeof(*sl));
    sl->cfg = *cfg;
    sl->current_step = init_step;
    sl->at_zero = init_step == 0u;
}

void ds4_draft_gamma_controller_init(ds4_draft_gamma_controller *c,
                                     const char *mode_env,
                                     const ds4_draft_gamma_config *cfg) {
    if (!c) return;
    memset(c, 0, sizeof(*c));
    if (mode_env && mode_env[0]) {
        c->mode = !strcmp(mode_env, "adaptive")
                      ? DS4_DRAFT_GAMMA_MODE_ADAPTIVE
                      : DS4_DRAFT_GAMMA_MODE_FIXED;
    } else {
        c->mode = ds4_draft_gamma_mode_from_env();
    }
    ds4_draft_gamma_config local;
    if (cfg) {
        local = *cfg;
    } else {
        ds4_draft_gamma_default_config(&local);
    }
    c->init_step = 0u;   /* set for real on the first gamma query per bucket */
    c->config_ready = true;
    for (uint32_t i = 0; i < DS4_DRAFT_GAMMA_MAX_SLOTS; i++) {
        slot_init(&c->slot[i], &local, 0u);
    }
}

void ds4_draft_gamma_controller_reset(ds4_draft_gamma_controller *c,
                                      uint32_t fixed_gamma) {
    if (!c) return;
    if (c->mode != DS4_DRAFT_GAMMA_MODE_ADAPTIVE) return;
    ds4_draft_gamma_config cfg = c->slot[0].cfg;
    uint32_t start = fixed_gamma;
    if (start > cfg.max_steps) start = cfg.max_steps;
    c->init_step = start;
    for (uint32_t i = 0; i < DS4_DRAFT_GAMMA_MAX_SLOTS; i++) {
        /* A fresh request starts every bucket at the gamma the fixed path
         * would have used, so the A/B begins from the same place. */
        slot_init(&c->slot[i], &cfg, start);
    }
}

uint32_t ds4_draft_gamma_controller_gamma(const ds4_draft_gamma_controller *c,
                                          uint32_t lanes,
                                          uint32_t fixed_gamma) {
    if (!c || c->mode != DS4_DRAFT_GAMMA_MODE_ADAPTIVE) return fixed_gamma;
    if (fixed_gamma == 0u) return 0u;   /* no drafting to size */
    const ds4_draft_gamma_slot *sl = &c->slot[ds4_draft_gamma_bucket_for_lanes(
                                                   lanes ? lanes : 1u)];
    uint32_t step = sl->current_step;
    if (step > fixed_gamma) step = fixed_gamma;   /* never exceed the engine */
    return step;
}

/*
 * The load-bearing logic. Called every update_interval batches once warmed.
 */
static void slot_decide(ds4_draft_gamma_slot *sl) {
    const uint32_t prev_step = sl->current_step;
    const uint32_t max_steps = sl->cfg.max_steps;
    uint32_t hard_cap = sl->hard_cap;
    if (hard_cap == 0u || hard_cap > max_steps) hard_cap = max_steps;

    /* Probe one step beyond observed acceptance. */
    uint32_t probe = (uint32_t)(sl->ema_accept_len + 0.5f) + 1u;
    if (probe > max_steps) probe = max_steps;
    if (probe > hard_cap) probe = hard_cap;

    const float down_edge =
        (float)prev_step - 0.5f + sl->cfg.down_hysteresis;
    const float up_edge =
        (float)sl->current_step - 0.5f + sl->cfg.up_hysteresis;

    uint32_t target;
    if (sl->ema_accept_len <= down_edge) {
        /* Drafts are being rejected early: walk toward fewer steps. */
        target = probe;
        if (prev_step > 0u && target >= prev_step) target = prev_step - 1u;
    } else if (sl->ema_accept_len > up_edge) {
        /* Drafts are landing: walk toward more steps. */
        target = probe;
        if (target <= sl->current_step) target = sl->current_step + 1u;
    } else {
        /* Inside the hysteresis band: hold, do not oscillate. */
        target = sl->current_step;
    }
    if (target > max_steps) target = max_steps;
    if (target > hard_cap) target = hard_cap;

    /* EMA ceiling: gamma is capped DOWNWARD ONLY, so a step-up can still be
     * explored and the EMA gets a chance to catch up. */
    if (sl->cfg.ceiling_coeff > 0.0f && target < sl->current_step) {
        uint32_t ceil_step =
            (uint32_t)(sl->cfg.ceiling_coeff * sl->ema_accept_len + 0.5f);
        if (target > ceil_step) target = ceil_step;
    }

    /* After a zero-step interval, probe the smallest positive step to re-test
     * whether drafting is worth resuming. The step-up edge at gamma 0 is -0.5,
     * so a healthy EMA walks it back up on its own; this makes the re-probe
     * explicit and gives the zero state a REST of one decision interval first.
     * Without the rest the controller alternates 1/0 every interval, which
     * spends a verify pass on exactly the drafts it just decided were not
     * worth their bandwidth. */
    if (sl->current_step == 0u) {
        if (sl->zero_hold_pending) {
            target = 1u;
            if (target > hard_cap) target = hard_cap;
            sl->zero_hold_pending = false;
            sl->zero_hold_batches = 0u;
            sl->zero_probes++;
        } else {
            target = 0u;
            sl->zero_hold_pending = true;
        }
    }

    if (sl->current_step != target) {
        if (target == 0u && sl->current_step != 0u) {
            sl->zero_intervals++;
        }
        sl->current_step = target;
    }
    sl->at_zero = sl->current_step == 0u;
    sl->decisions++;
    sl->batches_since_decide = 0u;
}

void ds4_draft_gamma_controller_observe(ds4_draft_gamma_controller *c,
                                        uint32_t lanes,
                                        uint32_t accepted_len,
                                        uint32_t hard_cap) {
    if (!c || c->mode != DS4_DRAFT_GAMMA_MODE_ADAPTIVE) return;
    ds4_draft_gamma_slot *sl =
        &c->slot[ds4_draft_gamma_bucket_for_lanes(lanes ? lanes : 1u)];
    const uint32_t cap = hard_cap ? hard_cap : sl->cfg.max_steps;

    sl->hard_cap = cap;
    if (!sl->ever_observed) {
        /* Seed the EMA with the first observation rather than dragging it up
         * from zero, which would fake a collapse on the first batch. */
        sl->ema_accept_len = (float)accepted_len;
        sl->ever_observed = true;
    } else {
        sl->ema_accept_len += sl->cfg.ema_alpha *
                              ((float)accepted_len - sl->ema_accept_len);
    }
    sl->batches_seen++;
    if (sl->batches_since_decide < 1000000u) sl->batches_since_decide++;
    if (!sl->warmed) {
        if (sl->batches_seen >= sl->cfg.warmup_batches) sl->warmed = true;
        return;
    }
    if (sl->batches_since_decide >= sl->cfg.update_interval) {
        slot_decide(sl);
    }
}

/*
 * A CYCLE IN WHICH NOTHING WAS DRAFTED.
 *
 * `observe` is fed from cycles that produced a draft, because only those carry
 * evidence about acceptance.  That is right, and it is also why gamma 0 was
 * ABSORBING (REVIEW-R5, finding G1): at gamma 0 the engine drafts nothing, so
 * the verifier takes its no-draft early return and reports `no_draft = true`,
 * which the caller does not feed.  `batches_since_decide` therefore never
 * advanced, `slot_decide` never ran again, and the re-probe below it was
 * unreachable from the engine.  The first collapse to zero in a lane bucket
 * disabled drafting there for the rest of the request.
 *
 * This is the other half of the feed: a cycle the CONTROLLER ITSELF chose to
 * skip.  It carries no acceptance evidence - so THE EMA IS NOT TOUCHED, and
 * the EMA the controller held when it collapsed is what the re-probe is then
 * tested against - but it does carry the passage of a decision interval, which
 * is the thing `slot_decide` is waiting for.
 *
 * It is deliberately narrow.  A no-draft cycle the controller did NOT ask for
 * (the support model absent, drafting off, a stochastic request) says nothing
 * about the controller's own decision, so a slot that is not at zero is left
 * alone; and an engine that cannot draft at all (`hard_cap` 0) has nothing to
 * re-probe toward.
 */
void ds4_draft_gamma_controller_note_no_draft(ds4_draft_gamma_controller *c,
                                              uint32_t lanes,
                                              uint32_t hard_cap) {
    if (!c || c->mode != DS4_DRAFT_GAMMA_MODE_ADAPTIVE) return;
    if (hard_cap == 0u) return;
    ds4_draft_gamma_slot *sl =
        &c->slot[ds4_draft_gamma_bucket_for_lanes(lanes ? lanes : 1u)];
    if (!sl->at_zero) return;

    sl->hard_cap = hard_cap;
    sl->batches_seen++;
    if (sl->batches_since_decide < 1000000u) sl->batches_since_decide++;
    if (!sl->warmed) {
        if (sl->batches_seen >= sl->cfg.warmup_batches) sl->warmed = true;
        return;
    }
    if (sl->batches_since_decide >= sl->cfg.update_interval) {
        slot_decide(sl);
    }
}

bool ds4_draft_gamma_controller_at_zero(const ds4_draft_gamma_controller *c,
                                        uint32_t lanes) {
    if (!c || c->mode != DS4_DRAFT_GAMMA_MODE_ADAPTIVE) return false;
    return c->slot[ds4_draft_gamma_bucket_for_lanes(lanes ? lanes : 1u)]
               .at_zero;
}

int ds4_draft_gamma_controller_dump(const ds4_draft_gamma_controller *c,
                                    char *buf,
                                    size_t buflen) {
    if (!c || !buf || buflen == 0) return 0;
    int n = snprintf(buf, buflen, "draft-gamma mode=%s init=%u",
                     ds4_draft_gamma_mode_name(c->mode),
                     (unsigned)c->init_step);
    if (c->mode != DS4_DRAFT_GAMMA_MODE_ADAPTIVE) return n;
    for (uint32_t i = 0; i < DS4_DRAFT_GAMMA_MAX_SLOTS; i++) {
        const ds4_draft_gamma_slot *sl = &c->slot[i];
        if (!sl->ever_observed && sl->decisions == 0u) continue;
        n += snprintf(buf + (n < (int)buflen ? n : (int)buflen - 1),
                      n < (int)buflen ? buflen - (size_t)n : 1u,
                      " | lanes>=%u gamma=%u ema=%.3f obs=%u dec=%u zero=%u"
                      " probe=%u hold=%d warm=%d",
                      (unsigned)(1u << i),
                      (unsigned)sl->current_step,
                      (double)sl->ema_accept_len,
                      (unsigned)sl->batches_seen,
                      (unsigned)sl->decisions,
                      (unsigned)sl->zero_intervals,
                      (unsigned)sl->zero_probes,
                      sl->zero_hold_pending ? 1 : 0,
                      sl->warmed ? 1 : 0);
        if (n >= (int)buflen) break;
    }
    return n;
}