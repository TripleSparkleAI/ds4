/*
 * test_draft_gamma.c -- behaviour check for the adaptive draft-gamma
 * controller (ds4_draft_gamma.c).
 *
 * Builds and runs standalone, with no GPU, no model and no Spark:
 *   make test-draft-gamma
 *
 * It exercises the state machine rather than a speedup: pass-through in fixed
 * mode, warmup, collapse to gamma 0 and the re-probe, the climb when drafts
 * land, the hysteresis hold, the downward-only EMA ceiling, per-lane-bucket
 * independence, and the engine cap. Every number it prints is a state trace,
 * not a tokens/s result. */
#include "ds4_draft_gamma.h"
#include <stdio.h>
#include <string.h>

static int failures = 0;
#define CHECK(cond, ...) do { if (!(cond)) { failures++; \
    printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); \
    printf("\n"); } } while (0)

static ds4_draft_gamma_config cfg_default(void) {
    ds4_draft_gamma_config cfg;
    ds4_draft_gamma_default_config(&cfg);
    return cfg;
}

static void feed(ds4_draft_gamma_controller *c, uint32_t lanes,
                 uint32_t accept, uint32_t cap, int times) {
    for (int i = 0; i < times; i++) {
        ds4_draft_gamma_controller_observe(c, lanes, accept, cap);
    }
}

/*
 * THE ENGINE'S OWN FEED CONTRACT, reproduced exactly.
 *
 * This is the point of case 11 below.  Every other case in this file calls
 * `observe` directly, on every cycle, WHETHER OR NOT A DRAFT WAS PRODUCED - and
 * the engine does not do that.  In ds4.c a cycle asks the controller for gamma;
 * gamma 0 means `prepare_dspark_draft_impl` drafts nothing and the verifier
 * takes its no-draft early return, which reports `no_draft = true`, and the EMA
 * is not fed from such a cycle.  A test that observes at gamma 0 is therefore
 * testing a path the engine cannot reach, and it passed for a controller that
 * was dead in the engine from its first collapse (R5 finding G1).
 *
 * One cycle, in the engine's order:
 *   gamma = controller_gamma(...)             prepare asks
 *   if gamma == 0: draft nothing              -> note_no_draft   (no EMA)
 *   else:          draft, verify accepts k    -> observe(k)
 */
static uint32_t engine_cycle(ds4_draft_gamma_controller *c, uint32_t lanes,
                             uint32_t fixed_gamma, uint32_t drafter_quality) {
    uint32_t gamma = ds4_draft_gamma_controller_gamma(c, lanes, fixed_gamma);
    if (gamma == 0u) {
        ds4_draft_gamma_controller_note_no_draft(c, lanes, fixed_gamma);
        return 0u;
    }
    uint32_t accepted = drafter_quality < gamma ? drafter_quality : gamma;
    ds4_draft_gamma_controller_observe(c, lanes, accepted, fixed_gamma);
    return gamma;
}

int main(void) {
    /* 1. fixed mode is a pass-through. */
    {
        ds4_draft_gamma_controller c;
        ds4_draft_gamma_controller_init(&c, "fixed", NULL);
        CHECK(ds4_draft_gamma_controller_gamma(&c, 1, 9) == 9, "fixed passthrough");
        feed(&c, 1, 0, 16, 200);
        CHECK(ds4_draft_gamma_controller_gamma(&c, 1, 9) == 9, "fixed untouched by observe");
        feed(&c, 1, 8, 16, 200);
        CHECK(ds4_draft_gamma_controller_gamma(&c, 1, 9) == 9, "fixed untouched after accepts");
    }
    /* 2. unset / garbage env stays fixed. */
    {
        ds4_draft_gamma_controller c;
        ds4_draft_gamma_controller_init(&c, "", NULL);
        CHECK(c.mode == DS4_DRAFT_GAMMA_MODE_FIXED, "unset stays fixed");
        ds4_draft_gamma_controller_init(&c, "nonsense", NULL);
        CHECK(c.mode == DS4_DRAFT_GAMMA_MODE_FIXED, "typo stays fixed");
        ds4_draft_gamma_controller_init(&c, "adaptive", NULL);
        CHECK(c.mode == DS4_DRAFT_GAMMA_MODE_ADAPTIVE, "adaptive engages");
    }
    /* 3. adaptive: warmup is observe-only. */
    {
        ds4_draft_gamma_controller c;
        ds4_draft_gamma_controller_init(&c, "adaptive", NULL);
        ds4_draft_gamma_controller_reset(&c, 4);
        feed(&c, 1, 6, 16, 9);   /* warmup_batches = 10 */
        CHECK(ds4_draft_gamma_controller_gamma(&c, 1, 16) == 4, "no decision during warmup");
        CHECK(c.slot[0].decisions == 0, "warmup decisions stay 0");
    }
    /* 4. drafts rejected early: gamma walks down to zero and RESTS there,
     *    re-probing the smallest positive step once per interval. */
    {
        ds4_draft_gamma_controller c;
        ds4_draft_gamma_controller_init(&c, "adaptive", NULL);
        ds4_draft_gamma_controller_reset(&c, 4);
        feed(&c, 1, 0, 16, 200);
        CHECK(c.slot[0].zero_intervals >= 1, "zero interval recorded");
        CHECK(c.slot[0].zero_probes >= 1, "zero interval is re-probed");
        CHECK(c.slot[0].at_zero || c.slot[0].current_step == 1,
              "collapse lands on zero or on the re-probe step (got %u)",
              c.slot[0].current_step);
        CHECK(c.slot[0].zero_probes * 2u <= c.slot[0].decisions,
              "the zero state rests more than it probes"
              " (probes=%u dec=%u)", c.slot[0].zero_probes, c.slot[0].decisions);
        printf("  collapse: step=%u ema=%.3f dec=%u zero=%u probe=%u\n",
               c.slot[0].current_step, (double)c.slot[0].ema_accept_len,
               c.slot[0].decisions, c.slot[0].zero_intervals,
               c.slot[0].zero_probes);
    }
    /* 5. healthy accepts: gamma climbs, probing one step beyond acceptance. */
    {
        ds4_draft_gamma_controller c;
        ds4_draft_gamma_controller_init(&c, "adaptive", NULL);
        ds4_draft_gamma_controller_reset(&c, 2);
        uint32_t prev = c.slot[0].current_step;
        int up_moves = 0;
        for (int i = 0; i < 400; i++) {
            /* a drafter that accepts everything it is asked for */
            uint32_t g = ds4_draft_gamma_controller_gamma(&c, 1, 16);
            feed(&c, 1, g, 16, 1);
            if (c.slot[0].current_step > prev) up_moves++;
            prev = c.slot[0].current_step;
        }
        CHECK(up_moves > 0, "gamma climbs when drafts land");
        CHECK(c.slot[0].current_step >= 4, "gamma climbed past its start (got %u)",
              c.slot[0].current_step);
        printf("  climb: step=%u ema=%.3f dec=%u up_moves=%d\n",
               c.slot[0].current_step, (double)c.slot[0].ema_accept_len,
               c.slot[0].decisions, up_moves);
    }
    /* 6. hysteresis: an EMA parked inside the band holds the step. With
     *    SGLang's defaults the band is (prev-0.75, cur-0.5], which integer
     *    accept lengths rarely sit inside, so widen up_hysteresis to make the
     *    band observable (the knob is the point, not the widening). */
    {
        ds4_draft_gamma_config cfg = cfg_default();
        cfg.up_hysteresis = 1.0f;    /* band becomes (prev-0.75, cur+0.5] */
        cfg.warmup_batches = 0;
        ds4_draft_gamma_controller c;
        ds4_draft_gamma_controller_init(&c, "adaptive", &cfg);
        ds4_draft_gamma_controller_reset(&c, 4);
        feed(&c, 1, 4, 16, 300);
        CHECK(c.slot[0].current_step == 4,
              "ema inside the band holds the step (got %u)",
              c.slot[0].current_step);
        printf("  hold: step=%u ema=%.3f dec=%u\n", c.slot[0].current_step,
               (double)c.slot[0].ema_accept_len, c.slot[0].decisions);
    }
    /* 6b. with the default bands, integer accept lengths give a bounded
     *     probe/hold cycle rather than a fixed point. Report it, do not hide
     *     it: the step never runs away in either direction. */
    {
        ds4_draft_gamma_config cfg = cfg_default();
        cfg.warmup_batches = 0;
        ds4_draft_gamma_controller c;
        ds4_draft_gamma_controller_init(&c, "adaptive", &cfg);
        ds4_draft_gamma_controller_reset(&c, 4);
        uint32_t lo = 99u, hi = 0u;
        for (int i = 0; i < 400; i++) {
            feed(&c, 1, 4, 16, 1);
            uint32_t g = c.slot[0].current_step;
            if (g < lo) lo = g;
            if (g > hi) hi = g;
        }
        CHECK(hi - lo <= 1u, "default-band steady state is bounded (lo=%u hi=%u)",
              lo, hi);
        printf("  default band: step in [%u, %u] ema=%.3f\n", lo, hi,
               (double)c.slot[0].ema_accept_len);
    }
    /* 7. ceiling caps downward only. */
    {
        ds4_draft_gamma_config cfg = cfg_default();
        cfg.ceiling_coeff = 0.6f;   /* aggressive cap, easy to see */
        cfg.warmup_batches = 0;
        cfg.update_interval = 1;
        ds4_draft_gamma_controller c;
        ds4_draft_gamma_controller_init(&c, "adaptive", &cfg);
        ds4_draft_gamma_controller_reset(&c, 8);
        feed(&c, 1, 1, 16, 100);
        /* ceil = 0.6 * 1 = 0.6 -> 1, so a down move cannot pass 1 */
        CHECK(c.slot[0].current_step <= 1, "ceiling bounds the down move (got %u)",
              c.slot[0].current_step);
        /* and it does not block a step-up: raise the EMA and watch it climb */
        feed(&c, 1, 4, 16, 100);
        CHECK(c.slot[0].current_step > 1, "ceiling does not block step-ups (got %u)",
              c.slot[0].current_step);
        printf("  ceiling: step=%u ema=%.3f\n", c.slot[0].current_step,
               (double)c.slot[0].ema_accept_len);
    }
    /* 8. per-lane buckets are independent. */
    {
        ds4_draft_gamma_controller c;
        ds4_draft_gamma_controller_init(&c, "adaptive", NULL);
        ds4_draft_gamma_controller_reset(&c, 6);
        feed(&c, 1, 0, 16, 200);    /* lane bucket 0 collapses */
        feed(&c, 32, 6, 16, 200);   /* high-concurrency bucket does not */
        CHECK(c.slot[0].current_step == 0, "bucket 0 collapsed");
        CHECK(c.slot[5].current_step > 0, "bucket 5 held (got %u)",
              c.slot[5].current_step);
        CHECK(ds4_draft_gamma_bucket_for_lanes(1) == 0, "bucket 1");
        CHECK(ds4_draft_gamma_bucket_for_lanes(3) == 1, "bucket 3");
        CHECK(ds4_draft_gamma_bucket_for_lanes(64) == 5, "bucket 64");
        printf("  buckets: b0=%u b1=%u b2=%u b3=%u b4=%u b5=%u\n",
               c.slot[0].current_step, c.slot[1].current_step,
               c.slot[2].current_step, c.slot[3].current_step,
               c.slot[4].current_step, c.slot[5].current_step);
    }
    /* 9. the engine cap can never be exceeded. */
    {
        ds4_draft_gamma_controller c;
        ds4_draft_gamma_controller_init(&c, "adaptive", NULL);
        ds4_draft_gamma_controller_reset(&c, 4);
        feed(&c, 1, 16, 5, 300);   /* accepts are high, engine cap is 5 */
        CHECK(ds4_draft_gamma_controller_gamma(&c, 1, 5) <= 5,
              "gamma never exceeds the engine cap (got %u)",
              ds4_draft_gamma_controller_gamma(&c, 1, 5));
        printf("  cap: step=%u gamma=%u\n", c.slot[0].current_step,
               ds4_draft_gamma_controller_gamma(&c, 1, 5));
    }
    /* 10. the dump renders. */
    {
        ds4_draft_gamma_controller c;
        ds4_draft_gamma_controller_init(&c, "adaptive", NULL);
        ds4_draft_gamma_controller_reset(&c, 3);
        feed(&c, 1, 3, 16, 40);
        char buf[1024];
        int n = ds4_draft_gamma_controller_dump(&c, buf, sizeof(buf));
        CHECK(n > 0, "dump produced text");
        printf("  dump: %s\n", buf);
    }

    /* 11. THE ENGINE PATH: gamma 0 must not be absorbing.
     *
     *     Driven only through engine_cycle(), so the controller is fed exactly
     *     as ds4.c feeds it and no more. A drafter that accepts nothing drives
     *     the collapse; then the drafter turns good and gamma must come back.
     *     Before the no-draft report existed this second half was unreachable:
     *     nothing drafts, so nothing observes, so nothing decides. */
    {
        ds4_draft_gamma_controller c;
        ds4_draft_gamma_controller_init(&c, "adaptive", NULL);
        ds4_draft_gamma_controller_reset(&c, 4);

        int reached_zero = 0;
        for (int i = 0; i < 200; i++) {
            if (engine_cycle(&c, 1, 16, 0) == 0u) reached_zero = 1;
        }
        /* The vacuity control for the recovery check below: if the collapse
         * never happened, "it recovered" would be true of a controller that
         * never left 4. Its own case. */
        CHECK(reached_zero,
              "engine path: a drafter accepting nothing collapses gamma to 0");
        uint32_t after_collapse = ds4_draft_gamma_controller_gamma(&c, 1, 16);
        printf("  engine collapse: gamma=%u ema=%.3f dec=%u zero=%u probe=%u\n",
               after_collapse, (double)c.slot[0].ema_accept_len,
               c.slot[0].decisions, c.slot[0].zero_intervals,
               c.slot[0].zero_probes);

        uint32_t probes_before = c.slot[0].zero_probes;
        uint32_t decisions_before = c.slot[0].decisions;

        /* Now the drafter is good. Same harness, same contract. */
        uint32_t best = 0u;
        for (int i = 0; i < 400; i++) {
            uint32_t g = engine_cycle(&c, 1, 16, 8);
            if (g > best) best = g;
        }
        printf("  engine recovery: gamma=%u best=%u ema=%.3f dec=%u probe=%u\n",
               ds4_draft_gamma_controller_gamma(&c, 1, 16), best,
               (double)c.slot[0].ema_accept_len, c.slot[0].decisions,
               c.slot[0].zero_probes);

        CHECK(c.slot[0].decisions > decisions_before,
              "engine path: the decision clock advances while gamma is 0"
              " (dec %u -> %u)", decisions_before, c.slot[0].decisions);
        CHECK(c.slot[0].zero_probes > probes_before,
              "engine path: the re-probe out of zero is reachable from the"
              " engine (probes %u -> %u)", probes_before,
              c.slot[0].zero_probes);
        CHECK(best > 0u,
              "engine path: gamma leaves 0 once drafts land again (best=%u)",
              best);
        CHECK(ds4_draft_gamma_controller_gamma(&c, 1, 16) > 0u,
              "engine path: gamma recovers and stays positive (got %u)",
              ds4_draft_gamma_controller_gamma(&c, 1, 16));
    }
    /* 11b. the no-draft report is narrow: a slot that is DRAFTING is told
     *      nothing by a no-draft cycle it did not ask for. Its own case. */
    {
        ds4_draft_gamma_controller c;
        ds4_draft_gamma_controller_init(&c, "adaptive", NULL);
        ds4_draft_gamma_controller_reset(&c, 4);
        feed(&c, 1, 4, 16, 40);
        float ema = c.slot[0].ema_accept_len;
        uint32_t seen = c.slot[0].batches_seen;
        uint32_t step = c.slot[0].current_step;
        CHECK(step > 0, "11b precondition: the slot is drafting (got %u)", step);
        for (int i = 0; i < 100; i++) {
            ds4_draft_gamma_controller_note_no_draft(&c, 1, 16);
        }
        CHECK(c.slot[0].batches_seen == seen,
              "a no-draft cycle the controller did not ask for is ignored"
              " (%u -> %u)", seen, c.slot[0].batches_seen);
        CHECK(c.slot[0].ema_accept_len == ema,
              "and it never touches the EMA");
        CHECK(!ds4_draft_gamma_controller_at_zero(&c, 1),
              "at_zero is false while the slot is drafting");
    }
    /* 11c. fixed mode is untouched by the new report. Its own case. */
    {
        ds4_draft_gamma_controller c;
        ds4_draft_gamma_controller_init(&c, "fixed", NULL);
        for (int i = 0; i < 100; i++) {
            ds4_draft_gamma_controller_note_no_draft(&c, 1, 16);
        }
        CHECK(ds4_draft_gamma_controller_gamma(&c, 1, 9) == 9,
              "fixed mode still passes through after no-draft reports");
        CHECK(!ds4_draft_gamma_controller_at_zero(&c, 1),
              "at_zero is false in fixed mode");
    }

    printf("%s: %d failure(s)\n", failures ? "FAIL" : "ok", failures);
    return failures ? 1 : 0;
}