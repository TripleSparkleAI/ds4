/* JEV ROUTER: the scout's guess from the model's own routing statistics.
 *
 * The scout (ds4_scout_logic.h) guesses layer L+1's experts as "the ones the
 * last token used at L+1" (stickiness). This header replaces that guess with
 * a one-pass decision from state the engine already holds: the ids this token
 * routed to at layer L, scored through a per-layer successor table that was
 * trained OFFLINE from a routing trace of this same model
 * (gguf-tools/jev-router/train_next_layer.py), plus a sticky bonus for the
 * last token's L+1 ids. Reads only, never math: greedy output is byte-
 * identical by construction. Off unless DS4_CUDA_JEV_ROUTER names a table.
 *
 * Table file (text, written by the trainer, one successor row per (layer,
 * expert) that has any):
 *
 *   ds4-jev-router 1
 *   layers <n_layers> experts <n_experts> k <k> sticky <w> min <m>
 *   <layer> <expert> <id> <prob> <id> <prob> ...   (up to k pairs)
 *
 * `layer` is the SOURCE layer L; its row lists the experts most often routed
 * to at L+1 by tokens that routed to `expert` at L. Predict: score every
 * candidate at L+1 as the sum of the successor probabilities over this
 * token's ids at L, plus `sticky` for each id in the last token's L+1 set;
 * keep candidates scoring at least `min`, best first, ties by lower id, at
 * most `cap`. Deterministic, a few hundred adds per layer.
 *
 * Pure C99, no CUDA types: tests/test_jev_router_logic.c exercises it on any
 * host. Table memory is n_layers * n_experts * k * (int16 + float), about
 * 4 MB at 128 x 1024 x 8, allocated once at load.
 */
#ifndef DS4_JEV_ROUTER_LOGIC_H
#define DS4_JEV_ROUTER_LOGIC_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DS4_JEV_MAX_LAYERS  128u
#define DS4_JEV_MAX_EXPERTS 1024u
#define DS4_JEV_MAX_K       16u

typedef struct {
    uint32_t n_layers, n_experts, k;
    float    sticky;       /* bonus for an id in the last token's L+1 set */
    float    min_score;    /* a candidate below this is not guessed */
    int16_t *succ;         /* [layer][expert][k], -1 = empty */
    float   *prob;         /* [layer][expert][k] */
} ds4_jev_router;

static inline void ds4_jev_router_free(ds4_jev_router *r) {
    if (!r) return;
    free(r->succ); free(r->prob);
    memset(r, 0, sizeof *r);
}

/* Returns 1 on a well-formed table, 0 otherwise (r is zeroed on 0). A row
 * naming a layer or expert outside the header's bounds fails the load: a
 * table the loader half-trusts is worse than none. */
static inline int ds4_jev_router_load_file(ds4_jev_router *r, FILE *f) {
    if (!r || !f) return 0;
    memset(r, 0, sizeof *r);
    char tag[32] = {0};
    unsigned ver = 0;
    if (fscanf(f, "%31s %u", tag, &ver) != 2 || strcmp(tag, "ds4-jev-router") != 0 || ver != 1)
        return 0;
    unsigned nl = 0, ne = 0, k = 0;
    float sticky = 0.f, min_score = 0.f;
    if (fscanf(f, " layers %u experts %u k %u sticky %f min %f", &nl, &ne, &k, &sticky, &min_score) != 5)
        return 0;
    if (!nl || nl > DS4_JEV_MAX_LAYERS || !ne || ne > DS4_JEV_MAX_EXPERTS || !k || k > DS4_JEV_MAX_K)
        return 0;
    const size_t cells = (size_t)nl * ne * k;
    r->succ = (int16_t *)malloc(cells * sizeof *r->succ);
    r->prob = (float *)malloc(cells * sizeof *r->prob);
    if (!r->succ || !r->prob) { ds4_jev_router_free(r); return 0; }
    for (size_t i = 0; i < cells; i++) { r->succ[i] = -1; r->prob[i] = 0.f; }
    r->n_layers = nl; r->n_experts = ne; r->k = k; r->sticky = sticky; r->min_score = min_score;
    unsigned layer = 0, expert = 0;
    while (fscanf(f, " %u %u", &layer, &expert) == 2) {
        if (layer >= nl || expert >= ne) { ds4_jev_router_free(r); return 0; }
        int16_t *s = r->succ + ((size_t)layer * ne + expert) * k;
        float   *p = r->prob + ((size_t)layer * ne + expert) * k;
        for (unsigned j = 0; j < k; j++) {
            int c = fgetc(f);
            while (c == ' ' || c == '\t') c = fgetc(f);
            if (c == '\n' || c == EOF) { if (c == '\n') ungetc(c, f); break; }
            ungetc(c, f);
            unsigned id = 0; float pr = 0.f;
            if (fscanf(f, "%u %f", &id, &pr) != 2 || id >= ne) { ds4_jev_router_free(r); return 0; }
            s[j] = (int16_t)id; p[j] = pr;
        }
    }
    return 1;
}

static inline int ds4_jev_router_load(ds4_jev_router *r, const char *path) {
    FILE *f = path ? fopen(path, "r") : NULL;
    if (!f) { if (r) memset(r, 0, sizeof *r); return 0; }
    const int ok = ds4_jev_router_load_file(r, f);
    fclose(f);
    return ok;
}

/* The guess for layer `layer`+1. cur[0..n_cur) are this token's ids at
 * `layer`; sticky[0..n_sticky) are the last token's ids at layer+1 (may be
 * NULL/0). Writes at most `cap` ids to out, best first, their scores to
 * out_conf when it is not NULL (the trace's per-expert confidence), and
 * returns how many.
 * Returns 0 when the table has no row for `layer` (a caller then falls back
 * to stickiness, which is what the scout did before this header existed). */
static inline uint32_t ds4_jev_router_predict(const ds4_jev_router *r, uint32_t layer,
                                              const int32_t *cur, uint32_t n_cur,
                                              const int32_t *sticky, uint32_t n_sticky,
                                              uint32_t cap, int32_t *out, float *out_conf) {
    if (!r || !r->succ || !out || !cap || layer >= r->n_layers || !cur) return 0;
    float score[DS4_JEV_MAX_EXPERTS];
    memset(score, 0, sizeof(float) * r->n_experts);
    int any = 0;
    for (uint32_t i = 0; i < n_cur; i++) {
        const int32_t e = cur[i];
        if (e < 0 || (uint32_t)e >= r->n_experts) continue;
        int dup = 0;
        for (uint32_t j = 0; j < i; j++) if (cur[j] == e) { dup = 1; break; }
        if (dup) continue;
        const int16_t *s = r->succ + ((size_t)layer * r->n_experts + (uint32_t)e) * r->k;
        const float   *p = r->prob + ((size_t)layer * r->n_experts + (uint32_t)e) * r->k;
        for (uint32_t j = 0; j < r->k && s[j] >= 0; j++) { score[s[j]] += p[j]; any = 1; }
    }
    if (!any) return 0;
    for (uint32_t i = 0; sticky && i < n_sticky; i++) {
        const int32_t e = sticky[i];
        if (e < 0 || (uint32_t)e >= r->n_experts) continue;
        int dup = 0;
        for (uint32_t j = 0; j < i; j++) if (sticky[j] == e) { dup = 1; break; }
        if (!dup) score[e] += r->sticky;
    }
    uint32_t n = 0;
    for (;;) {
        int32_t best = -1;
        for (uint32_t e = 0; e < r->n_experts; e++)
            if (score[e] >= r->min_score && score[e] > 0.f && (best < 0 || score[e] > score[best])) best = (int32_t)e;
        if (best < 0 || n >= cap) break;
        if (out_conf) out_conf[n] = score[best];   /* the summed successor probability: the forecast's confidence */
        out[n++] = best;
        score[best] = -1.f;   /* taken; ties resolve to the lower id by the strict > above */
    }
    return n;
}

#endif /* DS4_JEV_ROUTER_LOGIC_H */
