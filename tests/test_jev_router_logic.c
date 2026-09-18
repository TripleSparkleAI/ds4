/* Unit tests for the jev router (ds4_jev_router_logic.h).
 *
 * Pure C99: no CUDA, no GPU, no model. Builds and runs on any host.
 *
 * Subjects: the table LOADER (a well-formed table loads, every malformed
 * shape is refused whole: bad tag, bad version, a row past the header's
 * bounds, a successor id past the expert count) and PREDICT (successor
 * probabilities sum over the current ids, the sticky bonus adds, the cap
 * and the minimum score hold, ties go to the lower id, an unknown layer or
 * an empty row returns 0 so the caller falls back to stickiness).
 *
 * Red-proven 2026-09-18 on a Mac by planted mutations: dropping the
 * `id >= ne` refusal in the loader turned test_loader_refuses_bad_shapes
 * red (2 checks: the refusal and the nothing-allocated check that follows
 * it) and nothing else; dropping the sticky bonus turned
 * test_sticky_bonus_reorders red (2 checks) and nothing else. 33/33 green
 * with both restored.
 */
#include "../ds4_jev_router_logic.h"

#include <stdio.h>
#include <string.h>

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

static const char *k_table =
    "ds4-jev-router 1\n"
    "layers 3 experts 8 k 3 sticky 0.25 min 0.2\n"
    "0 1 5 0.6 6 0.3 7 0.1\n"
    "0 2 6 0.5 3 0.5\n"
    "1 4 0 1.0\n";

static int load_str(ds4_jev_router *r, const char *s) {
    FILE *f = tmpfile();
    if (!f) return 0;
    fputs(s, f);
    rewind(f);
    const int ok = ds4_jev_router_load_file(r, f);
    fclose(f);
    return ok;
}

static void test_loader_reads_a_table(void) {
    ds4_jev_router r;
    CHECK(load_str(&r, k_table) == 1, "well-formed table loads");
    CHECK(r.n_layers == 3 && r.n_experts == 8 && r.k == 3, "header dims");
    CHECK(r.sticky == 0.25f && r.min_score == 0.2f, "header weights");
    const int16_t *s = r.succ + ((size_t)0 * 8 + 1) * 3;
    CHECK(s[0] == 5 && s[1] == 6 && s[2] == 7, "row (0,1) successors");
    const int16_t *s2 = r.succ + ((size_t)0 * 8 + 2) * 3;
    CHECK(s2[0] == 6 && s2[1] == 3 && s2[2] == -1, "short row pads with -1");
    const int16_t *s3 = r.succ + ((size_t)2 * 8 + 0) * 3;
    CHECK(s3[0] == -1, "absent row is empty");
    ds4_jev_router_free(&r);
    CHECK(r.succ == NULL && r.n_layers == 0, "free zeroes");
}

static void test_loader_refuses_bad_shapes(void) {
    ds4_jev_router r;
    CHECK(load_str(&r, "ds4-scout 1\nlayers 1 experts 2 k 1 sticky 0 min 0\n") == 0, "bad tag");
    CHECK(load_str(&r, "ds4-jev-router 2\nlayers 1 experts 2 k 1 sticky 0 min 0\n") == 0, "bad version");
    CHECK(load_str(&r, "ds4-jev-router 1\nlayers 0 experts 2 k 1 sticky 0 min 0\n") == 0, "zero layers");
    CHECK(load_str(&r, "ds4-jev-router 1\nlayers 1 experts 2 k 1 sticky 0 min 0\n3 0 1 0.5\n") == 0,
          "row layer past header");
    CHECK(load_str(&r, "ds4-jev-router 1\nlayers 1 experts 2 k 1 sticky 0 min 0\n0 5 1 0.5\n") == 0,
          "row expert past header");
    CHECK(load_str(&r, "ds4-jev-router 1\nlayers 1 experts 2 k 1 sticky 0 min 0\n0 0 9 0.5\n") == 0,
          "successor id past expert count");
    CHECK(r.succ == NULL, "a refused load leaves nothing allocated");
    CHECK(ds4_jev_router_load(&r, "/nonexistent/ds4-jev-router.table") == 0, "missing file");
}

static void test_predict_sums_successors(void) {
    ds4_jev_router r;
    CHECK(load_str(&r, k_table) == 1, "load");
    const int32_t cur[2] = {1, 2};
    int32_t out[8];
    /* scores: 5:0.6  6:0.3+0.5=0.8  7:0.1(<min)  3:0.5 -> 6, 5, 3 */
    const uint32_t n = ds4_jev_router_predict(&r, 0, cur, 2, NULL, 0, 8, out, NULL);
    CHECK(n == 3, "three above the minimum");
    CHECK(out[0] == 6 && out[1] == 5 && out[2] == 3, "best first");
    const int32_t dup[3] = {1, 1, 2};
    const uint32_t nd = ds4_jev_router_predict(&r, 0, dup, 3, NULL, 0, 8, out, NULL);
    CHECK(nd == 3 && out[0] == 6 && out[1] == 5, "a duplicated current id counts once");
    ds4_jev_router_free(&r);
}

static void test_cap_and_fallback(void) {
    ds4_jev_router r;
    CHECK(load_str(&r, k_table) == 1, "load");
    const int32_t cur[2] = {1, 2};
    int32_t out[8];
    CHECK(ds4_jev_router_predict(&r, 0, cur, 2, NULL, 0, 2, out, NULL) == 2, "cap holds");
    CHECK(out[0] == 6 && out[1] == 5, "cap keeps the best");
    const int32_t unknown[1] = {7};
    CHECK(ds4_jev_router_predict(&r, 0, unknown, 1, NULL, 0, 8, out, NULL) == 0, "empty row: 0, caller falls back");
    CHECK(ds4_jev_router_predict(&r, 9, cur, 2, NULL, 0, 8, out, NULL) == 0, "layer past table: 0");
    CHECK(ds4_jev_router_predict(&r, 0, cur, 2, NULL, 0, 0, out, NULL) == 0, "cap 0: 0");
    const int32_t bad[2] = {-1, 99};
    CHECK(ds4_jev_router_predict(&r, 0, bad, 2, NULL, 0, 8, out, NULL) == 0, "out-of-range ids ignored");
    ds4_jev_router_free(&r);
}

static void test_sticky_bonus_reorders(void) {
    ds4_jev_router r;
    CHECK(load_str(&r, k_table) == 1, "load");
    const int32_t cur[1] = {1};           /* 5:0.6 6:0.3 7:0.1 */
    const int32_t sticky[2] = {6, 7};     /* +0.25 each: 6:0.55 7:0.35 */
    int32_t out[8];
    const uint32_t n = ds4_jev_router_predict(&r, 0, cur, 1, sticky, 2, 8, out, NULL);
    CHECK(n == 3, "sticky lifts 7 over the minimum");
    CHECK(out[0] == 5 && out[1] == 6 && out[2] == 7, "sticky reorders below the strong successor");
    /* sticky alone, with no successor row for the current id, still returns 0:
     * the table must have something to say before the bonus applies. */
    const int32_t none[1] = {7};
    CHECK(ds4_jev_router_predict(&r, 0, none, 1, sticky, 2, 8, out, NULL) == 0, "no row: no guess");
    ds4_jev_router_free(&r);
}

static void test_ties_go_to_the_lower_id(void) {
    ds4_jev_router r;
    CHECK(load_str(&r, k_table) == 1, "load");
    const int32_t cur[1] = {2};           /* 6:0.5 3:0.5 */
    int32_t out[8];
    const uint32_t n = ds4_jev_router_predict(&r, 0, cur, 1, NULL, 0, 8, out, NULL);
    CHECK(n == 2 && out[0] == 3 && out[1] == 6, "equal scores: lower id first");
    float conf[8] = {0};
    const int32_t cur2[2] = {1, 2};   /* 6:0.8 5:0.6 3:0.5 */
    const uint32_t n2 = ds4_jev_router_predict(&r, 0, cur2, 2, NULL, 0, 8, out, conf);
    CHECK(n2 == 3 && conf[0] > 0.79f && conf[0] < 0.81f && conf[1] > 0.59f && conf[2] > 0.49f,
          "confidence out is the summed successor probability, best first");
    ds4_jev_router_free(&r);
}

int main(void) {
    RUN(test_loader_reads_a_table);
    RUN(test_loader_refuses_bad_shapes);
    RUN(test_predict_sums_successors);
    RUN(test_cap_and_fallback);
    RUN(test_sticky_bonus_reorders);
    RUN(test_ties_go_to_the_lower_id);
    fprintf(stderr, "jev router logic: %d/%d checks passed\n", g_total - g_failed, g_total);
    return g_failed ? 1 : 0;
}
