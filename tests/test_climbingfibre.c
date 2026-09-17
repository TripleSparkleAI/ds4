/*
 * tests/test_climbingfibre.c - the four cache laws, checked rather than
 * asserted.
 *
 * This test does not need a model, a GPU or a corpus.  Every claim the
 * CLIMBINGFIBRE design makes about SAFETY is a claim about a small piece of
 * control flow, so it is testable on the host, in a second, with no box time.
 * That matters here: the CUDA build gate is owed (the Spark is busy), and a
 * safety property that could only be checked behind a busy build gate would be
 * prose.  This is the version that runs.
 *
 *   gcc -O2 -I. -o tests/test_climbingfibre tests/test_climbingfibre.c \
 *       ds4_climbingfibre.c
 *   ./tests/test_climbingfibre
 *
 * WHAT IS CHECKED
 *   1. LAW 1, structurally.  The module names no part of the deep path and
 *      cannot reach one: a source guard strips comments and fails if any of
 *      the deep path's names appear in the code.  The handle is opaque, so no
 *      caller can even see inside it.
 *   2. LAW 2, a miss is free.  A cold table and a switched-off table take the
 *      same path and produce the same stream.
 *   3. LAW 3, correctness never depends on the cache.  A decode driven by a
 *      scripted target emits the SAME token stream with the cache off, cold,
 *      and poisoned with entries that are deliberately wrong.  Only the amount
 *      of work changes.  A wrong entry wastes work; it cannot corrupt anything.
 *   4. LAW 4, learn from the misses.  A rejection teaches the table, and the
 *      very next proposal at that context is the target's own token.
 *   5. THE RETENTION POLICY is a parameter, not a constant: the same store is
 *      served, withheld, or halved depending on the horizon and the decay knob.
 *   6. THE SWITCH defaults OFF and OFF means INERT: no allocation, no learning,
 *      no proposal.
 */

#include "ds4_climbingfibre.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_fail;
static int g_checks;

static void ok(bool cond, const char *what) {
    g_checks++;
    if (!cond) {
        g_fail++;
        fprintf(stderr, "FAIL  %s\n", what);
    } else {
        fprintf(stderr, "pass  %s\n", what);
    }
}

/* --------------------------------------------------- 1. the source guard --- */

/* Strip C comments so that a comment ABOUT the deep path is not mistaken for a
 * reference TO it.  Everything the module says about the model is in prose; the
 * point of this check is that the CODE never says it. */
static char *strip_comments(const char *src) {
    size_t n = strlen(src);
    char *out = malloc(n + 1);
    size_t w = 0;
    int in_block = 0, in_line = 0, in_str = 0, in_chr = 0, esc = 0;
    for (size_t i = 0; i < n; i++) {
        char ch = src[i];
        if (in_block) {
            if (ch == '*' && i + 1 < n && src[i + 1] == '/') {
                in_block = 0;
                i++;
            }
            out[w++] = ' ';
            continue;
        }
        if (in_line) {
            if (ch == '\n') { in_line = 0; out[w++] = '\n'; }
            else out[w++] = ' ';
            continue;
        }
        if (in_str || in_chr) {
            out[w++] = ch;
            if (esc) esc = 0;
            else if (ch == '\\') esc = 1;
            else if ((in_str && ch == '"') || (in_chr && ch == '\'')) {
                in_str = in_chr = 0;
            }
            continue;
        }
        if (ch == '/' && i + 1 < n && src[i + 1] == '*') {
            in_block = 1;
            i++;
            out[w++] = ' ';
            continue;
        }
        if (ch == '/' && i + 1 < n && src[i + 1] == '/') {
            in_line = 1;
            i++;
            out[w++] = ' ';
            continue;
        }
        if (ch == '"') in_str = 1;
        if (ch == '\'') in_chr = 1;
        out[w++] = ch;
    }
    out[w] = '\0';
    return out;
}

static char *read_all(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz < 0) { fclose(f); return NULL; }
    char *buf = malloc((size_t)sz + 1);
    size_t got = fread(buf, 1, (size_t)sz, f);
    buf[got] = '\0';
    fclose(f);
    return buf;
}

static void check_no_deep_path_name(const char *path) {
    static const char *forbidden[] = {
        "ds4_session", "ds4_engine", "ds4_gpu", "ds4_tensor",
        "logits", "checkpoint", "kv_cache", "kvstore",
        "tensor", "cuda", "metal", "vocab",
        "ggml", "gguf", "weights",
    };
    char *raw = read_all(path);
    if (!raw) {
        char msg[256];
        snprintf(msg, sizeof(msg), "source guard: could not read %s", path);
        ok(false, msg);
        return;
    }
    char *code = strip_comments(raw);
    int hits = 0;
    for (size_t i = 0; i < sizeof(forbidden) / sizeof(forbidden[0]); i++) {
        if (strstr(code, forbidden[i])) {
            hits++;
            fprintf(stderr, "      %s mentions '%s' in CODE\n",
                    path, forbidden[i]);
        }
    }
    char msg[256];
    snprintf(msg, sizeof(msg),
             "LAW 1  %s names no part of the deep path in code", path);
    ok(hits == 0, msg);
    free(code);
    free(raw);
}

/* --------------------------------------------------------- 2. the switch --- */

static void test_switch_defaults_off(void) {
    unsetenv("DS4_CLIMBINGFIBRE");
    ds4_climbingfibre *c = ds4_climbingfibre_create();
    ok(c != NULL, "create() always returns a handle");
    ok(!ds4_climbingfibre_enabled(c),
       "LAW 6  the switch defaults OFF (DS4_CLIMBINGFIBRE unset)");

    const int32_t key[4] = { 11, 22, 33, 44 };
    int32_t out = -12345;
    uint32_t out_len = 99;
    ds4_climbingfibre_learn(c, key, 4, 7, 8);   /* must do nothing at all */
    bool served = ds4_climbingfibre_propose(c, key, 4, &out, 1, &out_len);
    ok(!served && out_len == 0,
       "LAW 6  OFF means never proposes");
    ok(out == -12345, "LAW 6  OFF writes nothing through the out pointer");
    ds4_climbingfibre_tick(c);
    ok(ds4_climbingfibre_step(c) == 0, "LAW 6  OFF does not even tick");

    /* The census is the only observable, and it must show an inert table. */
    FILE *f = tmpfile();
    ok(f != NULL, "tmpfile for the census");
    if (f) {
        ds4_climbingfibre_census(c, f);
        rewind(f);
        char buf[1024];
        size_t got = fread(buf, 1, sizeof(buf) - 1, f);
        buf[got] = '\0';
        ok(strstr(buf, "enabled=0") != NULL, "LAW 6  census says enabled=0");
        ok(strstr(buf, "learns=0") != NULL && strstr(buf, "served=0") != NULL,
           "LAW 6  OFF learns nothing and serves nothing");
        fclose(f);
    }
    ds4_climbingfibre_free(c);
}

/* ------------------------------------- 3/4. the laws, on a scripted decode --- */

/*
 * A scripted target.  The next token is a pure function of the last TWO tokens,
 * so a 4-gram key determines the next token exactly: the table's key is
 * informative by construction and a correct entry is always correct.
 */
#define CF_TEST_VOCAB 61
#define CF_TEST_LEN   4000
#define CF_KEY_N      4

static int scripted_next(const int32_t *hist, int len) {
    int a = len >= 1 ? hist[len - 1] : 0;
    int b = len >= 2 ? hist[len - 2] : 0;
    return (a * 3 + b * 7 + 5) % CF_TEST_VOCAB;
}

static void tail_key(const int32_t *hist, int len, int32_t *key) {
    for (int i = 0; i < CF_KEY_N; i++) {
        int idx = len - CF_KEY_N + i;
        key[i] = idx >= 0 ? hist[idx] : 0;
    }
}

enum { CF_MODE_OFF = 0, CF_MODE_COLD = 1, CF_MODE_POISON = 2 };

/*
 * One simulated speculative decode.
 *
 * A round emits the target's own token (the "seed"), then offers the table's
 * proposal for the position after it.  The scripted target is the verifier: if
 * the proposal equals the token the target itself would produce, it is
 * committed as an extra token; if it does not, it is REFUSED and the refusal is
 * what the table learns from.  The emitted stream is therefore always the
 * target's stream, whatever the table says.
 */
static void run_decode(int mode, int32_t *stream_out, int *n_out,
                       long *extra_out, long *learns_out, long *own_rejects,
                       long *served_out, uint32_t cap) {
    if (mode == CF_MODE_OFF) unsetenv("DS4_CLIMBINGFIBRE");
    else setenv("DS4_CLIMBINGFIBRE", "1", 1);
    setenv("DS4_CLIMBINGFIBRE_HORIZON", "forever", 1);
    setenv("DS4_CLIMBINGFIBRE_DECAY", "none", 1);
    setenv("DS4_CLIMBINGFIBRE_NGRAM", "4", 1);
    {
        char capbuf[32];
        snprintf(capbuf, sizeof(capbuf), "%u", cap);
        setenv("DS4_CLIMBINGFIBRE_CAP", capbuf, 1);
    }

    ds4_climbingfibre *cache = ds4_climbingfibre_create();
    if (!cache) { *n_out = 0; return; }

    int32_t truth[CF_TEST_LEN];
    int32_t hist[CF_TEST_LEN];
    for (int i = 0; i < CF_TEST_LEN; i++) {
        truth[i] = scripted_next(hist, i);
        hist[i] = truth[i];
    }

    if (mode == CF_MODE_POISON) {
        /* Pre-poison every key the stream will visit with a WRONG candidate,
         * stored with eight times the weight a single correction carries, so
         * the table is confidently and durably wrong for the whole run rather
         * than being outvoted by the first correction that arrives. */
        for (int rep = 0; rep < 8; rep++) {
            for (int i = CF_KEY_N; i < CF_TEST_LEN; i++) {
                int32_t key[CF_KEY_N];
                tail_key(hist, i, key);
                int wrong = (truth[i] + 1) % CF_TEST_VOCAB;
                ds4_climbingfibre_learn(cache, key, CF_KEY_N, wrong, wrong);
            }
        }
        for (int i = 0; i < 3; i++) ds4_climbingfibre_tick(cache);
    }

    int pos = 0;
    long extra = 0, serves = 0;
    while (pos < CF_TEST_LEN) {
        int32_t key[CF_KEY_N];
        int emitted = pos;
        stream_out[emitted] = truth[pos];
        pos++;

        if (pos >= CF_TEST_LEN) break;

        tail_key(truth, pos, key);
        ds4_climbingfibre_tick(cache);
        int32_t cand = -1;
        uint32_t cand_len = 0;
        bool have = ds4_climbingfibre_propose(cache, key, CF_KEY_N,
                                              &cand, 1, &cand_len);
        if (have) serves++;
        int32_t drafted = have && cand_len == 1 ? cand : -1;

        if (drafted >= 0 && drafted == truth[pos]) {
            /* ACCEPT: the target's own verifier agrees, so one more token is
             * committed in this round.  The token committed is the target's. */
            stream_out[pos] = truth[pos];
            pos++;
            extra++;
        } else {
            /* REJECT, or no draft at all.  The target's token is emitted and
             * the refusal (if there was one) is written back. */
            ds4_climbingfibre_learn(cache, key, CF_KEY_N, drafted, truth[pos]);
        }
    }
    *n_out = CF_TEST_LEN;
    *extra_out = extra;
    if (served_out) *served_out = serves;

    FILE *f = tmpfile();
    if (f) {
        ds4_climbingfibre_census(cache, f);
        rewind(f);
        char buf[2048];
        size_t got = fread(buf, 1, sizeof(buf) - 1, f);
        buf[got] = '\0';
        const char *p = strstr(buf, "learns=");
        if (learns_out) *learns_out = p ? atol(p + 7) : -1;
        p = strstr(buf, "own_rejected=");
        if (own_rejects) *own_rejects = p ? atol(p + 13) : -1;
        fclose(f);
    }
    ds4_climbingfibre_free(cache);
}

static void test_streams_are_identical(void) {
    /* 65536 keys: far above this scripted target's key population, so the table
     * can actually hold what it learns. */
    const uint32_t CF_TEST_CAP = 65536u;
    /* 64 keys: far BELOW it, so the table thrashes and every lookup misses. */
    const uint32_t CF_THRASH_CAP = 64u;

    int32_t *a = malloc(sizeof(int32_t) * CF_TEST_LEN);
    int32_t *b = malloc(sizeof(int32_t) * CF_TEST_LEN);
    int32_t *c = malloc(sizeof(int32_t) * CF_TEST_LEN);
    int32_t *d = malloc(sizeof(int32_t) * CF_TEST_LEN);
    int na = 0, nb = 0, nc = 0, nd = 0;
    long extra_a = 0, extra_b = 0, extra_c = 0, extra_d = 0;
    long learns = 0, own = 0, served = 0;

    run_decode(CF_MODE_OFF, a, &na, &extra_a, &learns, &own, &served,
               CF_TEST_CAP);
    run_decode(CF_MODE_COLD, b, &nb, &extra_b, &learns, &own, &served,
               CF_TEST_CAP);
    run_decode(CF_MODE_POISON, c, &nc, &extra_c, &learns, &own, &served,
               CF_TEST_CAP);
    run_decode(CF_MODE_COLD, d, &nd, &extra_d, &learns, &own, &served,
               CF_THRASH_CAP);

    ok(na == CF_TEST_LEN && nb == CF_TEST_LEN && nc == CF_TEST_LEN &&
       nd == CF_TEST_LEN,
       "all four runs reached the same length");
    ok(memcmp(a, b, sizeof(int32_t) * CF_TEST_LEN) == 0,
       "LAW 2  ON-but-cold emits the SAME stream as OFF");
    ok(memcmp(a, c, sizeof(int32_t) * CF_TEST_LEN) == 0,
       "LAW 3  ON-and-POISONED emits the SAME stream as OFF");
    ok(memcmp(b, c, sizeof(int32_t) * CF_TEST_LEN) == 0,
       "LAW 3  a wrong entry changes no output");
    ok(memcmp(a, d, sizeof(int32_t) * CF_TEST_LEN) == 0,
       "LAW 3  a table too small to hold its own keys emits the SAME stream "
       "as OFF: the failure mode of an undersized front cache is inertness, "
       "not corruption");

    ok(extra_a == 0, "OFF commits no extra tokens (no speculation at all)");
    ok(extra_b > 0,
       "LAW 4  the table earns tokens: a cold cache warms up and starts "
       "committing extras");
    ok(extra_c == 0,
       "LAW 3  the poisoned table commits exactly zero extras: a table that is "
       "always wrong wastes work and nothing else");
    ok(extra_d == 0,
       "CAPACITY  the 64-key table commits zero extras: it thrashes, so it "
       "buys nothing.  Capacity is a real cliff and it is a knob, not a "
       "constant");
    ok(extra_b > extra_c,
       "the poisoned-vs-learned gap is the effect size, and it is the only "
       "thing the poison changes");

    fprintf(stderr,
            "      simulated: off=+%ld  cold(65536 keys)=+%ld  "
            "poisoned=+%ld  cold(64 keys)=+%ld  extra tokens over %d\n",
            extra_a, extra_b, extra_c, extra_d, CF_TEST_LEN);

    free(a); free(b); free(c); free(d);
}

/*
 * ATTRIBUTION WITH DRAFTS IN FLIGHT.
 *
 * This is the one thing the asynchronous-drafter reading (Saguaro, 2026) really
 * changes.  Two proposals go out on different contexts; the REJECTION for the
 * FIRST one arrives only after the second has been made.  A single "the
 * proposal I just made" slot would credit that rejection to the wrong proposal
 * and report the engine's own draft as ours.  The ring must find the right one,
 * and an unrelated rejection must still be counted as foreign.
 */
static void test_attribution_with_drafts_in_flight(void) {
    unsetenv("DS4_CLIMBINGFIBRE");
    setenv("DS4_CLIMBINGFIBRE", "1", 1);
    setenv("DS4_CLIMBINGFIBRE_HORIZON", "forever", 1);
    setenv("DS4_CLIMBINGFIBRE_DECAY", "none", 1);
    setenv("DS4_CLIMBINGFIBRE_NGRAM", "4", 1);
    setenv("DS4_CLIMBINGFIBRE_CAP", "1024", 1);
    ds4_climbingfibre *c = ds4_climbingfibre_create();

    int32_t k1[4] = { 1, 1, 1, 1 };
    int32_t k2[4] = { 2, 2, 2, 2 };
    /* Seed with a CONFIRMATION so the seeding itself is not counted as a
     * rejection; the census below is about the two events that follow. */
    ds4_climbingfibre_learn(c, k1, 4, 100, 100);
    ds4_climbingfibre_learn(c, k2, 4, 200, 200);

    int32_t out = -1;
    uint32_t out_len = 0;
    ok(ds4_climbingfibre_propose(c, k1, 4, &out, 1, &out_len) && out == 100,
       "IN-FLIGHT  proposal 1 served");
    ds4_climbingfibre_tick(c);
    ok(ds4_climbingfibre_propose(c, k2, 4, &out, 1, &out_len) && out == 200,
       "IN-FLIGHT  proposal 2 served while 1 is still unaccounted for");

    /* The rejection for proposal 1 lands now, AFTER 2 was made. */
    ds4_climbingfibre_learn(c, k1, 4, 100, 7);
    /* And one the engine made itself, which must not be credited to us. */
    ds4_climbingfibre_learn(c, k2, 4, 999, 8);

    FILE *f = tmpfile();
    ok(f != NULL, "tmpfile for the attribution census");
    if (f) {
        ds4_climbingfibre_census(c, f);
        rewind(f);
        char buf[2048];
        size_t got = fread(buf, 1, sizeof(buf) - 1, f);
        buf[got] = '\0';
        const char *p = strstr(buf, "own_rejected=");
        long own = p ? atol(p + 13) : -1;
        p = strstr(buf, "foreign_rejected=");
        long foreign = p ? atol(p + 17) : -1;
        ok(own == 1,
           "IN-FLIGHT  the out-of-order rejection was attributed to OUR "
           "proposal");
        ok(foreign == 1,
           "IN-FLIGHT  the engine's own refused draft was NOT claimed by us");
        fclose(f);
    }
    ds4_climbingfibre_free(c);
}

static void test_learn_from_the_miss(void) {
    unsetenv("DS4_CLIMBINGFIBRE");
    setenv("DS4_CLIMBINGFIBRE", "1", 1);
    setenv("DS4_CLIMBINGFIBRE_HORIZON", "forever", 1);
    setenv("DS4_CLIMBINGFIBRE_DECAY", "none", 1);
    setenv("DS4_CLIMBINGFIBRE_NGRAM", "4", 1);
    ds4_climbingfibre *c = ds4_climbingfibre_create();

    const int32_t key[4] = { 3, 5, 8, 13 };
    int32_t out = -1;
    uint32_t out_len = 0;
    ok(!ds4_climbingfibre_propose(c, key, 4, &out, 1, &out_len),
       "LAW 2  a cold key MISSES, and a miss is free");

    /* A rejection: the shortcut guessed 7, the target produced 9. */
    ds4_climbingfibre_learn(c, key, 4, 7, 9);
    out_len = 0;
    bool served = ds4_climbingfibre_propose(c, key, 4, &out, 1, &out_len);
    ok(served && out_len == 1 && out == 9,
       "LAW 4  the rejection wrote the target's token back, and it is served");

    /* Now let the SAME key be wrong once: the refused guess is what the table
     * had been offering, so it must be depressed, not merely outvoted. */
    ds4_climbingfibre_learn(c, key, 4, 9, 4);
    out_len = 0;
    served = ds4_climbingfibre_propose(c, key, 4, &out, 1, &out_len);
    ok(served && out == 4,
       "LAW 4  a refused candidate is depressed and the correction takes over");

    /* And a demotion that reaches zero drops the entry rather than leaving a
     * confident zero-weight row behind. */
    int32_t k2[4] = { 1, 2, 3, 4 };
    ds4_climbingfibre_learn(c, k2, 4, 40, 41);   /* store 41, weight 1 */
    ds4_climbingfibre_learn(c, k2, 4, 41, 42);   /* store 42; 41 -> weight 0 */
    out_len = 0;
    served = ds4_climbingfibre_propose(c, k2, 4, &out, 1, &out_len);
    ok(served && out == 42,
       "LAW 4  a zero-weight candidate is dropped, not served");
    ds4_climbingfibre_free(c);
}

static void test_retention_is_a_parameter(void) {
    const int32_t key[4] = { 100, 200, 300, 400 };
    int32_t out = -1;
    uint32_t out_len = 0;

    /* horizon = 10 steps, decay = hard: past the horizon the entry is gone. */
    setenv("DS4_CLIMBINGFIBRE", "1", 1);
    setenv("DS4_CLIMBINGFIBRE_HORIZON", "10", 1);
    setenv("DS4_CLIMBINGFIBRE_DECAY", "hard", 1);
    setenv("DS4_CLIMBINGFIBRE_NGRAM", "4", 1);
    ds4_climbingfibre *hard = ds4_climbingfibre_create();
    ds4_climbingfibre_learn(hard, key, 4, 1, 2);
    for (int i = 0; i < 5; i++) ds4_climbingfibre_tick(hard);
    out_len = 0;
    ok(ds4_climbingfibre_propose(hard, key, 4, &out, 1, &out_len) && out == 2,
       "RETENTION  inside the horizon the entry is served");
    for (int i = 0; i < 10; i++) ds4_climbingfibre_tick(hard);
    out_len = 0;
    ok(!ds4_climbingfibre_propose(hard, key, 4, &out, 1, &out_len),
       "RETENTION  past the horizon with decay=hard the entry is gone");
    ds4_climbingfibre_free(hard);

    /* The same store under decay = none is still there. */
    setenv("DS4_CLIMBINGFIBRE_DECAY", "none", 1);
    ds4_climbingfibre *never = ds4_climbingfibre_create();
    ds4_climbingfibre_learn(never, key, 4, 1, 2);
    for (int i = 0; i < 100000; i++) ds4_climbingfibre_tick(never);
    out_len = 0;
    ok(ds4_climbingfibre_propose(never, key, 4, &out, 1, &out_len) && out == 2,
       "RETENTION  decay=none still serves after 100000 steps: the horizon "
       "is a parameter, not a constant");
    ds4_climbingfibre_free(never);

    /* decay = half: one correction is halved away, two survive as one. */
    setenv("DS4_CLIMBINGFIBRE_HORIZON", "10", 1);
    setenv("DS4_CLIMBINGFIBRE_DECAY", "half", 1);
    ds4_climbingfibre *half = ds4_climbingfibre_create();
    ds4_climbingfibre_learn(half, key, 4, 1, 2);   /* weight 1 */
    for (int i = 0; i < 12; i++) ds4_climbingfibre_tick(half);
    out_len = 0;
    ok(!ds4_climbingfibre_propose(half, key, 4, &out, 1, &out_len),
       "RETENTION  decay=half: a weight of 1 halves to 0 and is withheld");
    ds4_climbingfibre_free(half);

    ds4_climbingfibre *half2 = ds4_climbingfibre_create();
    ds4_climbingfibre_learn(half2, key, 4, 1, 2);  /* weight 1 */
    ds4_climbingfibre_learn(half2, key, 4, 1, 2);  /* weight 2 */
    for (int i = 0; i < 12; i++) ds4_climbingfibre_tick(half2);
    out_len = 0;
    ok(ds4_climbingfibre_propose(half2, key, 4, &out, 1, &out_len) && out == 2,
       "RETENTION  decay=half: a weight of 2 halves to 1 and survives");
    ds4_climbingfibre_free(half2);

    /* The named horizons are budgets, and they are ordered. */
    unsetenv("DS4_CLIMBINGFIBRE_HORIZON");
    unsetenv("DS4_CLIMBINGFIBRE_DECAY");
    setenv("DS4_CLIMBINGFIBRE_STEPS_PER_DAY", "1000", 1);
    setenv("DS4_CLIMBINGFIBRE_STEPS_PER_SESSION", "10", 1);
    static const char *names[4] = { "session", "day", "week", "forever" };
    uint64_t seen[4] = { 0, 0, 0, 0 };
    for (int i = 0; i < 4; i++) {
        setenv("DS4_CLIMBINGFIBRE_HORIZON", names[i], 1);
        ds4_climbingfibre *c = ds4_climbingfibre_create();
        FILE *f = tmpfile();
        ds4_climbingfibre_census(c, f);
        rewind(f);
        char buf[1024];
        size_t got = fread(buf, 1, sizeof(buf) - 1, f);
        buf[got] = '\0';
        const char *p = strstr(buf, "horizon=");
        seen[i] = p ? strtoull(p + 8, NULL, 10) : 0;
        fclose(f);
        ds4_climbingfibre_free(c);
    }
    ok(seen[0] == 10 && seen[1] == 1000 && seen[2] == 7000 &&
       seen[3] == UINT64_MAX,
       "RETENTION  session < day < week < forever are real budgets");
    fprintf(stderr,
            "      horizons (steps): session=%llu day=%llu week=%llu forever=%llu\n",
            (unsigned long long)seen[0], (unsigned long long)seen[1],
            (unsigned long long)seen[2], (unsigned long long)seen[3]);
}

/* Read one `name=` field out of a census line.  The handle is opaque by LAW 1,
 * so the census is the only way a test may see a counter - which is the right
 * shape: a number a test can read is a number an operator can read. */
static uint64_t census_u64(ds4_climbingfibre *c, const char *field) {
    FILE *f = tmpfile();
    if (!f) return UINT64_MAX;
    ds4_climbingfibre_census(c, f);
    rewind(f);
    char buf[2048];
    size_t got = fread(buf, 1, sizeof(buf) - 1, f);
    buf[got] = '\0';
    fclose(f);
    const char *p = strstr(buf, field);
    if (!p) return UINT64_MAX;
    return strtoull(p + strlen(field), NULL, 10);
}

/* --------------------------------- 7. decay that reaches zero RETIRES --- */

/*
 * THE DEFECT THIS PINS (R5 finding C1).  Under the default decay `half`, an
 * entry past the horizon was halved and ITS CLOCK WAS RESET.  Halving a weight
 * of 1 gives 0, so that probe was refused - and on the very next probe the
 * entry's age was 0, inside the horizon, so it was served again, with a weight
 * of zero, for ever.  Nothing ever aged out of the table under the default
 * policy; the halving was a one-step skip wearing a retention policy's name.
 *
 * The control is the SECOND probe, not the first: the unfixed code passes the
 * first one.  A test that stopped at "the halved entry is withheld" is exactly
 * the test that shipped, and it could not see this.
 */
static void test_decay_to_nothing_retires(void) {
    const int32_t key[4] = { 11, 22, 33, 44 };
    int32_t out = -1;
    uint32_t out_len = 0;

    setenv("DS4_CLIMBINGFIBRE", "1", 1);
    setenv("DS4_CLIMBINGFIBRE_NGRAM", "4", 1);
    setenv("DS4_CLIMBINGFIBRE_HORIZON", "10", 1);
    setenv("DS4_CLIMBINGFIBRE_DECAY", "half", 1);

    ds4_climbingfibre *c = ds4_climbingfibre_create();
    ds4_climbingfibre_learn(c, key, 4, 1, 2);        /* weight 1 */
    ok(census_u64(c, "live=") == 1,
       "RETIRE  the entry is live before the horizon passes");
    for (int i = 0; i < 12; i++) ds4_climbingfibre_tick(c);

    out_len = 0;
    ok(!ds4_climbingfibre_propose(c, key, 4, &out, 1, &out_len),
       "RETIRE  a weight of 1 halves to 0 and is withheld");

    /* THE CONTROL.  One more probe, at the same step or any later one. */
    out = -1;
    out_len = 0;
    bool served_again = ds4_climbingfibre_propose(c, key, 4, &out, 1, &out_len);
    ok(!served_again,
       "RETIRE  and it is STILL withheld on the next probe: an entry decay "
       "consumed is gone, not served with a weight of zero");
    if (served_again) {
        fprintf(stderr, "      served token %d again with out_len=%u\n",
                out, out_len);
    }

    ok(census_u64(c, "live=") == 0,
       "RETIRE  the retired entry is off the live count");

    /* And it is still gone many steps later - not merely skipped once more. */
    for (int i = 0; i < 50; i++) ds4_climbingfibre_tick(c);
    out_len = 0;
    ok(!ds4_climbingfibre_propose(c, key, 4, &out, 1, &out_len),
       "RETIRE  still gone 50 steps later");
    ds4_climbingfibre_free(c);

    /* THE VACUITY CONTROL, on its own case: the same table, the same probes,
     * an entry whose weight decay has NOT consumed is served.  Without this,
     * "returns false" would also pass for a module that served nothing. */
    ds4_climbingfibre *keep = ds4_climbingfibre_create();
    ds4_climbingfibre_learn(keep, key, 4, 1, 2);
    ds4_climbingfibre_learn(keep, key, 4, 1, 2);     /* weight 2 */
    for (int i = 0; i < 12; i++) ds4_climbingfibre_tick(keep);
    out = -1;
    out_len = 0;
    ok(ds4_climbingfibre_propose(keep, key, 4, &out, 1, &out_len) && out == 2,
       "RETIRE  vacuity control: a weight of 2 halves to 1 and is still served "
       "on both probes");
    out = -1;
    out_len = 0;
    ok(ds4_climbingfibre_propose(keep, key, 4, &out, 1, &out_len) && out == 2,
       "RETIRE  vacuity control: and again on the probe after that");
    ds4_climbingfibre_free(keep);
}

/* ------------------------------ 8. a long session does not degrade --- */

/*
 * THE DEFECT THIS PINS (R5 finding C2).  A dropped entry becomes a TOMB so the
 * probe chain through it survives, and ONLY EMPTY ends a chain.  Nothing ever
 * turned a tombstone back into EMPTY and there was no rehash, so the number of
 * non-EMPTY slots was monotone: once a session had inserted `cap` distinct
 * (key, token) pairs - 16384 by default, well inside a 65536-step session -
 * every miss walked the WHOLE table, on every speculative cycle, for the rest
 * of the session.
 *
 * The assertion is on the census `probes=` counter rather than on a clock, so
 * it is deterministic and says what it means: probe steps walked per lookup.
 * At the 3/4 live load the reclaimer holds, linear probing costs about 8 steps
 * for an unsuccessful search; the bound below is generous against that and
 * nowhere near the `cap` the unfixed code pays.
 */
static void test_long_session_does_not_degrade(void) {
    const uint32_t cap = 1024;
    setenv("DS4_CLIMBINGFIBRE", "1", 1);
    setenv("DS4_CLIMBINGFIBRE_NGRAM", "4", 1);
    setenv("DS4_CLIMBINGFIBRE_DECAY", "none", 1);
    setenv("DS4_CLIMBINGFIBRE_HORIZON", "forever", 1);
    setenv("DS4_CLIMBINGFIBRE_CAP", "1024", 1);

    ds4_climbingfibre *c = ds4_climbingfibre_create();
    ok(census_u64(c, "cap=") == cap, "DEGRADE  the small cap took effect");

    /* Churn: four times the table's capacity in distinct keys, which is what a
     * session longer than the table looks like. */
    for (int32_t i = 0; i < 4096; i++) {
        int32_t key[4] = { i, i + 1, i + 2, i + 3 };
        ds4_climbingfibre_learn(c, key, 4, 7, 8 + i);
        ds4_climbingfibre_tick(c);
    }

    uint64_t used = census_u64(c, "used=");
    uint64_t rehashes = census_u64(c, "rehashes=");
    ok(rehashes > 0, "DEGRADE  the tombstone sweep ran during the churn");
    ok(used < cap,
       "DEGRADE  the table still has EMPTY slots after 4x cap distinct "
       "inserts: non-EMPTY is no longer monotone");
    fprintf(stderr, "      after churn: used=%llu of cap=%u, rehashes=%llu\n",
            (unsigned long long)used, cap, (unsigned long long)rehashes);

    /* THE MEASUREMENT.  200 lookups on keys the table has never seen - the
     * miss that the unfixed code pays `cap` probes for. */
    const int probe_count = 200;
    uint64_t before = census_u64(c, "probes=");
    for (int32_t i = 0; i < probe_count; i++) {
        int32_t key[4] = { 900000 + i, 4, 5, 6 };
        int32_t out = -1;
        uint32_t out_len = 0;
        ds4_climbingfibre_propose(c, key, 4, &out, 1, &out_len);
    }
    uint64_t after = census_u64(c, "probes=");
    double per_miss = (double)(after - before) / (double)probe_count;
    fprintf(stderr, "      probes per miss: %.2f (cap=%u)\n", per_miss, cap);

    ok(per_miss < 64.0,
       "DEGRADE  a miss costs a bounded walk, not the whole table");

    /* The vacuity control, its own case: the counter is real and moves. */
    ok(after > before,
       "DEGRADE  vacuity control: the probe counter moved at all");

    ds4_climbingfibre_free(c);
    unsetenv("DS4_CLIMBINGFIBRE_CAP");
    unsetenv("DS4_CLIMBINGFIBRE_HORIZON");
    unsetenv("DS4_CLIMBINGFIBRE_DECAY");
}

static void test_no_allocation_when_off(void) {
    unsetenv("DS4_CLIMBINGFIBRE");
    ds4_climbingfibre *c = ds4_climbingfibre_create();
    FILE *f = tmpfile();
    ok(f != NULL, "tmpfile for the off-mode census");
    if (f) {
        ds4_climbingfibre_census(c, f);
        rewind(f);
        char buf[1024];
        size_t got = fread(buf, 1, sizeof(buf) - 1, f);
        buf[got] = '\0';
        ok(strstr(buf, "cap=0") != NULL && strstr(buf, "live=0") != NULL,
           "LAW 6  OFF allocates no table at all");
        fclose(f);
    }
    ds4_climbingfibre_free(c);
}

int main(void) {
    fprintf(stderr, "== CLIMBINGFIBRE: the four cache laws ==\n\n");

    check_no_deep_path_name("ds4_climbingfibre.h");
    check_no_deep_path_name("ds4_climbingfibre.c");

    /* The handle is opaque: no caller, and not even the test, can see a field
     * of it, so there is no way to write a table entry from outside. */
    {
        char *hdr = read_all("ds4_climbingfibre.h");
        char *code = hdr ? strip_comments(hdr) : NULL;
        ok(code && strstr(code, "typedef struct ds4_climbingfibre "
                                "ds4_climbingfibre;") != NULL,
           "LAW 1  the handle is a forward declaration");
        ok(code && strstr(code, "struct ds4_climbingfibre {") == NULL,
           "LAW 1  the handle's layout is not exposed");
        free(code);
        free(hdr);
    }

    test_switch_defaults_off();
    test_no_allocation_when_off();
    test_streams_are_identical();
    test_attribution_with_drafts_in_flight();
    test_learn_from_the_miss();
    test_retention_is_a_parameter();
    test_decay_to_nothing_retires();
    test_long_session_does_not_degrade();

    fprintf(stderr, "\n%d checks, %d failed\n", g_checks, g_fail);
    return g_fail == 0 ? 0 : 1;
}