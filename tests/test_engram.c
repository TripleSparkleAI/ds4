#define _DARWIN_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#include "ds4_engram.h"

#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static ds4_engram_layout layout(void) {
    static uint32_t map[256];
    for (int i = 0; i < 256; i++) map[i] = i / 2;
    ds4_engram_layout l = {.token_map = map, .vocab_size = 256,
                           .compressed_vocab_size = 128, .pad_id = 1};
    for (int layer = 0; layer < 2; layer++) {
        for (int j = 0; j < 4; j++)
            l.multipliers[layer][j] = 35184372088831ull - 2 * (j + 4 * layer);
        for (int j = 0; j < 24; j++) {
            l.primes[layer][j] = 16000057;
            l.rows[layer] += l.primes[layer][j];
        }
    }
    return l;
}

/* Independent full-history formulation: no rolling state or reuse of hashes. */
static void reference(const ds4_engram_layout *l, const int *tokens,
                      const uint8_t *mask, int count, uint32_t *out) {
    for (int pos = 0; pos < count; pos++) {
        for (int layer = 0; layer < 2; layer++) {
            uint32_t offset = 0;
            for (int col = 0; col < 24; col++) {
                __uint128_t hash = 0;
                bool blocked = false;
                for (int shift = 0; shift < col / 8 + 2; shift++) {
                    int p = pos - shift;
                    blocked |= p < 0 || (mask && !mask[p]);
                    uint32_t id = blocked ? l->pad_id : l->token_map[tokens[p]];
                    hash ^= (__uint128_t)id * l->multipliers[layer][shift];
                }
                *out++ = (uint32_t)(hash % l->primes[layer][col]) + offset;
                offset += l->primes[layer][col];
            }
        }
    }
}

static void test_hash(void) {
    enum {N = 513, WIDTH = 48};
    ds4_engram_layout l = layout();
    assert(ds4_engram_layout_valid(&l));
    ds4_engram_layout bad = l;
    bad.multipliers[1][3] = UINT64_MAX;
    assert(!ds4_engram_layout_valid(&bad));
    bad = l;
    bad.rows[0]--;
    assert(!ds4_engram_layout_valid(&bad));
    bad = l;
    bad.primes[0][0] = 0;
    assert(!ds4_engram_layout_valid(&bad));
    int tokens[N];
    uint8_t mask[N];
    uint32_t expected[N * WIDTH], actual[N * WIDTH];
    for (int i = 0; i < N; i++) {
        tokens[i] = (i * 97 + i / 3) % 256;
        mask[i] = (i % 17 != 0 && (i < 125 || i > 131));
    }
    for (int masked = 0; masked < 2; masked++) {
        const uint8_t *m = masked ? mask : NULL;
        reference(&l, tokens, m, N, expected);
        for (int chunk = 1; chunk <= N; chunk++) {
            ds4_engram_history h;
            ds4_engram_history_reset(&h);
            for (int i = 0; i < N; i += chunk) {
                int n = N - i < chunk ? N - i : chunk;
                ds4_engram_history snapshot = h;
                assert(ds4_engram_hash(&l, &h, tokens + i, m ? m + i : NULL,
                                       n, actual + i * WIDTH));
                ds4_engram_history after = h;
                h = snapshot;
                assert(ds4_engram_hash(&l, &h, tokens + i, m ? m + i : NULL,
                                       n, actual + i * WIDTH));
                assert(memcmp(&h, &after, sizeof(h)) == 0);
            }
            assert(memcmp(actual, expected, sizeof(actual)) == 0);
        }
    }
    ds4_engram_history h, before;
    ds4_engram_history_reset(&h);
    before = h;
    int invalid[] = {0, 256};
    assert(!ds4_engram_hash(&l, &h, invalid, NULL, 2, actual));
    assert(memcmp(&h, &before, sizeof(h)) == 0);
    invalid[1] = -1;
    assert(!ds4_engram_hash(&l, &h, invalid, NULL, 2, actual));
    assert(ds4_engram_hash(&l, &h, NULL, NULL, 0, NULL));
    assert(!ds4_engram_hash(&l, &h, tokens, NULL, SIZE_MAX, actual));
    h.tail[0] = 128;
    assert(!ds4_engram_hash(&l, &h, tokens, NULL, 1, actual));
}

static void test_rows(void) {
    char path[] = "/tmp/ds4-engram-XXXXXX";
    int fd = mkstemp(path);
    assert(fd >= 0);
    /* Exercise offsets beyond 32 bits without allocating a large file. */
    const uint64_t offset = (1ull << 33) + 32;
    uint8_t raw[3][DS4_ENGRAM_ROW_BYTES];
    for (int r = 0; r < 3; r++) {
        for (int i = 0; i < 256; i++) raw[r][i] = i;
        raw[r][127] = 0;
        raw[r][255] = 128;
        for (int i = 0; i < 8; i++) raw[r][256 + i] = 126 + r;
    }
    assert(pwrite(fd, raw, sizeof(raw), offset) == sizeof(raw));
    ds4_engram_table t;
    assert(ds4_engram_table_open(&t, path, offset, 3));
    assert(fcntl(t.fd, F_GETFD) & FD_CLOEXEC);
    uint32_t rows[] = {2, 0, 2, 1};
    float out[4 * 256];
    assert(ds4_engram_read(&t, rows, 4, out));
    for (int r = 0; r < 4; r++) {
        for (int i = 0; i < 256; i++) {
            int code = raw[rows[r]][i], exponent = (code >> 3) & 15;
            double v = exponent ? (1 + (code & 7) / 8.) * pow(2., exponent - 7) :
                                  (code & 7) / 512.;
            if (code & 128) v = -v;
            v *= pow(2., (int)rows[r] - 1);
            assert(out[r * 256 + i] == (float)v);
            if (!v) assert(!!signbit(out[r * 256 + i]) == !!(code & 128));
        }
    }
    enum { TOKENS = 2051, STRIDE = 48, WIDTH = DS4_ENGRAM_COLS * DS4_ENGRAM_DIM };
    uint32_t *batch_ids = malloc((size_t)TOKENS * STRIDE * sizeof(*batch_ids));
    float *batch = malloc(((size_t)TOKENS * WIDTH + 1) * sizeof(*batch));
    assert(batch_ids && batch);
    for (size_t i = 0; i < (size_t)TOKENS * STRIDE; i++) batch_ids[i] = UINT32_MAX;
    for (size_t i = 0; i < TOKENS; i++)
        for (size_t j = 0; j < DS4_ENGRAM_COLS; j++)
            batch_ids[i * STRIDE + j] = (i * 7 + j * 11) % 3;
    const size_t sizes[] = {1, 2, 31, 65, 257, 2047, 2048, 2049, TOKENS};
    float expected[3][DS4_ENGRAM_DIM];
    const uint32_t all_rows[] = {0, 1, 2};
    assert(ds4_engram_read(&t, all_rows, 3, &expected[0][0]));
    for (size_t n = 0; n < sizeof(sizes) / sizeof(*sizes); n++) {
        size_t count = sizes[n];
        batch[count * WIDTH] = 123456.0f;
        assert(ds4_engram_read_batch(&t, batch_ids, count, STRIDE, batch));
        for (size_t i = 0; i < count; i++) {
            for (size_t j = 0; j < DS4_ENGRAM_COLS; j++) {
                assert(memcmp(batch + i * WIDTH + j * DS4_ENGRAM_DIM,
                    expected[batch_ids[i * STRIDE + j]], sizeof(expected[0])) == 0);
            }
        }
        assert(batch[count * WIDTH] == 123456.0f);
    }
    /* A decode-sized read preserves row order, duplicates and output bounds
     * in both the serial and opt-in parallel modes. */
    batch[WIDTH] = 123456.0f;
    assert(ds4_engram_read(&t, batch_ids, DS4_ENGRAM_COLS, batch));
    for (size_t j = 0; j < DS4_ENGRAM_COLS; j++) {
        assert(!memcmp(batch + j * DS4_ENGRAM_DIM,
                       expected[batch_ids[j]], sizeof(expected[0])));
    }
    assert(batch[WIDTH] == 123456.0f);
    const uint32_t last = batch_ids[DS4_ENGRAM_COLS - 1];
    batch_ids[DS4_ENGRAM_COLS - 1] = 3;
    batch[0] = 123456.0f;
    assert(!ds4_engram_read(&t, batch_ids, DS4_ENGRAM_COLS, batch) && errno == EINVAL);
    assert(batch[0] == 123456.0f);
    batch_ids[DS4_ENGRAM_COLS - 1] = last;
    assert(ds4_engram_read_batch(&t, NULL, 0, 0, NULL));
    assert(!ds4_engram_read_batch(&t, batch_ids, 1, 23, batch) && errno == EINVAL);
    assert(!ds4_engram_read_batch(&t, batch_ids, 2, SIZE_MAX, batch) && errno == EINVAL);
    assert(!ds4_engram_read_batch(&t, batch_ids, SIZE_MAX, STRIDE, batch) && errno == EINVAL);
    batch_ids[(TOKENS - 1) * STRIDE] = 3;
    batch[0] = 123456.0f;
    assert(!ds4_engram_read_batch(&t, batch_ids, TOKENS, STRIDE, batch) && errno == EINVAL);
    assert(batch[0] == 123456.0f);
    batch_ids[(TOKENS - 1) * STRIDE] = 0;
    uint32_t bad = 3;
    errno = 0;
    assert(!ds4_engram_read(&t, &bad, 1, out) && errno == EINVAL);
    assert(ds4_engram_read(&t, NULL, 0, NULL));
    uint8_t nan = 127;
    assert(pwrite(fd, &nan, 1, offset) == 1);
    bad = 0;
    assert(!ds4_engram_read(&t, &bad, 1, out) && errno == EDOM);
    assert(!ds4_engram_read_batch(&t, batch_ids, 1, STRIDE, batch) && errno == EDOM);
    assert(!ds4_engram_read_batch(&t, batch_ids, 31, STRIDE, batch) && errno == EDOM);
    for (size_t part = 0; part < 4; part++) {
        uint32_t decode_ids[DS4_ENGRAM_COLS];
        for (size_t j = 0; j < DS4_ENGRAM_COLS; j++) decode_ids[j] = 2;
        decode_ids[part * (DS4_ENGRAM_COLS / 4)] = 0;
        assert(!ds4_engram_read(&t, decode_ids, DS4_ENGRAM_COLS, batch) && errno == EDOM);
    }
    assert(pwrite(fd, raw, sizeof(raw), offset) == sizeof(raw));
    nan = 255;
    assert(pwrite(fd, &nan, 1, offset + 256) == 1);
    assert(!ds4_engram_read(&t, &bad, 1, out) && errno == EDOM);
    assert(!ds4_engram_read_batch(&t, batch_ids, 1, STRIDE, batch) && errno == EDOM);
    assert(ftruncate(fd, offset + 260) == 0);
    assert(!ds4_engram_read(&t, &bad, 1, out) && errno == EIO);
    assert(!ds4_engram_read_batch(&t, batch_ids, 1, STRIDE, batch) && errno == EIO);
    assert(!ds4_engram_read_batch(&t, batch_ids, 31, STRIDE, batch) && errno == EIO);
    assert(!ds4_engram_read(&t, batch_ids, DS4_ENGRAM_COLS, batch) && errno == EIO);
    free(batch_ids);
    free(batch);
    ds4_engram_table_close(&t);
    ds4_engram_table_close(&t);
    assert(!ds4_engram_table_open(&t, path, offset, 1));
    assert(!ds4_engram_table_open(&t, path, UINT64_MAX - 1, 3));
    assert(!ds4_engram_table_open(&t, path, offset, 0));
    close(fd);
    assert(unlink(path) == 0);
}

static void test_all_scaled_values(void) {
    char path[] = "/tmp/ds4-engram-values-XXXXXX";
    const int fd = mkstemp(path);
    assert(fd >= 0);
    assert(unlink(path) == 0);
    ds4_engram_table table = {.fd = fd, .rows = 1};
    const uint32_t row = 0;
    uint8_t raw[DS4_ENGRAM_ROW_BYTES];
    float out[DS4_ENGRAM_DIM];
    for (uint32_t code = 0; code < 256; code++) {
        memset(raw, code, DS4_ENGRAM_DIM);
        for (uint32_t scale = 0; scale < 256; scale++) {
            memset(raw + DS4_ENGRAM_DIM, scale, DS4_ENGRAM_ROW_BYTES - DS4_ENGRAM_DIM);
            assert(pwrite(fd, raw, sizeof(raw), 0) == sizeof(raw));
            const int exponent = (code >> 3) & 15;
            double value = exponent ? (1.0 + (code & 7u) / 8.0) * pow(2.0, exponent - 7) :
                                      (code & 7u) / 512.0;
            if (code & 128u) value = -value;
            float expected = (float)(value * pow(2.0, (int)scale - 127));
            uint32_t bits;
            memcpy(&bits, &expected, sizeof(bits));
            bits = (bits + 0x7fffu + ((bits >> 16) & 1u)) & 0xffff0000u;
            memcpy(&expected, &bits, sizeof(expected));
            const bool valid = (code & 127u) != 127u && scale != 255u && isfinite(expected);
            errno = 0;
            assert(ds4_engram_read(&table, &row, 1, out) == valid);
            if (valid) {
                for (uint32_t j = 0; j < DS4_ENGRAM_DIM; j++)
                    assert(!memcmp(out + j, &expected, sizeof(expected)));
            } else assert(errno == EDOM);
        }
    }
    close(fd);
}

/* The reader count must never change a byte. A single token's worth of rows
 * is the size the batch reader used to leave serial, so it is swept here
 * alongside a multi-token read. The serial per-row reader is the reference
 * because it takes no reader count at all. Both knobs are swept: the reader
 * override and the row divisor that picks one wave against two rounds. */
static void test_reader_counts(void) {
    char path[] = "/tmp/ds4-engram-readers-XXXXXX";
    const int fd = mkstemp(path);
    assert(fd >= 0);
    enum { ROWS = 7, TOKENS = 40, STRIDE = DS4_ENGRAM_COLS,
           WIDTH = DS4_ENGRAM_COLS * DS4_ENGRAM_DIM };
    uint8_t raw[ROWS][DS4_ENGRAM_ROW_BYTES];
    for (int r = 0; r < ROWS; r++) {
        for (int i = 0; i < DS4_ENGRAM_DIM; i++)
            raw[r][i] = (uint8_t)((i * 5 + r) % 127);
        for (int i = 0; i < 8; i++)
            raw[r][DS4_ENGRAM_DIM + i] = (uint8_t)(120 + r);
    }
    assert(pwrite(fd, raw, sizeof(raw), 0) == sizeof(raw));
    ds4_engram_table t;
    assert(ds4_engram_table_open(&t, path, 0, ROWS));
    assert(unlink(path) == 0);

    uint32_t *ids = malloc((size_t)TOKENS * STRIDE * sizeof(*ids));
    float *reference = malloc((size_t)TOKENS * WIDTH * sizeof(*reference));
    float *actual = malloc((size_t)TOKENS * WIDTH * sizeof(*actual));
    assert(ids && reference && actual);
    for (size_t i = 0; i < (size_t)TOKENS * STRIDE; i++)
        ids[i] = (uint32_t)((i * 13) % ROWS);
    for (size_t tok = 0; tok < TOKENS; tok++)
        assert(ds4_engram_read(&t, ids + tok * STRIDE, DS4_ENGRAM_COLS,
                               reference + tok * WIDTH));

    /* This sweep sets the variable itself, so an ambient value in the
     * environment does not reach it. To hold one setting across the whole
     * binary, set it outside and do not run this function. */
    const char *counts[] = {"0", "1", "2", "3", "5", "12", "16", "64", "bogus", ""};
    const size_t token_counts[] = {1, 2, 40};
    for (size_t c = 0; c < sizeof(counts) / sizeof(*counts); c++) {
        assert(setenv("DS4_ENGRAM_READ_THREADS", counts[c], 1) == 0);
        for (size_t n = 0; n < sizeof(token_counts) / sizeof(*token_counts); n++) {
            const size_t tokens = token_counts[n];
            memset(actual, 0, (size_t)TOKENS * WIDTH * sizeof(*actual));
            assert(ds4_engram_read_batch(&t, ids, tokens, STRIDE, actual));
            assert(memcmp(actual, reference, tokens * WIDTH * sizeof(*actual)) == 0);
        }
    }
    assert(unsetenv("DS4_ENGRAM_READ_THREADS") == 0);
    /* Same invariant for the divisor: 1 is one wave, 2 is the old control arm.
     * A bogus or empty value falls back to the default. */
    const char *divisors[] = {"1", "2", "3", "bogus", ""};
    for (size_t c = 0; c < sizeof(divisors) / sizeof(*divisors); c++) {
        assert(setenv("DS4_ENGRAM_ROWS_PER_READER", divisors[c], 1) == 0);
        for (size_t n = 0; n < sizeof(token_counts) / sizeof(*token_counts); n++) {
            const size_t tokens = token_counts[n];
            memset(actual, 0, (size_t)TOKENS * WIDTH * sizeof(*actual));
            assert(ds4_engram_read_batch(&t, ids, tokens, STRIDE, actual));
            assert(memcmp(actual, reference, tokens * WIDTH * sizeof(*actual)) == 0);
        }
    }
    assert(unsetenv("DS4_ENGRAM_ROWS_PER_READER") == 0);
    for (size_t n = 0; n < sizeof(token_counts) / sizeof(*token_counts); n++) {
        const size_t tokens = token_counts[n];
        memset(actual, 0, (size_t)TOKENS * WIDTH * sizeof(*actual));
        assert(ds4_engram_read_batch(&t, ids, tokens, STRIDE, actual));
        assert(memcmp(actual, reference, tokens * WIDTH * sizeof(*actual)) == 0);
    }
    free(ids);
    free(reference);
    free(actual);
    ds4_engram_table_close(&t);
    close(fd);
}

/* Builds a small table and reads it back; the caller owns the ids and the
 * expected bytes, taken with the serial reader that takes no reader count. */
typedef struct {
    ds4_engram_table table;
    int fd;
    uint32_t *ids;
    float *expected, *actual;
    size_t tokens;
} decode_fixture;

enum { FIXTURE_ROWS = 7, FIXTURE_TOKENS = 40,
       FIXTURE_WIDTH = DS4_ENGRAM_COLS * DS4_ENGRAM_DIM };

static void fixture_open(decode_fixture *f) {
    char path[] = "/tmp/ds4-engram-decode-XXXXXX";
    f->fd = mkstemp(path);
    assert(f->fd >= 0);
    uint8_t raw[FIXTURE_ROWS][DS4_ENGRAM_ROW_BYTES];
    for (int r = 0; r < FIXTURE_ROWS; r++) {
        for (int i = 0; i < DS4_ENGRAM_DIM; i++)
            raw[r][i] = (uint8_t)((i * 5 + r) % 127);
        for (int i = 0; i < 8; i++)
            raw[r][DS4_ENGRAM_DIM + i] = (uint8_t)(120 + r);
    }
    assert(pwrite(f->fd, raw, sizeof(raw), 0) == sizeof(raw));
    assert(ds4_engram_table_open(&f->table, path, 0, FIXTURE_ROWS));
    assert(unlink(path) == 0);
    f->tokens = FIXTURE_TOKENS;
    f->ids = malloc(f->tokens * DS4_ENGRAM_COLS * sizeof(*f->ids));
    f->expected = malloc(f->tokens * FIXTURE_WIDTH * sizeof(*f->expected));
    f->actual = malloc(f->tokens * FIXTURE_WIDTH * sizeof(*f->actual));
    assert(f->ids && f->expected && f->actual);
    for (size_t i = 0; i < f->tokens * DS4_ENGRAM_COLS; i++)
        f->ids[i] = (uint32_t)((i * 13) % FIXTURE_ROWS);
    for (size_t tok = 0; tok < f->tokens; tok++)
        assert(ds4_engram_read(&f->table, f->ids + tok * DS4_ENGRAM_COLS,
                               DS4_ENGRAM_COLS, f->expected + tok * FIXTURE_WIDTH));
}

static void fixture_close(decode_fixture *f) {
    free(f->ids);
    free(f->expected);
    free(f->actual);
    ds4_engram_table_close(&f->table);
    close(f->fd);
}

/* THE DEMAND READ IS SERIAL. One token's DS4_ENGRAM_COLS rows per table is a
 * decode read, and the engine issues it on the critical path twice per token.
 * It must reach the pool zero times, whatever the reader knobs say. */
static void test_decode_read_never_dispatches(void) {
    /* Runs before any wide read, so the first pass proves the absolute claim:
     * a decode-only workload never creates the pool at all. */
    assert(ds4_engram_pool_dispatches() == 0);
    decode_fixture f;
    fixture_open(&f);
    const char *counts[] = {NULL, "0", "1", "2", "12", "24", "64"};
    for (size_t c = 0; c < sizeof(counts) / sizeof(*counts); c++) {
        if (counts[c]) assert(setenv("DS4_ENGRAM_READ_THREADS", counts[c], 1) == 0);
        else assert(unsetenv("DS4_ENGRAM_READ_THREADS") == 0);
        const uint64_t before = ds4_engram_pool_dispatches();
        memset(f.actual, 0, FIXTURE_WIDTH * sizeof(*f.actual));
        assert(ds4_engram_read_batch(&f.table, f.ids, 1, DS4_ENGRAM_COLS, f.actual));
        assert(ds4_engram_pool_dispatches() == before);
        assert(!memcmp(f.actual, f.expected, FIXTURE_WIDTH * sizeof(*f.actual)));
    }
    assert(unsetenv("DS4_ENGRAM_READ_THREADS") == 0);
    fixture_close(&f);
}

/* The control for the case above: the counter CAN move, so a zero there is a
 * measurement and not a dead instrument. The wide prefill read is deliberately
 * wide and keeps its dispatch. */
static void test_wide_read_still_dispatches(void) {
    decode_fixture f;
    fixture_open(&f);
    const uint64_t before = ds4_engram_pool_dispatches();
    memset(f.actual, 0, f.tokens * FIXTURE_WIDTH * sizeof(*f.actual));
    assert(ds4_engram_read_batch(&f.table, f.ids, f.tokens, DS4_ENGRAM_COLS, f.actual));
    assert(ds4_engram_pool_dispatches() > before);
    assert(!memcmp(f.actual, f.expected, f.tokens * FIXTURE_WIDTH * sizeof(*f.actual)));
    fixture_close(&f);
}

int main(void) {
    test_hash();
    /* Before anything else: these two read the process-wide dispatch counter,
     * and the first wants it still at zero. */
    test_decode_read_never_dispatches();
    test_wide_read_still_dispatches();
    unsetenv("DS4_ENGRAM_PARALLEL_DECODE");
    test_rows();
    setenv("DS4_ENGRAM_PARALLEL_DECODE", "1", 1);
    test_rows();
    unsetenv("DS4_ENGRAM_PARALLEL_DECODE");
    test_all_scaled_values();
    test_reader_counts();
    puts("Engram hashes, history, bounded disk rows, reader counts and a serial demand read: PASS");
    return 0;
}
