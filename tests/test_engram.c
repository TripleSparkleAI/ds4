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


/* --- the 4-bit sidecar ------------------------------------------------------ */

static void put_le32(uint8_t *p, uint32_t v) { for (int i = 0; i < 4; i++) p[i] = (uint8_t)(v >> (8 * i)); }
static void put_le64(uint8_t *p, uint64_t v) { for (int i = 0; i < 8; i++) p[i] = (uint8_t)(v >> (8 * i)); }

static uint16_t f32_to_f16_test(float f) {
    /* Exact for the values used here (small integers over powers of two). */
    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000u;
    int exponent = (int)((bits >> 23) & 255) - 127 + 15;
    const uint32_t mantissa = bits & 0x7fffffu;
    if (f == 0) return (uint16_t)sign;
    assert(exponent > 0 && exponent < 31 && (mantissa & 0x1fffu) == 0);
    return (uint16_t)(sign | ((uint32_t)exponent << 10) | (mantissa >> 13));
}

static void sidecar_header(uint8_t hdr[DS4_ENGRAM_4BIT_HEADER], const float *centroids,
                           uint32_t layer, uint32_t rows, uint64_t data, uint64_t source) {
    memset(hdr, 0, DS4_ENGRAM_4BIT_HEADER);
    memcpy(hdr, "DS4ENG4B", 8);
    put_le32(hdr + 8, 1); put_le32(hdr + 12, 2); put_le32(hdr + 16, DS4_ENGRAM_DIM);
    put_le32(hdr + 20, DS4_ENGRAM_4BIT_BLOCK); put_le32(hdr + 24, DS4_ENGRAM_ROW_BYTES_4BIT);
    put_le32(hdr + 28, 4096);
    for (int i = 0; i < 16; i++) {
        uint32_t bits;
        memcpy(&bits, &centroids[i], sizeof(bits));
        put_le32(hdr + 32 + 4 * i, bits);
    }
    /* entry 0: a decoy for layer 14; entry 1: the table under test */
    put_le32(hdr + 96, 14); put_le32(hdr + 100, 1000000); put_le64(hdr + 104, 4096); put_le64(hdr + 112, 1);
    put_le32(hdr + 128, layer); put_le32(hdr + 132, rows); put_le64(hdr + 136, data);
    put_le64(hdr + 144, source);
}

/* Independent reference: an explicit Sylvester W_128 built by the block
 * recursion, applied as a plain matrix product, in double. */
static void reference_dequant(const uint8_t row[DS4_ENGRAM_ROW_BYTES_4BIT],
                              const float *centroids, double out[DS4_ENGRAM_DIM]) {
    static signed char w[128][128];
    static int built = 0;
    if (!built) {
        w[0][0] = 1;
        for (int n = 1; n < 128; n *= 2)
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++) {
                    w[i][j + n] = w[i][j];
                    w[i + n][j] = w[i][j];
                    w[i + n][j + n] = (signed char)-w[i][j];
                }
        built = 1;
    }
    for (int b = 0; b < 2; b++) {
        double c[128];
        for (int k = 0; k < 64; k++) {
            c[2 * k] = centroids[row[b * 64 + k] & 15];
            c[2 * k + 1] = centroids[row[b * 64 + k] >> 4];
        }
        const uint16_t nb = (uint16_t)(row[128 + 2 * b] | row[129 + 2 * b] << 8);
        /* normal F16 or zero only in this test */
        const double norm = !(nb & 0x7fff) ? 0.0 : ldexp(1.0 + (nb & 1023) / 1024.0, ((nb >> 10) & 31) - 15) * ((nb & 0x8000) ? -1 : 1);
        for (int i = 0; i < 128; i++) {
            double acc = 0;
            for (int j = 0; j < 128; j++) acc += w[i][j] * c[j];
            out[b * 128 + i] = acc * norm / 128.0;
        }
    }
}

static void test_rows_4bit(void) {
    char path[] = "/tmp/ds4-engram4-XXXXXX";
    int fd = mkstemp(path);
    assert(fd >= 0);
    float centroids[16];
    for (int i = 0; i < 16; i++) centroids[i] = (i - 7.5f) * 0.375f;  /* increasing, symmetric */
    enum { ROWS = 5 };
    const uint64_t data = 8192, source = 123456789;
    uint8_t hdr[DS4_ENGRAM_4BIT_HEADER];
    sidecar_header(hdr, centroids, 1, ROWS, data, source);
    assert(pwrite(fd, hdr, sizeof(hdr), 0) == (ssize_t)sizeof(hdr));
    uint8_t raw[ROWS][DS4_ENGRAM_ROW_BYTES_4BIT];
    uint32_t seed = 12345;
    for (int r = 0; r < ROWS; r++) {
        for (int i = 0; i < 128; i++) {
            seed = seed * 1103515245u + 12345u;
            raw[r][i] = (uint8_t)(seed >> 16);
        }
        const uint16_t n0 = f32_to_f16_test(0.03125f * (r + 1)), n1 = f32_to_f16_test(r == 2 ? 0.0f : 1.5f);
        raw[r][128] = (uint8_t)n0; raw[r][129] = (uint8_t)(n0 >> 8);
        raw[r][130] = (uint8_t)n1; raw[r][131] = (uint8_t)(n1 >> 8);
    }
    assert(pwrite(fd, raw, sizeof(raw), (off_t)data) == (ssize_t)sizeof(raw));

    ds4_engram_table t;
    /* refusals: wrong rows (partial sidecar), wrong source, unknown layer, short file */
    assert(!ds4_engram_table_open_4bit(&t, path, 1, ROWS - 1, source) && errno == EINVAL);
    assert(!ds4_engram_table_open_4bit(&t, path, 1, ROWS, source + 1) && errno == EINVAL);
    assert(!ds4_engram_table_open_4bit(&t, path, 3, ROWS, source) && errno == EINVAL);
    assert(!ds4_engram_table_open_4bit(&t, path, 14, 1000000, 1) && errno == EINVAL); /* decoy: rows past EOF */
    assert(ds4_engram_table_open_4bit(&t, path, 1, ROWS, source));
    assert(t.row_bytes == DS4_ENGRAM_ROW_BYTES_4BIT && t.rows == ROWS && t.offset == data);
    for (int i = 0; i < 16; i++) assert(t.centroids[i] == centroids[i]);

    uint32_t ids[] = {4, 0, 2, 4, 1, 3};
    float out[6 * DS4_ENGRAM_DIM];
    assert(ds4_engram_read(&t, ids, 6, out));
    double worst = 0;
    for (int i = 0; i < 6; i++) {
        double expected[DS4_ENGRAM_DIM];
        reference_dequant(raw[ids[i]], centroids, expected);
        for (int j = 0; j < DS4_ENGRAM_DIM; j++) {
            const double err = fabs(out[i * DS4_ENGRAM_DIM + j] - expected[j]);
            if (err > worst) worst = err;
            if (err > 1e-6 * (1.0 + fabs(expected[j])))
                fprintf(stderr, "row %u j %d got %g expected %g\n", ids[i], j, out[i * DS4_ENGRAM_DIM + j], expected[j]);
            assert(err <= 1e-6 * (1.0 + fabs(expected[j])));
        }
    }
    /* the zero-norm block of row 2 is all zeros, and its sibling is not */
    {
        double expected[DS4_ENGRAM_DIM];
        reference_dequant(raw[2], centroids, expected);
        int nonzero = 0;
        for (int j = 0; j < 128; j++) { assert(expected[128 + j] == 0); nonzero += expected[j] != 0; }
        assert(nonzero > 0);
    }
    /* the batch path shares the same dequant */
    enum { STRIDE = DS4_ENGRAM_COLS + 1 };
    uint32_t batch_ids[3 * STRIDE];
    for (int i = 0; i < 3 * STRIDE; i++) batch_ids[i] = (uint32_t)(i * 7 % ROWS);
    float *batch = malloc(3 * DS4_ENGRAM_COLS * DS4_ENGRAM_DIM * sizeof(float));
    assert(batch && ds4_engram_read_batch(&t, batch_ids, 3, STRIDE, batch));
    for (int tok = 0; tok < 3; tok++)
        for (int col = 0; col < DS4_ENGRAM_COLS; col++) {
            double expected[DS4_ENGRAM_DIM];
            reference_dequant(raw[batch_ids[tok * STRIDE + col]], centroids, expected);
            for (int j = 0; j < DS4_ENGRAM_DIM; j++)
                assert(fabs(batch[(tok * DS4_ENGRAM_COLS + col) * DS4_ENGRAM_DIM + j] - expected[j]) <=
                       1e-6 * (1.0 + fabs(expected[j])));
        }
    free(batch);
    uint32_t bad = ROWS;
    assert(!ds4_engram_read(&t, &bad, 1, out) && errno == EINVAL);
    /* a NaN norm is refused, not propagated */
    uint8_t nan_norm[2] = {0x01, 0x7e};
    assert(pwrite(fd, nan_norm, 2, (off_t)(data + 130)) == 2);
    bad = 0;
    assert(!ds4_engram_read(&t, &bad, 1, out) && errno == EDOM);
    ds4_engram_table_close(&t);
    /* planted mutation: a centroid table that is not increasing is refused at open */
    float swapped[16];
    memcpy(swapped, centroids, sizeof(swapped));
    swapped[3] = centroids[4]; swapped[4] = centroids[3];
    sidecar_header(hdr, swapped, 1, ROWS, data, source);
    assert(pwrite(fd, hdr, sizeof(hdr), 0) == (ssize_t)sizeof(hdr));
    assert(!ds4_engram_table_open_4bit(&t, path, 1, ROWS, source) && errno == EINVAL);
    /* planted mutation: a bad magic */
    hdr[0] = 'X';
    assert(pwrite(fd, hdr, sizeof(hdr), 0) == (ssize_t)sizeof(hdr));
    assert(!ds4_engram_table_open_4bit(&t, path, 1, ROWS, source) && errno == EINVAL);
    close(fd);
    assert(unlink(path) == 0);
    printf("4-bit sidecar rows: PASS (worst abs error vs explicit W128 reference %.3g)\n", worst);
}

/* Optional cross-language check: DS4_ENGRAM_4BIT_FIXTURE=<dir> holding
 * `sidecar` (written by engram_hlwq_quantize.py --emit-fixture) and
 * `expected.f32` (the python decode of every row, float32 [rows][256]). */
static void test_fixture_4bit(void) {
    const char *dir = getenv("DS4_ENGRAM_4BIT_FIXTURE");
    if (!dir) return;
    char sidecar[4096], expected_path[4096];
    snprintf(sidecar, sizeof(sidecar), "%s/sidecar", dir);
    snprintf(expected_path, sizeof(expected_path), "%s/expected.f32", dir);
    FILE *fp = fopen(expected_path, "rb");
    assert(fp);
    fseek(fp, 0, SEEK_END);
    const long bytes = ftell(fp);
    rewind(fp);
    const uint32_t rows = (uint32_t)(bytes / (DS4_ENGRAM_DIM * sizeof(float)));
    float *expected = malloc((size_t)bytes);
    assert(expected && fread(expected, 1, (size_t)bytes, fp) == (size_t)bytes);
    fclose(fp);
    ds4_engram_table t;
    assert(ds4_engram_table_open_4bit(&t, sidecar, 1, rows, 0));
    float *got = malloc((size_t)rows * DS4_ENGRAM_DIM * sizeof(float));
    uint32_t *ids = malloc(rows * sizeof(*ids));
    assert(got && ids);
    for (uint32_t i = 0; i < rows; i++) ids[i] = i;
    assert(ds4_engram_read(&t, ids, rows, got));
    double worst = 0;
    for (size_t i = 0; i < (size_t)rows * DS4_ENGRAM_DIM; i++) {
        const double err = fabs(got[i] - expected[i]) / (1.0 + fabs(expected[i]));
        if (err > worst) worst = err;
    }
    assert(worst <= 2e-6);
    printf("4-bit fixture (%u rows, python encoder vs C dequant): PASS, worst rel error %.3g\n", rows, worst);
    free(expected); free(got); free(ids);
    ds4_engram_table_close(&t);
}

int main(void) {
    test_hash();
    unsetenv("DS4_ENGRAM_PARALLEL_DECODE");
    test_rows();
    setenv("DS4_ENGRAM_PARALLEL_DECODE", "1", 1);
    test_rows();
    unsetenv("DS4_ENGRAM_PARALLEL_DECODE");
    test_all_scaled_values();
    test_rows_4bit();
    test_fixture_4bit();
    puts("Engram hashes, history and bounded disk rows: PASS");
    return 0;
}
