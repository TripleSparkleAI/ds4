#define _DARWIN_C_SOURCE
#define _POSIX_C_SOURCE 200809L

#include "ds4_engram.h"

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <math.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

bool ds4_engram_layout_valid(const ds4_engram_layout *l) {
    if (!l || !l->token_map || !l->vocab_size ||
        !l->compressed_vocab_size || l->compressed_vocab_size > INT32_MAX ||
        l->pad_id >= l->compressed_vocab_size) return false;
    for (uint32_t i = 0; i < l->vocab_size; i++)
        if (l->token_map[i] >= l->compressed_vocab_size) return false;
    for (int layer = 0; layer < DS4_ENGRAM_LAYERS; layer++) {
        for (int i = 0; i < DS4_ENGRAM_NGRAM; i++) {
            uint64_t m = l->multipliers[layer][i];
            if (!(m & 1) || m > (uint64_t)INT64_MAX / l->compressed_vocab_size)
                return false;
        }
        uint64_t total = 0;
        for (int i = 0; i < DS4_ENGRAM_COLS; i++) {
            if (l->primes[layer][i] < 2) return false;
            total += l->primes[layer][i];
        }
        if (total != l->rows[layer]) return false;
    }
    return true;
}

void ds4_engram_history_reset(ds4_engram_history *h) {
    for (int i = 0; i < DS4_ENGRAM_NGRAM - 1; i++) h->tail[i] = DS4_ENGRAM_DEAD;
}

bool ds4_engram_hash(const ds4_engram_layout *l, ds4_engram_history *h,
                     const int *tokens, const uint8_t *mask, size_t count,
                     uint32_t *rows) {
    if (!l || !h || !l->token_map || (count && (!tokens || !rows)) ||
        count > SIZE_MAX / (DS4_ENGRAM_LAYERS * DS4_ENGRAM_COLS * sizeof(*rows)))
        return false;
    for (int i = 0; i < DS4_ENGRAM_NGRAM - 1; i++) {
        if (h->tail[i] < DS4_ENGRAM_DEAD ||
            (h->tail[i] >= 0 && (uint32_t)h->tail[i] >= l->compressed_vocab_size))
            return false;
    }
    for (size_t i = 0; i < count; i++) {
        if (tokens[i] < 0 || (uint32_t)tokens[i] >= l->vocab_size) return false;
    }
    for (size_t i = 0; i < count; i++) {
        int32_t current = mask && !mask[i] ? DS4_ENGRAM_DEAD :
                          (int32_t)l->token_map[tokens[i]];
        uint32_t ids[DS4_ENGRAM_NGRAM];
        bool blocked = false;
        for (int j = 0; j < DS4_ENGRAM_NGRAM; j++) {
            int32_t id = j ? h->tail[j - 1] : current;
            blocked |= id == DS4_ENGRAM_DEAD;
            ids[j] = blocked ? l->pad_id : (uint32_t)id;
        }
        for (int layer = 0; layer < DS4_ENGRAM_LAYERS; layer++) {
            uint64_t hash = (uint64_t)ids[0] * l->multipliers[layer][0];
            uint32_t offset = 0;
            for (int j = 1; j < DS4_ENGRAM_NGRAM; j++) {
                hash ^= (uint64_t)ids[j] * l->multipliers[layer][j];
                for (int head = 0; head < DS4_ENGRAM_HEADS; head++) {
                    int col = (j - 1) * DS4_ENGRAM_HEADS + head;
                    uint32_t prime = l->primes[layer][col];
                    *rows++ = (uint32_t)(hash % prime) + offset;
                    offset += prime;
                }
            }
        }
        for (int j = DS4_ENGRAM_NGRAM - 2; j > 0; j--) h->tail[j] = h->tail[j - 1];
        h->tail[0] = current;
    }
    return true;
}

bool ds4_engram_table_open(ds4_engram_table *t, const char *path,
                           uint64_t offset, uint32_t rows) {
    if (!t) return false;
    *t = (ds4_engram_table){.fd = -1};
    uint64_t bytes = (uint64_t)rows * DS4_ENGRAM_ROW_BYTES;
    if (!path || !rows || offset > INT64_MAX || bytes > INT64_MAX - offset) {
        errno = EINVAL;
        return false;
    }
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) return false;
    struct stat st;
    if (fstat(fd, &st) != 0) goto fail;
    if (!S_ISREG(st.st_mode) || st.st_size < 0 || offset + bytes > (uint64_t)st.st_size) {
        errno = EINVAL;
        goto fail;
    }
#ifdef __APPLE__
    if (fcntl(fd, F_NOCACHE, 1) != 0 || fcntl(fd, F_RDAHEAD, 0) != 0) goto fail;
#endif
    *t = (ds4_engram_table){.fd = fd, .offset = offset, .rows = rows};
    return true;
fail: {
        int saved = errno;
        close(fd);
        errno = saved;
        return false;
    }
}

void ds4_engram_table_close(ds4_engram_table *t) {
    if (!t) return;
    if (t->fd >= 0) close(t->fd);
    *t = (ds4_engram_table){.fd = -1};
}

static bool read_row(int fd, uint64_t offset, uint8_t row[DS4_ENGRAM_ROW_BYTES]) {
    size_t done = 0;
    while (done < DS4_ENGRAM_ROW_BYTES) {
        ssize_t n = pread(fd, row + done, DS4_ENGRAM_ROW_BYTES - done,
                          (off_t)(offset + done));
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) {
            if (n == 0) errno = EIO;
            return false;
        }
        done += (size_t)n;
    }
    return true;
}

static float e4m3(uint8_t byte) {
    int exponent = (byte >> 3) & 15, mantissa = byte & 7;
    float value = exponent ? ldexpf((float)(8 + mantissa), exponent - 10) :
                             ldexpf((float)mantissa, -9);
    return byte & 128 ? -value : value;
}

/* One process-wide pool for the Engram tables, created on first use and shared
 * by every table and every caller. The previous shape built and joined its own
 * threads (or ran dispatch_apply_f) once per call, so a decode step paid thread
 * churn on the very path it was trying to shorten. The caller is a worker too,
 * so the usable concurrency is ENGRAM_POOL_THREADS parked threads plus one. No
 * CUDA and no backend-specific symbol is used, so the same pool serves the
 * Metal, ROCm and CPU builds. A worker that cannot be created lowers the
 * concurrency, never the correctness. */
enum { ENGRAM_POOL_THREADS = 32 };

typedef struct {
    pthread_mutex_t lock;
    pthread_cond_t work, idle;
    pthread_t *workers;
    size_t nworkers;
    bool unavailable;
    void (*fn)(void *, size_t);
    void *context;
    size_t parts, next, active;
} engram_pool;

static engram_pool g_engram_pool = {
    .lock = PTHREAD_MUTEX_INITIALIZER,
    .work = PTHREAD_COND_INITIALIZER,
    .idle = PTHREAD_COND_INITIALIZER,
};
static pthread_mutex_t g_engram_pool_create = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t g_engram_pool_run = PTHREAD_MUTEX_INITIALIZER;

static void *engram_pool_worker(void *context) {
    engram_pool *pool = context;
    pthread_mutex_lock(&pool->lock);
    for (;;) {
        while (pool->next >= pool->parts) pthread_cond_wait(&pool->work, &pool->lock);
        const size_t part = pool->next++;
        pool->active++;
        void (*const fn)(void *, size_t) = pool->fn;
        void *const job = pool->context;
        pthread_mutex_unlock(&pool->lock);
        fn(job, part);
        pthread_mutex_lock(&pool->lock);
        if (--pool->active == 0) pthread_cond_broadcast(&pool->idle);
    }
}

/* Returns the worker count, zero if none could be created. */
static size_t engram_pool_create(void) {
    engram_pool *pool = &g_engram_pool;
    if (pool->workers || pool->unavailable) return pool->nworkers;
    pthread_mutex_lock(&g_engram_pool_create);
    if (!pool->workers && !pool->unavailable) {
        pthread_t *workers = calloc(ENGRAM_POOL_THREADS, sizeof(*workers));
        size_t started = 0;
        if (!workers) pool->unavailable = true;
        else {
            for (; started < ENGRAM_POOL_THREADS; started++)
                if (pthread_create(&workers[started], NULL, engram_pool_worker, pool)) break;
            pool->workers = workers;
            pool->nworkers = started;
            if (!started) pool->unavailable = true;
        }
    }
    pthread_mutex_unlock(&g_engram_pool_create);
    return pool->nworkers;
}

/* The concurrency cap: every parked worker plus the caller, which drains parts
 * itself so progress never depends on a wakeup. */
static size_t engram_pool_size(void) {
    return 1 + engram_pool_create();
}

/* Run parts [0, parts) across the pool. One caller at a time; the parts write
 * disjoint output, so completion order does not matter. A single part runs
 * inline and never starts a thread. */
static void engram_pool_run(void (*fn)(void *, size_t), void *context, size_t parts) {
    if (parts == 0) return;
    if (parts == 1) {
        fn(context, 0);
        return;
    }
    engram_pool *pool = &g_engram_pool;
    pthread_mutex_lock(&g_engram_pool_run);
    engram_pool_create();
    pthread_mutex_lock(&pool->lock);
    pool->fn = fn;
    pool->context = context;
    pool->parts = parts;
    pool->next = 0;
    pthread_cond_broadcast(&pool->work);
    while (pool->next < pool->parts) {
        const size_t part = pool->next++;
        pool->active++;
        pthread_mutex_unlock(&pool->lock);
        fn(context, part);
        pthread_mutex_lock(&pool->lock);
        if (--pool->active == 0) pthread_cond_broadcast(&pool->idle);
    }
    while (pool->active) pthread_cond_wait(&pool->idle, &pool->lock);
    pool->fn = NULL;
    pool->context = NULL;
    pool->parts = 0;
    pthread_mutex_unlock(&pool->lock);
    pthread_mutex_unlock(&g_engram_pool_run);
}

#ifdef __APPLE__
enum { ENGRAM_DECODE_READERS = 4 };

typedef struct {
    const ds4_engram_table *table;
    const uint32_t *rows;
    float *out;
    int error[ENGRAM_DECODE_READERS];
} engram_decode_read;

static void read_decode_part(void *context, size_t part) {
    engram_decode_read *read = context;
    const size_t begin = DS4_ENGRAM_COLS * part / ENGRAM_DECODE_READERS;
    const size_t end = DS4_ENGRAM_COLS * (part + 1) / ENGRAM_DECODE_READERS;
    if (!ds4_engram_read(read->table, read->rows + begin, end - begin,
                         read->out + begin * DS4_ENGRAM_DIM)) {
        read->error[part] = errno ? errno : EIO;
    }
}
#endif

bool ds4_engram_read(const ds4_engram_table *t, const uint32_t *rows,
                     size_t count, float *out) {
    if (!t || t->fd < 0 || (count && (!rows || !out)) ||
        count > SIZE_MAX / (DS4_ENGRAM_DIM * sizeof(*out))) {
        errno = EINVAL;
        return false;
    }
    for (size_t i = 0; i < count; i++) {
        if (rows[i] >= t->rows) {
            errno = EINVAL;
            return false;
        }
    }
#ifdef __APPLE__
    if (count == DS4_ENGRAM_COLS && getenv("DS4_ENGRAM_PARALLEL_DECODE")) {
        /* Decode reads 24 uncached rows. Partition the original order without
         * sorting or allocating; each reader owns disjoint output rows and the
         * shared pool joins all readers before returning to the caller. */
        engram_decode_read read = {.table = t, .rows = rows, .out = out};
        engram_pool_run(read_decode_part, &read, ENGRAM_DECODE_READERS);
        for (size_t i = 0; i < ENGRAM_DECODE_READERS; i++) {
            if (read.error[i]) {
                errno = read.error[i];
                return false;
            }
        }
        return true;
    }
#endif
    uint8_t raw[DS4_ENGRAM_ROW_BYTES];
    for (size_t i = 0; i < count; i++) {
        if (!read_row(t->fd, t->offset + (uint64_t)rows[i] * sizeof(raw), raw)) return false;
        for (int j = 0; j < DS4_ENGRAM_DIM; j++) {
            uint8_t code = raw[j], scale = raw[DS4_ENGRAM_DIM + j / 32];
            if ((code & 127) == 127 || scale == 255) {
                errno = EDOM;
                return false;
            }
            float value = ldexpf(e4m3(code), (int)scale - 127);
            uint32_t bits;
            memcpy(&bits, &value, sizeof(bits));
            bits = (bits + 0x7fffu + ((bits >> 16) & 1u)) & 0xffff0000u;
            memcpy(&value, &bits, sizeof(value));
            if (!isfinite(value)) {
                errno = EDOM;
                return false;
            }
            out[i * DS4_ENGRAM_DIM + j] = value;
        }
    }
    return true;
}

typedef struct {
    uint32_t row, output;
} engram_request;

static int request_order(const void *a, const void *b) {
    const engram_request *x = a, *y = b;
    return (x->row > y->row) - (x->row < y->row);
}

enum { ENGRAM_ROWS_PER_READER = 1, ENGRAM_MAX_READERS = ENGRAM_POOL_THREADS + 1 };

/* The reader divisor. The default is one row per reader, so a decode token's
 * DS4_ENGRAM_COLS rows per table are issued in ONE wave instead of two serial
 * rounds. */
static size_t engram_rows_per_reader(void) {
    const char *env = getenv("DS4_ENGRAM_ROWS_PER_READER");
    if (env) {
        char *end = NULL;
        const long configured = strtol(env, &end, 10);
        if (end != env && !*end && configured >= 1) return (size_t)configured;
    }
    return ENGRAM_ROWS_PER_READER;
}

/* Readers scale with the request count and are capped by the REAL pool size,
 * not by a constant: the cap is the number of contexts that can actually run a
 * part, so widening the pool widens the read. DS4_ENGRAM_ROWS_PER_READER=2
 * restores the old divisor and is the control arm in the same binary; a 24-row
 * read then gets 12 readers and two serial rounds, as it did before. The
 * override is read per call rather than cached: this runs once per token per
 * table, immediately before dozens of disk reads, so the lookup is free and no
 * first caller wins the setting for the lifetime of the process. */
static size_t engram_reader_count(size_t count, size_t pool) {
    const char *env = getenv("DS4_ENGRAM_READ_THREADS");
    size_t readers = count / engram_rows_per_reader();
    if (env) {
        char *end = NULL;
        const long configured = strtol(env, &end, 10);
        if (end != env && !*end && configured >= 0) readers = (size_t)configured;
    }
    if (readers > pool) readers = pool;
    if (readers > count) readers = count;
    return readers < 1 ? 1 : readers;
}

typedef struct {
    const ds4_engram_table *table;
    const engram_request *request;
    float *out;
    size_t count, readers;
    int error[ENGRAM_MAX_READERS];
} engram_batch;

static void read_batch_part(void *context, size_t part) {
    engram_batch *batch = context;
    const engram_request *request = batch->request;
    const size_t begin = batch->count * part / batch->readers;
    const size_t end = batch->count * (part + 1) / batch->readers;
    const float *previous = NULL;
    for (size_t i = begin; i < end; i++) {
        float *dst = batch->out + (size_t)request[i].output * DS4_ENGRAM_DIM;
        if (i > begin && request[i].row == request[i - 1].row) {
            memcpy(dst, previous, DS4_ENGRAM_DIM * sizeof(*dst));
        } else {
            if (!ds4_engram_read(batch->table, &request[i].row, 1, dst)) {
                batch->error[part] = errno ? errno : EIO;
                return;
            }
            previous = dst;
        }
    }
}

bool ds4_engram_read_batch(const ds4_engram_table *t, const uint32_t *rows,
                           size_t tokens, size_t stride, float *out) {
    if (!t || t->fd < 0 || (tokens && (!rows || !out || stride < DS4_ENGRAM_COLS)) ||
        tokens > SIZE_MAX / (DS4_ENGRAM_COLS * DS4_ENGRAM_DIM * sizeof(*out)) ||
        (tokens && tokens - 1 > (SIZE_MAX / sizeof(*rows) - DS4_ENGRAM_COLS) / stride)) {
        errno = EINVAL;
        return false;
    }
    for (size_t i = 0; i < tokens; i++) {
        for (size_t j = 0; j < DS4_ENGRAM_COLS; j++) {
            if (rows[i * stride + j] >= t->rows) {
                errno = EINVAL;
                return false;
            }
        }
    }
    if (!tokens) return true;
    enum { BATCH_TOKENS = 2048 };
    const size_t cap = tokens < BATCH_TOKENS ? tokens : BATCH_TOKENS;
    engram_request *request = malloc(cap * DS4_ENGRAM_COLS * sizeof(*request));
    if (!request) return false;
    bool ok = true;
    for (size_t start = 0; ok && start < tokens; start += cap) {
        const size_t n = tokens - start < cap ? tokens - start : cap;
        const size_t count = n * DS4_ENGRAM_COLS;
        for (size_t i = 0; i < count; i++) {
            request[i] = (engram_request){
                rows[(start + i / DS4_ENGRAM_COLS) * stride + i % DS4_ENGRAM_COLS],
                (uint32_t)i
            };
        }
        qsort(request, count, sizeof(*request), request_order);
        engram_batch batch = {.table = t, .request = request, .count = count,
            .out = out + start * DS4_ENGRAM_COLS * DS4_ENGRAM_DIM, .readers = 1};
        /* Concurrency hides random-read latency without caching the table.
         * Each part owns disjoint output rows; all finish before GPU use. The
         * pool is process-wide and already running, so this starts no thread. */
        batch.readers = engram_reader_count(count, engram_pool_size());
        engram_pool_run(read_batch_part, &batch, batch.readers);
        for (size_t i = 0; i < batch.readers; i++) {
            if (batch.error[i]) {
                errno = batch.error[i];
                ok = false;
                break;
            }
        }
    }
    int saved = errno;
    free(request);
    errno = saved;
    return ok;
}
