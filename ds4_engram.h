#ifndef DS4_ENGRAM_H
#define DS4_ENGRAM_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

enum {
    DS4_ENGRAM_LAYERS = 2,
    DS4_ENGRAM_NGRAM = 4,
    DS4_ENGRAM_HEADS = 8,
    DS4_ENGRAM_COLS = 24,
    DS4_ENGRAM_DIM = 256,
    DS4_ENGRAM_ROW_BYTES = 264,
    DS4_ENGRAM_DEAD = -1
};

typedef struct {
    const uint32_t *token_map;
    uint32_t vocab_size, compressed_vocab_size, pad_id;
    uint32_t rows[DS4_ENGRAM_LAYERS];
    uint64_t multipliers[DS4_ENGRAM_LAYERS][DS4_ENGRAM_NGRAM];
    uint32_t primes[DS4_ENGRAM_LAYERS][DS4_ENGRAM_COLS];
} ds4_engram_layout;

/* Newest first; copying this value also snapshots the complete hash state. */
typedef struct {
    int32_t tail[DS4_ENGRAM_NGRAM - 1];
} ds4_engram_history;

bool ds4_engram_layout_valid(const ds4_engram_layout *layout);
void ds4_engram_history_reset(ds4_engram_history *history);
/* Validate the layout once at model load. Mask zero breaks all n-grams that
 * cross that position. Output is [token][layer][column], including masked rows;
 * the graph must suppress Engram at masked positions, as in the reference. */
bool ds4_engram_hash(const ds4_engram_layout *layout,
                     ds4_engram_history *history,
                     const int *tokens, const uint8_t *mask, size_t count,
                     uint32_t *rows);

/* The optional 4-bit sidecar (DS4_ENGRAM_4BIT=<path>): the same rows re-encoded
 * with the HLWQ scheme (arXiv:2603.29078), 132 bytes a row = 128 packed nibbles
 * (low nibble = even index) + 2 F16 block norms. A 4096-byte header carries the
 * 16 Lloyd-Max centroids and, per table, the layer, rows, data offset and the
 * source tensor's offset in the GGUF. Written by
 * gguf-tools/engram-4bit/engram_hlwq_quantize.py. */
enum {
    DS4_ENGRAM_ROW_BYTES_4BIT = 132,
    DS4_ENGRAM_4BIT_CENTROIDS = 16,
    DS4_ENGRAM_4BIT_HEADER = 4096,
    DS4_ENGRAM_4BIT_BLOCK = 128
};

typedef struct {
    int fd;
    uint64_t offset;
    uint32_t rows;
    /* DS4_ENGRAM_ROW_BYTES (FP8, the GGUF) or DS4_ENGRAM_ROW_BYTES_4BIT (sidecar). */
    uint32_t row_bytes;
    float centroids[DS4_ENGRAM_4BIT_CENTROIDS];
} ds4_engram_table;

/* A separate uncached file descriptor, never an mmap or Metal model view.
 * Each GGUF I8 row is 256 E4M3 bytes followed by 8 original E8M0 scales. */
bool ds4_engram_table_open(ds4_engram_table *table, const char *path,
                           uint64_t offset, uint32_t rows);
/* Open one table of a 4-bit sidecar. The header's entry for `layer` must carry
 * exactly `rows` rows and name `source_offset` as the FP8 tensor it was built
 * from, so a sidecar built for another GGUF, or a partial one, is refused. */
bool ds4_engram_table_open_4bit(ds4_engram_table *table, const char *sidecar,
                                uint32_t layer, uint32_t rows, uint64_t source_offset);
void ds4_engram_table_close(ds4_engram_table *table);
/* Output uses F32 storage for the reference's BF16-rounded values. No whole
 * table allocation; caller owns count * DIM floats. Failure invalidates output. */
bool ds4_engram_read(const ds4_engram_table *table, const uint32_t *rows,
                     size_t count, float *out);
/* Read COLS rows per token, restoring token order after deduplicated disk reads.
 * Input stride is in row IDs; output is packed [token][COLS][DIM]. Temporary
 * storage is bounded to 384 KiB, independent of the table and prefix size.
 * On macOS, large batches use bounded concurrent pread readers. */
bool ds4_engram_read_batch(const ds4_engram_table *table, const uint32_t *rows,
                           size_t tokens, size_t stride, float *out);

#endif
