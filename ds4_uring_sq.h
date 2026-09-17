#ifndef DS4_URING_SQ_H
#define DS4_URING_SQ_H

/* io_uring submission-queue bookkeeping, with no kernel headers and no I/O.
 *
 * The submission protocol has two tails.  One is private: the index of the
 * next SQE the engine will fill.  The other lives in the mmap'd ring and is
 * the ONLY thing the kernel reads.  A caller that advances the private tail
 * and never stores the shared one has queued nothing at all: io_uring_enter
 * returns 0 without waiting, the caller sees its own private tail running
 * ahead of what was accepted, and a submit-until-drained loop never exits.
 *
 * The SQ side is factored out here so that loop can be driven, and that class
 * of defect caught, on a host with no kernel and no CUDA - see
 * tests/test_uring_sq.cpp.  ds4_cuda.cu includes this and owns the CQ side and
 * the syscall itself.
 *
 *   [[kg:CallPathGate]] - the ring below had never been compiled, let alone
 *   run; the accounting is therefore tested where it can be run.
 */

#include <errno.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    unsigned *sq_head;    /* kernel writes: SQEs it has consumed */
    unsigned *sq_tail;    /* WE write: SQEs the kernel may consume */
    unsigned *sq_mask;    /* ring_mask, entries - 1 */
    unsigned *sq_array;   /* SQE index array the kernel walks */
    unsigned  entries;
    unsigned  tail;       /* private: next SQE index we will fill */
    unsigned  submitted;  /* private: SQEs the kernel has accepted */
} ds4_uring_sq;

/* Claim the next SQ slot.  Returns the SQE index, or -1 when every entry is
 * still unconsumed.  Writes the SQ index array, which the kernel walks to find
 * the SQE for each tail position. */
static inline int ds4_uring_sq_claim(ds4_uring_sq *sq) {
    const unsigned head = __atomic_load_n(sq->sq_head, __ATOMIC_ACQUIRE);
    if ((unsigned)(sq->tail - head) >= sq->entries) return -1;
    const unsigned idx = sq->tail & *sq->sq_mask;
    sq->sq_array[idx] = idx;
    return (int)idx;
}

/* Count a filled SQE.  Still invisible to the kernel until publish(). */
static inline void ds4_uring_sq_fill(ds4_uring_sq *sq) { sq->tail++; }

/* SQEs the kernel has not accepted yet. */
static inline unsigned ds4_uring_sq_to_submit(const ds4_uring_sq *sq) {
    return sq->tail - sq->submitted;
}

/* Publish the tail.  MUST happen before every io_uring_enter, with release
 * ordering: the SQE writes have to be visible to the kernel before the tail
 * that admits them. */
static inline void ds4_uring_sq_publish(ds4_uring_sq *sq) {
    __atomic_store_n(sq->sq_tail, sq->tail, __ATOMIC_RELEASE);
}

static inline void ds4_uring_sq_accepted(ds4_uring_sq *sq, unsigned n) {
    sq->submitted += n;
}

/* Wrap-safe: the private counters are free-running unsigned. */
static inline int ds4_uring_sq_drained(const ds4_uring_sq *sq) {
    return ds4_uring_sq_to_submit(sq) == 0u;
}

/* The submit loop.  `enter` is io_uring_enter: it returns the number of SQEs
 * accepted, or a negative errno.  Returns 1 when everything queued has been
 * accepted, 0 on an error the caller must treat as a dead ring.
 *
 * `max_rounds` bounds the loop.  The kernel accepts what the published tail
 * admits, so a healthy ring drains in one or two rounds; a bound turns the
 * unpublished-tail defect into a loud failure instead of a busy spin that
 * hangs the first cache miss of the run. */
typedef int (*ds4_uring_enter_fn)(void *ctx, unsigned to_submit, unsigned wait_nr);

static inline int ds4_uring_sq_submit(ds4_uring_sq *sq, ds4_uring_enter_fn enter,
                                      void *ctx, unsigned wait_nr,
                                      unsigned max_rounds) {
    for (unsigned round = 0; round < max_rounds; round++) {
        const unsigned to_submit = ds4_uring_sq_to_submit(sq);
        ds4_uring_sq_publish(sq);
        const int rc = enter(ctx, to_submit, wait_nr);
        if (rc < 0) {
            if (rc == -EINTR) continue;
            return 0;
        }
        ds4_uring_sq_accepted(sq, (unsigned)rc);
        if (to_submit == 0u || ds4_uring_sq_drained(sq)) return 1;
    }
    return 0;
}

#ifdef __cplusplus
}
#endif

#endif /* DS4_URING_SQ_H */
