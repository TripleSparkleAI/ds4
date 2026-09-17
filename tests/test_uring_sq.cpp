/* Host-side gate on the io_uring submission-queue accounting in
 * ds4_uring_sq.h, which ds4_cuda.cu's expert-fetch ring is built from.
 *
 * No kernel, no CUDA: a fake kernel stands in for io_uring_enter and reads the
 * SHARED tail exactly as the real one does.  That is the whole point - an
 * engine that advances only its private tail queues nothing, the fake kernel
 * accepts nothing, and the submit loop never drains.  Before the fix that was
 * a busy spin on the first cache miss of the run.
 */

#include <errno.h>
#include <stdio.h>
#include <string.h>

#include "ds4_uring_sq.h"

static int g_fail;
static void check(int ok, const char *what) {
    if (!ok) { printf("FAIL: %s\n", what); g_fail++; }
}

/* A ring the test owns: the four shared words plus an index array. */
#define RING_ENTRIES 8u
typedef struct {
    unsigned head, tail, mask;
    unsigned array[RING_ENTRIES];
    unsigned consumed[RING_ENTRIES];  /* SQE indices the kernel took, in order */
    unsigned n_consumed;
    unsigned calls;                   /* enter() invocations */
    int      force_rc;                /* != 0: return this instead of working */
} fake_ring;

static void fake_ring_init(fake_ring *k, ds4_uring_sq *sq) {
    memset(k, 0, sizeof *k);
    k->mask = RING_ENTRIES - 1u;
    memset(sq, 0, sizeof *sq);
    sq->sq_head = &k->head;
    sq->sq_tail = &k->tail;
    sq->sq_mask = &k->mask;
    sq->sq_array = k->array;
    sq->entries = RING_ENTRIES;
}

/* The kernel side: consume everything the PUBLISHED tail admits. */
static int fake_enter(void *ctx, unsigned to_submit, unsigned wait_nr) {
    fake_ring *k = (fake_ring *)ctx;
    (void)wait_nr;
    k->calls++;
    if (k->force_rc) return k->force_rc;
    unsigned took = 0;
    while (k->head != k->tail && took < to_submit) {
        k->consumed[k->n_consumed++ % RING_ENTRIES] = k->array[k->head & k->mask];
        k->head++;
        took++;
    }
    return (int)took;
}

/* One entry that accepts a single SQE per call, to exercise the round loop. */
static int fake_enter_one(void *ctx, unsigned to_submit, unsigned wait_nr) {
    fake_ring *k = (fake_ring *)ctx;
    (void)wait_nr;
    k->calls++;
    if (to_submit == 0u || k->head == k->tail) return 0;
    k->consumed[k->n_consumed++ % RING_ENTRIES] = k->array[k->head & k->mask];
    k->head++;
    return 1;
}

/* ① The tail the kernel reads is published, so a batch drains. */
static void test_submit_drains(void) {
    fake_ring k;
    ds4_uring_sq sq;
    fake_ring_init(&k, &sq);
    for (unsigned i = 0; i < 4u; i++) {
        const int idx = ds4_uring_sq_claim(&sq);
        check(idx == (int)i, "claim hands out consecutive SQE indices");
        ds4_uring_sq_fill(&sq);
    }
    check(ds4_uring_sq_to_submit(&sq) == 4u, "four SQEs are waiting to submit");
    const int rc = ds4_uring_sq_submit(&sq, fake_enter, &k, 0u, sq.entries + 8u);
    check(rc == 1, "the submit loop reports the batch drained");
    check(k.tail == 4u, "the SHARED tail was published (the kernel sees 4)");
    check(k.n_consumed == 4u, "the kernel consumed every queued SQE");
    check(ds4_uring_sq_drained(&sq), "nothing is left unsubmitted");
    check(k.calls == 1u, "a published tail drains in one enter");
}

/* ② The SQ index array names the SQE for each tail position, in order. */
static void test_index_array(void) {
    fake_ring k;
    ds4_uring_sq sq;
    fake_ring_init(&k, &sq);
    for (unsigned i = 0; i < 3u; i++) { (void)ds4_uring_sq_claim(&sq); ds4_uring_sq_fill(&sq); }
    (void)ds4_uring_sq_submit(&sq, fake_enter, &k, 0u, sq.entries + 8u);
    int in_order = (k.n_consumed == 3u);
    for (unsigned i = 0; i < 3u && in_order; i++) in_order = (k.consumed[i] == i);
    check(in_order, "the kernel walked SQE 0,1,2 through the index array");
}

/* ③ A kernel that takes one at a time still drains, within the bound. */
static void test_partial_accept(void) {
    fake_ring k;
    ds4_uring_sq sq;
    fake_ring_init(&k, &sq);
    for (unsigned i = 0; i < 5u; i++) { (void)ds4_uring_sq_claim(&sq); ds4_uring_sq_fill(&sq); }
    const int rc = ds4_uring_sq_submit(&sq, fake_enter_one, &k, 0u, sq.entries + 8u);
    check(rc == 1, "a one-at-a-time kernel still drains the batch");
    check(k.n_consumed == 5u, "all five SQEs were consumed");
}

/* ④ A ring that accepts NOTHING forever is refused, not spun on.  That is
 *    exactly the shape the unpublished-tail defect took: the kernel saw no
 *    SQEs, returned 0 without waiting, and the loop re-issued for ever. */
static int fake_enter_accepts_nothing(void *ctx, unsigned to_submit, unsigned wait_nr) {
    fake_ring *k = (fake_ring *)ctx;
    (void)to_submit; (void)wait_nr;
    k->calls++;
    return 0;
}

static void test_dead_ring_is_refused(void) {
    fake_ring k;
    ds4_uring_sq sq;
    fake_ring_init(&k, &sq);
    for (unsigned i = 0; i < 2u; i++) { (void)ds4_uring_sq_claim(&sq); ds4_uring_sq_fill(&sq); }
    const int rc = ds4_uring_sq_submit(&sq, fake_enter_accepts_nothing, &k, 0u, 6u);
    check(rc == 0, "a ring that accepts nothing is refused, not spun on");
    check(k.calls == 6u, "the loop is bounded by max_rounds");

    fake_ring dead;
    ds4_uring_sq dsq;
    fake_ring_init(&dead, &dsq);
    (void)ds4_uring_sq_claim(&dsq); ds4_uring_sq_fill(&dsq);
    dead.force_rc = -EIO;
    const int rc2 = ds4_uring_sq_submit(&dsq, fake_enter, &dead, 0u, 8u);
    check(rc2 == 0, "an erroring ring is refused so the caller can fall back");
    check(dead.calls == 1u, "an error is not retried");
}

/* ⑤ EINTR is retried rather than treated as a dead ring. */
static void test_eintr_retries(void) {
    fake_ring k;
    ds4_uring_sq sq;
    fake_ring_init(&k, &sq);
    (void)ds4_uring_sq_claim(&sq); ds4_uring_sq_fill(&sq);
    k.force_rc = -EINTR;
    const int rc = ds4_uring_sq_submit(&sq, fake_enter, &k, 0u, 4u);
    check(rc == 0, "endless EINTR is bounded, not spun on forever");
    check(k.calls == 4u, "every round was retried up to the bound");
}

/* ⑥ The ring refuses a claim when every entry is still unconsumed. */
static void test_full_ring(void) {
    fake_ring k;
    ds4_uring_sq sq;
    fake_ring_init(&k, &sq);
    for (unsigned i = 0; i < RING_ENTRIES; i++) {
        check(ds4_uring_sq_claim(&sq) >= 0, "a free entry is handed out");
        ds4_uring_sq_fill(&sq);
    }
    check(ds4_uring_sq_claim(&sq) < 0, "a full ring refuses the next claim");
    (void)ds4_uring_sq_submit(&sq, fake_enter, &k, 0u, sq.entries + 8u);
    check(ds4_uring_sq_claim(&sq) >= 0, "a drained ring hands entries out again");
}

/* ⑦ The private counters are free-running unsigned: drained must not compare
 *    with >=, which breaks across the 2^32 wrap. */
static void test_counters_wrap(void) {
    ds4_uring_sq sq;
    memset(&sq, 0, sizeof sq);
    sq.submitted = 0xfffffffeu;
    sq.tail = 0x00000002u;         /* four SQEs across the wrap */
    check(ds4_uring_sq_to_submit(&sq) == 4u, "to_submit is wrap-safe");
    check(!ds4_uring_sq_drained(&sq), "a wrapped tail is not read as drained");
    ds4_uring_sq_accepted(&sq, 4u);
    check(ds4_uring_sq_drained(&sq), "and drains once accepted");
}

int main(void) {
    test_submit_drains();
    test_index_array();
    test_partial_accept();
    test_dead_ring_is_refused();
    test_eintr_retries();
    test_full_ring();
    test_counters_wrap();
    if (g_fail) { printf("io_uring SQ accounting: %d FAILED\n", g_fail); return 1; }
    puts("io_uring SQ accounting: PASS");
    return 0;
}
