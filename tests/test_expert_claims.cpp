/* Host-side gate on the streaming expert-cache claim ledger
 * (ds4_expert_claims.h), which cuda_stream_selected_cache_begin_load uses.
 *
 * The invariant: after a fill that did NOT complete, no gate in the lookup map
 * may point at a slot whose bytes were never written.  begin_load claims a
 * victim before it reads, so that a later miss in the same batch cannot take
 * the same slot - and its mid-loop `victim == UINT32_MAX` return left those
 * earlier claims published.  The next call that routed to one of those experts
 * took the hit path and ran the layer on whatever the slot held before.
 *
 * No CUDA here: the ledger is the whole of that bookkeeping, and this drives
 * the same object begin_load drives.
 */

#include <stdint.h>
#include <stdio.h>
#include <unordered_map>
#include <vector>

#include "ds4_expert_claims.h"

struct test_slot { uint64_t gate, up, down, used; };
typedef std::vector<test_slot> slots_t;
typedef std::unordered_map<uint64_t, uint32_t> map_t;
typedef ds4_expert_claim_ledger<slots_t, map_t> ledger_t;

static int g_fail;
static void check(int ok, const char *what) {
    if (!ok) { printf("FAIL: %s\n", what); g_fail++; }
}

/* One miss: take the victim, stamp the slot, publish the claim. */
static void claim_one(slots_t &slots, ledger_t &led, uint32_t victim,
                      uint64_t gate, uint64_t stamp) {
    test_slot s = { gate, gate + 100u, gate + 200u, stamp };
    slots[victim] = s;
    led.claim(victim, gate);
}

/* ① The bug: a fill aborted mid-loop leaves NO lookup pointing at a slot whose
 *    bytes were never written. */
static void test_abort_withdraws_every_claim(void) {
    slots_t slots(4);
    map_t by_gate;
    {
        ledger_t led(slots, by_gate);
        claim_one(slots, led, 0u, 1000u, 7u);
        claim_one(slots, led, 1u, 2000u, 7u);
        check(by_gate.size() == 2u, "both claims are visible while in flight");
        /* the third miss finds no victim: begin_load returns 0 here */
    }
    check(by_gate.empty(), "an aborted fill leaves no gate in the lookup map");
    check(slots[0].used == 0u && slots[1].used == 0u,
          "and the abandoned slots are free for the next fill");
}

/* ② A completed fill keeps its claims: the bytes are there. */
static void test_commit_keeps_the_claim(void) {
    slots_t slots(4);
    map_t by_gate;
    {
        ledger_t led(slots, by_gate);
        claim_one(slots, led, 0u, 1000u, 7u);
        claim_one(slots, led, 1u, 2000u, 7u);
        led.commit(0u);
        led.commit(1u);
    }
    check(by_gate.size() == 2u, "a completed fill keeps both entries");
    check(by_gate[1000u] == 0u && by_gate[2000u] == 1u, "each gate names its slot");
    check(slots[0].used == 7u && slots[1].used == 7u, "the slots stay stamped");
}

/* ③ A partly failed batch keeps only what landed - today's !all_ok behaviour. */
static void test_partial_batch(void) {
    slots_t slots(4);
    map_t by_gate;
    {
        ledger_t led(slots, by_gate);
        claim_one(slots, led, 0u, 1000u, 7u);
        claim_one(slots, led, 1u, 2000u, 7u);
        claim_one(slots, led, 2u, 3000u, 7u);
        led.commit(0u);           /* this one's three tensors read */
        led.commit(2u);           /* so did this one */
        /* victim 1's read failed: never committed */
    }
    check(by_gate.size() == 2u, "only the reads that landed stay in the map");
    check(by_gate.find(2000u) == by_gate.end(), "the failed read's gate is gone");
    check(slots[1].used == 0u, "and its slot is free again");
    check(by_gate.find(1000u) != by_gate.end() &&
          by_gate.find(3000u) != by_gate.end(), "the good entries survive");
}

/* ④ Withdrawal is exact: a claim never erases an entry it did not publish.
 *    The slot it claimed has since been taken by someone else. */
static void test_withdraw_does_not_evict_a_stranger(void) {
    slots_t slots(4);
    map_t by_gate;
    by_gate[9999u] = 0u;                      /* a pre-existing resident */
    slots[0].gate = 9999u; slots[0].used = 3u;
    {
        ledger_t led(slots, by_gate);
        led.claim(1u, 1000u);                 /* our claim, slot 1 */
        slots[1].gate = 1000u; slots[1].used = 7u;
        /* abort */
    }
    check(by_gate.size() == 1u && by_gate[9999u] == 0u,
          "the stranger's entry is untouched");
    check(slots[0].used == 3u, "and its slot keeps its stamp");
    check(by_gate.find(1000u) == by_gate.end(), "our own claim was withdrawn");
}

/* ⑤ The gate was re-pointed at another slot after we claimed it: withdrawing
 *    must not steal it back. */
static void test_withdraw_respects_a_repointed_gate(void) {
    slots_t slots(4);
    map_t by_gate;
    {
        ledger_t led(slots, by_gate);
        led.claim(1u, 1000u);
        slots[1].gate = 1000u; slots[1].used = 7u;
        by_gate[1000u] = 3u;                  /* someone re-pointed the gate */
        slots[3].gate = 1000u; slots[3].used = 9u;
    }
    check(by_gate.size() == 1u && by_gate[1000u] == 3u,
          "a gate now naming another slot is left alone");
    check(slots[3].used == 9u, "that slot keeps its stamp");
}

/* ⑥ Withdrawal is idempotent: an explicit withdraw then the destructor must
 *    not erase an entry republished in between. */
static void test_withdraw_is_idempotent(void) {
    slots_t slots(4);
    map_t by_gate;
    {
        ledger_t led(slots, by_gate);
        led.claim(1u, 1000u);
        slots[1].gate = 1000u; slots[1].used = 7u;
        led.withdraw_uncommitted();
        check(by_gate.empty(), "the explicit withdraw cleared it");
        by_gate[1000u] = 1u;                  /* a later fill republishes it */
        slots[1].used = 11u;
    }
    check(by_gate.size() == 1u, "the destructor does not withdraw it twice");
    check(slots[1].used == 11u, "and the republished stamp survives");
}

int main(void) {
    test_abort_withdraws_every_claim();
    test_commit_keeps_the_claim();
    test_partial_batch();
    test_withdraw_does_not_evict_a_stranger();
    test_withdraw_respects_a_repointed_gate();
    test_withdraw_is_idempotent();
    if (g_fail) { printf("expert claim ledger: %d FAILED\n", g_fail); return 1; }
    puts("expert claim ledger: PASS");
    return 0;
}
