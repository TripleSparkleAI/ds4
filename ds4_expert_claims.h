#ifndef DS4_EXPERT_CLAIMS_H
#define DS4_EXPERT_CLAIMS_H

/* The streaming expert-cache slot claims published by ONE cache fill.
 *
 * A claim is visible in the gate -> slot map the instant it is made, and it is
 * made BEFORE the expert's bytes are read, so that a later miss in the same
 * batch cannot choose the same victim.  That ordering has a debt attached:
 * every exit taken before the bytes land must WITHDRAW the claims it is
 * abandoning.  A published gate whose device memory was never written is read
 * as a HIT by the next call that routes to that expert, and the layer runs on
 * whatever the slot held before.
 *
 * Paying that debt at each return is a rule an author must remember; the mid
 * loop `victim == UINT32_MAX` return did not.  So the ledger pays it by
 * construction instead: a claim is PROVISIONAL until commit() names it, and
 * whatever is still provisional when the ledger leaves scope is withdrawn - on
 * the mid-loop abort, on an upload failure, on a partially failed batch, on a
 * thrown exception, and on any return added after this was written.
 *
 * Withdrawal is exact.  A claim erases its OWN gate, and only while that gate
 * still names the slot it claimed, so a ledger can never evict an entry it did
 * not publish.
 *
 * Templated on the slot container and the map so it can be driven on a host
 * with no CUDA - see tests/test_expert_claims.cpp.
 */

#include <stdint.h>
#include <stddef.h>
#include <vector>

template <typename Slots, typename Map>
class ds4_expert_claim_ledger {
public:
    ds4_expert_claim_ledger(Slots &slots, Map &by_gate)
        : slots_(slots), by_gate_(by_gate) {}
    ~ds4_expert_claim_ledger() { withdraw_uncommitted(); }

    /* Publish gate -> victim and record the debt. */
    void claim(uint32_t victim, uint64_t gate) {
        by_gate_[gate] = victim;
        entry e;
        e.victim = victim;
        e.gate = gate;
        e.settled = false;
        entries_.push_back(e);
    }

    /* The bytes for this victim landed: the claim stands. */
    void commit(uint32_t victim) {
        for (size_t i = 0; i < entries_.size(); i++)
            if (entries_[i].victim == victim) entries_[i].settled = true;
    }

    void withdraw_uncommitted() {
        for (size_t i = 0; i < entries_.size(); i++) {
            if (entries_[i].settled) continue;
            entries_[i].settled = true;           /* withdraw once */
            const uint64_t gate = entries_[i].gate;
            const uint32_t victim = entries_[i].victim;
            typename Map::iterator it = by_gate_.find(gate);
            if (it != by_gate_.end() && it->second == victim) by_gate_.erase(it);
            if (victim < slots_.size() && slots_[victim].gate == gate)
                slots_[victim].used = 0;
        }
    }

    size_t size() const { return entries_.size(); }

private:
    struct entry { uint32_t victim; uint64_t gate; bool settled; };
    Slots &slots_;
    Map &by_gate_;
    std::vector<entry> entries_;
    ds4_expert_claim_ledger(const ds4_expert_claim_ledger &);
    ds4_expert_claim_ledger &operator=(const ds4_expert_claim_ledger &);
};

#endif /* DS4_EXPERT_CLAIMS_H */
