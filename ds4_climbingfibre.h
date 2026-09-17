#ifndef DS4_CLIMBINGFIBRE_H
#define DS4_CLIMBINGFIBRE_H

/*
 * CLIMBINGFIBRE - the cerebellar front cache.
 *
 * A speculative decoder already computes a correction every time it rejects a
 * drafted token, and it throws that correction away.  At a rejection the engine
 * holds (context, what was drafted, what the target actually produced): a
 * labelled pair, produced at serving time, at zero marginal cost, because the
 * verify was happening anyway.  This module is the synapse those corrections
 * land on.  In the cerebellar reading the rejection is the CLIMBING FIBRE: the
 * teaching signal that drives plasticity at exactly the synapse that fired
 * wrongly.  See
 * experiments/track3-semiotic-codebook/NEW_IDEA_THE_CEREBELLAR_FRONT_CACHE_2026-09-01.md
 * sections 1 and 5.
 *
 * SUBSTRATE.  The key is a plain n-gram over the last N token ids.  That is
 * deliberate: MEASURED_DRAFTMEMORY_*
 * (experiments/track3-semiotic-codebook/, 2026-09-01) found the token-id n-gram
 * to be the strongest drafter address on both corpora, beating a centred
 * hidden-state address by z +11.17 on code and z +5.58 on prose.  Retrieval
 * drafting is occupied prior art (REST n-gram, DReSD hidden-state, RASD,
 * ReSpec) and THE DRAFTING SUBSTRATE IS NOT THE NOVEL PART.  What is claimed
 * here is narrower and is stated in the branch README: writing the verifier's
 * rejections back into the shortcut, ONLINE, with a retention policy.
 *
 * ====================================================================
 * THE FOUR CACHE LAWS, AND WHERE EACH ONE IS ENFORCED
 * ====================================================================
 *
 * LAW 1 - NEVER MODIFY THE DEEP PATH, ONLY SKIP IT.
 *   Enforced STRUCTURALLY, not by convention.  This module is handed token ids
 *   and scalars and nothing else.  It holds no pointer to a ds4_session, a
 *   ds4_engine, a weight tensor, a KV cache, a GPU tensor or a logits vector,
 *   and it includes no header that could name one.  It therefore CANNOT write
 *   into the model, because it is never given a handle to it.  The pinning of
 *   this claim is tests/test_climbingfibre.c, which scans this header and
 *   ds4_climbingfibre.c for the names of the deep path and fails if any appear.
 *
 * LAW 2 - A MISS MUST BE FREE.
 *   `ds4_climbingfibre_propose` returns false and writes nothing.  The caller
 *   leaves the speculator's "have a draft" flag clear, exactly as it was, so a
 *   miss costs one failed hash probe and changes no decision.  A run with the
 *   switch OFF and a run with the switch ON but the table cold take the same
 *   code path and emit the same tokens.
 *
 * LAW 3 - CORRECTNESS NEVER DEPENDS ON THE CACHE.
 *   Nothing here is ever injected.  A proposal is only ever a CANDIDATE that
 *   the target's own verifier then accepts or refuses.  The committed stream is
 *   always the target's, so a poisoned table can cost work and can never change
 *   an output.  tests/test_climbingfibre.c drives a scripted decode three ways
 *   (off, on-cold, on-with-every-entry-deliberately-wrong) and asserts the
 *   emitted token streams are identical.
 *
 * LAW 4 - LEARN FROM THE MISSES.
 *   `ds4_climbingfibre_learn` is called from the rejection sites.  It writes the
 *   pair (context, target's token) as a positive entry and DEMOTES the entry for
 *   the token that was refused - Albus's reading of the climbing fibre as a
 *   depression signal rather than a reward.
 *
 * ====================================================================
 * THE SEQUENTIAL-DRAFTER ASSUMPTION, NAMED
 * ====================================================================
 * This branch assumes the drafter runs SEQUENTIALLY: a proposal is made, it is
 * verified, and a rejection is written back before the next proposal is made.
 * That assumption is NOT PERMANENT and the reason it is not is measured prior
 * art, not speculation.  Saguaro (2026) runs the drafter ASYNCHRONOUSLY, in
 * parallel with verification, which hides the sequential drafter cost - the
 * exact cost that made our own DSpark drafter lose single-stream at 0.51x,
 * because it is a separate 37 GB MoE that pays its own bandwidth
 * (WIKI/research/deterministic-moe-reduction/04-self-spec-cheap-drafter.md:20
 * and :3-7).  LayerSkip's efficiency insight points the same way: verify and
 * draft run the same early layers in the same order, so verification can reuse
 * the draft's activations and KV and the drafter adds almost no bandwidth
 * (same file, :17-19).
 *
 * WHAT THAT CHANGES HERE, AND WHAT IT DOES NOT.
 *   The WRITE-BACK is already async-safe, by construction and not by luck: the
 *   key handed to learn() is built at the VERIFY site from the prefix that was
 *   actually verified, so a correction that lands many cycles after the draft
 *   it belongs to still lands on the right context.  Nothing reads "the current
 *   context" to decide where a correction goes.
 *   What DID need handling is ATTRIBUTION - telling a refused guess of OURS
 *   apart from a refused guess of the engine's when several drafts are
 *   outstanding.  A single "the proposal I just made" slot would mis-attribute
 *   every rejection that arrives out of order, so the module keeps a ring of
 *   CF_INFLIGHT outstanding proposals and matches each rejection to its own.
 *   See `ds4_climbingfibre_propose` / `cf_inflight_consume` in
 *   ds4_climbingfibre.c.
 *   A rejection signal was worth about FOUR percent, not twenty-seven
 *   (MEASURED_DECAYVSDEMOTE, 2026-09-01): plain uniform decay, which knows
 *   nothing about rejections, delivers 22.2 of the 26.7 points the rejection
 *   signal had been credited with, and the remaining 3.7 percent is an UPPER
 *   BOUND because the decay sweep was still monotone toward its finest rate
 *   when it stopped.  Nothing here should be read as promising more.
 *
 * THE LOOP CLOSES THROUGH THE FULL MODEL, WHICH IS THE TEACHER.  The `actual`
 * argument to learn() is always a token the target produced under its own
 * verification.  This is not a model learning from its own outputs; it is a fast
 * system corrected by a slow one, which is the safe form of the loop.
 *
 * ====================================================================
 * THE SWITCH
 * ====================================================================
 *   DS4_CLIMBINGFIBRE=1|0            master switch.  DEFAULT 0 (OFF).
 *                                    Off means the shortcut NEVER LEARNS and
 *                                    NEVER PROPOSES.  With it off, nothing
 *                                    about the engine's behaviour changes.
 *   DS4_CLIMBINGFIBRE_TOMBSTONES     are swept by rebuilding the table from its
 *                                    LIVE entries once they reach a quarter of
 *                                    it.  Not a knob - it is what keeps a
 *                                    lookup from degrading to a full walk once
 *                                    a session has inserted more than `cap`
 *                                    distinct keys.  The census reports
 *                                    `used=` (non-EMPTY slots), `rehashes=`
 *                                    and `probes=`.
 *   DS4_CLIMBINGFIBRE_HORIZON=       how stale an entry may be and still be
 *                                    served.  session | day | week | forever |
 *                                    <steps>.  DEFAULT session.  This is A
 *                                    PARAMETER, NOT A CONSTANT - the whole
 *                                    point of the knob is that the right value
 *                                    is a measurement we do not yet have.
 *   DS4_CLIMBINGFIBRE_STEPS_PER_DAY= decode steps in a "day", default 1048576.
 *   DS4_CLIMBINGFIBRE_STEPS_PER_SESSION= default 65536.
 *   DS4_CLIMBINGFIBRE_DECAY=         what retention does on expiry:
 *                                    hard  - a past-horizon entry is gone
 *                                    half  - its weight halves and its clock
 *                                            resets; AN ENTRY HALVED TO ZERO
 *                                            IS RETIRED, not kept at zero
 *                                            (DEFAULT)
 *                                    none  - never forget
 *   DS4_CLIMBINGFIBRE_NGRAM=2..8     key width, default 4.
 *   DS4_CLIMBINGFIBRE_CAP=<n>        table entries, default 16384.
 *   DS4_CLIMBINGFIBRE_STATS=1        print a census line at free.
 *
 * THE STEP BUDGETS ABOVE ARE A PLACEHOLDER SCALE, NOT A MEASUREMENT.  We have
 * not run the window sweep that would price "how stale may it be" (the
 * ROLLINGSPLIT experiment named in section 6 of the design document).  They are
 * named presets rather than magic numbers so that the parameter that actually
 * needs measuring is visible in one place.
 *
 * HONEST EFFECT SIZE.  MEASURED_DECAYVSDEMOTE (2026-09-01) found that plain
 * uniform decay - which knows nothing about rejections - delivers 22.2 of the
 * 26.7 points the rejection signal was credited with.  The rejection signal
 * itself is worth about 3.7 percent, and that is an UPPER BOUND: the decay
 * sweep was still monotone toward its finest rate when it stopped.  Nothing in
 * this module promises more than that.
 */

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

typedef struct ds4_climbingfibre ds4_climbingfibre;

/* Widest key the module can be configured for; callers size a stack buffer with
 * this so that a ngram change can never overflow the caller. */
#define DS4_CLIMBINGFIBRE_KEY_MAX 8

/* What retention does to an entry once it is past the horizon. */
typedef enum {
    DS4_CLIMBINGFIBRE_DECAY_HARD = 0, /* past the horizon it is gone */
    DS4_CLIMBINGFIBRE_DECAY_HALF = 1, /* its weight halves, clock resets */
    DS4_CLIMBINGFIBRE_DECAY_NONE = 2, /* never forget */
} ds4_climbingfibre_decay;

/* Always returns a handle, even when the switch is off: the OFF case must not
 * change any caller's control flow, so the hooks never have to NULL-check and
 * never have to take a different branch.  Returns NULL only on allocation
 * failure, and then every other entry point is a no-op. */
ds4_climbingfibre *ds4_climbingfibre_create(void);
void ds4_climbingfibre_free(ds4_climbingfibre *c);

/* True only when DS4_CLIMBINGFIBRE=1 was set at create time.  When this is
 * false LEARN AND PROPOSE ARE INERT: the first statement of each is the guard,
 * so nothing is read, written or counted. */
bool ds4_climbingfibre_enabled(const ds4_climbingfibre *c);

/* The configured key width.  Zero when the switch is off, so a caller that sizes
 * a buffer from this cannot ask for a key it cannot get. */
int ds4_climbingfibre_ngram(const ds4_climbingfibre *c);

/* The logical clock.  One tick per speculative cycle in which the cache is
 * consulted, so the retention policy is expressed in decode steps rather than
 * in wall-clock seconds and is reproducible offline. */
void ds4_climbingfibre_tick(ds4_climbingfibre *c);
uint64_t ds4_climbingfibre_step(const ds4_climbingfibre *c);

/*
 * LEARN FROM A MISS.  `key`/`key_len` are the token ids immediately preceding
 * the position that was predicted, oldest first.  `drafted` is what the
 * shortcut guessed; `actual` is the token the TARGET produced at that position.
 * Only `actual` is ever written as a candidate; `drafted` is only ever demoted.
 *
 * A no-op when the switch is off.  Writes only into this module's own array.
 */
void ds4_climbingfibre_learn(ds4_climbingfibre *c,
                             const int32_t *key, int key_len,
                             int32_t drafted, int32_t actual);

/*
 * PROPOSE.  Serves at most one token: the highest-weight non-expired candidate
 * stored against this key.  Returns true and sets *out_len to 1 on a hit,
 * false and *out_len to 0 on a miss.  A miss is FREE: it leaves every caller
 * observable unchanged.
 *
 * The candidate is a CANDIDATE only.  The caller must put it through the
 * target's verifier like any other draft; this module never says what the
 * output is.
 */
bool ds4_climbingfibre_propose(ds4_climbingfibre *c,
                               const int32_t *key, int key_len,
                               int32_t *out, uint32_t out_cap,
                               uint32_t *out_len);

/* One line to `out` (typically stderr) with the census.  Safe to call on a
 * disabled or NULL cache. */
void ds4_climbingfibre_census(const ds4_climbingfibre *c, FILE *out);

#endif /* DS4_CLIMBINGFIBRE_H */
