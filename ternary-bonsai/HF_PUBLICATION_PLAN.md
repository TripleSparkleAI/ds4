# HUGGING FACE PUBLICATION PLAN - for the final versions, plural

> ⛔⛔⛔ **NOTHING IS PUBLISHED. NOTHING HAS BEEN UPLOADED. NO REPOSITORY HAS BEEN CREATED. NO TOKEN
> HAS BEEN READ.** This estate prepares and stops; an upload is outward-facing in exactly the way a
> pull request is, and a human performs it or it does not happen.
>
> ⛔ **AND THIS FILE CANNOT BE FINISHED YET.** We do not hold the V4.1 Flash FP16 safetensors, so no
> quant exists, so there is nothing to host. **Every box below is unchecked, and the first
> precondition is not on this page at all** - it is
> `methods/01-P0-the-fp16-precondition.md`.

## ⛔ THE LICENCE IS A REAL BLOCKER AND IT IS NOT OURS TO DECIDE

A ternary quant of DeepSeek V4.1 Flash is a **derivative of the DeepSeek weights**. What may be
published is governed by **their** licence - not by ours, and **not by Bonsai's Apache-2.0**, which
covers PrismML's own artifacts and says nothing about a different publisher's model.

    [ ] ⛔ OPEN QUESTION, OWNER: THE NAVIGATOR. May a derivative of these weights be published,
        and under what terms?

⛔ **Do not read a licence and rule on it.** A lane reading a licence file and concluding "this looks
permissive" is the shape of mistake that is expensive and unrecoverable once an artifact is public.
**It is named here as an open question with an owner, and it stays that way until the owner answers.**

## ⚠ FINAL IS PLURAL, so this is a plan for a SET

The method has variants - which layers were ternarized, which imatrix corpus, which bit width - and
each is a different experimental arm. **There will be several finals worth standing behind**, so the
plan has to say how they are named, versioned and related, before the first one exists.

    [ ] the NAMING SCHEME across the set, so two finals cannot be confused
        ^ every distinguishing choice in the name: checkpoint, type, layer set, imatrix corpus
        ^ e.g. v41flash-ptq1_0-layers-0-2-8-9-40-42-imatrix-wiki103
    [ ] ONE REPO PER FINAL, or ONE REPO WITH REVISIONS - decide, and SAY WHY
        ^ revisions keep the set together and make a stranger's `revision=` pin load-bearing
        ^ separate repos make each card standalone and each download unambiguous
        ^ ⚠ this is a real fork in the plan and it is not decided
    [ ] how a reader is told which final SUPERSEDES which, and that one is not simply "newer"
    [ ] whether a superseded final is DELETED or kept with a banner
        ^ this estate's own habit is to keep and mark, never to delete a record

## What every card must carry

    [ ] quantised FROM which EXACT checkpoint - the repo, the revision, the sha
    [ ] the METHOD: the rotation, its block size, the imatrix corpus, the layer set
    [ ] the BPW, and the ggml TYPE by number: 143 / PTQ1_0 / 1.75 bpw
        ^ ⚠ NOT 142. PQ2_0 is 2.125 bpw and is a different type. wiki/02-the-ggml-types.md
    [ ] ⛔ WHAT READS IT. A type-143 file is unreadable without a kernel that implements 143, and a
        rotated file is wrong without the matching activation transform. A card that omits this
        hands somebody a file that silently produces nonsense.
    [ ] the evaluation table: NLL, top-token agreement, avg_lcp
    [ ] ★ THE EXPERT-SELECTION AGREEMENT NUMBER, per layer
    [ ] the provenance chain, lifted from the artifact's own row in artifacts/final/INDEX.md
    [ ] the artifact list with exact bytes and sha256 per file, same source
    [ ] the licence, once the navigator has answered

## ★ WHY EXPERT-SELECTION AGREEMENT BELONGS ON THE CARD

Nobody asked for this and it is the most useful thing a card of ours could carry. It is the
**MoE-specific** measurement: does the quantised model route to the same 6 of 256 experts as FP16,
layer by layer. ⚠ **NLL cannot see that failure** - a token computed by a different subnetwork is a
different answer, and averaged over a corpus it hides inside a good NLL
(`wiki/03-the-moe-risk.md`).

⇒ **a card carrying it is more useful than one carrying another perplexity figure**, because
perplexity is what every card already has and this is a number the field does not publish. ⚠ We have
not found such a probe anywhere; that is a claim about our search, not about the field.

## ⛔ What no card may claim

| forbidden | why |
|---|---|
| **"lossless"** | a ternary quant is an approximation. The word is banned in this folder and in anything we publish. |
| any **retention figure carried from the 27B dense result** | 98.2 % is a **dense** number. The whole open question is whether it survives an MoE. Carrying it across is the error the issue itself warns about. |
| a **speed number without its box, its load, its context and its cache state** | a t/s without its stamps is not comparable to anything, including to itself on another day |
| a **residency claim we have not measured** | main-weights-resident on 256 GB needs a 256 GB Mac and the fast kernel. **We have neither.** |
| a **streaming multiple** we have not measured | the 2.5x is the issue author's claim, attributed to him, never restated as ours |

## ⚠ HOSTING A 100-PLUS GB ARTIFACT IS ITSELF A PROJECT

**Saying "then we upload it" understates this badly**, so it is written down here:

    [ ] the upload TIME, measured on the actual link, not its advertised rate
    [ ] LFS behaviour at this size: per-file limits, shard count, resumability of a failed push
    [ ] whether a partial upload leaves a repo somebody else can load and be misled by
    [ ] ⛔ WHETHER THE SHARDS ARE USEFUL TO ANYONE AT ALL without the ternary kernel that reads them
        ^ this is the honest question underneath the whole file. A type-143 gguf with no kernel in
          mainline is a file almost nobody can run. Publishing it may be publishing a placeholder.
    [ ] ⇒ therefore: does the KERNEL (methods/05-P4-the-kernel.md) have to land BEFORE a card is
        worth writing? Not decided.
    [ ] storage and bandwidth cost, and who pays it

★ **That second-to-last box may reorder the whole plan**, and it is better raised now than after an
upload.

## The order, if it ever happens

    [ ] 1. the navigator answers the LICENCE question
    [ ] 2. a final exists in artifacts/final/ with a complete INDEX row
    [ ] 3. P1 passed on it, with the expert-agreement number, not on NLL alone
    [ ] 4. the naming and one-repo-or-revisions decision is made and written here
    [ ] 5. the card is DRAFTED in this folder, reviewed, and still not uploaded
    [ ] 6. a human uploads it

⛔ **Steps 1 to 5 are preparation and may be done by a lane. Step 6 may not.**
