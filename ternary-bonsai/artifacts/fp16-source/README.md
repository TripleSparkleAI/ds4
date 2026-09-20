# `fp16-source/` - the FP16 safetensors we quantise FROM

> ⛔⛔ **EMPTY, AND THIS IS THE PRECONDITION THE WHOLE PLAN WAITS ON.** We do not hold the
> DeepSeek V4.1 Flash FP16 safetensors. Issue 1085 is explicit that the quant must come from FP16
> and not from an existing quant, so **nothing in `../../methods/` past P0 can start.**
> `../../methods/01-P0-the-fp16-precondition.md`.

## What should be here

The V4.1 Flash FP16 safetensors shards, **or a subset of them**, plus the shard index.

★ **THE SHARD INDEX IS THE IMPORTANT FILE AND IT IS TRACKED.** `model.safetensors.index.json` maps
parameter name to shard file. It is a few hundred KB of JSON, `../../.gitignore` ignores
`*.safetensors` and not `*.json`, and **it answers the single biggest unpriced question in the
plan**: can a per-layer SUBSET be fetched, or does step 1 need the whole checkpoint? ⇒ **if you
fetch one file from this checkpoint, fetch that one, and commit it.**

## ⚠ Where the bytes go

**Not here, if they are large.** The Keep (`/media/hologram/M200/dwarfstar` on the spark) is where a
Keep-class artifact lives; this directory then holds a **STONE** naming the path, the sha256 and the
command that brings it home. The INDEX row says which arrangement is in force.

## ⚠ And the disk is the cost, not the card

A 120 GB disk request more than doubled a `$0.0994` rented box to `$0.2378/hr`, measured. These are
the largest artifacts in the project, so **any box that holds them is priced by its disk**.
`../../wiki/06-the-box-economics.md`.

## ⛔ Secrets

Any credential for fetching these is an **environment variable NAME** in every tracked file. Never
the value.
