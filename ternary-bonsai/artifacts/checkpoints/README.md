# `checkpoints/` - intermediate states of a quantisation run

> ⚠ **EMPTY. Nothing has been run.**

A quantisation run over a large checkpoint is long and interruptible, and a rented box bills every
second it exists. ⇒ **resumability is an economic property here, not a convenience.**

## What belongs here

Per-run intermediate state, enough that a box that dies mid-run does not restart from zero.

⚠ **And the run manifest is TRACKED even though the state is not.** A small JSON naming the run, its
seal, its layer range, its box, its card and driver, and which shards are done. `../../.gitignore`
ignores `*.pt` and `*.bin` and not `*.json`, deliberately.

## ⚠ PULL BEFORE DESTROY

A checkpoint on a rented box that is destroyed is gone. The destination sha is verified **before**
the source is touched, every time.

## The naming

    ✅ v41flash-ptq1_0-layers-8-9-run-2026-09-21-box-3060-a-shard-04-of-12.pt
    ⛔ ckpt.pt
