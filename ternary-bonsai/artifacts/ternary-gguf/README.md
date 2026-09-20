# `ternary-gguf/` - the PTQ1_0 output

> ⚠ **EMPTY. Nothing has been quantised.**

The output of a quantisation run: a **type 143 `PTQ1_0`** gguf at **1.75 bpw**.

## ⛔ 143, NOT 142

| ggml type | name | bpw |
|---|---|---:|
| 142 | `PQ2_0` | 2.125 |
| **143** | **`PTQ1_0`** | **1.75** |

⚠ **The issue's 1.72 bpw is 1.75 in the type we hold**, and that number is what the whole size
argument multiplies. ⚠ **And in this tree the 143 kernel body is 40 instructions of stub**, so a
file here is unreadable at speed until P4 - which is fine, because P1's quality measurement needs a
**correct** read and not a fast one. `../../wiki/02-the-ggml-types.md`.

## The naming - the layer set and the imatrix both go in the filename

Two ggufs that differ in which layers were ternarized, or in which imatrix was used, are **different
experimental arms** and must not be confusable.

    ✅ v41flash-ptq1_0-layers-0-2-8-9-40-42-imatrix-wiki103.gguf
    ⛔ v41flash-ternary-v2.gguf
