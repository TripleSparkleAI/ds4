# ✅ THE BYTE LAYOUT - 55.44 % of the file cannot be ternarized

MEASURED by job `1520-engram-table-geometry` on the rented RTX 5090 (clock UTC+0000), reading the
tensor table of our own V4.1 Flash Q2 gguf tensor by tensor, offsets and residual walked. Full
statement: `../measured/ISSUE_1085_bonsai-style-ternary-ptq-for-v41-flash_what-we-measured_2026-09-20.md` §1.

| | bytes | GB | GiB |
|---|---:|---:|---:|
| tensor table total | 365,701,254,608 | 365.7 | 340.6 |
| **`blk.N.engram_embd.weight`, two I8 tensors** | **202,758,032,400** | 202.8 | **188.8** |
| everything else | 162,943,222,208 | 162.9 | 151.8 |

## What it settles

★ **The issue's 189 GB figure is right.** It is **188.8 GiB**; the unit label in the issue is
loose, the quantity is exact. So the issue's own ceiling on total savings stands, in its own
terms: **55.44 % of the file is a lookup table that cannot be ternarized, and it is already I8.**

The issue's sizing arithmetic also checks out in its own units: **294 GB at 4.5 bpw scaled to
1.72 bpw is 112 GB**, exactly as stated.

⚠ **Our file cannot test the headline.** Ours is **Q2**; the residency claim is a **Q4 to
ternary** path. Our non-engram half is 151.8 GiB at roughly 2.6 bpw, which ternary would take to
about **102 GiB**. That is a different and smaller claim than the one the issue makes, and it must
not be reported as agreement with it.

## ⚠ The engram table is not what the obvious reading assumes

MEASURED, same job. Two tensors:

| tensor | dims |
|---|---|
| `blk.N.engram_embd.weight` (a) | **(264, 384006168)** |
| `blk.N.engram_embd.weight` (b) | **(264, 384016682)** |
| vocab, for comparison | **99,092** |

**Different row counts. Neither a power of two. No vocabulary power.** ⇒ **evidence against simple
hashed n-gram addressing**, which is the reading the name invites. Evidence against, not a
disproof: UNKNOWN what the addressing actually is.

★ **Why this matters to a quantization plan rather than only being a curiosity:** if the engram
table were a hashed n-gram lookup keyed on token ids, its rows would be independent and a
per-row precision decision would be safe. **Not knowing the addressing means not knowing whether
the table tolerates any re-quantization at all**, and it is 55.44 % of the file. It is already I8,
so nothing in issue 1085 touches it - but any later plan that proposes to shrink it is proposing
to shrink a structure nobody here has characterised.

## The consequence for the whole plan

**Even a perfect ternary result caps the total saving where the issue says it does.** The routed
MoE path is the target; the lookup table is the floor; and the floor is more than half the file.
That is stated in the issue and it is confirmed here.
