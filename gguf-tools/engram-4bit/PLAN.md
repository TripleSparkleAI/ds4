# PLAN - the Engram 4-bit re-encode, and why this branch carries no engine change

Branch `triple-engram-4bit`, cut from `triple-antirez-tip-latest` (`e6d9d3b83`).

This directory is a **design record**. It holds the arithmetic, the scope decision, the four costs
that are already on the record for this quant, and the format contract that a later implementation
would fill in. It holds **no working quantizer and no engine change**. The skeleton,
`engram_hlwq_quantize.py`, raises `NotImplementedError` at every entry point and encodes zero rows.
`engram_4bit_arithmetic.py` is runnable and prints every number quoted in the root README.

---

## 1. What the lever is

A checkpoint re-encode of the two Engram hash tables from FP8 to 4-bit Hadamard-Lloyd. The card
that prices it is `research/v41-flash-landscape/07-heterogeneous-spark-plus-5090/sources/hf-hlwq-engram-q4.md`
(`caiovicentino1/DeepSeek-V4.1-Flash-HLWQ-Engram-Q4`), studied in that folder's `CODE.md` and
`HOW-WE-USE-IT.md`.

| | FP8, as we ship it | 4-bit HLWQ |
|---|---:|---:|
| bytes a row (256 dims) | 264 (256 E4M3 + 8 E8M0) | 132 (128 packed nibbles + 2 x F16 norms) |
| both tables on disk and in RAM | 202.758 GB | 101.379 GB |
| checkpoint download | 510 GB | ~408 GB |
| relative RMSE a row | - | 0.0967 |
| mean cosine a row | - | 0.99536 |
| dequant a lookup | none on the card's path | 3 tensor ops |

Sources: card table at `sources/hf-hlwq-engram-q4.md:26-32`; the download and RAM figures at
`sources/hf-hlwq-engram-q4.md:24`. Our own row layout is identical to the card's FP8 column:
`ds4_engram.h:13-14` (`DS4_ENGRAM_DIM = 256`, `DS4_ENGRAM_ROW_BYTES = 264`) and `ds4_engram.h:48`
("Each GGUF I8 row is 256 E4M3 bytes followed by 8 original E8M0 scales"), with the scale index
`raw[DS4_ENGRAM_DIM + j / 32]` at `ds4_engram.c:202`. The encoding is named in our own metadata:
`deepseek41.engram.encoding = "e4m3_e8m0_32_row264"` (`ds4.c:6912`).

---

## 2. The arithmetic on our own file

Run `python3 gguf-tools/engram-4bit/engram_4bit_arithmetic.py`. Every input in it carries its source
as a comment. The output, verbatim, 2026-09-16:

```
THE TABLE, BY ARGUMENT
  rows, layer 1 and layer 14     384,006,168 + 384,016,682 = 768,022,850
  parameters in the pair         196.6 G (256 dims a row)
  FP8 at 264 B a row           202,758,032,400 B = 202.758 GB = 188.833 GiB
  4-bit at 132 B a row          101,379,016,200 B = 101.379 GB = 94.417 GiB
  saved                          101,379,016,200 B = 101.379 GB = 94.417 GiB

THE FILE
  file as recorded               340.6 GiB = 365.72 GB
  file after the re-encode       246.2 GiB = 264.34 GB
  Engram share of the file       55.4%

THE PER TOKEN READ
  rows read a decode step        2 tables x 24 cols = 48 rows
  bytes read a decode step       12,672 B
  saved a step                   6,336 B
  as a share of the census floor 6,336 B of 9.670 GB = 0.0000655 percent
  at the achieved 230.5 GB/s     27.5 ns a token

WHAT THE READ ACTUALLY COSTS, FROM THE RECORD
  the Engram read, a step        23.9 ms of 198.6 ms = 12.03%
  the rate that implies          12,672 B in 23.9 ms = 0.53 MB/s
```

### 2.1 The assumptions behind it, each one attackable

1. **Row counts are the engine's own constants.** `ds4.c:6925` on `triple-antirez-tip-latest`:
   `const uint32_t engram[] = {1, 14}, rows[] = {384006168, 384016682};`.
2. **Row width is the engine's own constant.** 264 B, `ds4_engram.h:14`. The 4-bit width, 132 B, is
   the card's, `sources/hf-hlwq-engram-q4.md:28`.
3. **We read 48 rows a decode step, not the table.** `DS4_ENGRAM_LAYERS = 2` and
   `DS4_ENGRAM_COLS = 24` (`ds4_engram.h:9` and `:12`), and a decode step calls
   `ds4_engram_read(&g->table[i], ids[i], 24, g->rows[i])` per table exactly once. The call sites
   are enumerated in the Engram lane's own finding: `CUDA_LANES_CHANNEL.md:973-981`, with the
   decode path at `ds4.c:41261` and the server decode path at `ds4.c:42063`. So the per token read
   is 48 x 264 B, and the table is 768 million rows of which we touch 48 a step.
4. **The file size is our own census record**, not a measurement taken here:
   `WIKI/theory/177-the-sparkport-day-a-cache-that-was-a-stub-and-a-token-that-is-latency-not-bytes-2026-09-13.md:61-62`
   states the file at 340.6 GiB with a mapped tensor span of 151.8 GiB and "the other ~189 GiB of
   the 340.6 GiB file is Engram tables outside the map". 768,022,850 x 264 B = 188.833 GiB, so the
   record's 189 GiB and this arithmetic agree to three figures. The tables are outside the map
   because `ds4.c:3383-3384` skips `blk.1.engram_embd.weight` and `blk.14.engram_embd.weight` out
   of the mapped span.
5. **The per token floor is the census's own 9.670 GB/token**, from `ds4-gguf-census.py` (Q3), and
   the achieved ceiling is the census's own 230.5 GB/s constant, never the 273 GB/s datasheet peak.
6. **The 23.9 ms is a quoted measurement, not one taken here.** `CUDA_LANES_CHANNEL.md:1635` records
   the profiled decode at n=32 with `engram=23.922` against `198.636` ms, 12.04 percent, and
   `CUDA_LANES_CHANNEL.md:960-980` states the shape: "12,672 bytes arrive in 48 SEPARATE `pread`
   CALLS at queue depth one, scattered uniformly across 202.8 GB. This is a LATENCY problem, not a
   bandwidth one".

### 2.2 Was the census tooling run, or is this arithmetic?

Both, and the split matters.

**What was run, without a GPU and without the 340.6 GiB file:** `ds4-gguf-census.py` is a pure
Python GGUF header reader. Its own gate passes on this machine with no model at all:
`python3 ds4-gguf-census.py --selftest` prints
`ds4-gguf-census selftest: PASS (synthetic GGUF parsed, buckets + block math verified)`. A full
census of the GGUF that *is* here completed in seconds and reproduced the 9.670 GB/token floor
exactly, with `f = 0.999` on the Q6_K gate and a projected non-routed saving of 1.763 GB.

**What could NOT be run, and why the Engram figures are arithmetic.** The file on this machine is
86.72 GB, 1328 tensors, and **it carries zero Engram tensors**: reading its header with the census's
own parser returns `engram tensors: 0`. It is not the 340.6 GiB artifact the Engram figures come
from. The census tool also has no Engram bucket (`classify()` at `ds4-gguf-census.py` folds any
unknown name into `other`). So no tool on this box can measure the Engram table bytes from a file.
The three Engram inputs above are our own record read header-only (`gguf_engram_offsets.py`, cited
at `CUDA_LANES_CHANNEL.md:960`) plus the engine's constants, and the per token figures are ARITHMETIC
on those. They are labelled as such in the root README's results table.

### 2.3 The one number that reframes the lever

The table is 55.4 percent of the file and 0.00007 percent of a token's read. Halving it saves
**94.417 GiB on disk** and **6,336 bytes a token**, which at our achieved 230.5 GB/s is **27.5 ns a
token**. Meanwhile the read that touches it costs a measured **23.9 ms of a 198.6 ms step**, because
it is 48 round trips to move twelve kilobytes, not a byte transfer.

That is not an argument against the re-encode. It is an argument about which axis it pays on: this
is a **footprint and download** lever, not a per token byte lever, and no engine change to the
lookup path can turn it into one.

---

## 3. The scope decision

**This is not an engine branch. It is a separate tooling effort, and this branch is its design
record.**

The reason is the artifact/engine split, and it is not a matter of taste:

1. **The re-encode changes the artifact.** It rewrites 2 of 1328 tensors and leaves the rest
   byte-identical, which is the card's own claim about its 46 untouched shards
   (`sources/hf-hlwq-engram-q4.md:22`). Nothing in it needs a GPU, a CUDA build, or the engine's
   runtime. It is a converter, and `gguf-tools/` is where this repo keeps its converters
   (`gguf-tools/README.md:1-12`: `deepseek4-quantize.c`, `quants.[ch]`, `imatrix/`,
   `quality-testing/`, and the Python assemblers beside them).
2. **Our engine cannot parse what the card ships.** The card's 4-bit tables are published as
   `*.hlwq_codes` U8 `[rows, 128]` plus `*.hlwq_norms` F16 `[rows, 2]` in safetensors side files
   (`sources/hf-hlwq-engram-q4.md:53-55`), served by a SGLang adapter. Our engine opens the two
   tables as raw byte ranges inside the GGUF at a fixed 264 B stride (`ds4_engram_table_open`,
   `ds4_engram.h:49-50`, `ds4_engram.c:93`, `ds4_engram.c:200`). Adopting this means writing a
   quantizer that emits OUR layout AND a dequant on our lookup path. That is two pieces of work in
   two places, which is precisely the case for separating them rather than parking both in an engine
   branch.
3. **There is nothing to switch.** The task rule is a switch only if there is code to switch. No
   re-encoded artifact exists, no quantizer exists, and the dequant path cannot be written against a
   format whose encoder does not exist. So this branch adds no engine code and no knob, which is
   also why the commit is a single `update readme`.
4. **The delivered order follows the gates we already hold.** Our roadmap's named byte lever is
   already gated; this one is not gated at all (no artifact, no quality measurement on our file,
   SM121 unvalidated). Doing the gated one first is the honest sequencing, and it is stated below.

**What this branch therefore carries:** the arithmetic (runnable), the four costs, the comparison
against the competing lever, the profiler question answered from our own record, and the format
contract as an unimplemented skeleton.

**What a real implementation would be, in order, each step gated:**

| step | what | where | gate before it starts |
|---|---|---|---|
| 1 | the quantizer, one row first, round trip on a held-out slice | `gguf-tools/engram-4bit/` | none, and it needs no GPU |
| 2 | the format contract in our layout: row width, encoding tag, geometry at `ds4.c:3183` | `ds4_engram.h`, `ds4_engram.c`, `ds4.c` | step 1 verified bit-exact |
| 3 | the host dequant branch in `ds4_engram_read` (`ds4_engram.c:198-219`) | `ds4_engram.c` | a CUDA build (OWED, see section 6) |
| 4 | quality: teacher-forced NLL against the FP8 tables, idle to idle, code and needle strata apart | our eval harness | step 3 built |
| 5 | only then, a switch | n/a | step 4 passing |

---

## 4. The four costs, each attributed to the source that states it

**(a) The runtime family is not ours, so we write both halves.**
`sources/hf-hlwq-engram-q4.md:102`: "Validated only through SGLang with the row-store adapter on
SM120. MLX/GGUF runtimes have their own Engram quantizers (e.g. mixed 4/8-bit with 6-bit Engram);
this repo does not target them." Same finding in our own study at
`research/v41-flash-landscape/07-.../CODE.md:106-109` and `HOW-WE-USE-IT.md:121-124`, which adds the
license consequence: the base is MIT and the method is credited to HLWQ by Caio Vicentino
(`sources/hf-hlwq-engram-q4.md:107`, arXiv:2603.29078), so attribution travels with any port.

**(b) SM120 validated, SM121 is ours, and the prefill patch's applicability is unknown.**
`sources/hf-hlwq-engram-q4.md:34` and `:83`: produced and validated on 4x RTX PRO 6000 (SM120). The
card also ships an SM120 prefill patch and says datacenter Blackwell does not need it
(`sources/hf-hlwq-engram-q4.md:44`, and `CODE.md:102-103` naming `patches/`, context
flashinfer#5095). Our chip is GB10, SM121, and the card says nothing about it. Nothing in the card
establishes that the patch, or the validation, carries.

**(c) The quality cost is asymmetric, and it is stated in nats a token.**
`sources/hf-hlwq-engram-q4.md:87-91`, teacher-forced NLL a token, FP8 to HLWQ Q4: pt-BR prose
0.0668 to 0.0601 (better), code 0.1086 to 0.1233 (**+0.0147, about +0.015**), needle at 3 to 52 k
context 0.0840 to 0.1228 (**+0.0388, about +0.04**). All needles are still retrieved, 5 of 5 at
3.4 k, 9.6 k, 22 k, 39 k and 52 k tokens, identical answers to FP8 with wording differing once
(`:93`). The card's own caution is that under concurrent traffic the same metric moves by up to
+/-0.015 by itself, so the comparison must be idle to idle (`:85`). Against our own weights, the
0.015 is the same order as the measurement floor the card itself declares, and the 0.04 is not.

**(d) The card's win was measured against a host-RAM row store, which is a different shape from
ours.** The card serves the tables from host RAM through a `cudaLaunchHostFunc` row store, with an
NVMe mode if host RAM is under 128 GB (`CODE.md:99-100`, and `sources/hf-hlwq-engram-q4.md:79`), and
its dequant is three GPU tensor ops against a ~40 ms forward (`sources/hf-hlwq-engram-q4.md:97`).
Our path is a plain host `pread` loop with a host dequant
(`ds4_engram.c:125-138`, `:198-219`), and it dequantizes with 256 `ldexpf` calls a row today. A
4-bit row would replace that with a 256-entry gather plus two 128 x 128 rotations a row, which is
about 128 times the arithmetic a row on the same host thread. The card's "invisible" is a statement
about a GPU kernel; it is not a statement about our host path.

### 4.1 The open question, and the answer our own record already holds

The open question was whether our Engram path is bound by **table bytes** or by **lookup ops**,
because the card's win was measured against a host-RAM row store rather than ours. Our own lane
answered it before this branch existed, in writing:

- `CUDA_LANES_CHANNEL.md:960-970`: "12,672 bytes arrive in 48 SEPARATE `pread` CALLS at queue depth
  one ... This is a LATENCY problem, not a bandwidth one - 48 round trips to move twelve kilobytes.
  Anyone reaching for a bandwidth lever here is aiming at the wrong axis."
- The measured exposure is 23.9 ms of 198.6 ms, 12.04 percent (`CUDA_LANES_CHANNEL.md:1635`).
- 12,672 B in 23.9 ms is **0.53 MB/s**, which is 0.00023 percent of the 230.5 GB/s achieved ceiling.

So: **bound by lookup ops, by about four orders of magnitude.** Halving the row bytes cannot halve
that 23.9 ms. The lever that does attack it is concurrency on the read, which is already its own
branch (`triple-engram-read-threads`, and the finding at `CUDA_LANES_CHANNEL.md:639-680` that a
decode token asks for 24 rows a table and the old divisor of 2 with a cap of 16 forced two serial
rounds). The Engram byte lever and the Engram latency lever are on different axes and do not
compete; the byte lever simply cannot move the number that matters here.

---

## 5. The competing byte lever, and which one first

Our roadmap names its own byte lever, and this branch must not pretend to be the only one:

| lever | axis | status | a token | on disk |
|---|---|---|---|---|
| **non-routed k-quant repack** | per token bytes read whole | **Q6_K gate PASSED, f = 0.999** | 1.763 GB of 9.670 GB, **18.23 percent**, `~1.22x @c=1 PROJECTED` | 7.742 to 5.979 GB |
| **Engram 4-bit re-encode** (this plan) | footprint and download | ungated: no artifact, no SM121, no quality run on our file | 6,336 B, **0.0000655 percent** | 202.758 to 101.379 GB |

Sources for the repack row: `WIKI/theory/48-decode-speedup-levers.md:383` ("Non-routed k-quant
repack ... The ONE large single-box byte lever. Q6_K gate PASSED (f=0.999); ~1.22x @c=1 PROJECTED,
small-band"), `WIKI/theory/82-census-corrections-2026-07-18.md:44-48` and `:90` ("Conservative
projected saving 1.763 GB/token; with the repack (9.67 to 7.91 GB) at today's implied 129.6 GB/s:
~16.4 t/s, ~1.22x @c=1, PROJECTED"), and the census's own Q1 line, reproduced in section 2.2 above.
The two derivations of 1.763 GB agree independently.

**Which first: the repack, and it is not close.** The repack cuts 18.23 percent of the bytes a token
actually moves, on the exact tensors that are read whole every token (7.742 GB of a 9.670 GB floor),
and it is already gated. The Engram re-encode cuts 0.0000655 percent of them. Per byte of disk
saved, the repack returns about 278,000 times more per token benefit, which is the number the
arithmetic script prints.

**And the routed-expert side is nearly exhausted, which is why this comparison matters at all.**
`WIKI/theory/71-moe-decode-kv-quant-frontier-2026.md:84` is titled "3. Low-bit weight quant -
beating the IQ2 floor", and our own census puts routed experts at **2.2500 bits a weight** against
the IQ2_XXS floor of **2.0625** (`WIKI/theory/82-census-corrections-2026-07-18.md:62`, and
`WIKI/theory/05-gguf-and-quant.md:75` and `:103` for the floor). That is **0.1875 bits a weight**
of headroom on 89.9 percent of the file, so the routed axis is nearly spent and the remaining byte
levers really are the non-routed repack and, on a different axis, footprint levers like this one.

**Where the Engram re-encode does earn its place**, once the repack has landed: 101.379 GB off the
file and off the download is a capacity fact, not a speed fact. It is what makes a 246 GiB artifact
movable, stageable and resident where a 340.6 GiB one is not, and it is the single largest byte
reduction available to us that does not touch a per token path at all. That is a real reason to do
it, and it is a different reason from the one the card gives.

---

## 6. What is owed, stated plainly

- **The CUDA build gate is OWED.** A CUDA build needs the Spark and the Spark is running an
  unrelated series, so no build was attempted and none is claimed. Nothing on this branch compiles
  into an engine change anyway, since the branch carries no engine code.
- **No tokens/s number exists for this lever and none is claimed.** The two `____` cells in the root
  README are owed, not zero, and there is no arm to measure until steps 1 to 4 of section 3 exist.
- **The quality run is OWED**: teacher-forced NLL on our own prompts against the FP8 tables, idle to
  idle, with the code and needle strata reported separately, because the card's cost is asymmetric
  and lands exactly on the two strata we care about.
- **The quantizer is unimplemented**, deliberately. `engram_hlwq_quantize.py --check-format` states
  the format contract and prints that it encodes zero rows.
- **One thing this branch does NOT settle**: whether a 4-bit row on our host dequant path is fast
  enough to be free. The arithmetic says it is about 128 times the per row work of the current
  256 `ldexpf` calls, and the card's "three tiny ops" was measured on a GPU. That is a measurement
  for whoever builds step 3, not a claim here.

---

## 7. Sources, one list

- Card: `research/v41-flash-landscape/07-heterogeneous-spark-plus-5090/sources/hf-hlwq-engram-q4.md`
  (lines cited inline). Study in the same folder: `CODE.md`, `HOW-WE-USE-IT.md`, `README.md`.
- Our engine, branch `triple-antirez-tip-latest`: `ds4_engram.h:9,12,13,14,48,49-50`,
  `ds4_engram.c:93,125-138,198-219`, `ds4.c:3183,3383-3384,6912,6925`.
- Our census: `ds4-gguf-census.py` (Q1, Q3, Q5 output quoted from a run on 2026-09-16 against the
  86.72 GB GGUF present on this machine), and
  `WIKI/theory/177-the-sparkport-day-...-2026-09-13.md:61-62`.
- Our Engram read record: `CUDA_LANES_CHANNEL.md:639-700` and `:940-1010` and `:1635`.
- Our competing lever: `WIKI/theory/48-decode-speedup-levers.md:383`,
  `WIKI/theory/82-census-corrections-2026-07-18.md:44-48,62,90`,
  `WIKI/theory/05-gguf-and-quant.md:75,103`,
  `WIKI/theory/71-moe-decode-kv-quant-frontier-2026.md:84`.
- Prior art on the Engram read path in our own research tree:
  `research/v41-flash-landscape/03-triple-spark-mxfp4-engram/CODE.md:95-96`, which already records
  "our tables total 202.76 GB against their 189 GiB", independently of the arithmetic here.
