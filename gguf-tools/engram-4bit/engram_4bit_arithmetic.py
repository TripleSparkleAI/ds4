#!/usr/bin/env python3
"""engram_4bit_arithmetic.py - the byte arithmetic for a 4-bit Engram table.

Reads no model and needs no GPU. Every input is a constant with its source
written beside it, so any figure this prints can be checked against the file it
came from. Run it with no arguments.

  python3 gguf-tools/engram-4bit/engram_4bit_arithmetic.py

WHAT THIS IS NOT: it is not a census. The Engram tables are not present in the
GGUF on this machine (the file here, 86.72 GB, carries 1328 tensors and zero
Engram tensors), and `ds4-gguf-census.py` has no Engram bucket. The table
geometry below is therefore taken from the engine's own constants and from our
header-only reads of the 340.6 GiB file, and the per token figures are ARITHMETIC
on top of those, not a measurement of this branch.
"""

# --- arguments of the row, from the engine -----------------------------------
# ds4.c:6925 on triple-antirez-tip-latest: the two Engram tables' row counts.
ROWS_LAYER1 = 384_006_168
ROWS_LAYER14 = 384_016_682
# ds4_engram.h:14 - DS4_ENGRAM_ROW_BYTES = 264 (256 E4M3 bytes + 8 E8M0 scales).
FP8_ROW_BYTES = 264
# ds4_engram.h:9 and :12 - DS4_ENGRAM_LAYERS = 2, DS4_ENGRAM_COLS = 24.
# A decode step issues COLS rows per table, so 48 rows a step.
LAYERS = 2
COLS = 24
# The 4-bit target row: 128 packed nibbles + 2 fp16 norms.
# research/v41-flash-landscape/07-heterogeneous-spark-plus-5090/
#   sources/hf-hlwq-engram-q4.md:28.
Q4_ROW_BYTES = 132

# --- arguments of the file, from our own census ------------------------------
# WIKI/theory/177-the-sparkport-day-...-2026-09-13.md:61-62: the 340.6 GiB file,
# mapped tensor span 151.8 GiB, the other ~189 GiB Engram tables outside the map.
FILE_GIB_RECORDED = 340.6
# ds4-gguf-census.py Q3 output, 2026-09-16: TOTAL = 9.670 GB/token.
CENSUS_GB_PER_TOKEN = 9.670
# ds4-gguf-census.py: achieved sequential ceiling on GB10, never the 273 peak.
ACHIEVED_GBPS = 230.5

# --- arguments of the measured read, from the CUDA lane channel --------------
# CUDA_LANES_CHANNEL.md:1635, and :962-966: the Engram read is 48 serial
# 264-byte preads, 23.9 ms of a 198.6 ms step, 12.04 percent.
ENGRAM_STEP_MS = 23.9
DECODE_STEP_MS = 198.6

# --- the competing lever, from our own roadmap -------------------------------
# ds4-gguf-census.py Q1 output, 2026-09-16: f = 0.999, projected saving 1.763 GB.
# WIKI/theory/82-census-corrections-2026-07-18.md:44-48 and :90.
REPACK_SAVING_GB_PER_TOKEN = 1.763
# WIKI/theory/82-census-corrections-2026-07-18.md:62-63: bits per weight by bucket.
ROUTED_BPW = 2.2500
IQ2_XXS_BPW = 2.0625

GB = 1_000_000_000
GIB = 1024 ** 3


def main() -> None:
    rows = ROWS_LAYER1 + ROWS_LAYER14
    fp8 = rows * FP8_ROW_BYTES
    q4 = rows * Q4_ROW_BYTES
    saved = fp8 - q4

    print("THE TABLE, BY ARGUMENT")
    print(f"  rows, layer 1 and layer 14     {ROWS_LAYER1:,} + {ROWS_LAYER14:,}"
          f" = {rows:,}")
    print(f"  parameters in the pair         {rows * 256 / 1e9:.1f} G (256 dims a row)")
    print(f"  FP8 at {FP8_ROW_BYTES} B a row           {fp8:,} B"
          f" = {fp8 / GB:.3f} GB = {fp8 / GIB:.3f} GiB")
    print(f"  4-bit at {Q4_ROW_BYTES} B a row          {q4:,} B"
          f" = {q4 / GB:.3f} GB = {q4 / GIB:.3f} GiB")
    print(f"  saved                          {saved:,} B"
          f" = {saved / GB:.3f} GB = {saved / GIB:.3f} GiB")
    print(f"  the saving is exactly one half of the table bytes:"
          f" {saved * 2 == fp8}")

    print()
    print("THE FILE")
    after_gib = FILE_GIB_RECORDED - saved / GIB
    print(f"  file as recorded               {FILE_GIB_RECORDED:.1f} GiB"
          f" = {FILE_GIB_RECORDED * GIB / GB:.2f} GB")
    print(f"  file after the re-encode       {after_gib:.1f} GiB"
          f" = {after_gib * GIB / GB:.2f} GB")
    print(f"  Engram share of the file       {saved * 2 / GIB / FILE_GIB_RECORDED:.1%}")

    print()
    print("THE PER TOKEN READ")
    per_step = LAYERS * COLS * FP8_ROW_BYTES
    per_step_q4 = LAYERS * COLS * Q4_ROW_BYTES
    print(f"  rows read a decode step        {LAYERS} tables x {COLS} cols"
          f" = {LAYERS * COLS} rows")
    print(f"  bytes read a decode step       {per_step:,} B")
    print(f"  bytes after the re-encode      {per_step_q4:,} B")
    print(f"  saved a step                   {per_step - per_step_q4:,} B")
    print(f"  as a share of the census floor {per_step - per_step_q4:,} B of"
          f" {CENSUS_GB_PER_TOKEN:.3f} GB = "
          f"{(per_step - per_step_q4) / (CENSUS_GB_PER_TOKEN * GB):.3e}"
          f" = {(per_step - per_step_q4) / (CENSUS_GB_PER_TOKEN * GB) * 100:.7f} percent")
    ns = (per_step - per_step_q4) / (ACHIEVED_GBPS * GB) * 1e9
    print(f"  at the achieved {ACHIEVED_GBPS} GB/s     {ns:.1f} ns a token")

    print()
    print("WHAT THE READ ACTUALLY COSTS, FROM THE RECORD")
    print(f"  the Engram read, a step        {ENGRAM_STEP_MS} ms of {DECODE_STEP_MS} ms"
          f" = {ENGRAM_STEP_MS / DECODE_STEP_MS:.2%}")
    rate_mbs = per_step / (ENGRAM_STEP_MS / 1e3) / 1e6
    print(f"  the rate that implies          {per_step:,} B in {ENGRAM_STEP_MS} ms"
          f" = {rate_mbs:.2f} MB/s")
    print(f"  against the {ACHIEVED_GBPS} GB/s ceiling  "
          f"{rate_mbs / 1e3 / ACHIEVED_GBPS:.3e}"
          f" = {rate_mbs / 1e3 / ACHIEVED_GBPS * 100:.6f} percent of it")
    print(f"  so the exposure cannot be halved by halving the row bytes:")
    print(f"    it is {LAYERS * COLS} round trips, not a byte count")

    print()
    print("THE OTHER BYTE LEVER, FOR COMPARISON")
    repack_pct = REPACK_SAVING_GB_PER_TOKEN / CENSUS_GB_PER_TOKEN
    engram_pct = (per_step - per_step_q4) / (CENSUS_GB_PER_TOKEN * GB)
    print(f"  non-routed repack, a token     {REPACK_SAVING_GB_PER_TOKEN:.3f} GB of"
          f" {CENSUS_GB_PER_TOKEN:.3f} GB = {repack_pct:.2%}")
    print(f"  Engram re-encode, a token      {engram_pct * 100:.7f} percent")
    print(f"  ratio, repack over Engram      {repack_pct / engram_pct:,.0f} to 1")
    print(f"  routed experts sit at          {ROUTED_BPW:.4f} bpw against the"
          f" IQ2_XXS floor of {IQ2_XXS_BPW:.4f}")
    print(f"  headroom above that floor      {ROUTED_BPW - IQ2_XXS_BPW:.4f} bits a weight")


if __name__ == "__main__":
    main()
