#!/usr/bin/env python3
"""engram_hlwq_quantize.py - the Engram 4-bit re-encode. NOT IMPLEMENTED.

THIS FILE ENCODES NO ROW AND DECODES NO ROW TODAY. It carries the format
contract for the re-encode and the function shapes a later implementer fills in.
Every quantizer entry point raises NotImplementedError on purpose. There is no
switch, no environment variable and no code path here that an engine could take.

WHY IT IS HERE AND NOT IN THE ENGINE
  The re-encode changes the ARTIFACT, not the engine. It rewrites two tensors in
  a checkpoint and leaves the other 1326 byte-identical. Nothing about it needs a
  GPU, a build, or the engine's runtime. The engine-side half (a dequant on the
  lookup path) is a separate, smaller change and is not written here; see PLAN.md.

THE FORMAT (the whole contract, and the only thing this file states as fact)
  In, per row, exactly as the engine reads it today:
      264 B = 256 x E4M3 weight bytes followed by 8 x E8M0 scale bytes,
      one E8M0 scale per 32 weights.
      Source: ds4_engram.h:13-14 and :48 on triple-antirez-tip-latest, and the
      GGUF metadata key deepseek41.engram.encoding whose value is the string
      "e4m3_e8m0_32_row264" (ds4.c:6912).
  Out, per row:
      132 B = 128 packed nibbles (two 4-bit codes a byte, low nibble = even
      index) followed by 2 x F16 block norms.
      Source: research/v41-flash-landscape/07-heterogeneous-spark-plus-5090/
      sources/hf-hlwq-engram-q4.md:28.
  Row count, the two tables and only these two:
      blk.1.engram_embd.weight   384,006,168 rows
      blk.14.engram_embd.weight  384,016,682 rows
      Source: ds4.c:6925 on triple-antirez-tip-latest.
  Per row, the transform (four steps, two 128-dim blocks):
      1. normalize the block by its L2 norm, keep the norm as F16;
      2. rotate by the normalized Walsh-Hadamard matrix H128;
      3. scale by sqrt(128) and snap each coordinate to the nearest of 16
         Lloyd-Max centroids for N(0,1);
      4. pack two 4-bit codes a byte.
      100 fixed-point conditional-expectation iterations build the centroids.
      Source: sources/hf-hlwq-engram-q4.md:66-75. Method: HLWQ by Caio
      Vicentino, arXiv:2603.29078, cited at that file's line 107.
  Dequant is the mirror: centroids[codes] -> (.) @ H128 -> x norm. Three tensor
  ops, no side data, because H is orthogonal and its own inverse (same file,
  line 73).

  AND THE WHOLE FILE IS A SECOND QUANTIZATION. The tables are already FP8 in the
  official checkpoint, so this is FP8 re-encoded, not FP8 lost (same file, line
  101). Inherent relative RMSE per row is about 9.7 percent (line 30).

WHAT THE ENGINE WOULD NEED, AND IS NOT WRITTEN HERE
  1. a row-width and encoding identity on the table handle, so a table can be
     opened as 264 or 132 B a row;
  2. a dequant branch in ds4_engram_read (ds4_engram.c:198-219 today) that runs
     the mirror above on the host, where the current path runs 256 ldexpf calls
     a row;
  3. the geometry site at ds4.c:3183, which subtracts the 8 scale bytes a row
     from the parameter count and would subtract 4 instead;
  4. a quality gate on the new artifact: teacher-forced NLL against the FP8
     tables, idle to idle, because the cost is asymmetric (code and retrieval
     worse, prose slightly better).
"""

import argparse
import sys

# --- the format, as constants, so nothing downstream has to restate it --------

ENGRAM_LAYERS = (1, 14)
ENGRAM_DIM = 256
ENGRAM_ROW_BYTES_IN = 264
ENGRAM_SCALE_BYTES_IN = 8
ENGRAM_ROW_BYTES_OUT = 132
ENGRAM_CODES_OUT = 128
ENGRAM_NORMS_OUT = 2
ENGRAM_NORM_BYTES = 2
ENGRAM_BLOCK = 128
ENGRAM_CENTROIDS = 16
ENGRAM_TABLES = (
    ("blk.1.engram_embd.weight", 384_006_168),
    ("blk.14.engram_embd.weight", 384_016_682),
)

# The four tensor names a finished re-encode would publish, per the card's own
# file list (sources/hf-hlwq-engram-q4.md:53-55): the codes and the norms.
OUT_SUFFIX = (".hlwq_codes", ".hlwq_norms")


def format_contract() -> dict:
    """State the format. Asserts only the byte identity, encodes nothing."""
    assert ENGRAM_ROW_BYTES_IN == ENGRAM_DIM + ENGRAM_SCALE_BYTES_IN, (
        "the input row is 256 E4M3 bytes plus 8 E8M0 scales")
    assert ENGRAM_ROW_BYTES_OUT == ENGRAM_CODES_OUT + (
        ENGRAM_NORMS_OUT * ENGRAM_NORM_BYTES), (
        "the output row is 128 packed nibbles plus 2 F16 norms")
    assert 2 * ENGRAM_CODES_OUT == ENGRAM_DIM, "two 4-bit codes a byte, 256 dims"
    assert ENGRAM_BLOCK * 2 == ENGRAM_DIM, "two 128-dim blocks a row"
    assert 2 ** 4 == ENGRAM_CENTROIDS, "4 bits, 16 Lloyd-Max centroids"
    rows = sum(rows for _, rows in ENGRAM_TABLES)
    return {
        "row_bytes_in": ENGRAM_ROW_BYTES_IN,
        "row_bytes_out": ENGRAM_ROW_BYTES_OUT,
        "rows": rows,
        "bytes_in": rows * ENGRAM_ROW_BYTES_IN,
        "bytes_out": rows * ENGRAM_ROW_BYTES_OUT,
        "bytes_saved": rows * (ENGRAM_ROW_BYTES_IN - ENGRAM_ROW_BYTES_OUT),
    }


def build_centroids(iterations: int = 100) -> list:
    """Build the 16 Lloyd-Max centroids for N(0,1). NOT IMPLEMENTED.

    The card fixes `iterations` at 100 conditional-expectation steps in
    fixed-point (sources/hf-hlwq-engram-q4.md:70). The card's own centroid
    tables ship in hlwq_config_layer{1,14}.json (line 56), so a first
    implementation should consume those rather than rebuild them, and this
    function exists only so the plan has a named home for the rebuild.
    """
    raise NotImplementedError(
        "no centroid table is built here; take hlwq_config_layer{1,14}.json")


def hadamard128() -> list:
    """The normalized H128. NOT IMPLEMENTED."""
    raise NotImplementedError("H128 is not built here")


def quantize_row(row: bytes, centroids: list) -> bytes:
    """264 B in, 132 B out. NOT IMPLEMENTED.

    A correct implementation is deterministic and needs no calibration data, no
    Hessian and no gradient (sources/hf-hlwq-engram-q4.md:75). It must be
    validated for BIT-EXACT dequant round trips on a held-out slice before it
    touches either table, and it must refuse to write a partial table.
    """
    raise NotImplementedError("no row is quantized here")


def quantize_table(src: str, dst: str, tensor: str, rows: int,
                   centroids: list) -> None:
    """Re-encode one of the two tables. NOT IMPLEMENTED.

    Only blk.1.engram_embd.weight and blk.14.engram_embd.weight may be named
    here. Every other tensor in the checkpoint stays byte-identical, which is the
    card's own claim about its 46 untouched shards
    (sources/hf-hlwq-engram-q4.md:22).
    """
    if tensor not in [name for name, _ in ENGRAM_TABLES]:
        raise ValueError(f"{tensor} is not one of the two Engram hash tables")
    raise NotImplementedError("no table is re-encoded here")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Engram 4-bit re-encode. NOT IMPLEMENTED; see PLAN.md.")
    parser.add_argument("--check-format", action="store_true",
                        help="print the format contract and exit; encodes nothing")
    parser.add_argument("--source", help="checkpoint to read (not used yet)")
    parser.add_argument("--output", help="checkpoint to write (not used yet)")
    args = parser.parse_args(argv)

    if args.check_format:
        contract = format_contract()
        print("Engram 4-bit re-encode, format contract")
        print(f"  row in          {contract['row_bytes_in']} B"
              " (256 E4M3 + 8 E8M0)")
        print(f"  row out         {contract['row_bytes_out']} B"
              " (128 packed nibbles + 2 F16 norms)")
        print(f"  rows            {contract['rows']:,} across"
              f" {len(ENGRAM_TABLES)} tables")
        print(f"  bytes in        {contract['bytes_in']:,} B"
              f" = {contract['bytes_in'] / 1e9:.3f} GB")
        print(f"  bytes out       {contract['bytes_out']:,} B"
              f" = {contract['bytes_out'] / 1e9:.3f} GB")
        print(f"  saved           {contract['bytes_saved']:,} B"
              f" = {contract['bytes_saved'] / 1e9:.3f} GB")
        print("  rows encoded by this file today: 0")
        return 0

    if args.source or args.output:
        parser.error("the quantizer is not implemented; see PLAN.md")

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
