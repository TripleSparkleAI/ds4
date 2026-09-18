#!/usr/bin/env python3
"""engram_hlwq_quantize.py - re-encode the two Engram hash tables to 4-bit, as a SIDECAR.

# <claudes_code_comments>
# ** Function List **
# lloyd_max_centroids(levels, iterations) - 16 Lloyd-Max levels for N(0,1), fixed point
# hadamard(n) - the unnormalized Sylvester Walsh-Hadamard matrix W_n, entries +-1
# e4m3_lut() - the 256-entry E4M3 decode table, the engine's own formula
# decode_fp8_rows(raw) - 264 B rows -> BF16-rounded float32 [n,256], as the engine reads them
# encode_rows(x, centroids) - float32 [n,256] -> (codes u8 [n,128], norms f16 [n,2])
# decode_rows(codes, norms, centroids) - the mirror, float32 [n,256]
# pack_rows(codes, norms) - 132 B rows: 128 packed nibbles + 2 x F16 norms
# unpack_rows(raw) - 132 B rows -> (codes, norms)
# parse_gguf(path) - header only: metadata, tensors, data offset
# engram_tensors(path) - the two tables' absolute offsets and row counts from a GGUF
# write_header(fp, centroids, tables) - the 4096 B sidecar header
# read_header(path) - parse and validate a sidecar header
# convert(src, dst, limit_rows, chunk) - the converter, refuses to leave a partial file
# check(src, sidecar, samples) - spot-check random rows: FP8 vs sidecar, RMSE + cosine
# emit_fixture(directory, rows) - synthetic sidecar + expected floats for the C test
# selftest() - synthetic table round trip + three planted mutations that must go RED
# main(argv) - CLI
#
# ** Technical Review **
# The two Engram tables in a DeepSeek V4.1 Flash GGUF (blk.1 / blk.14 engram_embd.weight)
# are stored as 264 B rows: 256 E4M3 bytes + 8 E8M0 scales (ds4_engram.h:48). The engine
# decodes a row to BF16-rounded floats (ds4_engram.c, ds4_engram_read). This tool reads
# those rows, decodes them EXACTLY as the engine does, and re-encodes each row with the
# HLWQ 4-bit scheme (Caio Vicentino, arXiv:2603.29078): per 128-dim block, divide by the
# block's L2 norm (kept as F16), rotate by the normalized Hadamard H = W/sqrt(128), scale by
# sqrt(128) (so y = W u, unit-variance coordinates), snap to the nearest of 16 Lloyd-Max
# centroids for N(0,1), pack two codes a byte. Dequant is norm * W c / 128 because H is
# orthogonal and symmetric. The result is a SIDECAR file (never a rewrite of the GGUF): a
# 4096 B header carrying the centroids and the per-table geometry, then the packed rows.
# The engine loads it behind DS4_ENGRAM_4BIT=<path> (ds4_engram.c, ds4_engram_read).
# Constants: DIM 256, BLOCK 128, ROW_IN 264, ROW_OUT 132, 16 centroids, header 4096 B.
# The converter writes to <dst>.partial and renames only on completion; a --limit-rows run
# records the limited row count in the header so the engine refuses it against the real
# table (row count mismatch), which is the intended failure.
# </claudes_code_comments>
"""

import argparse
import math
import os
import struct
import sys
import time

import numpy as np

# --- the format, as constants ---------------------------------------------------

ENGRAM_LAYERS = (1, 14)
ENGRAM_DIM = 256
ENGRAM_BLOCK = 128
ENGRAM_ROW_BYTES_IN = 264
ENGRAM_ROW_BYTES_OUT = 132
ENGRAM_CODES_OUT = 128
ENGRAM_CENTROIDS = 16
SIDECAR_MAGIC = b"DS4ENG4B"
SIDECAR_VERSION = 1
SIDECAR_HEADER_BYTES = 4096
SIDECAR_ALIGN = 4096
LLOYD_ITERATIONS = 1000  # the card says 100; from a quantile start 100 is not converged (2.7175), 1000 is (2.7326)

# --- the codebook ---------------------------------------------------------------


def _phi(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def _Phi(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def lloyd_max_centroids(levels=ENGRAM_CENTROIDS, iterations=LLOYD_ITERATIONS):
    """Lloyd-Max levels for N(0,1): fixed-point conditional-expectation iteration.

    Thresholds are midpoints of neighbouring centroids; each centroid is the
    conditional mean of N(0,1) on its cell. From the quantile initialisation
    100 iterations (the card's count) leave the outer level at 2.7175; 300 and
    beyond agree with Max (1960) at 2.7326, so the default is 1000. The
    codebook travels in the sidecar header, so the engine never rebuilds it.
    """
    # quantile initialisation, symmetric
    c = []
    for i in range(levels):
        p = (i + 0.5) / levels
        # inverse normal CDF by bisection (no scipy)
        lo, hi = -8.0, 8.0
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if _Phi(mid) < p:
                lo = mid
            else:
                hi = mid
        c.append(0.5 * (lo + hi))
    for _ in range(iterations):
        t = [-math.inf] + [0.5 * (c[i] + c[i + 1]) for i in range(levels - 1)] + [math.inf]
        new = []
        for i in range(levels):
            a, b = t[i], t[i + 1]
            pa = 0.0 if a == -math.inf else _phi(a)
            pb = 0.0 if b == math.inf else _phi(b)
            Pa = 0.0 if a == -math.inf else _Phi(a)
            Pb = 1.0 if b == math.inf else _Phi(b)
            new.append((pa - pb) / (Pb - Pa))
        c = new
    return np.asarray(c, dtype=np.float32)


_HADAMARD = {}


def hadamard(n=ENGRAM_BLOCK):
    """Sylvester Walsh-Hadamard W_n, entries +-1, symmetric, W W = n I."""
    if n not in _HADAMARD:
        w = np.array([[1]], dtype=np.int32)
        while w.shape[0] < n:
            w = np.block([[w, w], [w, -w]])
        assert w.shape == (n, n)
        _HADAMARD[n] = np.ascontiguousarray(w.astype(np.float32))
    return _HADAMARD[n]


def _wht(v):
    """v [m,128] -> v @ W. numpy's BLAS on macOS raises spurious divide/overflow
    warnings on this float32 matmul while returning correct values (the C
    cross-check in tests/test_engram.c agrees to 6e-8); the result is asserted
    finite instead of trusting the flag."""
    with np.errstate(all="ignore"):
        out = v @ hadamard()
    if not np.all(np.isfinite(out)):
        raise FloatingPointError("Walsh-Hadamard product is not finite")
    return out


# --- the FP8 side, the engine's exact decode ------------------------------------


def e4m3_lut():
    lut = np.zeros(256, dtype=np.float32)
    for byte in range(256):
        exponent, mantissa = (byte >> 3) & 15, byte & 7
        value = math.ldexp(8 + mantissa, exponent - 10) if exponent else math.ldexp(mantissa, -9)
        lut[byte] = -value if byte & 128 else value
    return lut


_E4M3 = e4m3_lut()


def bf16_round(x):
    """Round float32 to BF16 (nearest even), returned as float32; the engine does this."""
    bits = x.astype(np.float32).view(np.uint32)
    bits = (bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))) & np.uint32(0xFFFF0000)
    return bits.view(np.float32)


def decode_fp8_rows(raw):
    """raw: uint8 [n, 264] -> float32 [n, 256]. Mirrors ds4_engram_read's FP8 branch."""
    raw = np.asarray(raw, dtype=np.uint8).reshape(-1, ENGRAM_ROW_BYTES_IN)
    codes = raw[:, :ENGRAM_DIM]
    scales = raw[:, ENGRAM_DIM:].astype(np.int32)  # [n, 8], one per 32 weights
    if np.any((codes & 127) == 127) or np.any(scales == 255):
        raise ValueError("FP8 row carries a NaN code or a NaN scale (EDOM in the engine)")
    value = _E4M3[codes] * np.ldexp(np.float32(1.0), (scales - 127)).repeat(32, axis=1).astype(np.float32)
    value = bf16_round(value)
    if not np.all(np.isfinite(value)):
        raise ValueError("FP8 row decodes to a non-finite value (EDOM in the engine)")
    return value


# --- the 4-bit side -------------------------------------------------------------


def encode_rows(x, centroids):
    """float32 [n,256] -> codes uint8 [n,128] (one code a byte, unpacked), norms float16 [n,2]."""
    x = np.asarray(x, dtype=np.float32).reshape(-1, ENGRAM_DIM)
    n = x.shape[0]
    blocks = x.reshape(n * 2, ENGRAM_BLOCK)
    norms = np.linalg.norm(blocks.astype(np.float64), axis=1).astype(np.float32)
    norms16 = norms.astype(np.float16)
    # dequant uses the F16 norm, so encode against the F16-rounded norm too
    safe = np.where(norms16 > 0, norms16.astype(np.float32), np.float32(1.0))
    u = blocks / safe[:, None]
    y = _wht(u)  # y = W u, unit-variance coordinates (H u * sqrt(128))
    mids = 0.5 * (centroids[:-1] + centroids[1:])
    codes = np.searchsorted(mids, y).astype(np.uint8)  # nearest centroid
    codes[norms16 == 0] = int(np.argmin(np.abs(centroids)))
    return codes.reshape(n, 2 * ENGRAM_BLOCK), norms16.reshape(n, 2)


def decode_rows(codes, norms, centroids):
    """The mirror: x = norm * W c / 128 per block. codes uint8 [n,256] unpacked, norms f16 [n,2]."""
    codes = np.asarray(codes, dtype=np.uint8).reshape(-1, ENGRAM_DIM)
    n = codes.shape[0]
    c = centroids[codes].reshape(n * 2, ENGRAM_BLOCK)
    x = _wht(c) * (np.float32(1.0) / np.float32(ENGRAM_BLOCK))
    x = x * np.asarray(norms, dtype=np.float16).astype(np.float32).reshape(n * 2, 1)
    return x.reshape(n, ENGRAM_DIM)


def pack_rows(codes, norms):
    """-> uint8 [n, 132]: low nibble = even index, high nibble = odd index; then 2 x F16."""
    codes = np.asarray(codes, dtype=np.uint8).reshape(-1, ENGRAM_DIM)
    packed = (codes[:, 0::2] & 15) | ((codes[:, 1::2] & 15) << 4)
    norm_bytes = np.asarray(norms, dtype="<f2").reshape(-1, 2).view(np.uint8).reshape(-1, 4)
    return np.concatenate([packed, norm_bytes], axis=1)


def unpack_rows(raw):
    raw = np.asarray(raw, dtype=np.uint8).reshape(-1, ENGRAM_ROW_BYTES_OUT)
    packed = raw[:, :ENGRAM_CODES_OUT]
    codes = np.empty((raw.shape[0], ENGRAM_DIM), dtype=np.uint8)
    codes[:, 0::2] = packed & 15
    codes[:, 1::2] = packed >> 4
    norms = raw[:, ENGRAM_CODES_OUT:].copy().view("<f2").reshape(-1, 2)
    return codes, norms


# --- GGUF header, read only -----------------------------------------------------

KV_TYPE_FMT = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<B",
               10: "<Q", 11: "<q", 12: "<d"}


def _read_str(f):
    (n,) = struct.unpack("<Q", f.read(8))
    return f.read(n).decode("utf-8", "replace")


def _read_kv(f, t):
    if t == 8:
        return _read_str(f)
    if t == 9:
        (at,) = struct.unpack("<I", f.read(4))
        (n,) = struct.unpack("<Q", f.read(8))
        if n > 200000:
            if at == 8:
                for _ in range(n):
                    _read_str(f)
            else:
                f.seek(n * struct.calcsize(KV_TYPE_FMT[at]), 1)
            return f"<array[{n}]>"
        return [_read_kv(f, at) for _ in range(n)]
    fmt = KV_TYPE_FMT[t]
    (v,) = struct.unpack(fmt, f.read(struct.calcsize(fmt)))
    return v


def parse_gguf(path):
    meta, tensors = {}, []
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError("not a GGUF file")
        (ver,) = struct.unpack("<I", f.read(4))
        if ver != 3:
            raise ValueError(f"GGUF v{ver} unsupported (need v3)")
        n_t, n_kv = struct.unpack("<QQ", f.read(16))
        for _ in range(n_kv):
            k = _read_str(f)
            (t,) = struct.unpack("<I", f.read(4))
            meta[k] = _read_kv(f, t)
        for _ in range(n_t):
            name = _read_str(f)
            (nd,) = struct.unpack("<I", f.read(4))
            dims = struct.unpack(f"<{nd}Q", f.read(8 * nd))
            tt, off = struct.unpack("<IQ", f.read(12))
            tensors.append((name, dims, tt, off))
        header_end = f.tell()
    align = meta.get("general.alignment", 32)
    data_off = (header_end + align - 1) // align * align
    return meta, tensors, data_off


def engram_tensors(path):
    """-> [(layer, rows, abs_offset)] for blk.1 and blk.14, in that order."""
    meta, tensors, data_off = parse_gguf(path)
    enc = meta.get("deepseek41.engram.encoding")
    if enc != "e4m3_e8m0_32_row264":
        raise ValueError(f"deepseek41.engram.encoding is {enc!r}, expected e4m3_e8m0_32_row264;"
                         " this GGUF carries no FP8 Engram tables this tool understands")
    out = []
    for layer in ENGRAM_LAYERS:
        name = f"blk.{layer}.engram_embd.weight"
        hit = [t for t in tensors if t[0] == name]
        if not hit:
            raise ValueError(f"{name} not in {path}")
        _, dims, tt, off = hit[0]
        if len(dims) != 2 or dims[0] != ENGRAM_ROW_BYTES_IN:
            raise ValueError(f"{name} dims {dims}, expected (264, rows)")
        out.append((layer, int(dims[1]), data_off + off))
    return out


# --- the sidecar ----------------------------------------------------------------


def table_offsets(rows_list):
    offsets, pos = [], SIDECAR_HEADER_BYTES
    for rows in rows_list:
        offsets.append(pos)
        pos = (pos + rows * ENGRAM_ROW_BYTES_OUT + SIDECAR_ALIGN - 1) // SIDECAR_ALIGN * SIDECAR_ALIGN
    return offsets, pos


def write_header(fp, centroids, tables):
    """tables: [(layer, rows, data_offset, source_offset)]"""
    fp.seek(0)
    hdr = bytearray(SIDECAR_HEADER_BYTES)
    struct.pack_into("<8sIIIIII", hdr, 0, SIDECAR_MAGIC, SIDECAR_VERSION, len(tables),
                     ENGRAM_DIM, ENGRAM_BLOCK, ENGRAM_ROW_BYTES_OUT, SIDECAR_ALIGN)
    struct.pack_into("<16f", hdr, 32, *[float(c) for c in centroids])
    for i, (layer, rows, off, src) in enumerate(tables):
        struct.pack_into("<IIQQQ", hdr, 96 + 32 * i, layer, rows, off, src, 0)
    fp.write(hdr)


def read_header(path):
    with open(path, "rb") as f:
        hdr = f.read(SIDECAR_HEADER_BYTES)
    if len(hdr) < SIDECAR_HEADER_BYTES:
        raise ValueError("sidecar header short")
    magic, version, ntab, dim, block, row_bytes, align = struct.unpack_from("<8sIIIIII", hdr, 0)
    if magic != SIDECAR_MAGIC or version != SIDECAR_VERSION:
        raise ValueError(f"bad sidecar magic/version {magic!r} {version}")
    if (dim, block, row_bytes, align) != (ENGRAM_DIM, ENGRAM_BLOCK, ENGRAM_ROW_BYTES_OUT, SIDECAR_ALIGN):
        raise ValueError("sidecar geometry mismatch")
    centroids = np.asarray(struct.unpack_from("<16f", hdr, 32), dtype=np.float32)
    if not np.all(np.diff(centroids) > 0):
        raise ValueError("sidecar centroids are not strictly increasing")
    tables = []
    for i in range(ntab):
        layer, rows, off, src, _ = struct.unpack_from("<IIQQQ", hdr, 96 + 32 * i)
        tables.append((layer, rows, off, src))
    return centroids, tables


def convert(src, dst, limit_rows=0, chunk=1 << 18, quiet=False):
    centroids = lloyd_max_centroids()
    tables = engram_tensors(src)
    rows_out = [min(rows, limit_rows) if limit_rows else rows for _, rows, _ in tables]
    offsets, total = table_offsets(rows_out)
    partial = dst + ".partial"
    t0 = time.time()
    with open(src, "rb") as fin, open(partial, "wb") as fout:
        fout.truncate(total)
        write_header(fout, centroids, [(layer, rows_out[i], offsets[i], src_off)
                                       for i, (layer, _, src_off) in enumerate(tables)])
        for i, (layer, _, src_off) in enumerate(tables):
            done = 0
            while done < rows_out[i]:
                n = min(chunk, rows_out[i] - done)
                fin.seek(src_off + done * ENGRAM_ROW_BYTES_IN)
                raw = np.frombuffer(fin.read(n * ENGRAM_ROW_BYTES_IN), dtype=np.uint8)
                if raw.size != n * ENGRAM_ROW_BYTES_IN:
                    raise IOError(f"short read in blk.{layer} at row {done}")
                x = decode_fp8_rows(raw)
                codes, norms = encode_rows(x, centroids)
                fout.seek(offsets[i] + done * ENGRAM_ROW_BYTES_OUT)
                fout.write(pack_rows(codes, norms).tobytes())
                done += n
                if not quiet and (done % (chunk * 16) == 0 or done == rows_out[i]):
                    el = time.time() - t0
                    print(f"  blk.{layer}: {done:,}/{rows_out[i]:,} rows  {el:.0f}s", flush=True)
        fout.flush()
        os.fsync(fout.fileno())
    os.replace(partial, dst)
    return total


def check(src, sidecar, samples=4096, seed=0):
    centroids, tables = read_header(sidecar)
    src_tables = engram_tensors(src)
    rng = np.random.default_rng(seed)
    report = []
    with open(src, "rb") as fin, open(sidecar, "rb") as fsc:
        for (layer, rows, off, src_off), (slayer, srows, soff) in zip(tables, src_tables):
            if layer != slayer or src_off != soff:
                raise ValueError(f"sidecar table blk.{layer} does not describe this GGUF")
            if rows != srows:
                print(f"  blk.{layer}: sidecar rows {rows:,} != GGUF rows {srows:,} (partial sidecar)")
            ids = np.sort(rng.integers(0, rows, size=min(samples, rows)))
            ref, got = [], []
            for r in ids:
                fin.seek(src_off + int(r) * ENGRAM_ROW_BYTES_IN)
                ref.append(np.frombuffer(fin.read(ENGRAM_ROW_BYTES_IN), dtype=np.uint8))
                fsc.seek(off + int(r) * ENGRAM_ROW_BYTES_OUT)
                got.append(np.frombuffer(fsc.read(ENGRAM_ROW_BYTES_OUT), dtype=np.uint8))
            x = decode_fp8_rows(np.stack(ref))
            codes, norms = unpack_rows(np.stack(got))
            y = decode_rows(codes, norms, centroids)
            report.append((layer, len(ids)) + row_error(x, y))
    return report


def row_error(x, y):
    """-> (relative RMSE per row, mean; mean cosine per row)"""
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    num = np.sqrt(((x - y) ** 2).sum(axis=1))
    den = np.sqrt((x ** 2).sum(axis=1))
    ok = den > 0
    rel = float(np.mean(num[ok] / den[ok]))
    cos = float(np.mean((x[ok] * y[ok]).sum(axis=1) / (den[ok] * np.sqrt((y[ok] ** 2).sum(axis=1)))))
    return rel, cos


# --- selftest -------------------------------------------------------------------


def _synthetic_fp8_rows(n, rng):
    """Gaussian rows encoded to E4M3 + E8M0 per 32-block, as the checkpoint stores them."""
    x = rng.standard_normal((n, ENGRAM_DIM)).astype(np.float32) * np.float32(0.02)
    raw = np.zeros((n, ENGRAM_ROW_BYTES_IN), dtype=np.uint8)
    for b in range(8):
        blk = x[:, b * 32:(b + 1) * 32]
        amax = np.abs(blk).max(axis=1)
        e = np.floor(np.log2(np.maximum(amax, 1e-30))).astype(np.int32) - 8  # E4M3 max 448 ~ 2^8.8
        e = np.clip(e, -127, 127)
        raw[:, ENGRAM_DIM + b] = (e + 127).astype(np.uint8)
        scaled = blk / np.ldexp(np.float32(1.0), e)[:, None]
        # nearest E4M3 by table search (finite codes only)
        finite = np.array([v for v in range(256) if (v & 127) != 127])
        vals = _E4M3[finite]
        order = np.argsort(vals)
        idx = np.searchsorted(vals[order], scaled)
        idx = np.clip(idx, 1, len(order) - 1)
        left, right = order[idx - 1], order[idx]
        pick = np.where(np.abs(vals[left] - scaled) <= np.abs(vals[right] - scaled), left, right)
        raw[:, b * 32:(b + 1) * 32] = finite[pick]
    return raw


def selftest():
    fails = []

    def expect(cond, name):
        print(f"  {'ok  ' if cond else 'RED '} {name}")
        if not cond:
            fails.append(name)

    c = lloyd_max_centroids()
    expect(c.shape == (16,) and np.all(np.diff(c) > 0), "centroids: 16, strictly increasing")
    expect(abs(float(c[7] + c[8])) < 1e-6, "centroids: symmetric about zero")
    expect(abs(float(c[15]) - 2.7326) < 1e-3, f"centroids: outer level {c[15]:.4f} = 2.7326 (Max 1960, 16 levels)")
    w = hadamard()
    expect(np.allclose(_wht(w), 128 * np.eye(128)) and np.array_equal(w, w.T), "hadamard: symmetric, W W = 128 I")

    rng = np.random.default_rng(7)
    raw = _synthetic_fp8_rows(2048, rng)
    x = decode_fp8_rows(raw)
    expect(np.all(np.isfinite(x)) and x.shape == (2048, 256), "fp8 decode: finite, [2048,256]")
    codes, norms = encode_rows(x, c)
    packed = pack_rows(codes, norms)
    expect(packed.shape == (2048, 132), "pack: 132 B a row")
    c2, n2 = unpack_rows(packed)
    expect(np.array_equal(c2, codes) and np.array_equal(n2.view(np.uint16), norms.view(np.uint16)),
           "pack/unpack: codes and norms byte-exact")
    y = decode_rows(c2, n2, c)
    rel, cos = row_error(x, y)
    expect(rel < 0.13, f"round trip: mean relative RMSE {rel:.4f} < 0.13 (card: 0.0967)")
    expect(cos > 0.99, f"round trip: mean cosine {cos:.5f} > 0.99 (card: 0.99536)")
    # a zero block must round-trip to zeros, not NaN
    z = np.zeros((1, 256), dtype=np.float32)
    zc, zn = encode_rows(z, c)
    expect(np.all(decode_rows(zc, zn, c) == 0), "zero row: decodes to zeros")

    # sidecar header round trip through a real file
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.engram4")
        with open(p, "wb") as fp:
            fp.truncate(SIDECAR_HEADER_BYTES)
            write_header(fp, c, [(1, 2048, 4096, 12345), (14, 2048, 4096 + 2048 * 132, 99999)])
        hc, ht = read_header(p)
        expect(np.array_equal(hc, c) and ht == [(1, 2048, 4096, 12345), (14, 2048, 4096 + 2048 * 132, 99999)],
               "header: writes and reads back")

    # planted mutation 1: double every stored norm -> the decode error must blow past the bound
    bad_norms = (norms.astype(np.float32) * 2).astype(np.float16)
    rel_bad, _ = row_error(x, decode_rows(codes, bad_norms, c))
    expect(rel_bad > 0.5, f"planted norm mutation goes RED: rel RMSE {rel_bad:.3f} > 0.5")
    # planted mutation 2: a corrupted centroid table (non-monotone) is refused by the header reader
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "bad.engram4")
        cbad = c.copy()
        cbad[3], cbad[4] = cbad[4], cbad[3]
        with open(p, "wb") as fp:
            fp.truncate(SIDECAR_HEADER_BYTES)
            write_header(fp, cbad, [(1, 1, 4096, 0), (14, 1, 8192, 0)])
        refused = False
        try:
            read_header(p)
        except ValueError:
            refused = True
        expect(refused, "planted centroid mutation goes RED: header reader refuses")
    # planted mutation 3: a scale byte flipped in the FP8 input changes the encoded row
    raw2 = raw.copy()
    raw2[0, ENGRAM_DIM] = raw2[0, ENGRAM_DIM] + 3
    c3, n3 = encode_rows(decode_fp8_rows(raw2[:1]), c)
    expect(not (np.array_equal(c3[0], codes[0]) and np.array_equal(n3[0], norms[0])),
           "planted FP8 scale mutation changes the encoded row")

    print(f"engram_hlwq_quantize selftest: {'PASS' if not fails else 'FAIL ' + ', '.join(fails)}")
    return 0 if not fails else 1


def emit_fixture(directory, rows=512, seed=3):
    """Write <dir>/sidecar (layer 1, source offset 0) and <dir>/expected.f32 for
    tests/test_engram.c (DS4_ENGRAM_4BIT_FIXTURE=<dir>): the python encoder's
    rows decoded by the python mirror, for the C dequant to reproduce."""
    os.makedirs(directory, exist_ok=True)
    c = lloyd_max_centroids()
    rng = np.random.default_rng(seed)
    x = decode_fp8_rows(_synthetic_fp8_rows(rows, rng))
    codes, norms = encode_rows(x, c)
    offsets, total = table_offsets([rows, 1])
    with open(os.path.join(directory, "sidecar"), "wb") as fp:
        fp.truncate(total)
        write_header(fp, c, [(1, rows, offsets[0], 0), (14, 1, offsets[1], 0)])
        fp.seek(offsets[0])
        fp.write(pack_rows(codes, norms).tobytes())
    decode_rows(codes, norms, c).astype("<f4").tofile(os.path.join(directory, "expected.f32"))
    print(f"fixture: {rows} rows in {directory}")
    print(f"NEXT -> DS4_ENGRAM_4BIT_FIXTURE={directory} ./tests/test_engram")


def main(argv=None):
    p = argparse.ArgumentParser(description="Engram 4-bit sidecar converter (HLWQ scheme).")
    p.add_argument("--selftest", action="store_true", help="synthetic round trip + planted mutations")
    p.add_argument("--source", help="the DeepSeek V4.1 Flash GGUF carrying FP8 Engram tables")
    p.add_argument("--output", help="the sidecar to write (.engram4)")
    p.add_argument("--limit-rows", type=int, default=0,
                   help="convert only the first N rows of each table (a partial sidecar the engine refuses)")
    p.add_argument("--chunk", type=int, default=1 << 18, help="rows per chunk (default 262144)")
    p.add_argument("--check", metavar="SIDECAR", help="spot-check a sidecar against --source")
    p.add_argument("--samples", type=int, default=4096)
    p.add_argument("--check-format", action="store_true", help="print the format and the codebook")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--emit-fixture", metavar="DIR", help="write a synthetic sidecar + expected.f32 for tests/test_engram.c")
    a = p.parse_args(argv)

    if a.selftest:
        return selftest()
    if a.emit_fixture:
        emit_fixture(a.emit_fixture)
        return 0
    if a.check_format:
        c = lloyd_max_centroids()
        print(f"row in {ENGRAM_ROW_BYTES_IN} B (256 E4M3 + 8 E8M0) -> row out {ENGRAM_ROW_BYTES_OUT} B"
              f" (128 packed nibbles + 2 F16 norms); header {SIDECAR_HEADER_BYTES} B, align {SIDECAR_ALIGN}")
        print(f"centroids (Lloyd-Max 16, N(0,1), {LLOYD_ITERATIONS} iterations):")
        print("  " + " ".join(f"{v:+.5f}" for v in c))
        if a.source:
            for layer, rows, off in engram_tensors(a.source):
                print(f"  blk.{layer}.engram_embd.weight rows {rows:,} at {off:,}")
        return 0
    if a.check:
        if not a.source:
            p.error("--check needs --source")
        for layer, n, rel, cos in check(a.source, a.check, a.samples):
            print(f"  blk.{layer}: {n} rows sampled, mean rel RMSE {rel:.4f}, mean cosine {cos:.5f}")
        return 0
    if a.source and a.output:
        total = convert(a.source, a.output, a.limit_rows, a.chunk, a.quiet)
        print(f"wrote {a.output}: {total:,} B = {total / 1e9:.3f} GB")
        print(f"NEXT -> python3 {sys.argv[0]} --source {a.source} --check {a.output}")
        return 0
    p.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
