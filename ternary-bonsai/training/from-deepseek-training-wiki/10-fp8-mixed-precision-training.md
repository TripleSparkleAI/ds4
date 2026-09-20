# 10 · FP8 mixed-precision training, in depth — tile scaling, E4M3, FP32 promotion + DeepGEMM code

> Part of the **training framework** ([`00-index`](00-index.md)); the systems depth behind the FP8
> summary in [`theory/37 §4`](../theory/37-deepseek-architecture-deep.md#4-fp8-mixed-precision-training).
> DeepSeek-V3 is **the first publicly documented FP8-native training of a frontier-scale (671B) LLM**.
> This page goes to the *kernel level*: the **fine-grained tile/block scaling** (1×128 / 128×128), the
> **E4M3 choice**, and the accumulation-precision fix — **promote partial sums to FP32 on CUDA cores
> every N_C=128** — shown as **real DeepGEMM code**. This is a genuine engineering frontier and the
> **`FP8Training` / `DeepGEMM` hub** of the graph.

```
   ◇  1 0   ·   F P 8   T R A I N I N G   ( the low-bit frontier )
  ╠══════════════════════════════════════════════════════════════════
  ║  activations: 1×128 tiles (per token, per 128 ch)   ← tame outliers
  ║  weights:     128×128 blocks (per 128 in × 128 out)
  ║  E4M3 on ALL tensors (fine-grained scale gives the range back)
  ║  accumulate: FP8 WGMMA → promote to FP32 CUDA-core every N_C=128
  ╚══  DeepGEMM: ~300-line kernel, up to 1350+ FP8 TFLOPS on H800
```

---

## Table of contents
- [1. Why FP8 training is hard (the two failure modes)](#1-why-fp8-training-is-hard-the-two-failure-modes)
- [2. Fine-grained quantization — 1×128 / 128×128 tiles](#2-fine-grained-quantization--1128--128128-tiles)
- [3. E4M3 everywhere (not the usual hybrid)](#3-e4m3-everywhere-not-the-usual-hybrid)
- [4. The accumulation fix — FP32 promotion every N_C=128 (with real DeepGEMM code)](#4-the-accumulation-fix--fp32-promotion-every-n_c128-with-real-deepgemm-code)
- [5. Low-precision storage + FP8 dispatch](#5-low-precision-storage--fp8-dispatch)
- [6. DeepGEMM — the FP8 GEMM library, as code](#6-deepgemm--the-fp8-gemm-library-as-code)
- [7. What this grounds for DwarfStar](#7-what-this-grounds-for-dwarfstar)
- [8. Sources](#8-sources)

---

## 1. Why FP8 training is hard (the two failure modes)

FP8 has ~2–3 mantissa bits and a narrow exponent range. Two things break naïvely:

1. **Dynamic range / outliers.** A single per-tensor scale can't cover both the bulk of activations and
   the rare large outliers — quantize to the outlier and you crush the bulk to zero; quantize to the bulk
   and you clip/overflow the outlier. **Fix → fine-grained tile scaling** (§2).
2. **Accumulation error.** FP8 tensor-core MMA accumulates in a *limited* bit width; over a long
   `K`-dimension reduction the rounding error compounds until the GEMM diverges from its FP32 reference.
   **Fix → promote partials to FP32 on CUDA cores every N_C=128** (§4).

DeepSeek-V3 §3.3 solves both, and validates FP8 at 671B scale — *"for the first time, [we] validate the
feasibility and effectiveness of FP8 training on an extremely large-scale model."* (§3.3, [verified]).

---

## 2. Fine-grained quantization — 1×128 / 128×128 tiles

**Source:** V3 report [arXiv:2412.19437](https://arxiv.org/abs/2412.19437) §3.3 (Fig 7a) · [verified].
Instead of one scale per tensor, DeepSeek scales in small tiles so each tile's scale can track its own
local range:

> *"For **activations**, we group and scale elements on a **1×128 tile basis** (i.e., **per token per
> 128 channels**). For **weights**, we group and scale elements on a **128×128 block basis** (i.e., per
> 128 input channels per 128 output channels)."* (§3.3, Fig 7a)

| Tensor | Scale granularity | Meaning |
|--------|-------------------|---------|
| **Activation** | **1×128 tile** | one FP32 scale per **token**, per **128 channels** |
| **Weight** | **128×128 block** | one FP32 scale per **128 input × 128 output** chunk |

This is *far* finer than per-tensor scaling, so a token with an outlier channel-group gets its own scale
and doesn't force the whole tensor into a bad quantum. The `128` grain is not arbitrary — it aligns with
the tensor-core tile shape and the `N_C=128` accumulation interval (§4), so scaling and accumulation
share one natural block size.

---

## 3. E4M3 everywhere (not the usual hybrid)

The usual FP8 recipe uses a **hybrid**: E4M3 (4 exponent / 3 mantissa — more precision, less range) for
the forward pass, E5M2 (5 exp / 2 mant — more range, less precision) for gradients in the backward. V3
does **not**:

> *"… we adopt the **E4M3 format on all tensors** for higher precision."* (§3.3, [verified])

Because the *fine-grained tile scaling* (§2) already controls dynamic range, they can afford to keep the
higher-*precision* E4M3 on every tensor — the scaling gives the range back that E4M3's narrow exponent
would otherwise lack. This is the payoff of §2: fine-grained scale → E4M3 everywhere → more mantissa bits
everywhere.

---

## 4. The accumulation fix — FP32 promotion every N_C=128 (with real DeepGEMM code)

This is the sharpest engineering detail on the page. **Source:** V3 report §3.3 (Fig 7b) · [verified]:

> *"During MMA (Matrix Multiply-Accumulate) execution on Tensor Cores, intermediate results are
> accumulated using the limited bit width. Once an interval of **N_C is reached**, these partial results
> will be **copied to FP32 registers on CUDA Cores**, where full-precision FP32 accumulation is
> performed."* (§3.3) — with **N_C = 128** (*"equivalent to 4 WGMMAs"*).

So the reduction is **two-level**: the fast-but-imprecise FP8 tensor cores do the heavy lifting in short
`N_C=128` bursts, then a **FP32 accumulator on the CUDA cores** soaks up each burst's partial — bounding
the compounding rounding error. **This exact mechanism is visible in DeepGEMM's kernel** — `accum[]` is
the tensor-core (WGMMA) register set, `final_accum[]` is the FP32 promotion register set
([`deep_gemm/include/deep_gemm/fp8_gemm.cuh`](https://github.com/deepseek-ai/DeepGEMM/blob/main/deep_gemm/include/deep_gemm/fp8_gemm.cuh),
original FP8 release commit `7a70b43`) · [verified]:

```cpp
// two register sets: WGMMA tensor-core result vs FP32 CUDA-core promotion
float accum[WGMMA::kNumAccum], final_accum[WGMMA::kNumAccum] = {0};
...
// (1) commit WGMMA instructions — FP8 tensor-core MMA fills accum[]  (one N_C burst = 4 WGMMAs)
warpgroup_arrive();
#pragma unroll
for (int k = 0; k < BLOCK_K / WGMMA::K; ++k) {
    auto desc_a = make_smem_desc(smem_a[s] + math_wg_idx * WGMMA::M * BLOCK_K + k * WGMMA::K, 1);
    auto desc_b = make_smem_desc(smem_b[s] + k * WGMMA::K, 1);
    WGMMA::wgmma(desc_a, desc_b, accum, k);
}
warpgroup_commit_batch();
warpgroup_wait<0>();
// (2) PROMOTE with scales — accum[] → FP32 final_accum[], applying the 1×128 & 128×128 scales
#pragma unroll
for (int i = 0; i < WGMMA::kNumAccum / 4; ++i) {
    final_accum[i*4+0] += (predicate ? scale_0_0 : scale_0_1) * accum[i*4+0];
    final_accum[i*4+1] += (predicate ? scale_0_0 : scale_0_1) * accum[i*4+1];
    final_accum[i*4+2] += (predicate ? scale_1_0 : scale_1_1) * accum[i*4+2];
    final_accum[i*4+3] += (predicate ? scale_1_0 : scale_1_1) * accum[i*4+3];
}
```

Read it as the report's Fig 7b in C++: the WGMMA loop (1) is the "limited-bit-width tensor-core
accumulation"; the "Promote with scales" loop (2) is *"copied to FP32 registers on CUDA cores, where
full-precision FP32 accumulation is performed."* The `scale_*` factors are the §2 tile/block scales
applied at promotion time. The README states the intent in one line: *"To address the imprecise FP8
tensor core accumulation, it employs **CUDA-core two-level accumulation (promotion)**."*

> ⚠ **Provenance precision.** Code is **verbatim** from the raw `.cuh` at commit `7a70b43` (the original
> FP8 release — `main` has since been refactored into a broader BLAS library with renamed functions; see
> §6). The two blocks were returned in separate fetches and stitched for readability (the scale-derivation
> lines between them are elided); **re-pin against the raw file** before byte-quoting. The V3 §3.3 quotes
> were read from the ar5iv HTML mirror — eyeball the official PDF for exact punctuation.

---

## 5. Low-precision storage + FP8 dispatch

Two more places FP8/BF16 saves bytes (V3 §3.3.3 + §3.2, [verified]):

- **Optimizer storage** — the AdamW **first + second moments in BF16**; **master weights + gradients
  stay FP32** (§3.3.3). Halves optimizer-state memory, pairs with the ZeRO-1 sharding of
  [`08 §5`](08-training-framework-parallelism.md).
- **FP8 dispatch on the wire** — during EP all-to-all, *"tokens are dispatched using fine-grained FP8
  quantization, **reducing communication volume by 50% compared to BF16**"* (hardware paper
  [arXiv:2505.09343](https://arxiv.org/abs/2505.09343) §3.2). So FP8 is not just a *compute* format — on
  the bandwidth-capped H800 it also halves the **comm** bytes ([`11`](11-communication-overlap-engineering.md)).
  DeepEP's low-latency path uses **FP8 dispatch, BF16 combine** ([`11 §3`](11-communication-overlap-engineering.md)).

---

## 6. DeepGEMM — the FP8 GEMM library, as code

**Repo:** [github.com/deepseek-ai/DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) (MIT) · [verified].
The open-sourced FP8 GEMM library that powers V3/R1 training + inference. From the original FP8 README
(commit `7a70b43`): *"clean and efficient FP8 GEMMs with fine-grained scaling … supports both normal and
Mix-of-Experts (MoE) grouped GEMMs … Hopper tensor cores … no compilation need during installation, by
compiling all kernels at runtime using a lightweight Just-In-Time (JIT) module … only one core kernel
function comprising around **~300 lines of code**."*

**Performance (H800 SXM5, README table @ `7a70b43`)** · [verified]: up to **2.7×** vs an expert-tuned
CUTLASS baseline (the M=64,N=2112,K=7168 row: 206 TFLOPS, 1688 GB/s); peak **~1358 TFLOPS** on large
dense shapes; MoE grouped GEMMs **~815–1297 TFLOPS**. (Later releases report up to ~1550 TFLOPS.)

**The dense API** — the docstring literally states the §2 scaling
([`deep_gemm/jit_kernels/gemm.py`](https://github.com/deepseek-ai/DeepGEMM/blob/main/deep_gemm/jit_kernels/gemm.py)):

```python
def gemm_fp8_fp8_bf16_nt(lhs: Tuple[torch.Tensor, torch.Tensor],   # (FP8 [m,k], FP32 scale [m,⌈k/128⌉])
                         rhs: Tuple[torch.Tensor, torch.Tensor],   # (FP8 [n,k], FP32 scale [⌈n/128⌉,⌈k/128⌉])
                         out: torch.Tensor) -> None:               # BF16 [m,n]
    """Do a normal GEMM with FP8 inputs and BF16 output,
       with 1x128 LHS scaling and 128x128 RHS scaling."""
```

**The two MoE grouped variants** — `contiguous` for training/prefill (experts concatenated along M,
`m_indices` maps each row → its expert), `masked` for decode (fixed `[num_groups, m_max, …]` + runtime
mask, CUDA-graph friendly) ([`deep_gemm/jit_kernels/m_grouped_gemm.py`](https://github.com/deepseek-ai/DeepGEMM/blob/main/deep_gemm/jit_kernels/m_grouped_gemm.py)):

```python
def m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(lhs, rhs, out, m_indices) -> None:
    """Grouped GEMM (contiguous) — 1x128 LHS scaling and 128x128 RHS scaling."""  # training / prefill

def m_grouped_gemm_fp8_fp8_bf16_nt_masked(lhs, rhs, out, masked_m, expected_m) -> None:
    """Grouped GEMM (masked) — 1x128 LHS scaling and 128x128 RHS scaling."""      # decode (masked, graph-safe)
```

> ⚠ **Version drift.** The `main` branch was later refactored into a broader "BLAS kernel library"
> (FP8/FP4/BF16, SM90+SM100, renamed funcs like `fp8_gemm_nt`). **Everything above is the original
> FP8-era release — cite commit `7a70b43`, not bare `main`,** or the names/tables won't match. Nothing
> is downloaded (manifest `code-deepgemm`, disk_policy `ignore` + git-clone command ready).

---

## 7. What this grounds for DwarfStar

- **FP8-train ↔ q2-infer, same discipline.** DwarfStar runs a *2-bit inference* quant (IQ2_XXS,
  [`theory/05`](../theory/05-gguf-and-quant.md)) — a different regime from FP8 *training*, but the same
  two ideas underlie both: **fine-grained scaling** (i-quants use an importance matrix; V3 uses tiles) +
  **a preserved higher-precision correction path** (V3's FP32 promotion; ds4's exact-verify against a
  full-precision reference, [`theory/14`](../theory/14-exact-verification.md)). Aggressive low-bit only
  works with a high-precision escape hatch.
- **The draft-quant law (G5).** DSpark's *draft* may use its own quant to change acceptance τ, but the
  *verify* is exact ([`theory/14`](../theory/14-exact-verification.md) G5) — the inference analogue of
  "quantize the fast path (FP8 MMA), keep the slow path exact (FP32 promotion)."
- **V4-Flash ships FP4 experts + FP8 rest** ([`06`](06-v4-flash-training-notes.md),
  [`theory/24 §2`](../theory/24-deepseek-v3-v4-context.md)) — the next rung of this same low-bit ladder;
  its *training* recipe is vendor-unstated, but the FP4/FP8 *representation* is config-confirmed on the
  disk ds4 runs.

---

## 8. Sources

1. ⭐ **DeepSeek-V3 report §3.3** — [arXiv:2412.19437](https://arxiv.org/abs/2412.19437) · [verified].
   FP8 training: fine-grained 1×128/128×128 quantization (Fig 7a), E4M3-on-all-tensors, the N_C=128
   FP32-promotion accumulation (Fig 7b), §3.3.3 BF16-moments/FP32-master storage, "first at this scale."
   **FAVOURITE of this page.**
2. ⭐ **DeepGEMM repo** — [github.com/deepseek-ai/DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) (MIT)
   · [verified], commit `7a70b43`. `fp8_gemm.cuh` (the two-level accumulation kernel — the code form of
   Fig 7b), `jit_kernels/gemm.py` + `m_grouped_gemm.py` (the API + docstrings stating 1×128/128×128),
   README (the 2.7×/~1358-TFLOPS table, "~300 lines", JIT, TMA). **FAVOURITE — Fig 7b as real code.**
3. **DeepSeek-V3 hardware paper** — [arXiv:2505.09343](https://arxiv.org/abs/2505.09343) §3.2 · [verified].
   FP8 dispatch −50% comm vs BF16 — the bandwidth (not just compute) payoff of FP8.

> **Provenance honesty.** All FP8 scheme facts are V3 §3.3 ([verified] arXiv, quoted). DeepGEMM code is
> **verbatim** from commit `7a70b43`, **line numbers not byte-pinned**, the kernel blocks **stitched**
> for readability (flagged §4). Perf numbers are the README's own H800 table. **No V4 FP4/FP8 training
> recipe is published** — only the FP4-experts/FP8-rest *representation* is config-confirmed
> ([`06`](06-v4-flash-training-notes.md)).

Cross-refs: [`00`](00-index.md) · [`theory/37 §4`](../theory/37-deepseek-architecture-deep.md#4-fp8-mixed-precision-training)
· [`07`](07-cluster-and-machine-config.md) · [`08`](08-training-framework-parallelism.md) ·
[`11`](11-communication-overlap-engineering.md) · [`theory/05`](../theory/05-gguf-and-quant.md) ·
[`theory/14`](../theory/14-exact-verification.md) · [`13`](13-training-theory-why-it-works.md).
