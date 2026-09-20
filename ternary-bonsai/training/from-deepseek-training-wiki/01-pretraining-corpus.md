# 01 · Pretraining corpus — the 14.8T-token mix, tokenizer, mixture, dedup, long-context

> Part of the **training framework** ([`00-index`](00-index.md)). The first rodeo: the raw
> pretraining data DeepSeek learns language on, before any instruction/RL. This page is the
> **DATA + pipeline**; the *architecture* it trains lives in [`theory/37`](../theory/37-deepseek-architecture-deep.md)
> / [`theory/24`](../theory/24-deepseek-v3-v4-context.md), and the FP8 training *method* in
> [`theory/37 §4`](../theory/37-deepseek-architecture-deep.md#4-fp8-mixed-precision-training).
> Fetch pointer: `./fetch.sh --list` (ids `ds-v3-pretrain-corpus`, `fineweb-edu`, `the-stack-v2`,
> `open-web-math`).

```
   ◇  0 1   ·   P R E T R A I N   C O R P U S
  ╱   V2  8.1T tok (zh ~12% > en, vocab 100K)
 ╱    V3  14.8T tok (up-weighted math+code, vocab 129,280, 2.788M H800-hrs)
 ╱    V4  ~32T tok  ⚠ HF-card-stated, no arXiv report
∿∿∿∿∿∿∿∿∿∿   the actual mix is UNRELEASED — public analogues show the shape only
```

## 1. What data, and how much (CONFIRMED)

| Model | Corpus | Tokenizer | Cross-lang ratio | Source |
|-------|-------:|-----------|------------------|--------|
| **DeepSeek-V2** | **8.1T tokens** | BBPE, **100K** vocab | **zh ≈ 1.12× en** (the ONE published ratio) | [arXiv:2405.04434](https://arxiv.org/abs/2405.04434) §3.1.1 |
| **DeepSeek-V3** | **14.8T tokens** | byte-level BPE, **129,280** vocab (report rounds "128K") | not published | [arXiv:2412.19437](https://arxiv.org/abs/2412.19437) §1, §4.2 |
| **DeepSeek-V4-Flash** | **~32T tokens** ⚠ *card-stated* | vocab **129,280** (config-confirmed, not report-stated) | not published | HF card (see [`06`](06-v4-flash-training-notes.md)) |

- **V3 = 14.8T tokens** — "we pre-train DeepSeek-V3 on 14.8T high-quality and diverse tokens"
  (§4.2 Hyper-Parameters + §1). Exact figure = 14.8 trillion. **CONFIRMED.**
- **V3 compute** (§1, Table 1): Pre-Training **2,664K** H800 GPU-hours; Context Extension **119K**;
  Post-Training **5K**; **total 2,788K = 2.788M H800 GPU-hours.** **CONFIRMED.** FP8-native training
  (§3.3), relative loss error <0.25% vs BF16.
- **V4 ~32T** — see [`06`](06-v4-flash-training-notes.md); it is **HF-card-stated, not arXiv-confirmed**.

## 2. The mixture (qualitative only — no numeric ratios published)

V3 §4.1 (Data Construction) describes the mix *qualitatively*, **NOT** as numeric percentages:
> "…optimize the pre-training corpus by **enhancing the ratio of mathematical and programming
> samples**, while **expanding multilingual coverage** beyond English and Chinese."

- **No English/Chinese/code/math percentage split is published for V3 or V4.** Treat any specific
  `%` you see elsewhere as **UNCONFIRMED**. The only published cross-language number is **V2's**
  "Chinese tokens ≈ 12% more than English" (§3.1.1).
- **Document packing** (§4.1) is used "for data integrity"; no cross-sample attention masking issues.

## 3. Tokenizer

- **Byte-level BPE.** V3 uses an **extended 128K vocabulary** (§4.1); the on-disk `config.json`
  gives the exact **`vocab_size = 129280`** (1,280 reserved/special slots over 128,000). V4-Flash
  inherits **129,280** (config-confirmed; the V4 report text does not state vocab size).
- The V3 pretokenizer adds tokens that **combine punctuation + line breaks**; to mitigate the
  resulting token-boundary bias, they **randomly split a proportion** of these during training (§4.1).
- V2 used the DeepSeek-67B tokenizer (BBPE, **100K** vocab) — the lineage the 128K vocab grew from.

## 4. Dedup + quality filtering (as far as public)

V3 §4.1: "our data processing pipeline is **refined to minimize redundancy while maintaining corpus
diversity**." **No thresholds, no MinHash/Bloom parameters, no quality-classifier details are
disclosed.** (A common third-party "24GB tokenizer corpus / ~35K Chinese tokens" claim is a
*third-party estimate*, not from any DeepSeek report — do not treat as official.) → **vendor-unstated.**

## 5. Long-context extension data (CONFIRMED — steps, not doc counts)

- **V3** (§4.3): **YaRN**, two extra phases of **1000 steps each**, expanding context
  **4K → 32K → 128K** (phase-1 seqlen 32K, phase-2 128K). V4-Flash natively targets **1M** context
  ([`theory/24 §2`](../theory/24-deepseek-v3-v4-context.md), `config.json max_position 1,048,576`).
- **V2** (§3.1.4): 4K → 128K via YaRN, +1000 steps at seqlen 32K.
- **Per-phase document counts are NOT published** — only step counts. → the doc counts are unstated.

## 6. The fetch pointer + the public analogues (see the SHAPE, not the data)

**The real DeepSeek pretrain corpus is NOT released** (`ds-v3-pretrain-corpus`, policy `ignore` —
documentation only). To eyeball the *kind* of data each slice is, `fetch.sh` points at public
**analogues** — these are proxies, **not** DeepSeek's data:

| Slice | Analogue (fetch id) | Size | What a row is | Preview |
|-------|--------------------|-----:|---------------|---------|
| Quality-filtered web | `fineweb-edu` | ~8.8 TB (1.3T tok) | web doc + **edu quality score** | [samples/PREVIEWS.md](samples/PREVIEWS.md#fineweb-edu--quality-filtered-english-web-stage-01-analogue) |
| Code | `the-stack-v2` | ~67.5 TB | Software-Heritage file-IDs (blobs via SWH/S3) | (see manifest note) |
| Math web | `open-web-math` | ~60 GB (14.7B tok) | math-heavy web document | [samples/PREVIEWS.md](samples/PREVIEWS.md#open-web-math--math-web-text-stage-01-analogue) |

```bash
./fetch.sh --sample fineweb-edu     # ~5 rows via the HF datasets-server (tiny, git-ignored)
./fetch.sh fineweb-edu              # LARGE → prints the command, does NOT download
```

**Row shapes we actually pulled** (2026-07-04): `fineweb-edu` = `text, url, language, token_count,
score, int_score` (a filtered web doc + its edu score); `open-web-math` = `url, text, date, metadata`.
Full previews in [`samples/PREVIEWS.md`](samples/PREVIEWS.md).

> **Why analogues at all?** DeepSeek released the *weights + tokenizer* but never the *corpus* or the
> *mixture weights*. FineWeb-Edu (filtered web), The Stack v2 (code), OpenWebMath (math) are what a
> researcher can download to inspect the *filtered-web + code + math + multilingual* recipe the
> reports describe — the shape, not the substance.

## 7. Ranked sources

1. ⭐ **DeepSeek-V3 report** — [arXiv:2412.19437](https://arxiv.org/abs/2412.19437) §1/§4.1/§4.2/§4.3,
   Table 1 (14.8T tokens, 128K vocab, 2.788M H800-hrs, YaRN long-ctx, FP8). **FAVOURITE of this page.**
2. **DeepSeek-V2 report** — [arXiv:2405.04434](https://arxiv.org/abs/2405.04434) §3.1.1/§3.1.4
   (8.1T tokens, zh≈12%>en, 100K vocab — the lineage + the only published lang ratio).
3. **On-disk configs** — `DeepSeek-V3/config.json` + `DeepSeek-V4-Flash/config.json`
   (`vocab_size 129280` exact). The authoritative source for the tokenizer size.
4. **Public analogues** — [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) ·
   [The Stack v2](https://huggingface.co/datasets/bigcode/the-stack-v2) ·
   [OpenWebMath](https://huggingface.co/datasets/open-web-math/open-web-math) (the shape, not the data).

Cross-refs: [`00`](00-index.md) · [`02`](02-mtp-objective.md) · [`theory/37`](../theory/37-deepseek-architecture-deep.md) ·
[`theory/24`](../theory/24-deepseek-v3-v4-context.md) · [`theory/33`](../theory/33-deepseek-mtp-training-inference.md).
