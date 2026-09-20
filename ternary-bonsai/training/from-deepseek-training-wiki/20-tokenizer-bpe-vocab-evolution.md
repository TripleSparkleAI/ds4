# 20 · The tokenizer — byte-level BPE and the vocab-size evolution

> Part of the **training framework** ([`00-index`](00-index.md)); a **data-internals** page of the
> lineage ([`14`](14-deepseek-model-lineage.md)). [`01 §3`](01-pretraining-corpus.md) states the vocab
> sizes in passing; **this page owns the tokenizer in depth** — the byte-level BPE mechanism, the exact
> vocab-size evolution across versions, and the pre-tokenizer internals **read directly off the
> on-disk V4-Flash checkpoint** (the authoritative source, since the tokenizer *ships with the
> weights*). On-disk facts read 2026-07-05.

```
   ◇  2 0   ·   T O K E N I Z E R   ·   B Y T E - L E V E L   B P E
  ╱   V1/V2   BBPE · 100K vocab   (the DeepSeek-67B tokenizer)
 ╱    V3       byte-level BPE · 129,280 vocab (report rounds "128K")
 ╱    V4-Flash 129,280 (config-confirmed) · BPE 128,000 base + 1,283 added
∿∿∿∿∿∿∿∿∿∿   read off the shipped checkpoint — the tokenizer is part of the weights
```

---

## 1. What "byte-level BPE" means here (the mechanism)

DeepSeek uses a **byte-level Byte-Pair-Encoding (BBPE)** tokenizer — the same family as GPT-2/GPT-4,
not SentencePiece-unigram. The mechanics, confirmed from the on-disk `tokenizer.json`:

- **`model.type = "BPE"`** — a merge-based tokenizer: start from bytes, greedily apply learned merge
  rules. The on-disk V4-Flash model carries **128,000 base vocab entries and 127,741 merge rules**.
- **Byte-level (not char-level).** The final pre-tokenizer step is **`ByteLevel`** (and the decoder is
  `ByteLevel`), so every input is first mapped to a **reversible byte alphabet** — meaning *any* byte
  sequence (any Unicode, any binary-ish text, any language) is representable with **no `<unk>` token**.
  That "no out-of-vocabulary" property is why one tokenizer serves English, Chinese, code, and math
  symbols uniformly.
- **The pre-tokenizer split sequence** (on-disk, in order) — this is the "how text is chopped before
  BPE" recipe, and it is *DeepSeek-specific*:
  1. **`Split \p{N}{1,3}` (Isolated)** — split runs of **digits into groups of 1–3**, so numbers
     tokenize consistently (important for arithmetic / math reasoning — a model tokenizes "12345" the
     same way every time).
  2. **`Split [CJK ranges]+` (Isolated)** — isolate **CJK / Hiragana / Katakana** spans, giving Chinese
     and Japanese their own boundaries (the multilingual lineage from the 100K-vocab days).
  3. **`Split [GPT-style regex]`** — the familiar contraction/punctuation/whitespace pattern
     (`'s`, ` ?\p{L}+`, ` ?\p{P}+`, `\s*[\r\n]+`, …) that GPT-2/4 use.
  4. **`ByteLevel` (use_regex=false)** — map to the byte alphabet.

> **The V3 punctuation-boundary detail ([`01 §3`](01-pretraining-corpus.md)).** V3's report notes the
> pretokenizer adds tokens that **combine punctuation + line breaks**, and to counter the resulting
> **token-boundary bias** they **randomly split a proportion** of these during training. The on-disk
> pattern above (`\s*[\r\n]+`, ` ?[\p{P}\p{S}]+[\r\n]*`) is exactly the punctuation+newline combining it
> refers to.

---

## 2. The vocab-size evolution (the lineage fact)

| Model(s) | Tokenizer | Vocab size | Source |
|----------|-----------|-----------:|--------|
| DeepSeek-LLM 7B/67B, **DeepSeek-Coder** | BBPE | **~100K** (32K for early Coder ablations) | [2401.02954](https://arxiv.org/abs/2401.02954) / [2401.14196](https://arxiv.org/abs/2401.14196) |
| **DeepSeek-V2** | BBPE (DeepSeek-67B tokenizer) | **100,000** | [2405.04434](https://arxiv.org/abs/2405.04434) §3.1.1 |
| **DeepSeek-V3** | byte-level BPE (extended) | **129,280** (report says "128K") | [2412.19437](https://arxiv.org/abs/2412.19437) §4.1 + config |
| **DeepSeek-V4-Flash** | byte-level BPE (inherited) | **129,280** | on-disk `config.json` (report silent) |

- **V2 → V3 grew the vocab 100K → ~128K.** The V3 report describes an **"extended 128K vocabulary"**
  (§4.1). The exact on-disk number is **129,280** — the report *rounds to 128K* while the config is
  precise. The gap (129,280 − 128,000 = **1,280**) is **reserved / special slots** over the 128,000 base
  BPE entries.
- **V4-Flash inherits 129,280 exactly** — `config.json: "vocab_size": 129280` (read off the shipped
  checkpoint 2026-07-05). **The V4 report text does not state the vocab size**; the config is the
  authoritative source ([`01 §3`](01-pretraining-corpus.md)).
- **On-disk accounting (V4-Flash `tokenizer.json`):** **128,000 base BPE vocab** + **1,283 added tokens**
  (the special/control tokens like `<｜begin▁of▁sentence｜>`, `<｜end▁of▁sentence｜>`, tool/role
  markers) ≈ the 129,280 total (the small difference is reserved padding slots). `bos_token_id=0`,
  `eos_token_id=1`.

---

## 3. Why the vocab grew (and why it matters to V4)

- **Bigger vocab → fewer tokens per document → cheaper 1M-context.** A 128K vocab packs more bytes per
  token than a 32K/100K one, so a fixed 1M-token context window holds *more text*. For a model whose
  headline feature is **1M context** (V4-Flash, `model_max_length = 1,048,576` in the on-disk
  `tokenizer_config.json`), tokenizer efficiency is a direct cost lever.
- **Multilingual + code + math coverage.** The digit-grouping and CJK-isolation splits (above) are the
  concrete mechanism behind the reports' "expand multilingual coverage" and "up-weight math/code"
  ([`01 §2`](01-pretraining-corpus.md), [`23`](23-data-mix-ratios.md)) — the tokenizer is *co-designed*
  with the data mix.
- **The DwarfStar link.** ds4 renders prompts and computes byte-prefix SHA1 KV-cache keys over the
  tokenized stream; the exact vocab (129,280) and special-token ids are what the engine's prompt
  rendering + MTP/DSpark draft heads are built against ([`theory/02`](../theory/02-architecture.md),
  [`theory/24 §2`](../theory/24-deepseek-v3-v4-context.md)). A tokenizer mismatch would break greedy
  identity ([`theory/14`](../theory/14-exact-verification.md)) — so "read the tokenizer off the shipped
  checkpoint" is a verification-law concern, not a detail.

## 4. Honest provenance

- **CONFIRMED (on-disk, read 2026-07-05):** `vocab_size=129280`, `model.type=BPE`, 128,000 base vocab +
  127,741 merges + 1,283 added tokens, the 4-step pre-tokenizer split sequence, `ByteLevel` encoder+
  decoder, `model_max_length=1048576`, BOS/EOS ids — all read directly from the V4-Flash-DSpark
  `config.json` / `tokenizer.json` / `tokenizer_config.json`.
- **CONFIRMED (report-cited):** V2's 100K vocab (2405.04434 §3.1.1); V3's "extended 128K" (2412.19437
  §4.1) + the punctuation-boundary random-split mitigation.
- **⚠ Vendor-unstated:** the **V4 report does not state a vocab size** — 129,280 for V4-Flash comes from
  the *config*, not the report. The precise reserved-slot count (1,280) is an *arithmetic* fact
  (129,280 − 128,000), consistent with V3; treat it as config-derived, not report-stated. The "32K
  early-Coder-ablation vocab" is a Coder-paper ablation detail, not the shipped Coder vocab.

## 5. Ranked sources

1. ⭐ **On-disk V4-Flash tokenizer** — `DeepSeek-V4-Flash-DSpark/{config.json,tokenizer.json,
   tokenizer_config.json}` · [read 2026-07-05]. The authoritative source (the tokenizer ships with the
   weights). **FAVOURITE of this page.**
2. **DeepSeek-V3 report** — [arXiv:2412.19437](https://arxiv.org/abs/2412.19437) §4.1 · [verified].
   The "extended 128K vocabulary" + punctuation-boundary random-split.
3. **DeepSeek-V2 report** — [arXiv:2405.04434](https://arxiv.org/abs/2405.04434) §3.1.1 · [verified].
   100K vocab (the lineage the 128K grew from).
4. **DeepSeek-Coder / LLM** — [arXiv:2401.14196](https://arxiv.org/abs/2401.14196) /
   [arXiv:2401.02954](https://arxiv.org/abs/2401.02954) · [verified]. The early tokenizer.

Cross-refs: [`00`](00-index.md) · [`14`](14-deepseek-model-lineage.md) · [`01`](01-pretraining-corpus.md) ·
[`17`](17-deepseek-v2-mla-moe.md) · [`21`](21-data-pipeline-dedup-filtering.md) · [`23`](23-data-mix-ratios.md) ·
[`theory/24`](../theory/24-deepseek-v3-v4-context.md) · [`theory/14`](../theory/14-exact-verification.md).
