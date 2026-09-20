# 06 · V4-Flash training notes — what is PUBLIC vs vendor-unstated (the honest ledger)

> Part of the **training framework** ([`00-index`](00-index.md)). The sixth rodeo is the one where
> **most of the answer is "not published."** This page is a deliberately BLUNT ledger: for
> DeepSeek-V4-Flash (and its DSpark draft stack), what training detail is actually PUBLIC (with a
> source), and what is VENDOR-UNSTATED (flagged, not guessed). It builds on the honesty caveats
> already in [`theory/24 §5`](../theory/24-deepseek-v3-v4-context.md) and
> [`theory/37 §7`](../theory/37-deepseek-architecture-deep.md#7-how-v4flash--dspark-descend-from-all-this).

```
   ◇  0 6   ·   V 4 - F L A S H   —   P U B L I C   vs   U N S T A T E D
  ╠══════════════════════════════════════════════════════════════════
  ║   PUBLIC (HF card):  ~32T pretrain tokens · two-stage post-train ·
  ║                       reasoning modes · FP4+FP8 · retains MoE+MTP
  ║   UNSTATED:          data mixture · post-train volumes · DSpark
  ║                       DRAFT-STACK TRAINING RECIPE (essentially none)
  ╚══  V4 report VERIFIED: arXiv:2606.19348 (confirmed 2026-07-06) — cite it
```

## 1. ✅ CORRECTED (2026-07-06): the DeepSeek-V4 report IS a real arXiv paper

**`arXiv:2606.19348` = "DeepSeek-V4: Towards Highly Efficient Million-Token Context
Intelligence" — VERIFIED via the arXiv export API on 2026-07-06** (independently re-checked by
the orchestrator: the id resolves to that exact title). The earlier "do NOT cite / date-
inconsistent / structurally impossible" caveat was a **knowledge-cutoff artifact** — fetches
before the paper posted (June 2026) couldn't see it, so it was wrongly flagged as a
hallucination. That caveat is now **RETIRED**: cite `arXiv:2606.19348` as the V4 report.
(The `2606` = June 2026 id is correct for a mid-2026 release.) For the sparse-attention
lineage you may also cite **DeepSeek-V3.2 / DSA
[arXiv:2512.02556](https://arxiv.org/abs/2512.02556)**. (This repeats and reinforces
[`theory/24 §5`](../theory/24-deepseek-v3-v4-context.md).)

## 2. PUBLIC — what the HF card actually states (with source)

Source: the **[DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash)** +
**[DeepSeek-V4-Pro](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro)** model cards
(released 2026-04-24 preview; MIT license):

| Claim | Card wording (quotable) | Status |
|-------|-------------------------|--------|
| Pretrain tokens | "**more than 32T** diverse and high-quality tokens" (both cards) | ⚠ **HF-card-stated, NOT arXiv-confirmed** |
| Post-training | "a **two-stage paradigm: independent cultivation of domain-specific experts** (through **SFT and RL with GRPO**), followed by **unified model consolidation via on-policy distillation**" | HF-card-stated |
| Reasoning modes | **Non-think** (fast) · **Think High** (logical analysis) · **Think Max** (fullest reasoning) | HF-card-stated |
| Sampling | recommended **temperature = 1.0, top_p = 1.0**; Think-Max wants context ≥ **384K** | HF-card-stated |
| Precision | "**FP4 + FP8 Mixed**" — MoE experts FP4, most other params FP8 | HF-card-stated (matches on-disk config) |
| Architecture | retains **DeepSeekMoE** + **MTP**; hybrid **CSA + HCA** attention; **mHC**; **Muon** optimizer | HF-card-stated |
| Sizes | Flash **284B total / 13B active**; Pro **1.6T / 49B active**; both **1M** context | HF-card + on-disk config |

> The community "Technical Report Summary" (Pro discussion #129) paraphrases some of this as "32–33T"
> and claims a "reverse-KL" distillation loss — those are **community paraphrases, not card-confirmed**.
> Use **"more than 32T"** and do **not** assert the loss form.

## 3. Config-GROUNDED structure vs UNSTATED training (the crucial split for DSpark)

The **on-disk checkpoint** (`DeepSeek-V4-Flash-DSpark/inference/config.json`, read directly
2026-07-04) grounds the DSpark draft stack's **STRUCTURE** — but a config gives shapes, **not a
training recipe**:

**Config-grounded (structure — authoritative from the checkpoint):**
- `n_mtp_layers: 3` (3 draft blocks) · `dspark_block_size: 5` (γ=5 block draft) ·
  `dspark_target_layer_ids: [40, 41, 42]` (capture the last 3 of 43 layers) ·
  `dspark_markov_rank: 256` (the rank-256 Markov head) · `dspark_noise_token_id: 128799`.
- These match [`theory/24 §2/§3`](../theory/24-deepseek-v3-v4-context.md) and
  [`theory/12`](../theory/12-dspark-forward-spec.md) — the C-port blueprint.

**VENDOR-UNSTATED (the training of that stack — essentially no public source):**
- **How the DSpark draft blocks were trained** — the loss, the target-output caching, the schedule,
  the data, how the Markov/confidence heads were fit — is **not documented publicly.** The
  `DeepSeek-V4-Flash-DSpark` card says only that DSpark **"is not a new model … the same checkpoint
  with an additional speculative decoding module attached,"** and points to the
  [`deepseek-ai/DeepSpec`](https://github.com/deepseek-ai/DeepSpec) repo (a "minimal inference
  example" + a `DSpark_paper.pdf` + `train.py`). The repo's one-line workflow — "train a draft model
  against the cached target outputs" — is generic draft-distillation; it does **not** spell out the
  semi-AR / Markov / confidence-head training. **The authoritative source is the on-disk checkpoint**
  ([`theory/12`](../theory/12-dspark-forward-spec.md)); the *training* recipe remains unstated.

## 4. VENDOR-UNSTATED — the full flagged list (do not guess these)

- **Exact V4 pretrain token count** (only "more than 32T"), **data mixture**, **per-domain/language
  ratios**, **data cutoff date**, **dedup thresholds** — none published.
- **V4 post-training details**: which/how-many expert domains, the RL reward design, the exact
  distillation objective (reverse-KL is community-claimed, **not** card-confirmed), **any
  post-training data volumes** — named at a high level only.
- **The entire DSpark draft-stack TRAINING recipe** (see §3) — no authoritative public source.
- **Per-mode chat-template internals** (trigger tokens, per-mode stop sequences, per-mode sampling
  deltas) — live in the checkpoint's `tokenizer_config.json` / `chat_template`; read locally to be
  authoritative, don't quote from memory.
- **Any V4 hardware / compute / wall-clock training figures** — not published (unlike V3's 2.788M
  H800-hrs, which IS published).

## 5. What this framework therefore does

For every V4-Flash training item the [manifest](manifest.json) marks it **`vendor-internal` +
`ignore`** (documentation only, nothing to fetch), and the SIZE TABLE in [`00`](00-index.md) shows the
`~32T` figure with the ⚠ card-stated flag. The **public analogues** (pages [`01`](01-pretraining-corpus.md),
[`03`](03-sft-instruction-tuning.md), [`04`](04-rl-grpo-rewards.md), [`05`](05-distillation.md)) are the
only way to eyeball the *shape* — and they are DeepSeek-family-*inspired* open reproductions, not V4's
data. **No dataset here is claimed to be V4-Flash's actual training data, because none is public.**

## 6. Sources

1. **HF model cards** — [DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash) ·
   [DeepSeek-V4-Pro](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro) ·
   [DeepSeek-V4-Flash-DSpark](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-DSpark) — the only
   public statements (~32T, two-stage post-train, reasoning modes, FP4+FP8, MoE+MTP retained).
2. **HF PDF** — `DeepSeek_V4.pdf` on the V4-Pro repo (the report; **not** an arXiv paper). Community
   summary [Pro discussion #129](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/discussions/129)
   (paraphrase — use with care).
3. **On-disk checkpoint** — `DeepSeek-V4-Flash-DSpark/inference/config.json` (§3 structure; the
   authoritative source, [`theory/12`](../theory/12-dspark-forward-spec.md)).
4. **DSpark code** — [`deepseek-ai/DeepSpec`](https://github.com/deepseek-ai/DeepSpec) (`train.py`,
   `DSpark_paper.pdf` — may contain detail; unverified, not card-confirmed).
5. **arXiv grounding for the sparse-attn lineage only** — [DSA / V3.2, arXiv:2512.02556](https://arxiv.org/abs/2512.02556).

Cross-refs: [`00`](00-index.md) · [`theory/24 §5`](../theory/24-deepseek-v3-v4-context.md) ·
[`theory/37 §7`](../theory/37-deepseek-architecture-deep.md) · [`theory/12`](../theory/12-dspark-forward-spec.md) ·
[`theory/03`](../theory/03-dspark.md).
