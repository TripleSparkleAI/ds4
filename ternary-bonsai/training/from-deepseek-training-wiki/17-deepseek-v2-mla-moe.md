# 17 · DeepSeek-V2 — the MLA + DeepSeekMoE debut (the architecture V4 inherits)

> Part of the **training framework** ([`00-index`](00-index.md)); the trunk step of the lineage
> ([`14`](14-deepseek-model-lineage.md)) that **V4-Flash directly descends from**. The MLA and
> DeepSeekMoE *equations* are derived in [`theory/37 §1–§2`](../theory/37-deepseek-architecture-deep.md#1-mla--multi-head-latent-attention);
> [`01`](01-pretraining-corpus.md) covers V2's 8.1T corpus + tokenizer. **This page owns the DEBUT
> framing** — V2 is where the two pillars first ship *together* as a trained model, plus V2's own
> training recipe (data, long-context, RL). Verified 2026-07-05.

```
   ◇  1 7   ·   D E E P S E E K - V 2   ·   T H E   A R C H   D E B U T
  ╱   236B total / 21B active  ·  8.1T tokens  ·  128K ctx
 ╱    ★ MLA — cache ONE latent/token (~57× KV vs MHA) + decoupled-RoPE key
 ╱    ★ DeepSeekMoE — fine-grained routed experts + always-on shared experts
∿∿∿∿∿∿∿∿∿∿   the exact attention+MoE shape V4-Flash still runs
```

**Source:** [arXiv:2405.04434](https://arxiv.org/abs/2405.04434) "DeepSeek-V2: A Strong, Economical,
and Efficient Mixture-of-Experts Language Model", DeepSeek-AI, submitted 2024-05 · [verified].

---

## 1. Why V2 is the trunk's true root

DeepSeek-**LLM** ([`15`](15-deepseek-llm-coder.md)) was a dense LLaMA-shaped model; **V4 does not
descend from it.** V4 descends from **V2**, because V2 is where the **two innovations V4 still uses**
first appear *in a trained model*:

- **MLA (Multi-head Latent Attention)** — the KV-cache compression that makes 1M context affordable.
- **DeepSeekMoE** — fine-grained routed experts + always-on shared experts (the MoE paper
  [2401.06066](https://arxiv.org/abs/2401.06066) proposed it; **V2 is its first frontier-scale
  training**).

Everything after V2 — Coder-V2, V2.5, V3, V3.1, V3.2, V4 — is a *continuation or refinement* of this
attention+MoE base. That is why this page sits at the center of the lineage.

**V2 config (report-stated):** **236B total parameters, 21B activated** per token; **8.1T-token**
pretrain corpus (Chinese tokens ≈ 1.12× English — the one published cross-language ratio,
[`01 §1`](01-pretraining-corpus.md)); **128K** context after extension.

---

## 2. MLA — the KV win, at recipe altitude

**The equations** (KV joint compression Eqs 9–11, decoupled RoPE Eqs 14–19) are derived in
[`theory/37 §1`](../theory/37-deepseek-architecture-deep.md#1-mla--multi-head-latent-attention). The
*recipe-level* facts that belong here:

- MLA caches **one low-rank latent `c^{KV}_t` (dim `d_c`) plus one shared decoupled-RoPE key** per
  token per layer — **`d_c + d^R_h = 512 + 64 = 576` elements**, vs MHA's `2·n_h·d_h = 32768`. That is
  the **~57× KV reduction** V2 Table 1 reports.
- **Why it needed the decoupled RoPE trick.** The cheap-cache property relies on *absorbing* the
  up-projection `W^{UK}` into the query at inference — but RoPE is position-dependent and multiplicative,
  which breaks that absorption. V2's fix: split each head into a **content part** (no RoPE, compressed)
  and a **decoupled RoPE part** carried on extra dims with a *single shared* RoPE key
  ([`theory/37 §1`](../theory/37-deepseek-architecture-deep.md#1-mla--multi-head-latent-attention)).
  This is subtle enough that it is *the* thing a re-implementer gets wrong.
- **The DwarfStar link.** The DSpark *draft* deliberately does **not** use MLA (`compress_ratio == 0`,
  a plain 128-token sliding window, [`theory/24 §3`](../theory/24-deepseek-v3-v4-context.md)) — drafting
  on raw recent context is cheaper than reconstructing latents. So the draft *skips* the V2 invention
  while the *verify* pass runs the full MLA-descended target.

---

## 3. DeepSeekMoE — fine-grained + shared, at recipe altitude

**The equations** (routing Eqs 9–11) are in [`theory/37 §2`](../theory/37-deepseek-architecture-deep.md#2-deepseekmoe--fine-grained--shared-experts).
Recipe-level:

- **Fine-grained segmentation.** Split each standard expert into `m` smaller ones (FLOPs constant, more
  combinatorial routing → sharper specialization).
- **Shared-expert isolation.** Carve out `K_s` experts **always on** for every token, to absorb common
  knowledge and de-duplicate the routed experts.
- **V2's MoE config:** the routed/shared/active counts scale up in V3 (256 routed / 8 active / 1 shared,
  Sigmoid affinity) and V4-Flash (256 routed / 6 active / 1 shared) — but the **template is V2's**
  ([`theory/37 §2`](../theory/37-deepseek-architecture-deep.md#2-deepseekmoe--fine-grained--shared-experts)).
- **Affinity note (easy to get wrong):** V2/DeepSeekMoE use **Softmax** affinity; **V3 switched to
  Sigmoid** with the selected gates normalized among the chosen experts. Do not use one equation for
  both — the lineage *changed* the affinity function.

---

## 4. V2's own training recipe (what belongs to V2, not the arch)

- **Pretrain:** 8.1T tokens ([`01`](01-pretraining-corpus.md)); tokenizer BBPE, **100K vocab** — the
  lineage the later 128K/129,280 vocab grew from ([`20`](20-tokenizer-bpe-vocab-evolution.md)).
- **Long-context (V2 §3.1.4):** **YaRN** extension **4K → 128K**, with **+1000 steps at seqlen 32K** —
  the same YaRN-step recipe V3 later reuses (two 1000-step phases, [`01 §5`](01-pretraining-corpus.md)).
- **Alignment:** SFT then RL. V2's RL uses **GRPO** (from DeepSeekMath, [`16`](16-deepseek-math-grpo-origin.md))
  — one of the *earliest* trunk uses of GRPO, before R1 made it famous. This is the through-line: the
  math branch's RL algorithm was already in the general trunk by V2.

---

## 5. The through-line to V4 (why this is the pivot page)

```
   V2  ──(MLA + DeepSeekMoE)──▶  V3 (256 experts, Sigmoid, aux-loss-free, FP8, MTP)
    │                              │
    └──(continue-pretrain)──▶ Coder-V2   └──▶ R1 (GRPO reasoning) ──▶ V3.1 ──▶ V3.2 (DSA) ──▶ V4
```

- **V4-Flash's attention is an MLA descendant** (CSA/HCA are V4's own compressed-attention names, but
  the *low-rank-KV philosophy* is MLA's, [`theory/37 §7`](../theory/37-deepseek-architecture-deep.md#7-how-v4flash--dspark-descend-from-all-this)).
- **V4-Flash's FFN is DeepSeekMoE** (256 routed / 6 active / 1 shared, `moe_intermediate_size` 2048 —
  on-disk config, [`theory/24 §2`](../theory/24-deepseek-v3-v4-context.md)).
- Every intermediate model **continue-pretrains** from a V2-line checkpoint rather than rebuilding —
  the pattern that keeps the family a single coherent trunk ([`15 §3`](15-deepseek-llm-coder.md),
  [`22`](22-v2p5-v3p1-v3p2-incremental.md)).

## 6. Honest provenance

- **CONFIRMED:** arXiv:2405.04434 exists (fetched 2026-07-05); 236B/21B, 8.1T tokens, MLA ~57× KV,
  DeepSeekMoE fine-grained+shared, 100K vocab, YaRN 4K→128K +1000 steps @ 32K are **report-stated**
  (cross-checked against [`01`](01-pretraining-corpus.md) and [`theory/37`](../theory/37-deepseek-architecture-deep.md),
  which quote the same sections). MLA equations are derived (not restated) in theory/37 §1.
- **No numbers reconstructed here.** The V2→V3 affinity change (Softmax→Sigmoid) and the expert-count
  growth are cross-referenced to the V3 report and theory/37, not invented.

## 7. Ranked sources

1. ⭐ **DeepSeek-V2 report** — [arXiv:2405.04434](https://arxiv.org/abs/2405.04434) · [verified]. MLA
   (§2.1, Eqs 9–19) + DeepSeekMoE + 8.1T corpus + YaRN long-ctx. **FAVOURITE of this page** — the arch
   V4 inherits.
2. ⭐ **DeepSeekMoE** — [arXiv:2401.06066](https://arxiv.org/abs/2401.06066) · [verified]. The
   fine-grained + shared-expert routing V2 first trained at scale.
3. **DeepSeek-V3 report** — [arXiv:2412.19437](https://arxiv.org/abs/2412.19437) · [verified]. The
   Sigmoid-affinity + 256-expert + aux-loss-free continuation of V2's MoE.
4. **theory/37 §1–§2** ([link](../theory/37-deepseek-architecture-deep.md)) — the MLA/MoE equations
   this page cites rather than repeats.

Cross-refs: [`00`](00-index.md) · [`14`](14-deepseek-model-lineage.md) · [`01`](01-pretraining-corpus.md) ·
[`20`](20-tokenizer-bpe-vocab-evolution.md) · [`22`](22-v2p5-v3p1-v3p2-incremental.md) ·
[`theory/37`](../theory/37-deepseek-architecture-deep.md) · [`theory/24`](../theory/24-deepseek-v3-v4-context.md).
