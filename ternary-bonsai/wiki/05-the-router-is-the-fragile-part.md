# ✅ THREE MEASURED FACTS THAT SAY THE ROUTER IS WHERE THIS BREAKS

Three readings from our own tree, all MEASURED, all bearing on the router. Together they are a
strong prior about where ternary PTQ fails on an MoE, and they come from the people who built the
model rather than from our opinion.

## a · The model's own designers already treat the router as precision-sensitive

Jobs `110/111-router-tensor-census`, reading `ffn_gate_inp.weight` - the router itself:

| file | `ffn_gate_inp.weight` dtype | the experts around it |
|---|---|---|
| ds4flash | **F16** | 2-bit |
| V4.1 | **F32** | 2-bit |

⇒ **the router ships at 8 to 16 times the experts' precision.** Nobody chose that by accident.

⚠ **This corrected a claim in our own queue.** Row 4a asserted those tensors were F32 on **both**
files. On ds4flash it is **F16**, and a probe written to the stated dtype would have read an F16
tensor as F32 and produced numbers with nothing wrong-looking about them. **Recorded here as a
caught error, not tidied away.**

## b · The router is not uniform across layers, so a uniform policy is already wrong

| fact | figure | source |
|---|---|---|
| `exp_probs_b.bias` **absent** at | **layers 0, 1 and 2 of 43** on ds4flash | jobs `110/111` |
| the router bias spread **is a per-layer property** | **p = 0.00100**, both files | sealed cell `121-router-bias-centred-null` |

The p-value came from a mean-preserving two-sided null **with a vacuity control and a positive
control on a planted effect**, which is why it is quotable.

⇒ the issue's *"ternarize a few expert layers"* is the **right shape**, and **which** layers is
**not arbitrary**. The first three differ structurally from the rest, so a sample that happens to
include them measures something different from one that does not, and a sample that happens to
avoid them will look better for a reason that has nothing to do with ternary.

★ **The honest sample, and it belongs in a seal before running:** two early, two middle, two late,
named in advance. `../methods/03-P1-the-quality-measurement.md`.

## c · What this does NOT license

⛔ **These three facts do not say ternary will fail.** They say: if it fails, the router is the
first place to look, and a plan that quantizes the router at the same precision as the experts is
contradicting the model's own shipped choice. **That is a prior, not a result.**
