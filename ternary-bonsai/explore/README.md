# EXPLORE - the fleet measurement, and why it is a separate thing

> ⛔ **NOT FIRED. NO BOX HAS BEEN RENTED.** Every box unchecked.

| page | what it measures |
|---|---|
| `00-P2-the-jimothy-box-economics-explore.md` | what a range of boxes actually costs and delivers, per answer |

## ★ Why this is its own directory rather than a step in `../methods/`

Because it **measures the fleet, not the model**, and it therefore runs on a **workload we already
understand** rather than on ternary. ⚠ **Mixing a fleet measurement with a model measurement gives
one number that answers neither** - you cannot tell a slow box from a slow method.

★ **And it does not wait on P0.** It can run the moment somebody wants it, which makes it the only
compute-spending step on this board that is not blocked by the FP16 precondition.

## What it is FOR

Every later step asks **"which tier, and how many"**. Right now that question has no numbers behind
it, and the one figure we do hold says the obvious answer can be wrong:

✅ MEASURED on the rented 5090: the expert-cache ask **25GB is already saturated** - it plans
**1,928** slots and delivers **1,127**, and **18GB delivers the same 1,127**.

★ **The ask was never the constraint. A bigger box is not automatically a faster answer**, and this
table is how we stop guessing.
