# `imatrix/` - calibration matrices

> ⚠ **EMPTY. Nothing has been calibrated.** Gated on `../fp16-source/`.

An importance matrix records, over a calibration corpus, how much each weight actually matters to
the output, so the quantizer spends its precision where it changes the answer.

## ⛔ AN IMATRIX IS VALID ONLY FOR THE ROTATION IT WAS CALIBRATED UNDER

PrismML's own loader **refuses a mismatched calibration bias by design** on its KV path, and exports
`LLAMA_ATTN_ROT_DISABLE=1` when a bias is present because the bias was calibrated with K-rotation
off. ⇒ **treat the rotation choice and the imatrix as one sealed pair**, never two independent
settings. `../../wiki/01-the-method-hadamard-rotation-and-imatrix-calibration.md`.

## ⛔ AND THE CALIBRATION CORPUS MUST BE DISJOINT FROM THE EVALUATION CORPUS

This estate has already been caught reading a train/test overlap as a result: a champion's gain
turned out to be a property of a lexicon whose reachability was high **only because it was the
training half of the same corpus.** ⇒ **the seal says which corpus, and says they are disjoint,
before the run.** `../../wiki/07-calibration-is-where-ptq-touches-training.md`.

## The naming, so two matrices cannot be confused

Every distinguishing choice goes in the filename: the corpus, the layer set, the rotation state.

    ✅ v41flash-imatrix-wiki103-layers-0-2-8-9-40-42-rot-hadamard-b128.imatrix
    ⛔ imatrix.dat
