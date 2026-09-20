# GLMPREFILL PREREGISTERED RULE - the 0e9cc2d43 differential on CUDA

Sealed **2026-09-20T08:53Z** (spark local 2026-09-20 18:53 AEST, UTC+10) BEFORE any worktree on the
spark, any build, any binary, any bench run and any reading of a t/s. Nothing above a line marked
AMENDMENT is ever edited; amendments append as A1, A2, ... and say what they add.

The question comes from antirez/ds4 issue 1029 (nomishbhardwaj, 2026-09-12, no replies). Its
measurement is on an M5 Max under Metal. **Nothing in this seal is posted anywhere.**

---

## THE QUESTION

Issue 1029 bisected a 6 to 8 % GLM-5.3-Flash cold-prefill loss to ONE commit and stated the one
thing it could not answer:

> so this may be an unnoticed side effect of the added reduction-storage synchronization /
> padded-selection handling on Apple GPUs rather than an accepted trade-off.

The issue's own direct two-commit differential, `9d9e129` against `0e9cc2d`, Metal, resident Q2,
2 interleaved rounds each:

| depth | 9d9e129 | 0e9cc2d | delta |
| --- | ---: | ---: | ---: |
| 8K cold | 415 t/s | 382 t/s | -8.0 % |
| 16K cold | 386 t/s | 359 t/s | -7.0 % |
| 24K cold | 376 t/s | 355 t/s | -5.5 % |

Mean **-6.8 %**. Decode 0-24K flat on both.

**Q1 Does the same two-commit differential cost CUDA prefill on a GB10, at the same three depths?**
**Q2 If it does, is the magnitude the same as Metal's, smaller, or larger?**
**Q3 Does decode stay flat here too, i.e. is the effect confined to prefill?**
**Q4 What dense-attention limit does each binary print at load, on this box, at each ctx?**

Q4 exists because the limit is the mechanism and the issue could not see it: the OLD binary's load
line prints the cap and NOT the limit, so a reader of a 9d9e129 log cannot know what number the
commit changed.

---

## THE OBJECTS

```
  arm      worktree ~/dwarfstar/_worktrees/glmpf-new @ 0e9cc2d43ec7fed31d648658b3e9cb74a4ccad4b
           "Fix GLM attention masking and reductions across GPU backends", antirez, 2026-09-05
  control  worktree ~/dwarfstar/_worktrees/glmpf-old @ 9d9e1296d, which IS 0e9cc2d43^
           "Budget ROCm allocations against usable pinned memory"
           Verified before sealing: `git merge-base --is-ancestor 9d9e1296d 0e9cc2d43` is true and
           `git merge-base --is-ancestor 0e9cc2d43 8db1d1d15` is true, so the pair straddles the
           commit and both are ancestors of antirez main.
  build    `make cuda-spark -j12` in each worktree, nothing else, no env switches, NOTHING OF OURS.
           Neither tree is patched. If either refuses to build, that is the cell's finding and it
           is reported as a refusal with the error verbatim, never worked around.
  box      DGX Spark GB10, 20 cores, 121 GiB unified, driver 580.159.03, CUDA 13.0 V13.0.88,
           under ~/spark_model.lock held BY THE RUNNER for this job alone.
  model    ~/dwarfstar/gguf/GLM-5.3-Flash-Q2.gguf, 96,505,816,384 B, RESIDENT (no --ssd-streaming).
           sha256 prefix recorded in the cell's own stamp; the ledger's published value is
           e81fd624...05b32 and a mismatch VOIDS the cell.
           89.87 GiB of model against 117 GiB MemAvailable read before sealing, so it fits; the
           cell records MemAvailable again before every run and the reader may discount.
```

---

## THE MEASURANDS

### PREFILL, and it is COLD BY CONSTRUCTION

`prefill_tps` from antirez's own `ds4-bench` CSV, **one depth per process invocation**:

```
  ./ds4-bench -m <GLM q2> --cuda --prompt-file speed-bench/promessi_sposi.txt \
      --ctx-start <D> --ctx-max <D> --gen-tokens 128 --csv <file>
```

with D in **8192, 16384, 24576**.

**Why this is the issue's measurand and not a continued prefill.** `ds4_bench.c` initialises
`previous = 0` and reports `prefill_tokens = frontier - previous` for the first frontier, so a run
whose `--ctx-start` equals its `--ctx-max` performs exactly ONE prefill, of D tokens, into an empty
session. A fresh process per depth is what the issue calls "KV purged per block"; here there is no
purge because there is nothing to purge. Read at `8db1d1d15:ds4_bench.c` before sealing.

**Why ds4-bench and not the server path.** Issue 1029 used `ds4-server`'s chat path and its
server-reported `avg=`. This cell uses `ds4-bench` and says so, for three reasons stated before any
number is read: the tool is antirez's own and its CSV column is named `prefill_tps`, so the figure
needs no harness of ours to be believed; a bench invocation is one process with one argv, which a
queue cell can express and a chat session cannot; and the issue's vision encoder and its
`--kv-disk-dir` are absent here, which removes two variables this cell is not asking about.
**The consequence is stated rather than hidden: our absolute t/s is NOT comparable to the issue's
absolute t/s.** Only the WITHIN-BOX RATIO between the two binaries is, and the ratio is the answer
to Q1.

### DECODE, the control that must stay flat

`gen_steady_tps` and `gen_tps` from the same CSV rows, at all three depths. The commit touches the
prefill path's dense boundary; if decode moves as much as prefill, the mechanism is not the one this
cell names and the cell says so.

### THE DENSE-LIMIT LINE, which is Q4 and costs nothing

Both binaries print at load when `ctx_size > ctx_cap`. The two spellings, verbatim from the diff:

```
  9d9e129   "ds4: GLM session ctx=%u (model max=%u); full-attention prefill/work cap=%u; compact indexed decode is used beyond the cap"
  0e9cc2d   "ds4: GLM session ctx=%u (model max=%u); prefill/work cap=%u; dense attention limit=%u"
```

Every run's log is grepped for its own line and both are quoted in the result. At ctx 8192, 16384
and 24576 the condition `ctx_size > ctx_cap` holds for any cap at or below 8192, so the line is
expected in all six log families.

### THE FLOOR, computed from THIS CELL'S OWN adjacent same-binary pairs and from nothing else

Order, fixed in the job's own `run.sh` and therefore not reorderable by the queue:

```
  cycle 0   OLD 8K, NEW 8K, OLD 16K, NEW 16K, OLD 24K, NEW 24K      <- WARM-UP, DISCARDED WHOLE
  cycle 1   OLD 8K, NEW 8K, OLD 16K, NEW 16K, OLD 24K, NEW 24K
  cycle 2   OLD 8K, NEW 8K, OLD 16K, NEW 16K, OLD 24K, NEW 24K
  cycle 3   OLD 8K, NEW 8K, OLD 16K, NEW 16K, OLD 24K, NEW 24K
```

24 runs, interleaved A/B/A/B, 3 counted cycles per condition after one discarded cycle, exactly as
the issue interleaved and discarded its own first round as cold.

Per depth, the arm figure is **MIN ACROSS the 3 counted cycles** for each binary (the standing
measurement law: take the minimum, never the mean, because contention noise is one-sided), and

```
  D_depth  = (min_new - min_old) / min_old * 100
  floor_depth = max over the 4 adjacent same-binary pairs at that depth of
                |t_{i+1} - t_i| / mean(t_i, t_{i+1}) * 100
                (2 pairs from OLD's cycles 1-2 and 2-3, 2 pairs from NEW's)
  sign_depth = how many of the 3 cycles have new < old at that depth, out of 3
```

**VERDICT per depth:** `LOSES` if `D <= -floor`; `WINS` if `D >= +floor`; otherwise `TIES`.
**THE CELL'S HEADLINE VERDICT** is one word over the three depths:
`REPRODUCES ON CUDA` if at least 2 of 3 depths read LOSES;
`DOES NOT REPRODUCE` if 0 of 3 read LOSES and at least 2 of 3 read TIES or WINS;
`INCONCLUSIVE` otherwise, with the floor printed beside it.

### STAMPS, on every run's own row

box, both clocks, driver 580.159.03 read live, CUDA version, model path and bytes and sha prefix,
ctx, git sha of the binary that ran, the exact argv, load before and after, GPU utilisation before
and after, MemAvailable before and after, rc, and the run's own dense-limit line.

---

## WHAT IS ALREADY READ FROM THE SOURCE, AND IS THEREFORE A REPRODUCTION CHECK, NOT A PREDICTION

All six read at the two commits on the laptop before sealing. The spark must merely agree.

**R1 THE DENSE LIMIT MOVES FROM 4096 TO 2051, AND IT IS BACKEND-INDEPENDENT.**
`glm_graph_dense_compact_attention_limit` in `ds4.c` changes from

```c
    if (g && g->glm53 && !g->full_kv_cache) return g->ctx_cap;
    return glm_graph_indexer_top_k_limit();
```

to

```c
    return g && g->glm53 ? glm53_graph_indexer_selected_limit() :
                          glm_graph_indexer_top_k_limit();
```

and `glm53_graph_indexer_selected_limit()` is `glm_graph_indexer_top_k_limit() +
DS4_GLM53_INDEX_POOL_SIZE - 1u` with `DS4_GLM53_INDEX_POOL_SIZE 4u`. The commit message names the
result: "the model-defined 2051-token dense boundary", so `top_k` is 2048 for this model.
`g->ctx_cap` is `glm_graph_full_attention_cap(ctx_size, ssd_streaming)`, which for a non-streaming
session is `DS4_GLM_METAL_FULL_ATTN_DEFAULT_CONTEXT = 4096u`, reduced only to `ctx_size` when
`ctx_size` is smaller and only at `ctx_size >= 65536` by the long-context clamp. **At our three
depths, resident, the old limit is 4096 and the new limit is 2051.**
**And `g->full_kv_cache` is ALWAYS false**: `glm_graph_expanded_kv_cache_enabled` is
`(void)ssd_streaming; return false;`. So the old branch's guard never fires and the old limit really
is `ctx_cap`, not `top_k`. This clause is why the change is large rather than three tokens wide.

**R2 ON CUDA THE PADDED-SELECTION REROUTE IS A NO-OP, AND THIS IS THE CELL'S SHARPEST SOURCE READ.**
`ds4.c` reroutes GLM 5.3 from `..._lora_valid_tensor` to `..._lora_tensor`. In `ds4_cuda.cu` the same
commit RENAMES the existing `ds4_gpu_glm_attention_indexed_batch_lora_valid_tensor` to
`ds4_gpu_glm_attention_indexed_batch_lora_tensor` with its body untouched, then adds a new
`..._valid_tensor` that does nothing but forward every argument to `..._tensor`. Verified by
extracting both function bodies and diffing them: **BODIES IDENTICAL**, zero differing lines. So on
CUDA both names resolve to the same code and the reroute cannot change what runs.

**R3 ON METAL THE SAME REROUTE IS A KERNEL SWAP, AND THE COMMIT DID NOT TOUCH THE FILE THAT MAKES IT
ONE.** In `ds4_metal.m`, unchanged by this commit and carrying both entry points at the PARENT
commit already, `..._lora_tensor` calls `..._lora_layout_tensor(..., false)` and
`..._lora_valid_tensor` calls it with `true`; the final parameter is named `selected_rows_valid`, and
inside `_layout_tensor` it picks the PIPELINE:

```c
    if (use_vec_lora && selected_rows_valid && full_head_groups)  -> ..._group8_vec_valid_fullheads
    else if (use_vec_lora && selected_rows_valid)                 -> ..._group8_vec_valid
    else if (use_vec_lora)                                        -> ..._group8_vec
    else                                                          -> ..._group8
```

So the reroute drops GLM 5.3 on Metal off two specialised pipelines onto the generic one. **This is
an observation about source, not a measurement, and this cell measures no Metal.**

**R4 THE ROCm SLICE CLAMP IS NOT ON CUDA.** The added `if (!slice_causal && slice > 256u) slice =
256u;` sits inside `#ifdef DS4_ROCM_BUILD`.

**R5 THE ONLY CUDA DEVICE-CODE CHANGE IS ONE BARRIER, AND IT IS IN THE DENSE KERNEL.** `ds4_cuda.cu`
gains a single `__syncthreads()` after `const float max_score = reduce[0];` in
`glm_dense_attn_causal_softmax_f16_kernel`. Metal gains the same shape in four places in
`metal/dsv4_misc.metal` plus one vision line. **The kernel the CUDA barrier is added to is the
kernel that R1 now runs over a range less than half as wide.**

**R6 THE CHANGED BAND IS REACHABLE ON THE CUDA PREFILL PATH.** `dense_limit` is consumed in
`glm_graph_forward_indexed_tokens`, and `glm53_graph_use_indexed_prefill` on a non-Apple, non-ROCm
build is `g->glm53 && g->indexed_prefill_cap != 0` with no `full_kv_cache` term, so the indexed
prefill path is the CUDA prefill path for this model. **A source read establishes that the path is
reachable; it does NOT establish that it runs, and no nsys capture is in this seal.** That
distinction is this repo's CALL-PATH GATE and it is written here so the result cannot quietly
upgrade a source read into a profile.

**THE MECHANISM R1 TO R6 IMPLY, stated before the measurement so it can be wrong.** In the NEW
binary, prefill positions in the band [2051, 4096) leave the dense causal path and take the sparse
selected path, which must also SCORE and TOP-K SELECT for those positions; the OLD binary attended
densely and skipped the selection. That is an extra cost of fixed ABSOLUTE size, about 2,045
positions wide, independent of depth. **A fixed absolute cost over growing total work shrinks as a
percentage with depth**, and the issue's own direct differential is monotone in exactly that
direction: -8.0, -7.0, -5.5 at 8K, 16K, 24K. Above 4096 both binaries are sparse and identical.

---

## THE PREDICTIONS

Probabilities are the lane's own and are scored HIT or MISS against the measured number.

### P - THE PREFILL DIFFERENTIAL, the headline

- **P1  0.60  THE SIGN IS NEGATIVE AT ALL THREE DEPTHS: `min_new < min_old` at 8K and 16K and
  24K.** Ground: R1 is backend-independent and R6 puts it on the CUDA prefill path, so the extra
  selection work exists here too. Not higher, because the sparse path READS FEWER KV ROWS in that
  band (2051 against up to 4095) and a CUDA sparse kernel efficient enough to pay for its own
  selection would make the sign positive.
- **P2  0.45  THE MEAN OF THE THREE DELTAS LANDS IN -4.0 .. -0.5 %, central -2.0 %.** Explicitly
  SMALLER in magnitude than Metal's -6.8 % mean. Ground: R2 and R3 together say one of the two
  mechanisms in this commit is a kernel swap on Metal and literally nothing on CUDA, so CUDA should
  pay at most the dense-limit half. Confidence is under one half because the split between the two
  mechanisms' costs is unknown and could be 0/100 either way.
- **P3  0.50  THE MAGNITUDE SHRINKS WITH DEPTH: |D_8K| >= |D_16K| >= |D_24K|.** Ground: the fixed
  absolute cost argued above. This is the prediction most worth being wrong about, because a
  violation says the cost is not the band's selection work.
- **P4  0.25  AT LEAST ONE DEPTH READS `LOSES` AGAINST ITS OWN FLOOR.** Deliberately LOW. A 2 %
  effect against a floor this estate has repeatedly measured at 2 to 3 % on this box is a TIE by
  construction, so the honest expectation is that the cell can see the SIGN and cannot resolve the
  MAGNITUDE. Saying so before the run is the point.
- **P5  0.55  THE HEADLINE VERDICT IS `INCONCLUSIVE`,** for exactly the reason in P4.
  `REPRODUCES ON CUDA` 0.25, `DOES NOT REPRODUCE` 0.20.

### N - THE NULL, and it is the load-bearing one

- **N1  0.70  EVERY ADJACENT SAME-BINARY PAIR IS INSIDE THE FLOOR, BY CONSTRUCTION.** The floor IS
  the max over those pairs, so this is a statement that no pair is a wild outlier: no adjacent
  same-binary pair at any depth exceeds **6.0 %**. A pair above 6 % means the box moved under the
  round and the cell reports that instead of a differential.
- **N2  0.75  THE FLOOR AT EACH DEPTH LANDS IN 0.5 .. 4.0 %, central 2.0 %.** Ground: this estate's
  own recent same-box floors on decode read 2.27 % (KP3), 3.1049 % (KP1FIRE) and 2.49 % (R28). No
  prefill floor has ever been taken on this box, so the band is wide and the confidence is not high.
- **N3  0.80  DECODE IS FLAT: the `gen_steady_tps` differential at every depth is INSIDE that
  depth's own decode floor,** computed the same way from the same adjacent pairs. Ground: the commit
  changes a prefill boundary, and the issue measured decode flat on Metal.

### Q - THE LOAD LINE

- **Q1p  0.85  THE OLD BINARY PRINTS `full-attention prefill/work cap=4096`** at all three ctx.
  Ground: R1's arithmetic. A different number falsifies R1's cap derivation, which would be a
  better finding than the differential.
- **Q2p  0.85  THE NEW BINARY PRINTS `prefill/work cap=4096; dense attention limit=2051`** at all
  three ctx.
- **Q3p  0.90  THE LINE IS PRESENT IN ALL SIX LOG FAMILIES**, because `ctx_size > ctx_cap` holds at
  8192, 16384 and 24576 against a cap of 4096.

### B - THE BUILDS

- **B1  0.85  BOTH WORKTREES BUILD CLEAN WITH `make cuda-spark -j12`, rc=0, NOTHING OF OURS.**
  Ground: both commits are ancestors of `8db1d1d15`, which this box builds today, and the CUDA
  surface of the pair differs by one `__syncthreads()` and one function rename.
- **B2  0.60  EACH BUILD'S WALL TIME LANDS IN 300 .. 1200 s, central 600.** WEAKLY GROUNDED and
  flagged as such: the only ds4 CUDA build wall time this estate has recorded on this box is 205 s
  for one tree, and a `-j12` build of two trees in sequence has no prior reading.
- **B3  0.70  THE TWO `ds4-bench` BINARIES DIFFER IN Berkeley `size` text.** A byte-equal pair would
  mean the commit changed nothing the compiler emitted, which R5 says is false.

---

## WHAT VOIDS WHAT

- **V1** A bench run with `rc != 0` is VOID and reported as void, never dropped quietly. One void
  run at a depth voids that depth's differential; two or more voids at one depth void that depth
  entirely and the cell reports the other depths and says which is missing.
- **V2** A counted run whose log contains `aligned artifact` is VOID: the first load of this file
  builds CUDA derived artifacts on disk and a run that pays that cost is a different machine. This
  is why cycle 0 is discarded whole. The cell greps every log and reports any hit.
- **V3** A worktree that is not exactly `9d9e1296d` or `0e9cc2d43` VOIDS the whole cell. Verified by
  `git rev-parse` in each tree at the END of the cell, not only at the start.
- **V4** A model sha256 prefix that does not match the ledger's `e81fd624` VOIDS the whole cell.
- **V5** Losing `~/spark_model.lock` voids everything measured after the loss. The runner holds the
  lock for this job; the cell asserts ownership and refuses rather than running unheld.
- **V6** `OOM`, `Killed`, `adjusted to fit`, or any CPU-fallback line in a counted log is reported
  beside that run's number and the run is marked suspect. A resident 89.87 GiB model on a 121 GiB
  box has about 27 GiB for KV at 24K, and if that is not enough the cell says so rather than
  reporting a number taken under adjustment.
- **V7 NO RUN IS DROPPED FOR LOAD.** Load, GPU utilisation and MemAvailable are stamped on every run
  and reported. A reader may discount; the lane may not.

---

## WHAT THIS CELL CANNOT SAY, WHATEVER IT MEASURES

- **It cannot price anything on Metal.** This is a CUDA box. R3 is a source read and stays one.
- **It cannot say which of the commit's two mechanisms costs what on Metal.** It can say that one of
  them is a no-op on CUDA (R2, verified by diff) and therefore that a CUDA number prices the OTHER
  one alone. That is the whole reason a CUDA reading answers the issue's open question at all.
- **It cannot say a kernel RUNS.** R6 establishes reachability from source. No nsys capture is in
  this seal, so nothing here is a profile.
- **It cannot compare its absolute t/s to issue 1029's.** Different box, different backend,
  different harness, no vision encoder, no `--kv-disk-dir`. Only the within-box ratio transfers.
- **It cannot attribute the M5's loss.** It can only say whether the same differential is visible
  here, and at what size against this round's own floor.
- **It cannot resolve an effect smaller than its floor**, and P4 and P5 say in advance that this is
  the likely outcome.
