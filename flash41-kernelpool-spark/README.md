# flash41-kernelpool-spark

Our own version of antirez/ds4 PR #1073 (kernelpool's DeepSeek V4.1 Flash Metal optimisations
and DSpark support) for the DGX Spark GB10 CUDA path. Branch `flash41-kernelpool-spark`, cut
from antirez `main` at `8db1d1d15`, stacked on nothing else. Pushed only to the `sparkle` fork,
never to antirez's repository; no PR and no comment is opened from here.

## THE PORT: one branch, one flag per mechanism, default OFF

The navigator's order (2026-09-19): all in one branch, with a flag for each part kernelpool
modified, so any one or any subset can be tested. Every CUDA-relevant or engine-generic mechanism
from the PR lands behind its own `DS4_KP_<NAME>` environment flag, read once per process:
`=1` on, `=0` explicitly off, and `DS4_KP_ALL=1` turns on every flag not explicitly `0`. With
nothing set the V4.1 graph runs main's operator sequence, launch for launch. When any flag is on
the engine prints one line at graph creation, `ds4: V4.1 kernelpool flags on: ...`, so a run log
names its own arm. The helper is `ds41_kp_env_on` in `ds4.c`; the shims that switch on the CUDA
side live in `ds4_deepseek41_cuda.cuh`.

| flag | his sha | mechanism | CUDA status at main | landed |
|---|---|---|---|---|
| `DS4_KP_QUEUE_LAYERS` | `2a281b080` | single-node decode encodes all layers in one command stream, a flush every N layers (`DS4_CUDA_V41_DECODE_FLUSH_LAYERS`, else `DS4_METAL_V41_DECODE_FLUSH_LAYERS`, default 2; 0 = one stream per token); both Engram rows uploaded before the stream, no layer-13 drain | changes CUDA behaviour: `ds4_gpu_end_commands` and `ds4_gpu_flush_commands` are both `cudaDeviceSynchronize`, so main's ~40 syncs per token (one per layer on this 40-layer file) become ~20 at the default and 2 at flush 0 | **LANDED** `6131e0b6d` |
| `DS4_KP_ROUND_IN_KERNEL` | `d95f8b610` | bf16 rounding applied on the store inside the producing kernel instead of a separate pass | the PR's CUDA shims were op + pass (no change); this branch adds real store epilogues: `rms_norm_weight_round_kernel`, `hc_weighted_sum_round_kernel`, `hc_expand_round_kernel` in `ds4_cuda.cu`, one launch where main ran two at the attention norm, ffn norm, both hc weighted sums and the hc expand, per layer; bit-identical by construction (same RNE of the same fp32). The q8 matmul (matvec at n=1, cuBLAS at n>1) and the rope keep two launches | **LANDED** `08c4d85ba` (KP3 in the design doc) |
| `DS4_KP_SKIP_SELECT` | `e76839617` | skip the per-token block top-2048 and mask launches while every block is a candidate (`(n_comp+7)/8 <= 2048`, i.e. up to 16,384 compressed rows) | changes CUDA behaviour: two launches fewer per index-source layer per token at the standing frontiers; lossless (the mask it skips writing was all zero) | **LANDED** `22dd14cca` |
| `DS4_KP_SELECT_BATCH` | `ce5a812fd` | prefill index batch: candidate blocks scored, selected and masked once per batch instead of once per row | CUDA gets `ds4_gpu_dsv41_candidate_topk_batch` / `_mask_batch` as a per-row loop over the existing top-k and mask kernels (no new kernel); prefill only | **LANDED** `48a03f251` |
| `DS4_KP_SHORT_ROWS` | `3b7f8f224` | rows with at most 512 visible index keys take every key in order; batched top-k from the second row of a batch instead of from 1,024 compressed rows | new tiny CUDA kernel `dsv41_indexer_all_kernel`; applies to the first 512 x ratio tokens of a prompt only, ~0 at the standing frontiers | **LANDED** `f5177f3f1` |
| `DS4_KP_HEAD_EARLY` | `4fbbc4a45` (queued-head half) | the vocab head is encoded inside the last layer's stream before its drain; logits read straight after | one begin/end pair fewer per token, on CUDA one `cudaDeviceSynchronize` fewer. Upstream gates it on `queue_layers`; here it stands with or without `DS4_KP_QUEUE_LAYERS` (never under layer_resident, imatrix or TP). The commit's other half, the wide prefill sweep from 3072 tokens, is NOT ported (prefill schedule, outside this port) | **LANDED** `8d0db8d1b` |
| `DS4_KP_DRAFT_GATE` | `1c6920666` | stop proposing DSpark drafts while they lose to plain decoding | **NO-OP, NOT LANDED**: main `8db1d1d15` has no V4.1 DSpark cycle at all. `ds41_session_spec` and `ds41_draft_propose`, which this commit edits, are introduced by `d8d1523b3` (4,093 lines, the Metal fused-decode commit). The flag would gate code that does not exist on this branch, so no code carries the name; setting it does nothing and the stamp line does not list it | not landed |
| `DS4_KP_RADIX_SELECT` | `af7c02b59` | radix select for the 2048-of-N index candidate pick, replacing a sort, on Metal | **NO-OP by census**: the commit touches `ds4_metal.m`, `metal/argsort.metal` and a Metal test only. CUDA's `ds4_gpu_indexer_topk_tensor` already selects without a sort (pow2 bitonic at `n_comp <= 4096`, chunk + merge above, CUB at 8192). No code carries the name; setting it does nothing | not landed |
| `DS4_KP_ALL` | - | every landed flag above that is not explicitly `0` | - | **LANDED** `c46a6ed3d` |

Every landed flag: `make cuda-spark` rc=0 on the GB10 (driver 580.159.03, CUDA 13.0), one build
per commit, logs `~/dwarfstar/.devlogs/flagport_build{1..8}.log` on the spark. `ds4-server .text`
by commit: main 24,862,747 · QUEUE_LAYERS 24,865,747 · ROUND_IN_KERNEL 24,898,903 · SKIP_SELECT
24,890,991 · SELECT_BATCH 24,888,415 · SHORT_ROWS 24,886,903 · HEAD_EARLY 24,887,447 · stamp
24,881,767 (the tip of the branch; +19,020 B over main, inside the seal's 0.2 MB gate).

⚠ **Off is main's operator sequence; it is NOT yet proven byte-identical by a run.** The claim
"byte-identical to main when all are off" is by construction at every gated site (each site
was read against `8db1d1d15`; the commit messages name them) and by the ROUND_IN_KERNEL
argument (same RNE of the same fp32 in the epilogue as in the pass). The G1 gate that turns the
construction into a measurement is the matrix runner's `off` arm against `tip`, and it has not
been run: the lock was KP1's for the whole of this lane.

⚠ **Untouched by this port, by order:** `routed_moe_launch`, the `cuda_stream_*` cache
functions and `glm_graph_*` (hits-first and glm53 live there). Derive:
`git diff 8db1d1d15..HEAD -- ds4_cuda.cu | grep -E '^[+-].*(routed_moe_launch|cuda_stream_|glm_graph_)'`
returns 0 rows.

### Not ported, and why

Metal-only and TP-only commits from the PR get no flag. From the census (row numbers there):
Metal-only 3 `edceb7ab7` (one-dispatch router; CUDA already one launch), 4 `10118742a`,
8 `d744ebb9e` (Q4_K resident tiles; not our file), 12 `d8d1523b3` (the fused Metal decode +
DSpark verify rows; its CUDA half is 9 `return 0` stubs and compositions of main's ops, and its
DSpark plumbing is what DRAFT_GATE would have needed), 16 `1923131d4`, 21 `e49643fff` (shared
expert beside routed; CUDA already a second stream), 22 `262b4a66e` and 23 `a6b50ff6c`
(simdgroup matmuls; cuBLAS at n>1), 26 `bd85ba6ac`, 34 `fd7091916`, 35 `4b1b1519e`, 36
`af7c02b59` (RADIX_SELECT above). TP-only 14, 15, 18, 19, 20, 24, 25, 30 (two Macs over a wire;
the GB10 is one box). Engine-generic but not decode and not taken: 6 `ccbf2c083` (prefill
decoder threshold), 10, 11, 32 (converter), 27 `7da9535f2` (server slots), 28 `b7bcccdea`
(DSpark EOS), 29 `29ce27171` (Engram prefill table), and the 3072-sweep half of 13.

## THE MATRIX RUNNER: `kp_matrix.sh`

```
bash flash41-kernelpool-spark/kp_matrix.sh --arms tip,QUEUE_LAYERS,ROUND_IN_KERNEL,ALL --cycles 3
bash flash41-kernelpool-spark/kp_matrix.sh --arms off,ALL --cycles 3 --name kpid   # the byte-identity control arm
bash flash41-kernelpool-spark/kp_matrix.sh --help | --selftest | --dry-run --arms ...
```

Runs `ds4-bench` on the standing sweep (ctx 2048 warm-up frontier, 4096/6144, gen 128, `--cuda
--ssd-streaming --ssd-streaming-cache-experts 90GB`, `promessi_sposi.txt`) for a SET of arms,
sequentially, KP1's shape: one discarded tip warm-up, then per cycle every arm followed by a tip
bracket. `tip` is `_worktrees/kp-main`'s binary, `off` the port's binary with no flag, `ALL`
`DS4_KP_ALL=1`, and `NAME` or `NAME+NAME` the named flags. One CSV per run in ds4-bench's own
`--csv` format under `~/sweeps/<name>_<arm>_<cycle>.csv`, a runlog with driver, both binaries'
`.text`, the model's header ident (arch, name, size, mtime) and per-run load/gpu/memavail, a
SIGCONT-only net keyed to the script's pid, and the `~/spark_model.lock` mkdir protocol: a held
lock prints its owner and exits 2. Refuses a model whose header arch is not `deepseek41`.
`--selftest` is 19 checks (arm parsing, plan shape, lock refusal and release, gguf ident, a stub
bench through the run path), green on the M5 and on the spark.

## THE MODEL FILE (censused 2026-09-19)

`~/dwarfstar/gguf/DeepSeek-V4.1-Flash-Q2.gguf`, 365,713,686,528 B, is a runnable DeepSeek V4.1
Flash q2: `general.architecture = deepseek41`, 1,046 tensors, 40 layers, 384 routed experts, 6
per token, IQ2_XXS gate/up (93.4 GB) + Q2_K down (59.5 GB) + Q8_0 attention/shared/head
(7.6 GB) + F16/F32 (2.5 GB) = 163 GB of transformer weights over 754.6 G elements (a ~552B
text model, not 284B; 284B is V4 Flash), plus 202.76 GB in two I8 Engram tables
(384,006,168 + 384,016,682 rows x 264 B, encoding `e4m3_e8m0_32_row264`, layers 1 and 14).
The tensor table sums to 365.70 GB, matching the file to 0.003 %. It is the file R28, R29,
GLM-R1 and KP1 ran; no converter run is needed.

## The docs

| file | what |
|---|---|
| `CENSUS_pr1073_what-is-metal-what-is-cuda-what-is-portable_2026-09-20.md` | the 36 commits classified (11 Metal-only, 4 CUDA-touching, 11 engine-generic, 8 TP-only, 2 docs/tests), each Metal mechanism against what the CUDA path already does, overlap with hits-first and glm53 by function name |
| `BUILD_as-is-cuda_2026-09-20.md` | the as-is PR does not compile on CUDA (rc=2, six undefined identifiers, then two unresolved TP symbols); the two-commit fix ref `pr1073-cudafix` = `0bbca98ee`; .text 24,862,747 (main) -> 24,945,559 |
| `2026-09-20-KP1-PREREGISTERED-RULE.txt` | the sealed round: main@8db1d1d1 vs the fixed PR head, both `--cuda` at the standing 90GB streaming regime on `DeepSeek-V4.1-Flash-Q2.gguf`, 3 interleaved cycles, G1 greedy identity as the gate, predictions stated |
| `DESIGN_the-port-ranked-by-mechanism_2026-09-20.md` | the Metal mechanisms ranked by (share x confidence) / cost; KP2 (the flush knob) first because it is free, KP3 (bf16 epilogue on the norm and hc kernels) the first real port; what is not portable |
| `PR1073_COMMENT_DRAFT_we-tried-this-on-cuda_2026-09-20.md` | a comment for the PR, prepared as a file and NOT posted; measured numbers only, `[NOT YET MEASURED]` where a number is owed |
| `kp_matrix.sh` | the flag-matrix bench runner, above |

The fix ref `pr1073-cudafix` is not on this branch on purpose: it is PR #1073's history plus two
commits and exists only so the KP1 seal has an arm that builds. This branch is `8db1d1d15` + the
docs + the seven flag commits above.
