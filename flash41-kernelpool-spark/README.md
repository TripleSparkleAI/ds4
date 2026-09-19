# flash41-kernelpool-spark

Our own version of antirez/ds4 PR #1073 (kernelpool's DeepSeek V4.1 Flash Metal optimisations
and DSpark support) for the DGX Spark GB10 CUDA path. Branch `flash41-kernelpool-spark`, cut
from antirez `main` at `8db1d1d15`, stacked on nothing else. Docs land first; code lands per
port, one commit each, with a sealed round and a before/after CSV. Pushed only to the
`sparkle` fork, never to antirez's repository; no PR and no comment is opened from here.

| file | what |
|---|---|
| `CENSUS_pr1073_what-is-metal-what-is-cuda-what-is-portable_2026-09-20.md` | the 36 commits classified (11 Metal-only, 4 CUDA-touching, 11 engine-generic, 8 TP-only, 2 docs/tests), each Metal mechanism against what the CUDA path already does, overlap with hits-first and glm53 by function name |
| `BUILD_as-is-cuda_2026-09-20.md` | the as-is PR does not compile on CUDA (rc=2, six undefined identifiers, then two unresolved TP symbols); the two-commit fix ref `pr1073-cudafix` = `0bbca98ee`; .text 24,862,747 (main) -> 24,945,559 |
| `2026-09-20-KP1-PREREGISTERED-RULE.txt` | the sealed round: main@8db1d1d1 vs the fixed PR head, both `--cuda` at the standing 90GB streaming regime on `DeepSeek-V4.1-Flash-Q2.gguf`, 3 interleaved cycles, G1 greedy identity as the gate, predictions stated. Queued behind lane GLMTEST's lock at sealing time |
| `DESIGN_the-port-ranked-by-mechanism_2026-09-20.md` | the Metal mechanisms ranked by (share x confidence) / cost; KP2 (the flush knob) first because it is free, KP3 (bf16 epilogue on the norm and hc kernels) the first real port; what is not portable |
| `PR1073_COMMENT_DRAFT_we-tried-this-on-cuda_2026-09-20.md` | a comment for the PR, prepared as a file and NOT posted; measured numbers only, `[NOT YET MEASURED]` where a number is owed |

The fix ref is not on this branch on purpose: `pr1073-cudafix` is PR #1073's history plus two
commits and exists only so the seal has an arm that builds. This branch stays `8db1d1d15` + docs
until the first port lands.
