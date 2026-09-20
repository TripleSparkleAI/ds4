# WIKI INDEX - ternary PTQ for DeepSeek V4.1 Flash

The knowledge half of `ternary-bonsai/`. Every page states what is measured, what is claimed by
somebody else, and what nobody knows. Three labels, never mixed: **MEASURED** = a number taken on
real hardware, with its job and its box. **CLAIMED** = somebody else's statement, attributed.
**UNKNOWN** = nobody in the sources says, and nothing was invented to fill the gap.

| page | what it settles |
|---|---|
| `01-the-method-hadamard-rotation-and-imatrix-calibration.md` | what ternary PTQ actually does, and which parts of it we have never run |
| `02-the-ggml-types.md` | ⛔ the type trap: 142 is 2.125 bpw, 143 is the ternary one at 1.75 |
| `03-the-moe-risk.md` | ★ why NLL alone cannot see the MoE failure mode, and what to measure instead |
| `04-the-byte-layout-and-the-engram-ceiling.md` | ✅ 55.44 % of the file is un-ternarizable I8 lookup |
| `05-the-router-is-the-fragile-part.md` | ✅ three measured facts that say the router is where this breaks |
| `06-the-box-economics.md` | the JIMOTHY tiers and the money rule, quoted from the skill, never invented |
| `07-calibration-is-where-ptq-touches-training.md` | ⚠ the one place PTQ and training meet, and the fence around it |

## Sources on disk

| file | what it is | read |
|---|---|---|
| `SOURCE_prismml-model-formats_read-2026-09-18.md` | PrismML's own `MODEL-FORMATS.md`, the three bands and the backend registry | 2026-09-18 |
| `SOURCE_prismml-findings_read-2026-09-18.md` | lane PRISMHARVEST's findings over 21 sources, exact byte counts from the HF API | 2026-09-18 |
| `SOURCE_prismml-sources-list_read-2026-09-18.md` | the URL list behind both of the above | 2026-09-18 |

⚠ **Three files, not the 23 that `prism/research/` holds on the research branch.** The other 20
are about serving Bonsai 2 27B concurrently on the Spark, which is a different project with a
different object; bringing them here would put two subjects in one folder. They are reachable
with `git show autonomous/dspark-concurrency-and-spark-queue:prism/research/<file>`. The three
brought in are the ones that bear on the TYPES and the BYTE COUNTS, which is what this folder
argues about.
