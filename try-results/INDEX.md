# try-spec-offload results

One line per leg.  Date, branch, what was ON, and the file.

**Every table below is OWED.**  The four legs have not been run: the CUDA build
gate and the bench both need the DGX Spark, and the Spark is serialized to an
unrelated series.  The file names are fixed now so the legs cannot be renamed
after the fact, and so the interleaving order is written down before the numbers
exist.

**The run order, fixed in advance.**  Legs are run in this sequence on ONE binary
vintage, then the whole block repeats:

    1. cold / dspark off      3. warm / dspark off
    2. cold / dspark on       4. warm / dspark on

A leg measured against a control from another build is not a measurement.  The
pair partners are named in each file: the off leg carries the SAME argv as its on
leg, `--dspark --mtp-model` included, and differs only by `DS4_MTP_SPEC_DISABLE=1`.

| date | branch | arm | dspark | prime | file |
| --- | --- | --- | --- | --- | --- |
| 2026-09-16 | triple-spec-under-offload | cold | on | no | `2026-09-16-triple-spec-under-offload-dspark-on-cold.txt` |
| 2026-09-16 | triple-spec-under-offload | cold | off | no | `2026-09-16-triple-spec-under-offload-dspark-off-cold.txt` |
| 2026-09-16 | triple-spec-under-offload | warm | on | yes | `2026-09-16-triple-spec-under-offload-dspark-on-warm.txt` |
| 2026-09-16 | triple-spec-under-offload | warm | off | yes | `2026-09-16-triple-spec-under-offload-dspark-off-warm.txt` |
