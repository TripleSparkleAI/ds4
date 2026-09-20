# MODEL-FORMATS.md - the exact GGUF quant names and which one loads where

- URL: https://github.com/PrismML-Eng/Bonsai-demo/blob/main/MODEL-FORMATS.md (5,098 bytes)
- Date read: 2026-09-18

## Bonsai 2 27B: three bands, no mainline-compatible one

Verbatim table:

| band | bits/weight | size | where |
|---|---|---|---|
| `PTQ1_0` | 1.75 | 5.9 GB | Ternary-Bonsai-2-27B-gguf |
| `PQ2_0` | 2.13 | 7.2 GB | same repo; what the demo downloads, faster prompt processing |
| `Q2_0` | 2.25 | 7.6 GB | Ternary-Bonsai-2-27B-gguf-dev, testing only |

"All three store their weights in a rotated basis and need the activation transform that only
this demo's binaries carry, from the PrismML fork. There is no 'works everywhere' option the
way group-64 `Q2_0` is for the previous generation."

`PQ2_0` and `PTQ1_0` fail safely on stock llama.cpp because their ggml type ids sit past
upstream's `GGML_TYPE_COUNT`. **`Q2_0` does not fail safely**: upstream already knows the `Q2_0`
type and supports the `qwen35` architecture, so mainline loads the file without a warning and
outputs gibberish. That is why the Bonsai 2 `Q2_0` band is published separately as
`Ternary-Bonsai-2-27B-Q2_0-prism-fork-required.gguf`, with the requirement in the filename.

## The previous generation's three formats

| format | ggml type id | group size | who reads it |
|---|---|---|---|
| legacy `Q2_0` (deprecated) | 42 | 128 | pre-v7 fork releases only (`prism` = prism-v5 branch) |
| official `Q2_0` | 42 | 64 | mainline llama.cpp AND prism-v7+ |
| `PQ2_0` | 142 | 128 | prism-v7+ fork builds, on supported backends |

Exact gen-1 filenames, 27B row: deprecated `Ternary-Bonsai-27B-Q2_0.gguf`, official group-64
`Ternary-Bonsai-27B-Q2_g64.gguf` (note the 27B name differs from the smaller sizes, which use
`_Q2_0_g64`), fork `Ternary-Bonsai-27B-PQ2_0.gguf`.

The legacy-format refusal message on prism-v7 builds: "this file matches the legacy Prism Q2_0
layout (group size 128 stored as ggml type id 42), but this build reads Q2_0 as the official
group-64 format".

## PQ2_0 backend support registry

Mirrors `pq2_0_ready_backend` in `scripts/common.sh`:

| backend | PQ2_0 kernels |
|---|---|
| Metal (macOS) | yes |
| CUDA | yes |
| ROCm / HIP | yes |
| CPU (x86 VNNI, ARM NEON) | yes |
| Vulkan | not yet (port planned) |
| SYCL | not yet |

**Consequence for the Spark:** CUDA has PQ2_0 kernels, so the fork's preferred band is available.
If a build falls back to Vulkan on the Spark (which the binary downloader will do, see
08_build_cuda_linux.md), PQ2_0 has no kernels there.
