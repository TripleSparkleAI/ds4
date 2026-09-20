# FINDINGS - serving Ternary Bonsai 2 27B concurrently on a DGX Spark GB10

- Lane PRISMHARVEST, research to disk only. Date read for every source: 2026-09-18.
- 21 source files in this folder, one per source, each carrying its URL and its numbers.
- Three labels are used and never mixed: **CLAIMED** = PrismML's own statement. **MEASURED** = a third party took the number on real hardware. **UNKNOWN** = nobody in these sources says, and nothing was invented to fill the gap.

---

## 1 - The model files, exactly

MEASURED from the Hugging Face API, exact byte counts (11_hf_file_sizes.md). Repo
`prism-ml/Ternary-Bonsai-2-27B-gguf`, Apache 2.0, lastModified 2026-09-17T18:44:01Z.

| band | file | bytes | GB | bits/weight |
|---|---|---|---|---|
| PTQ1_0 (dense trits) | `Ternary-Bonsai-2-27B-PTQ1_0.gguf` | 5,946,648,928 | 5.947 | 1.75 |
| PQ2_0 (2-bit slots) | `Ternary-Bonsai-2-27B-PQ2_0.gguf` | 7,206,168,928 | 7.206 | 2.13 |
| F16 reference | `Ternary-Bonsai-2-27B-F16.gguf` | 53,808,408,928 | 53.808 | 16.0 |
| vision projector | `Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf` | 629,246,976 | 0.629 | - |
| vision projector BF16 | `Ternary-Bonsai-2-27B-mmproj-BF16.gguf` | 931,145,856 | 0.931 | - |

A third band, `Q2_0` at 2.25 bits/weight and 7.6 GB, is published **outside** the main repo as
`Ternary-Bonsai-2-27B-Q2_0-prism-fork-required.gguf` in `Ternary-Bonsai-2-27B-gguf-dev`, testing
only. Its size is CLAIMED, not byte-verified here.

**The "5.9 GB" headline is the PTQ1_0 band. The demo downloads PQ2_0 by default, so 7.21 GB is
what lands on disk and what the server loads unless told otherwise** (09_common_sh_ctx_ngl.md:
`select_model_gguf` tries `*-PQ2_0.gguf` first, then `*-PTQ1_0.gguf`).

Names are **not** `Q1_0` / `Q2_0` ternary as the brief guessed. They are `PTQ1_0` (ggml type
id not published; fails safely upstream) and `PQ2_0` (ggml type id **142**, group 128). The
older generation's `Q2_0` is ggml type id **42**, group 64, and reads on mainline. Full
three-format history and the gen-1 filenames are in 04_model_formats.md.

⚠ **`Q2_0` for Bonsai 2 does not fail safely on stock llama.cpp.** Upstream knows the type and
the `qwen35` architecture, loads the file with no warning, and **outputs gibberish**. PTQ1_0 and
PQ2_0 have type ids past upstream's `GGML_TYPE_COUNT` and are refused outright.

⚠ **There is no dspark drafter file in the Bonsai 2 27B repo.** The gen-1 repo has two
(`dspark-Q4_1.gguf` 1.946 GB, `dspark-bf16.gguf` 7.292 GB). Whether a Bonsai 2 drafter exists is
**UNKNOWN**; `SPECULATIVE.md` says "newer releases ship ready-to-use drafters", which the file
listing does not support. Do not plan a Bonsai 2 speculative arm on that sentence.

## 2 - The fork, the branch, the commit

- Fork: `https://github.com/PrismML-Eng/llama.cpp`, branch **`prism`**, described in its own README as "developed as `prism-v7`". It tracks current mainline and adds the low-bit formats on top.
- `prism` HEAD when read: **`5d80cff0b8cb9f2bf823cfc4e71e3abb97f290d6`**, 2026-09-17.
- Latest releases: `prism-b10687-5d80cff` (2026-09-17), `prism-b10685-7dffb15`, `prism-b10684-6ac5eb0`, `prism-b10683-d8f26ee` (2026-09-09).
- **Bonsai-demo pins nothing on the source path.** `build_cuda_linux.sh` runs `git clone -b prism` and takes whatever HEAD is. Only `download_binaries.sh` pins a tag, and it pins **`prism-b10683-d8f26ee`**. So "which commit does Bonsai-demo pin" has two answers and the source answer is "none".
- Never build `prism-v6` (stale mid-migration snapshot). Never mix the fork's `ggml-*` libraries with a stock llama.cpp build. A frozen `prism-v5` line exists at release `prism-b9601-68faa14` for the deprecated legacy band only.

Third-party pins that are known to work: `e311ed38f` (build 10660) for PrismML's own GB10
entry; `62061f9` (b9591) for the 16 GB bench repo; `54e8e26` on `pr/q2_0-cuda` for the
kubesimplify Spark run.

## 3 - The CUDA build on sm_121, which is the first real obstacle

⚠⚠ **THE PREBUILT BINARIES DO NOT COVER THE SPARK AT ALL, AND THE FALLBACK IS SILENT.**
MEASURED by reading `download_binaries.sh` (08_build_cuda_linux.md). Its guard, verbatim:

```
# Guard CUDA/ROCm to x64 only (no arm64 builds available)
```

The Spark is aarch64 with CUDA. It falls back to the **Vulkan arm64** or plain **CPU arm64**
archive with a warning. PQ2_0 has **no Vulkan kernels** (04_model_formats.md registry), so
`./setup.sh` on a Spark produces either a binary that cannot read the demo's default band or a
CPU build. **Building from source is mandatory, not an optimisation.**

⚠ **And the build script's default architecture list omits the Spark.** It is
`80;86;89;90;100;120a` on CUDA 13+ and `80;86;89;90;120a` on CUDA 12. The GB10 is compute
capability **12.1**, that is **sm_121**; `120a` is sm_120a. PrismML's own GB10 benchmark entry
states it was "built for CUDA architecture **`121a`**". So:

```
./scripts/build_cuda_linux.sh --archs "121a"
```

Whether a `120a` fat binary runs on sm_121 at all, or at reduced speed via PTX JIT, is
**UNKNOWN** and untested in any source. `121a` is the documented-working value.

The configure line the script emits:

```
cmake -B build-cuda -G Ninja \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_COMPILER="$CUDA_PATH/bin/nvcc" \
    -DCMAKE_CUDA_ARCHITECTURES="121a" \
    -DCMAKE_BUILD_TYPE=Release
```

then `cmake --build build-cuda -j<nproc capped at 16>` for the `all` target. Needs nvcc, cmake,
ninja-build, and patchelf for the `$ORIGIN` RUNPATH step. Output lands in `bin/cuda`, which the
server script searches second, ahead of any downloaded Vulkan binary.

⚠ One build trap MEASURED by a third party (21): "a GPU-less configure caches
`CUDA_DRIVER=NOTFOUND` and later builds keep failing until you wipe CMakeCache". Configure with
the GPU attached.

## 4 - The server command, and the concurrency flag

MEASURED by reading `scripts/start_llama_server.sh` at HEAD (07_server_script.md). What it
execs for Bonsai 2 27B:

```
llama-server -m <model> --host 127.0.0.1 --port 8080 -ngl 99 -fa on -c <RAM-tiered> \
    --temp 1.0 --top-p 0.95 --top-k 20 \
    --jinja \
    [--mmproj <proj>] [--image-max-tokens N] \
    [-md <drafter> --spec-type draft-dspark --spec-draft-n-max 4 -ngld 999 -np 1] \
    [--cache-type-k q4_0 --cache-type-v q4_0] [--kv-mean-center <bias>] \
    --webui-config-file scripts/webui-config.json \
    "$@"
```

⚠ **The script never passes `--parallel` or `-np`, except `-np 1` inside the speculative flag
string.** On a plain run the slot count is whatever the built llama-server defaults to.
`AGENTS.md` claims that default is 4; that value has **not** been read off a binary here, so
treat the effective default as **UNKNOWN until `llama-server --help` is run on the Spark's own
build**. It is one command and it settles the question.

`"$@"` is last, so the sweep needs no fork-specific flag:

```
./scripts/start_llama_server.sh --parallel 8
```

### Is `--parallel` / continuous batching supported on the ternary CUDA path? YES.

Three independent lines of evidence, and this was the brief's sharpest question:

1. **The kernels exist and are used at every batch size.** PrismML-Eng/llama.cpp **#164, MERGED**: the PTQ1_0 MMQ tile path had landed behind `MMQ_PTQ1_0_MAX_BATCH_SIZE = 64`; "the cap is now effectively off (MMQ at every batch, like the other low-bit types)", with `GGML_CUDA_PTQ1_0_MMQ_MAX_BATCH=<n>` left as an A/B knob. HIP stays excluded. In the batch band a concurrent decode lives in, MMQ **beats** the dequantize-plus-cuBLAS fallback: 9B ubatch 64 goes 1,332 to 3,392 t/s, 128 goes 1,883 to 4,232, 256 goes 2,956 to 4,761; at 512 the fallback is 5,206 against 4,739. "Decode is unchanged either way."
2. **The one official statement**, `AGENTS.md`: `--parallel N`, "Server slots (default 4)", "More slots = more concurrent users, **same total context pool**".
3. **Somebody has run it.** 8 concurrent streams "without errors" on a 16 GB card (21, and Bonsai-demo #101).

⚠ The open issue **#140** is about the **upstream `TQ1_0`** type, which has no MMQ tile loader
and dequantizes to F16 on every batched matmul. That is a different type from the fork's
`PQ2_0`/`PTQ1_0`. Do not read #140 as a statement about the bands we will serve.

## 5 - The known concurrency limitation, named and quantified

There is **no mutex and no single-stream lock in the llama.cpp fork's server path.** The
limitation is **latency under a shared context pool**, and it is MEASURED.

MEASURED, RTX 5060 Ti 16 GB, CUDA, Linux (Bonsai-demo #101 and repo 21):

| streams | aggregate decode | per stream | TTFT p95, cold about 3.4k-token prefill |
|---|---|---|---|
| 1 | 44 t/s | 44 t/s | - |
| 4 | UNKNOWN | UNKNOWN | **12.7 s** |
| 8 | **about 91 t/s** | about 11.4 t/s | **25.1 s** |

The reporter's own summary: "at 8 streams the fork's slot batching reaches about 91 t/s
aggregate but TTFT p95 degrades to about 25 s, while the smaller model on vLLM holds TTFT p95
about 0.3 s flat at the same concurrency. **The serving stack, not the model, decides who wins
the multi-stream seat.**" And: "Multi-user latency-bound serving: wrong tool today."

So batching gives **2.07x aggregate at 8 streams** on that card while per-stream falls to 26% of
solo. Throughput scales; first-token latency collapses. The context pool is **shared and
divided**, confirmed independently by that repo's own benchmark advice: "For AIME use a single
slot and ctx >= 64k; for MMLU/LCB **2 slots @ 32k** works."

⚠ **Speculation and concurrency are mutually exclusive today.** Not advice, mechanism: the start
script hard-codes `-np 1` into the DSpark flag string, `SPECULATIVE.md` states "**Single slot**
(`-np 1`): one request at a time", and #101 says "the fork also cannot combine the DSpark
drafter with more than one slot". DSpark also disables cross-request prompt-cache reuse.

Two other serialization findings that must **not** be cited as evidence about this engine:

- `herlangga72/ternary-bonsai-inference` is a **separate pure-Rust engine** whose README says "Requests are serialized. The engine is single-stream: one generation at a time behind a mutex". True of that engine, irrelevant to llama.cpp (18).
- `PrismML-Eng/image-studio`'s `asyncio.Lock` serializes **backend switching in an image-generation FastAPI app**, and its GPU arm "holds no in-process state" (19). SOURCES.md's one-line gloss overstates it.

Related open defects: Bonsai-demo **#147** repeated 14K prompts get no prefix-cache reuse on the
27B multimodal path; **#161** asks for the hybrid prompt-cache and checkpoint behaviour to be
documented at all; llama.cpp **#77** an infinite prompt-processing loop past 43k tokens.

## 6 - The KV budget, which is the first thing a sweep hits

MEASURED by reading `scripts/common.sh` (09). The default context is RAM-tiered and the Spark
lands on the top tier:

| RAM | default `-c` |
|---|---|
| up to 11 GB | 8192 |
| up to 23 GB | 16384 |
| up to 35 GB | 32768 |
| up to 71 GB | 65536 |
| above 71 GB, 27B | **131072** |

The Spark reports **121 GiB**, so the script picks **`-c 131072`**. CLAIMED per-token FP16 KV on
the 27B is **64 KiB** (against about 140 KiB on the full-attention 8B), so 131072 tokens is
about **8 GiB** of KV. Slots divide that pool, so `--parallel 8 -c 131072` is 16,384 tokens per
slot. **The arithmetic, not the kernels, is the first constraint to design around.**

The script deliberately never emits `-c 0`; its own comment: that would take the model's full
262144 training context and OOM constrained machines.

### KV4

`BONSAI_KV4=1` passes `--cache-type-k q4_0 --cache-type-v q4_0`, requires flash attention
(already on via `-fa on`), and CLAIMS a **3.5x** KV cut: "from 64 KiB per token to about 18 KiB
per token on the 27B, so a 100K-token context needs about **1.8 GiB instead of 6.3 GiB**".
Marked experimental, llama.cpp backend only, "a memory tool, not a speed tool: decode is
slightly slower than the default FP16 KV cache".

Quality booster: `./scripts/make_kv_bias.sh` runs `llama-kv-mean-center` over a calibration
corpus (a tiny built-in synthetic one is enough; a user corpus can be passed) and writes
`<Model>-kv-bias.gguf`; the server picks it up automatically and applies it at "zero decode-time
cost (one subtract when the cache is written)". It is model-specific and must be rebuilt per
model. The server exports `LLAMA_ATTN_ROT_DISABLE=1` when a bias is present because the bias is
calibrated with K-rotation off, and the loader refuses a mismatched bias by design.

⚠⚠ **KV4 is the obvious way to buy slots and it has a measured prefill cliff on CUDA.**
Bonsai-demo **#145**, MEASURED on an RTX 4090, driver 580.173.02, CUDA 12.6, arch 89,
`Ternary-Bonsai-27B-Q2_g64.gguf`, one slot:

| KV K / V | pp512 | pp2048 | pp4096 | tg128 |
|---|---|---|---|---|
| F16 / F16 | 3427.7 | 3495.6 | 3459.5 | 91.4 |
| Q8_0 / Q8_0 | 3104.6 | 3341.1 | 3320.5 | 85.4 |
| Q4_0 / Q8_0 | 921.8 | 279.4 | **71.8** | not measured |
| Q8_0 / Q4_0 | 647.4 | 147.4 | aborted | not measured |

At pp4096 that is about **46x slower** than Q8/Q8. On a real server at `--ctx-size 262144
--parallel 1`, a 13,932-token prompt took **285,435.7 ms = 48.81 tok/s** with Q4-K/Q8-V, against
**4,270.7 ms = 3,282.83 tok/s** with Q8/Q8: five minutes versus under six seconds. Physical batch
512 was the best real-server setting of 512/1024/2048. A fix is in flight upstream
(ggml-org/llama.cpp#27140) and PrismML said they will pick it up once merged.
Whether the **fork's** own Q4-plus-mean-centering path carries this defect is the exact question
the issue asks and it is **unanswered: UNKNOWN**. **Measure Q8/Q8 as well as Q4/Q4.**

Also: Bonsai-demo **#139** reports `BONSAI_KV4` plus `BONSAI_SPECULATIVE` on the 27B ternary
failing with a KV cache type error.

## 7 - Published tokens/s on the DGX Spark GB10

### Ternary-Bonsai-27B, generation 1. There is NO Bonsai 2 27B number on any Spark, anywhere.

MEASURED, PrismML's own merged community-benchmarks entry (12). Driver **580.173.02**, CUDA
toolkit **13.0.88**, DGX OS Ubuntu 24.04.4 aarch64, 20 CPUs (10 Cortex-X925 / 10 Cortex-A725),
**121 GiB** unified LPDDR5X, GB10 compute capability **12.1**, fork commit **`e311ed38f`** (build
10660) built for **`121a`**, all layers offloaded, `-fa on`, **one server slot**, `-r 5`:

| file | size | test | t/s |
|---|---|---|---|
| `Ternary-Bonsai-27B-PQ2_0.gguf` | 6.66 GiB | pp512 | **1005.19 +/- 9.76** |
| `Ternary-Bonsai-27B-PQ2_0.gguf` | 6.66 GiB | tg128 | **29.16 +/- 0.04** |
| `Ternary-Bonsai-27B-Q2_g64.gguf` | 7.05 GiB | pp512 | 1008.80 +/- 13.11 |
| `Ternary-Bonsai-27B-Q2_g64.gguf` | 7.05 GiB | tg128 | 28.01 +/- 0.03 |

MEASURED independently, kubesimplify (13), DGX Spark, driver **580.159.03**, CUDA 13.0, sm_121,
5 repetitions, models about 7.17 GB ternary and about 3.79 GB 1-bit:

| Model | Engine | pp512 | tg128 |
|---|---|---|---|
| Q1_0 (1-bit) | Mainline llama.cpp @ `12127de` | 922.9 +/- 11.8 | 40.9 +/- 0.2 |
| Q1_0 (1-bit) | PrismML `prism` @ `62061f9` | 991.6 +/- 13.7 | **42.8 +/- 0.1** |
| Ternary Q2_g64 | PrismML `pr/q2_0-cuda` @ `54e8e26` | 900.7 +/- 19.1 | 26.6 +/- 0.5 |
| Ternary Q2_0 | PrismML `prism` @ `62061f9` | 937.7 +/- 11.0 | **28.5 +/- 0.1** |

**Two independent parties agree on the Spark's ternary 27B single-stream decode: 28.0 to 29.2
t/s.** The 1-bit band is 40.9 to 45.4 t/s. PrismML's league table row for the Spark is ternary
1,005 pp512 / 29.2 tg128 and 1-bit 1,024 / 45.4; the 8B 1-bit row is 3,978 / 159.

### The bandwidth sanity check

The Rust engine's README independently states the model "streams about 7 GB per token", which
matches the 7.165 GB PQ2_0 file. 273 GB/s divided by about 7 GB per token is about **39 tok/s**
as the roofline. Measured is 28.0 to 29.2, so the fork's CUDA decode runs at roughly **75%** of
the published bandwidth ceiling at batch 1. That is the headroom a batched sweep is aiming at,
and it is a ratio, not a promise.

⚠ Against it, CLAIMED by PrismML on the model card: on "H100, A100, and the Blackwell cards ...
batch-1 decode is not bandwidth-starved" but limited by "instruction throughput and launch
overhead". The GB10 is Blackwell-class at 273 GB/s rather than 1.8 TB/s, so that may not
transfer. It is a reason to profile rather than to assume the roofline.

### The DSpark conflict on the Spark, unresolved

| source | without | with | change |
|---|---|---|---|
| PrismML community-benchmarks GB10 entry, temp 0, seed 42, code prompt, `-np 1 -c 16384` | 28.60 t/s | **70.01 t/s** | **2.45x** |
| PrismML GB10 entry, blended 16 prompts | 28.71 | 67.47 | 2.35x |
| kubesimplify, same class of machine | 28.2 t/s | **17.6 t/s** | **-37%** |

Both are DGX Sparks. They disagree by more than 4x in ratio terms. The mechanism is documented
in Bonsai-demo **#105**: DSpark acceptance decays monotonically with prompt depth and with
temperature, and goes net-negative. MEASURED on an A5000 at `-c 32768`, temp 0.7: 1.11x at depth
0 (accept 0.449), 0.89x at 8k (0.335), 0.67x at 16k (0.234), **0.37x at 24k (accept 0.009)**. And
DSpark **silently stops engaging above 32k context** because the drafter stages the whole target
context into one block and its own `n_ctx_train` is 4096; the server drops to the
autoregressive floor with no error to the client.

⇒ **Do not cite a single DSpark speedup for the Spark.** The published win was taken at
temperature 0 and shallow depth, which is where the drafter is strongest. And it is single-slot
by construction, so it is not on the concurrency path at all.

## 8 - Concurrency shapes to measure against

No Bonsai concurrency sweep exists on a DGX Spark. Two things bracket it.

MEASURED, kubesimplify, RTX PRO 6000 96 GB, sm_120, driver 610.43.02, CUDA 13.0, via
**`llama-batched-bench`** (13). The author's caveat: "These are synthetic batched-benchmark
results, not an end-to-end production load test."

| sequences | Ternary Q2_0 aggregate | per sequence | Q1_0 aggregate | per sequence |
|---|---|---|---|---|
| 1 | 119.3 | 119.3 | 140.2 | 140.2 |
| 10 | 411.9 | 41.2 | 471.5 | 47.1 |
| 32 | 671.8 | 21.0 | 786.1 | 24.6 |

1 to 32 gives **5.63x** aggregate on the ternary band while per-sequence falls to 17.6% of solo.
The knee is passed by 10 sequences: 3.45x by 10, only 1.63x more from 10 to 32.

MEASURED, kubesimplify, **vLLM on a DGX Spark, 2026-05-21**, other models (14). This is the
shape a good serving stack gets out of this box:

| concurrency | Gemma 4 31B IT (NVFP4), aggregate | per request |
|---|---|---|
| 1 | 6.15 | 6.15 |
| 4 | 24.41 | 6.10 |
| 8 | 47.13 | 5.89 |
| 16 | 92.10 | 5.76 |
| 32 | 166.36 | 5.20 |

1 to 32 gives **27.0x** aggregate while per-request holds at **84.6%** of solo. Qwen2.5-3B BF16
went 26.14 t/s at c=1 to 1,462.30 t/s at c=64, a 55.9x rise. The author's statement of the
limit: "memory, not compute, is what caps your concurrency on Spark", with KV competing for the
shared 128 GB pool. The Spark's bandwidth is stated as **about 273 GB/s**, "approximately 12x
slower than an H100's 3.35 TB/s".

⚠ That vLLM sweep has **no** published serve command, client command, version, driver or TTFT.
It is a shape, not a reproducible configuration. **UNKNOWN.**

For completeness, the only DGX Spark aggregate-versus-per-request claim at a named concurrency
(16, NVIDIA forums, vendor CLAIMED, unverified in thread, 2x Spark cluster, "C4" undefined but
arithmetically 4): DeepSeek V4 Flash unquantized 55.99 decode t/s aggregate / 14.00 per request;
Gemma 4 26B 63.75 / 15.94; Nemotron 3 Nano Omni 30B NVFP4 90.83 / 22.71. A thread reply claims
DSpark reaches "up to 70 tok/s C1" on DeepSeek V4 Flash on the same cluster class.

## 9 - vLLM: settled, and it is a finding rather than an option

**There is no public ternary vLLM kernel.** PrismML's own answer in Bonsai-demo **#101**
(khosravipasha), verbatim: "For vllm you can unpack the weights for fp16 and use that in vllm to
have consistent setup across eval, we do the same for the models we evaled. For accuracy of
serving vs unpacked we use KL logits to validate the kernels end to end which we usually get
near 0 KL for unpacked vs packed runtimes. That being said we are interested longer term in
supporting vllm/sglang."

So the model card's "Evaluated with EvalScope + vLLM on NVIDIA H100" ran on **fp16-unpacked**
weights, which is what the `prism-ml/*-unpacked` Hugging Face repos are. An fp16 27B is about
54 GB: it fits the Spark's 121 GiB but forfeits the entire point of the exercise. The issue is
CLOSED with no date and no commitment.

⇒ The prism README's line "vLLM has no ternary path; that is a finding, not an option" is
**confirmed from the vendor's own mouth**, and it should be cited to #101.

## 10 - What Hermes (Nous) means here

**Hermes is a client, not a serving component** (20). It is `NousResearch/hermes-agent`, an
open-source terminal agent with persistent memory and learned skills, pointed at any
OpenAI-compatible `/v1/chat/completions`. Setup is `hermes setup model`, choose "Custom
OpenAI-compatible endpoint", base URL `http://localhost:8080/v1`, any non-empty API key; config
lands in `~/.hermes/config.yaml`. The docs recommend the ternary 27B "which is trained for
native tool calling".

It matters as the **workload shape**, and it is the adversarial one: agent turns are multi-turn
(so they need the cross-request prompt cache that DSpark disables and that #147 reports failing
on repeated 14K prompts), carry long contexts (so they sit in the depth band where DSpark goes
net-negative), and N agents is N streams (which is where TTFT p95 reaches 25 s). **Serving
concurrent Hermes-style agents is precisely the deployment the public record says this stack
loses today.**

⚠ **Two different things share the name Hermes in this project.** Here it is the terminal agent
pointed at a local endpoint. Elsewhere in dwarfstar it is the Nous inference portal used as the
ds4-flash judge door. Never conflate them in a prism document.

## 11 - Doc drift worth knowing before trusting any page

`docs.prismml.com`'s "Run the Server" table is stale against the repo at HEAD (02). It lists
`-c 0`, `--temp 0.5 --top-p 0.85`, and warns "the scripts bind to 0.0.0.0". At HEAD the script
binds **127.0.0.1**, never emits `-c 0` (it computes the RAM tier instead), and uses
`--temp 1.0 --top-p 0.95 --top-k 20` for the bonsai2 family. The docs page even carries an
unresolved author note: "TODO: confirm benchmark sampling settings match the demo defaults".
**Read the scripts, not the docs site.** The docs site also never mentions `BONSAI_KV4` or
`--parallel` at all.

`alphasignal` (15) restates PrismML's figures second-hand and four of them disagree with the
model card: 1.76 versus 1.72 bits/weight, 83.9/85.4 versus 84.78/86.32 benchmark average, 143
versus 129.9 t/s on RTX 5090, 44 versus 47.0 t/s on M5 Max. **Cite the model card, never
alphasignal, for any number.**

---

## 12 - The open questions this harvest could not close

Each is UNKNOWN, with the one command or measurement that would settle it.

1. **The effective `--parallel` default on the Spark's own build.** `AGENTS.md` claims 4; nothing else says. Run `bin/cuda/llama-server --help | grep -A2 parallel`.
2. **Whether a `120a` build runs on sm_121**, and at what cost. Only `121a` is documented working. Build both and compare tg128.
3. **Any Bonsai 2 27B number on a GB10, at any concurrency.** None exists publicly. Every Spark figure in this harvest is generation 1, Ternary-Bonsai-27B. The prism bring-up will produce the first.
4. **Whether a Bonsai 2 dspark drafter exists.** No file in the repo; the doc implies one. Re-check the repo listing.
5. **Whether the fork's Q4-plus-mean-centering KV path carries #145's prefill cliff.** The issue asks; nobody answered. Sweep Q8/Q8 against Q4/Q4 on the fork's own build.
6. **What actually limits N on the ternary path at batch greater than 1 on this box**: KV budget, prefill contention, or the MMQ kernel's occupancy. #164 shows MMQ is used at every batch and is faster than the fallback in the ubatch 64 to 256 band, which argues against the kernel being the limit, but no profile on GB10 exists. `GGML_CUDA_PTQ1_0_MMQ_MAX_BATCH` is a ready-made A/B knob for the PTQ1_0 band.
7. **Why DSpark on the Spark is +2.45x in one source and -37% in another.** Depth and temperature explain the mechanism (#105); which configuration each party actually ran is not fully stated.
8. **TTFT under concurrency on the Spark.** The only TTFT numbers anywhere are 12.7 s at 4 streams and 25.1 s at 8 on a 16 GB consumer card with a 32k shared pool. The Spark has 121 GiB and can afford a far larger pool, so its TTFT curve is genuinely unknown and may be much better.

## 13 - The bring-up, as the sources dictate it

Not a plan, just what these documents force. Step 1 of the prism README, with its landmines
removed:

```
git clone https://github.com/PrismML-Eng/Bonsai-demo.git && cd Bonsai-demo
# do NOT rely on setup.sh for the binary: no arm64 CUDA prebuild exists
./scripts/build_cuda_linux.sh --archs "121a"        # needs nvcc, cmake, ninja, patchelf
./scripts/download_models.sh                        # fetches PQ2_0 (7.21 GB) + mmproj by default
./scripts/start_llama_server.sh                     # -c 131072 -ngl 99 -fa on, one host, port 8080
# and for step 2:
./scripts/start_llama_server.sh --parallel N        # "$@" passes it straight through
```

Every number this produces carries load, GPU, memory, driver, context and checkpoint stamps per
the prism README's own laws, and the checkpoint stamp must say **Bonsai 2 27B PQ2_0** because
every comparable public Spark figure is generation 1 and a comparison across that seam is
confounded.
