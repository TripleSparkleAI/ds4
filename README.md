<p align="center">
  <img src="logo.svg" alt="DwarfStar logo" width="220">
</p>

**DwarfStar** aims to be the best way to run a few excellent large
language models on consumer hardware (that is, hardware that people
can actually own). To reach this goal, we are building
a small native inference engine optimized first for
**DeepSeek V4 Flash** (including the experimental vision model),
**DeepSeek V4.1 Flash** (Metal, and text inference on CUDA),
and additionally **GLM 5.2 and 5.3**, **GLM 5.3 Flash** and
**DeepSeek V4 PRO**, and **Qwen3.8 Flash Next** (Metal and CUDA). The code is self-contained and
deliberately narrow, not a general GGUF runner: you need to use the
GGUF files the project produces, that are part of the project
itself.

We test things in integration: model loading, prompt rendering,
tool calls, KV state, the HTTP server, and the coding agent are built and tested together.
The repository also includes tools and data for GGUF, imatrix, quality, and speed.

## Supported hardware

* **Metal**, the primary target, on Macs with 96 GB or more. Smaller machines
  can use SSD streaming. SSD streaming is also needed in order to run very
  large models such as full GLM 5.x (not Flash) on 128GB systems.
* **NVIDIA CUDA**, the DGX Spark is our main gaol. DwarfStar also supports multi-GPU systems that are not supported by other backends, for instance it can run DeepSeek v4 Flash on Ada Lovelace cards.
* **ROCm** on Strix Halo systems such as the Framework Desktop.

This project would not exist without **llama.cpp and GGML**, make sure to read
the acknowledgements section, a big thank you to Georgi Gerganov and all the
other contributors.

**Model support is intentionally opportunistic**. The project follows the best open
weights for useful local machine sizes, especially 128 GB laptops and 256/512 GB
workstations. A model may be removed when a better replacement arrives.

# So, what can I do with this software?

* You can run a very capable models in your consumer hardware, a MacBook, a DGX Spark, or a Strix Halo for example. Even if you have not enough RAM, with SSD streaming, you can run it at a decent speed.
* You can use multiple CUDA cards as a multi-user LLM server. Ada Lovelace, including L40S, is supported: newer models can run here even when their other inference implementations require newer GPUs. Our eight-L40S Flash setup has reached about 126 t/s aggregate generation with 16 sessions.
* Using two 128 GB Macs connected with RDMA, you can run 4-bit DeepSeek Flash or GLM 5.3 Flash with tensor parallelism. Larger GLM 5.2 quants need larger machines, such as Mac Studios.
* You can also use pipeline paralellism to glue together multiple systems to sum their RAM and run larger models.

## Motivations

* Capable open-weight models now fit on high-end personal machines.
* DeepSeek V4 Flash and PRO, GLM 5.2, tolerate aggressive routed-expert quantization.
* Compressed KV caches and fast local SSDs make long contexts practical.
* The idea of an inference system specialized for a few models.

# AI full disclosure

* This software is developed with **strong assistance from AI coding agents** and with humans leading the ideas, testing, and debugging. We say this openly because it shaped how the project was built. If you are not happy with AI-developed code, this software is not for you. The acknowledgement below is equally important: this would not exist without `llama.cpp` and GGML, largely written by hand.

## Acknowledgements to llama.cpp and GGML

`ds4.c` does not link against GGML, but it **exists thanks to the path opened by the
llama.cpp project and the kernels, quantization formats, GGUF ecosystem, and hard-won
engineering knowledge developed there**.
We are thankful and indebted to [`llama.cpp`](https://github.com/ggml-org/llama.cpp)
and its contributors. Their implementation, kernels, tests, and design choices were
an essential reference while building this DeepSeek V4 specific inference path.
Some source-level pieces are retained or adapted here under the MIT license: GGUF
quant layouts and tables, CPU quant/dot logic, and certain kernels. For this
reason, and because we are genuinely grateful, we keep the GGML authors copyright
notice in our `LICENSE` file.

## Status

The software is currently very fast changing. Consider it beta quality.
Before each release, a big QA run is executed, however instabilities
and regressions are definitely possible.

# How to use this project?

I (Salvatore) believe that the way projects should be shipped and used changed because of AI. The main differences today are:

1. With AI, users can modify the software in significant ways with low efforts, costs, and even lacking deep domain knowledge about the task they want to accomplish. For instance, a DwarfStar user with a specific hardware setup can ask a coding agent to improve the inference speed of this software for the specific hardware setup, asking the model to reach the maximum prefill and generation speed without impacting correctness, and also asking to do a deep QA pass.
2. Similiarly, because of "1", software may be shipped in a different way than before. It must be more a working template for the biggest use cases, without trying to cover every possible setup. If DwarfStar showcases a few good implementations of tensor parallel execution, the code will work as a rail for implementing the same feature in specific conditions, for a new model, and so forth.

So, while this project attempts to be usable for the featured models and the most common hardware setups, I ask you, if you have access to coding agents, to consider using coding agents as an interface to discover the project, make modifications, create personalized setups. This way you can likely do more than what we ship, and certain things that are not documented or implemented, and that you require, are potentially very easy to achieve.

## Start Here

```sh
git clone https://github.com/antirez/ds4.git
cd ds4
```

Choose your build. The platform guides cover prerequisites, memory sizing,
and hardware-specific setups:

| Platform guide | Build |
| --- | --- |
| [Metal on Apple Silicon](docs/METAL.md) | `make` |
| [DGX Spark](docs/DGX_SPARK.md) | `make cuda-spark` |
| [Strix Halo / Framework Desktop](docs/STRIX_HALO.md) | `make strix-halo` |
| [One or more CUDA cards, including Ada/L40S](docs/CUDA_MULTI_GPU.md) | `make cuda-generic` |

For a first run on a 96 or 128 GB machine, download DeepSeek V4 Flash Q2:

```sh
./download_model.sh ds4f-q2
```

Downloads go in `gguf/`. Repeat the command to resume an interrupted download.
Leave memory for the context and runtime buffers as well as the model.
See [other models](docs/MODELS.md) or use [SSD streaming](docs/SSD_STREAMING.md)
on a smaller Mac.

## Everyday Use

Once built and with a model downloaded:

```sh
./ds4
./ds4 -p "Explain Redis streams in one paragraph."
./ds4-agent
./ds4-server --ctx 32768
```

The default model is `ds4flash.gguf`, a link updated by main-model downloads.
Pass `-m FILE` to choose explicitly. Commands normally run from the repository
root; use `--chdir /path/to/ds4` when launching elsewhere.

The server listens at `http://127.0.0.1:8000` by default; see [serving](docs/SERVER.md)
for API access and multiple sessions.

The interactive CLI keeps a multi-turn conversation. Use `/help`, `/read FILE`,
`/ctx N`, and `/quit`. Ctrl+C interrupts generation and returns to the prompt.
Run each binary with `--help` for its full options.

### Native coding agent

`ds4-agent` runs inference directly, without a separate HTTP server. It keeps
the token history and live model state together, shows prefill progress, and
uses the model's native tool format. DeepSeek and GLM have their own templates.

Use `/hints on` for occasional, brief explanations of the programming choices
behind the work, and `/hints off` to stop them. Changes take effect at the next
conversation boundary without rebuilding the cached context. New and resumed
sessions start with hints off.

Sessions are stored in `~/.ds4/kvcache`:

| Command | Action |
| --- | --- |
| `/save` | Save the current session |
| `/list` | List saved sessions |
| `/switch <sha>` | Resume a session |
| `/del <sha>` | Delete a saved session |
| `/strip <sha>` | Keep text and title, removing the large KV payload |

Compatible local KV snapshots avoid rebuilding the prompt. Stripped sessions
and network TP restores require prefill. Sessions containing images cannot yet
be saved. Saved conversations and traces may contain private information.

For Pi, OpenCode, Codex CLI, or Claude Code, use `ds4-server` instead and follow
the [client setup guide](docs/CLIENTS.md).

### Models, images, and speculation

[Models and vision](docs/MODELS.md) lists the supported downloads and memory
requirements. DeepSeek Vision Experimental uses a different checkpoint from
Flash 0731; GLM 5.3 Flash and Qwen3.8 Flash Next add vision to the same text
model through a separate encoder.

DeepSeek V4.1 Flash text and vision run on Metal; text also runs on a DGX Spark.
Q2 runs with SSD streaming on one 128 GB Mac or Spark, or resident across two
Macs or two Sparks using RDMA. Q4 needs SSD streaming or a 512 GB Mac.
Engram tables remain on disk in every mode, so use a fast
local SSD. See the [model guide](docs/MODELS.md#deepseek-v41-flash) for downloads
and setup.

With the matching encoder passed as `--vision FILE`, use `/read image.png`
in the CLI or `view_image` in the native agent.

Qwen3.8's smaller Q2 release has **41.73 GiB** of main/MTP weights,
with imatrix IQ2_XXS gate/up experts and padded Q2_K down projections.
It is the starting option for 64 GB Macs.
The GGUF also contains 95.37 GiB of original BF16 n-grams, read directly
from disk rather than loaded into RAM. Keep it on a fast SSD. Start with 8K context:

```sh
./download_model.sh qwen38-q2
./ds4 --ctx 8192 --prefill-chunk 1024
```

The download fetches one 137.10 GiB file and updates `ds4flash.gguf`.
Add `--mtp` for speculative decoding. The larger
`qwen38-q4k` target is also available. Download the optional vision encoder
with `./download_model.sh qwen38-vision` and pass it with `--vision`.
See [Qwen setup](docs/QWEN38_FLASH_NEXT.md) for details.

Speculative decoding is opt-in. GLM and Qwen use `--mtp`; V4 Flash DSpark needs a matching
support GGUF. It can improve generation, but not every workload benefits.
Read [speculative decoding](docs/SPECULATIVE_DECODING.md) for setup and the
difference between default opportunistic sampling and `--mtp-exact-sampling`.

### Output and power

Thinking is enabled by default. Use `--nothink` or `/nothink` for direct
answers, and `--think` or `/think` to enable it again.
For V4.1, `ds4` and `ds4-agent` also accept
`--think-level 25` or `/think 25`: 1 to 100 sets the reasoning effort, and
0 disables thinking. `--think` selects 75, `--think-max` selects 100.
Changing the level in a conversation rebuilds its cached prefix.
The normal sampling defaults are temperature 1, top-p 1, and min-p 0.05;
`--temp 0` selects greedy output.

For DeepSeek V4, `--power N` trades throughput for lower sustained GPU load.
The default is 100. V4.1 and GLM currently require `--power 100`.

DeepSeek V4 Flash and GLM 5.3 Flash also support directional steering. Load a
vector with `--dir-steering-file FILE`; `/steer F` adjusts its scale for
subsequent tokens in a local CLI or agent session, without rebuilding the
existing KV cache. See [steering documentation](dir-steering/README.md).

`--prefix-file FILE` preloads complete `USER:` / `ASSISTANT:` pairs before
the live conversation. A turn marker must start a line, roles must alternate,
and the last turn must be `ASSISTANT:`.

## Capability Evaluation

`ds4-eval` runs embedded capability regression tests against a real GGUF.
These are DwarfStar integration checks, not official leaderboard scores.

```sh
./ds4-eval -m ds4flash.gguf --trace /tmp/ds4-eval.txt
./ds4-eval -m ds4flash.gguf --suite hard-smoke
./ds4-eval -m ds4flash.gguf --suite hard --retry-incomplete
```

The default suite is `core`; `--suite all` runs core and hard cases.
`--list-cases` lists tests without loading a model. `--plain` selects
non-interactive output, and `--regrade-trace FILE` scores an existing trace
without generating again. Sources and licenses are in [EVAL_DATA.md](EVAL_DATA.md).
For inference correctness and release checks, read [testing](docs/TESTING.md).

## Speed

This recorded DeepSeek V4 Flash Q2 sweep uses an M5 Max with 128 GB RAM,
2048-token continued-prefill intervals, and 128 greedy generation tokens per
frontier. It is a baseline, not a fresh benchmark of every commit.

![M5 Max Flash Q2 throughput](speed-bench/m5_max_ts.svg)

See [performance and benchmarking](docs/PERFORMANCE.md) for the full numbers,
DGX Spark results, comparison conditions, and benchmark commands.

## Detailed Guides

- [Models and vision](docs/MODELS.md): Flash, PRO, GLM, Qwen, and matching encoders.
- [Qwen3.8 Flash Next](docs/QWEN38_FLASH_NEXT.md): model setup, MTP, vision, and validation.
- [SSD streaming](docs/SSD_STREAMING.md): run larger than RAM and size the cache.
- [Inference across machines](docs/DISTRIBUTED.md): two-Mac TP/RDMA and layer pipelines.
- [Speculative decoding](docs/SPECULATIVE_DECODING.md): DSpark, GLM and Qwen MTP, and sampling.
- [Serving](docs/SERVER.md): APIs, images, batching, and disk KV caches.
- [Coding agent clients](docs/CLIENTS.md): Pi, OpenCode, Codex CLI, and Claude Code.
- [Performance](docs/PERFORMANCE.md): reproducible measurements and recorded baselines.
- [Testing and development](docs/TESTING.md): regression tests, debugging, and model-building tools.

Read [CONTRIBUTING.md](CONTRIBUTING.md) before sending a pull request.

## Logo

The DwarfStar logo was designed by hand by Salvatore Sanfilippo, made more
graphical with AI, and manually reworked by Ben Gnomino, whose human touch made
it rock.

---

**✦ ✧ ✦ ✧ ✦ ✧ ✦ ✧ ✦ ✧ ✦ ✧ ✦   T R I P L E S P A R K L E   ✦ ✧ ✦ ✧ ✦ ✧ ✦ ✧ ✦ ✧ ✦ ✧ ✦**

**✦ above: the upstream README, unchanged · below: this branch's card and numbers**

Rebased onto triple-tip-2026-09-16 (12997e9c8) on 2026-09-17; tests make test 42 pass lines, 5 pre-existing failures (the Qwen3.8 and GLM 5.3 model-absent skips, counted as failures by ds4_test and identical on every branch).

```
  ┌──────────────────────────────────────────────────────────────────────
  │
  │  BRANCH     triple-iq2-lut-fix                                   TOOLING
  │
  │  WHAT       a source check, not a lever: does our tree carry upstream's
  │             IQ2 dequant-LUT defect, where the codebook was staged in
  │             shared memory only for n_embd <= 4096 and then read below the
  │             guard unconditionally
  │
  │  LATEST     never in a sealed round, and it cannot enter one. The arm lists
  │             of 2026-09-17-ROUND3-, -R4- and -R5-PREREGISTERED-RULE.txt name
  │             this branch zero times. Its own finding is a source read taken
  │             2026-09-16 and re-counted on this rebased tree 2026-09-17
  │
  │  GEN        not measured - no engine code, nothing for the bench to run
  │  PREFILL    not measured - no engine code, nothing for the bench to run
  │
  │  VERDICT    TOOLING, NOT A LEVER - our tree is NOT AFFECTED and no patch is
  │             owed. 5 of 5 IQ2 kernels stage the codebook unconditionally at
  │             HEAD, so there is nothing to fix and nothing to time
  │
  │  SWITCH     NONE - no code changed, no knob added
  │  OUTPUT     not re-run - the branch changes no executable byte
  │
  └──────────────────────────────────────────────────────────────────────
```

**Why this branch is not in the measurement pass.** It changes one file, `README.md`, and no
other. There is no kernel, no switch, no argv and no build difference, so a generation or prefill
row would describe a change that does not exist, and a round that included it would be timing the
tip twice under two names. It is in no round's arm list for exactly that reason.

## The defect

- `WIKI/research/neutronstar/NEUTRONSTAR_02_expert-streaming.md` section 3: on the CUDA batch path
  the MoE expert-tile kernels loaded the IQ2 dequantization tables into shared memory only under
  `if (xq_blocks <= 16u)`, where `xq_blocks = expert_in_dim / CUDA_QK_K`, so 16 blocks is exactly
  n_embd 4096.
- They then read `s_iq2_grid` and `s_iq2_signs` below that guard unconditionally.
- For n_embd above 4096 the dot products ran on uninitialized shared memory: fluent garbage at
  full speed, no crash and no warning.
- Upstream fixed it as `1d8a26c` (`antirez/ds4#513`) and asked for our twin to be checked. This
  branch is that check, and nothing else.
- ⚠ The `WIKI/` path above is not in this branch's tree. These branches are rooted on the upstream
  tip, which carries no `WIKI/`, so that citation resolves in the dwarfstar working branch and not
  here.

## What the check found, re-counted on this tree 2026-09-17

- The defect was in our lineage and was removed by `a04f46fa42` ("DeepSeek v4.1 Flash support for
  CUDA", 2026-09-13), whose diff lifts the two staging loops out of the guard in all five kernels,
  raises the activation stage to 32 blocks, and carries the comment "Wide inputs bypass the shared
  activation cache, not the lookup tables."
- At HEAD of `ds4_cuda.cu`: `grep -c '__shared__ uint64_t s_iq2_grid'` is **5**,
  `grep -c 's_iq2_grid\[i\] = cuda_iq2xxs_grid'` is **5**, and `grep -c 's_iq2_grid'` is **20**, so
  5 declarations plus 5 loads leaves **10 consumers**.
- All five kernels, by name and by line, with the load above the `__syncthreads()` and every
  consumer below it: `moe_gate_up_mid_decode_lut_qwarp32_kernel` (declared 20611, staged 20621),
  `..._decode_lut_owned_qwarp32_kernel` (20679, 20687),
  `moe_gate_up_mid_expert_tile8_row32_kernel` (21194, 21217), `..._row2048_kernel` (21284, 21307),
  `..._rowspan_kernel` (21378, 21401).
- The three expert-tile kernels are the multi-token batch path, dispatched for `n_tokens > 1` via
  `use_expert_tiles`. All three stage unconditionally.
- `use_decode_lut_gate` carries no `xq_blocks` term: at 25010 it reads
  `(n_tokens == 1u || small_exact_batch) && getenv("DS4_CUDA_MOE_NO_DECODE_LUT_GATE") == NULL`.
- A tree-wide search for the two table names returns only `ds4_cuda.cu` and
  `ds4_iq2_tables_cuda.inc`, so there is no second copy of the pattern to audit.

**⚠ Corrected here.** The previous card said "the one remaining `if (xq_blocks <= 32u)` guards the
activation copy `sxq` only". Counted at HEAD there are **two** `xq_blocks <= 32u` guards (20613,
20681) and **sixteen** `xq_blocks <= 16u` guards, plus one `xq_blocks <= 16u` term in the Q4 MMA
dispatch at 25235 which is a different path. The verdict does not move: every one of those guards
encloses an `sxq` activation copy and not a table load, which is the property the check is about.
The singular "one remaining" was the number that was wrong, not the finding.

```
   T H E   R O L L - C A L L   O F   F I V E ─────────────────────────

   kernel                                   decl   stage   guarded?
   moe_gate_up_mid_decode_lut_qwarp32       20611  20621   no  ✓
   ..._decode_lut_owned_qwarp32             20679  20687   no  ✓
   ..._expert_tile8_row32                   21194  21217   no  ✓   n_tokens>1
   ..._expert_tile8_row2048                 21284  21307   no  ✓   n_tokens>1
   ..._expert_tile8_rowspan                 21378  21401   no  ✓   n_tokens>1

   every stage sits ABOVE its __syncthreads(), every consumer BELOW it
   the 18 surviving xq_blocks guards all enclose an sxq activation copy
   5 of 5 clean, so LOAD and CONSUME agree at every xq_blocks
```

## Who would have been exposed on a pre-fix tree

- DeepSeek V4.1 Flash, n_embd 5120, 20 blocks, and V4 Pro, 7168, 28 blocks, on the CUDA IQ2 routed
  MoE expert-tile batch path with `n_tokens > 1`.
- V4 Flash, n_embd 4096, 16 blocks, was never in scope.
- GLM 5.2, 6144, uses `glm_routed_moe_gateup_expert_tile8_kernel` and never touches the IQ2 tables.

## Owed

- Nothing for the verdict, which is a property of the source text and was re-derived above.
- A CUDA rebuild of the unchanged file on the Spark would confirm compilation and say nothing about
  the finding. A cheap runtime cross-check needs no new code: a 2-token batch on V4.1 Flash against
  the known-good greedy sha.
