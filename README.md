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
  │  BRANCH     triple-spec-under-offload                            NOT YET
  │
  │  WHAT       does DSpark speculative decoding PAY on a box whose weights
  │             stream from disk? A residency axis crossed with a spec axis:
  │             RAW (cold, page cache evicted) and WARM WORKING SET (resident
  │             before the measured block). Four legs, one binary, no engine
  │             code added
  │
  │  LATEST     never in a sealed round. The arm lists of 2026-09-17-ROUND3-,
  │             -R4- and -R5-PREREGISTERED-RULE.txt name this branch zero times,
  │             and no round's argv carries --dspark. Its own four result files
  │             under try-results/ are TEMPLATES: every numeric cell is blank on
  │             purpose, and a grep for a two-decimal number across all four
  │             returns 0 lines
  │
  │  GEN        not measured - no engine code, nothing for the bench to run
  │  PREFILL    not measured - no engine code, nothing for the bench to run
  │
  │  VERDICT    NOT YET - the four legs and the CUDA build both need the Spark,
  │             and nothing has been run. The warm arm is the one nobody has
  │             ever run: every DSpark A/B we hold ran WITHOUT residency,
  │             because the F16 drafter at 117.73 GiB with the target does not
  │             fit beside the 115 GiB the GB10 offers
  │
  │  SWITCH     --dspark --mtp-model <file>; the off leg is
  │             DS4_MTP_SPEC_DISABLE=1 on the same argv. The residency axis uses
  │             the pre-existing --ssd-streaming, --ssd-streaming-cold and
  │             --ssd-streaming-cache-experts N|NGB (ds4_help.c lines 174 and
  │             175 at HEAD)
  │  OUTPUT     not re-run
  │
  └──────────────────────────────────────────────────────────────────────
```

**Why this branch is not in the measurement pass.** It adds no engine code. Its files are
`try-spec-offload.sh` (828 lines of POSIX sh), four blank result templates, an index and this card.
The standing round builds one binary per arm and runs `ds4-bench` with no switches set, so an arm
built from this branch would be the tip compiled twice. The question here needs its own four-leg
runner on the Spark, and the Spark is serialized to another series.

## The question

- Not "is DSpark lossless" and not "does DSpark help on a resident box". On a disk-bound box, does
  the draft's saved forward passes outweigh the extra bytes the verify pass pulls through the
  device?
- Our record says every DSpark A/B ran without residency: the converted drafter is F16 at
  39,696,568,160 bytes, so 117.73 GiB with the target against 115 GiB available before KV
  (`WIKI/theory/03-dspark.md` section 5).
- Upstream describes its support file as adding "about 5.6 GiB of weights plus runtime state"
  (`docs/SPECULATIVE_DECODING.md` line 35, in this branch's own tree). The runner records the server
  command, so the two drafters can be told apart afterwards.
- ⚠ The `WIKI/` paths here are not in this branch's tree. The branch is rooted on the upstream tip,
  which carries no `WIKI/`.

## What the field knows, and why it does not close this

- `WIKI/theory/89-serving-reframe-bug-shaped-2026-07-19.md` section 6: the community runs ds4-flash
  with DSpark on GB10s and wins. Dual-GB10 about 31 to 34 t/s, a forum report about 60 to 67 against
  40 to 45 plain, a reproduction at 26 to 40 to 60 t/s, so 1.51x then 2.29x.
- ⚠ All of those are dual-box TP=2 plus CUDA graphs plus a higher quant. Our single-box IQ2
  `ds4-server` has none of the three, so none of them is a number about our configuration.
- `WIKI/theory/48-decode-speedup-levers.md` section 8 still ranks more spec-decode as "net-negative,
  settled, do not" for this box. That ranking is the position under test.

## The tension inside our own record

- Closed branch `triple-word-finisher` measured a k-token verify pass reading the SAME NVMe bytes as
  k single passes: bytes saved 0.0000 at k of 2, 4 and 8, miss reuse 0 of 1,219, acceptance real at
  2.37x code and 1.17x prose, and the union of k steps' misses equal to their sum. If that transfers,
  a streaming box gains nothing from drafting.
- Against it, SpecOffload (`WIKI/research/engine-harvests/05-new-engines-scout-round2.md`) fills idle
  cycles during the fetch, and `03-adaptive-gamma-under-concurrency.md` records a rejected draft as
  pure wasted verify bandwidth, worse when the box is already saturated.
- The sign can fall either way. That is the point of running it.

```
   T H E   F O U R   L E G S ─────────────────────────────────────────

   leg  residency  dspark  drafter loaded   verify path
    1     cold      off    yes, same argv   killed by DS4_MTP_SPEC_DISABLE=1
    2     cold      on     yes, same argv   runs
    3     warm      off    yes, same argv   killed by DS4_MTP_SPEC_DISABLE=1
    4     warm      on     yes, same argv   runs

   cold = --ssd-streaming-cold + page cache evicted: about 112 GiB of the
          unrelated Pro gguf pushed through the cache with dd, about 105 s at
          the measured 1.2 GB/s SSD read (WIKI/theory/48 section 4a)
   warm = default preload + one DISCARDED prime pass at the fixed context

   ctx 8321 · the same ten tests · order 1 2 3 4, then the block again, one vintage
   every numeric cell of all four result files is blank today: grep for a
   two-decimal number across try-results/*.txt returns 0
```

## What the branch adds

- `try-spec-offload.sh`, a sibling of the stack branch's `try.sh`: POSIX sh, `${VAR:-default}` knobs,
  markdown-table result files, the model unloaded on every exit path.
- Four blank result files under `try-results/`, named by date, branch and what was on, plus an index.
- No kernel, no data path, no arithmetic change.
- `--dspark-strict` is a third state and is NOT the pair partner: the pair keeps one argv and closes
  the verify path with the kill switch, which gates the speculative branch in the server
  (`ds4_server.c`), the agent and the CLI.

## Owed

- `make cuda-spark` on the Spark.
- The four legs and their repeat.
- NVMe bytes per decode token from a device-side reading.
  `WIKI/theory/89-serving-reframe-bug-shaped-2026-07-19.md` section 3 names the decode capture at
  B=1 across draft lengths as the arbiter: flat means the non-routed read amortises, linear means it
  is re-read per position.
- Accept length tau on the on legs, and greedy identity off against on.
- One known limitation, stated in the runner: the off legs keep the drafter loaded, so they measure
  the verify pass and not the drafter's footprint. A no-drafter baseline is a different argv, labelled
  `dspark-absent`.
