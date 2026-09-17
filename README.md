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

Rebased onto triple-tip-2026-09-16 (12997e9c8) on 2026-09-17; tests make climbingfibre-check 51 checks 0 failed, cc -fsyntax-only clean on 3 touched C files. `make test` now reaches 26 verdicts, all pass, and stops at ds4_test, which wants a model a worktree does not have; before this branch added ds4_climbingfibre.o to the host CPU object lists it stopped earlier, at the tests/test_session_state link, a failure reproduced at the pre-rebase sha d93005082.

```
  ┌──────────────────────────────────────────────────────────────────────
  │
  │  BRANCH     triple-climbingfibre                                 NOT YET
  │
  │  WHAT       a front cache in front of the frozen model that learns from the
  │             speculative verifier's own REJECTIONS, online, with a retention
  │             horizon. It proposes one draft token; it never injects anything
  │
  │  LATEST     never in a sealed round
  │
  │  GEN        not measured
  │  PREFILL    not measured
  │
  │  VERDICT    NOT YET - no CUDA build of this branch exists, so no arm has
  │             ever run against a control. The only acceptance figures come
  │             from a scripted 61-token target, +139 accepted tokens cold and
  │             +0 with every entry poisoned, which is a mechanism check
  │
  │  SWITCH     DS4_CLIMBINGFIBRE=1, default 0. Off allocates nothing, never
  │             learns and never proposes
  │             DS4_CLIMBINGFIBRE_HORIZON=session|day|week|forever|<steps>
  │             _DECAY=hard|half|none · _NGRAM 4 · _CAP 16384
  │  OUTPUT     not re-run
  │
  └──────────────────────────────────────────────────────────────────────
```

## The mechanism

- Every speculative verify already computes the correction and throws it away. At a rejection the
  engine holds (context, drafted token, the target's actual token): a labelled pair at zero
  marginal cost.
- This branch writes that pair into a plain 4-gram table keyed on token ids and demotes the token
  that was refused.
- It consults the table in the DSpark proposer ONLY when the engine offered no draft of its own.
- The verifier that accepts or refuses the proposal is the same model it always ran, so a wrong
  entry costs work and cannot change an output.
- Hook sites in `ds4.c`: the first-token refusal (`sample_argmax` against `drafts[0]`), the
  within-block refusal (`row_tops[i-1] != drafts[i]`, which fires only at a real mismatch), and
  the proposer inside the DSpark path.
- `climb_key` takes an explicit prefix bound, so a correction landing late still lands on the
  context that was actually verified.
- A ring of 8 outstanding proposals attributes each rejection to its own proposal, which is what
  an asynchronous drafter (Saguaro, 2026) would otherwise break.
- Design: `experiments/track3-semiotic-codebook/NEW_IDEA_THE_CEREBELLAR_FRONT_CACHE_2026-09-01.md`
  section 5.

```
   THE SIGNAL IS ALREADY ON THE FLOOR, once per rejection

   verify step   drafted ─▶ [ target says NO ] ─▶ correction computed
                                                  │
                            as shipped ───────────┴──▶ discarded
                            this branch ──────────┴──▶ demote(drafted)
                                                       write(context -> actual)

   and the read side is fenced: consulted ONLY when the engine drafted nothing,
   so a wrong entry buys a wasted verify pass and never an output byte
```

## What is measured, and what is not

- `tests/test_climbingfibre.c`, 775 lines, `make climbingfibre-check`: **51 checks, 0 failed**.
  OFF, ON-cold and ON-with-every-entry-wrong emit the same stream; a 64-key table too small for
  its keys is inert rather than corrupting; a source guard strips comments and fails on any
  deep-path name in `ds4_climbingfibre.c`, which is 617 lines.
- The degrade arm is in that count: after 4x cap distinct inserts the table still has EMPTY slots,
  used 820 of cap 1024 with 13 rehashes, 25.27 probes per miss, and its own vacuity control
  asserts the probe counter moved at all.
- ⚠ An earlier card said 39 checks. That was before the retirement-and-tombstone-sweep repair
  (`465934e49`) added its arm; 51 is the current run on this Mac, 2026-09-17.
- Scripted target, 4000 tokens: **+139 extra accepted tokens cold, +0 with a poisoned table**.
  These come from a 61-token vocabulary, not the model, and are not a t/s claim.
- The prior on effect size is about 4 %, not 27: `MEASURED_DECAYVSDEMOTE_*_2026-09-01.md` found
  uniform decay delivers 22.2 of the 26.7 points once credited to the rejection signal, leaving
  3.7 % as an upper bound.
- The substrate is not the claim: `MEASURED_DRAFTMEMORY_*_2026-09-01.md` found the token-id n-gram
  beat a centred hidden-state address on both corpora, z +11.17 on code and z +5.58 on prose.
- Retrieval drafting is occupied (REST, DReSD at 4.64x, RASD, ReSpec). The narrow claim left is
  online write-back of REJECTIONS with a retention policy.
- The `session`, `day` and `week` step budgets are placeholders. ROLLINGSPLIT (design doc
  section 6) would price them and has not been run on this branch.

## The Makefile repair this branch owed, and the half of it still owed

- `ds4_climbingfibre.o` was added to `CORE_OBJS` and not to the CPU object lists, so
  `tests/test_session_state`, `tests/test_engine_mgpu_placement` and `tests/test_sampling` failed
  to link on `ds4_climbingfibre_free`.
- The list is spelled out in FOUR places, not one: `CPU_CORE_OBJS` under `ifeq Darwin`,
  `CPU_CORE_OBJS` in the `else` (CUDA) branch, and literally again in the two test rules that do
  not use the variable. All four now carry the object.
- ⚠ Only the Darwin path is PROVEN. The else branch's line is the same edit and has never been
  compiled, because this host has no CUDA toolkit.

## Owed

- A CUDA build on the Spark, then an interleaved A/B against the unpatched tip at one power mode,
  then the greedy-identical check.
- What has been verified is a `-Wall -Wextra -std=c99` compile of `ds4.c` with the hooks in place
  plus the host self test. That is a compile check, not a gate.
- Reach is the graph backends and the greedy path only.
- Files: `ds4_climbingfibre.h` (222 lines), `ds4_climbingfibre.c` (617),
  `ds4.c` (three hooks, two helpers), `tests/test_climbingfibre.c` (775), `Makefile`, `.gitignore`.
