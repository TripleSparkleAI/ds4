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

✦ TRIPLESPARKLE ✦  this branch's card is below; antirez's README above is unchanged

```
  ┌──────────────────────────────────────────────────────────────────────
  │  BRANCH   triple-all-fastest                                POSITIVE
  │  WHAT     the new tip plus hitsfirst, the measured fastest tree
  │           one lever, ON by default, and nothing else
  │           set by rule from the sealed rounds, not by claim
  │  SWITCH   DS4_CUDA_HITS_FIRST=0 restores wait-then-launch
  │  OUTPUT   not gated
  │  BASE     triple-tip-2026-09-16 @12997e9c
  └──────────────────────────────────────────────────────────────────────

  RESULTS
  round  variant                date        arm t/s  tip t/s    delta  sign   floor  verdict  n
  r12                           2026-09-18    10.29     9.54   +9.07%   4/4    3.81  BEATS    4
  r12    cache=60GB             2026-09-18     8.86     9.54   -7.14%   0/4    3.81  LOSES    4
  r10                           2026-09-18     9.70     9.73   -0.64%   2/4    2.18  TIES     4
  r9                            2026-09-17    10.41     9.42  +12.09%   4/4   17.27  TIES     4
  r8                            2026-09-17    10.34     8.96  +12.31%   4/4   10.69  BEATS    4
  r4                            2026-09-17    10.37     9.65   +6.47%   4/4    2.71  BEATS    4
  gen tokens/s at min across ctx 4096 and 6144 · DGX Spark GB10 · each repeat against its bracketing tip runs · floor = max adjacent tip pair
  r12 2026-09-18-R12-RESULT-winners-qd16-default-and-scout-all-beat-the-tip-the-60gb-cache-loses-seven.md
  r10 2026-09-18-R10-RESULT-the-combination-works-on-a-quiet-box-and-hitsfirst-alone-dipped-twice.md
  r9  2026-09-17-R9-RESULT-the-default-holds-hitsfirst-is-the-tightest-tree-on-a-loud-instrument.md
  r8  2026-09-17-R8-RESULT-hitsfirst-alone-beats-through-the-noise-the-winners-tree-does-not.md
  r4  2026-09-17-R4-RESULT-hitsfirst-pool-and-the-newtip-stack-beat-the-tip-readahead-order-loses-eleven.md
```

NOTES
- The default STANDS as the tip plus hitsfirst: five sealed rounds, three wins, two ties, no sealed loss.
- Round 12, quiet box, four repeats: this tree +9.07, 4 of 4, against a 3.81 floor.
- winners@QD16 read 0.6 % above it, inside the floor, on two quiet boxes now: at least as fast, not faster.
- The sealed rule settles a tie by the smaller tree, and one lever is smaller than three: hitsfirst keeps it.
- The 60 GB expert cache setting is closed: -7.14 here, negative every repeat, on both trees it was tried on.

## The index: every branch card

One card per branch, generated by `track0_harness/card_tool.py index` from each branch's
`card.json` at its HEAD (CARD_STANDARD_v3.md), grouped by the status the card declares.
Regenerate, never hand-edit. Regenerated 2026-09-18 04:27Z, 28 cards.

```
  branch                           status      best delta  latest  verdict  rows
  triple-all-fastest               POSITIVE       +12.31%  r12     BEATS       6
  triple-hitsfirst                 POSITIVE       +12.31%  r12     BEATS       5
  triple-winners                   POSITIVE       +10.25%  r12     BEATS       5
  triple-pool                      POSITIVE        +9.59%  r9      TIES        3
  triple-all                       POSITIVE        +7.02%  r9      TIES        3
  triple-all-fastest-newtip        POSITIVE        +6.64%  r9      TIES        5
  triple-pair                      POSITIVE        +5.58%  r10     BEATS       1
  triple-prefetch-pool             POSITIVE        +2.28%  r4      PARTIAL     1
  triple-scout                     WORTH ZERO      +7.30%  r12     BEATS       2
  triple-draincut                  WORTH ZERO      +1.48%  r5      TIES        1
  triple-pagecache                 WORTH ZERO      +1.47%  r5      TIES        1
  triple-climbingfibre             WORTH ZERO      +1.28%  r5      TIES        1
  triple-lfu-offsetkey             WORTH ZERO      -0.20%  r6      TIES        1
  triple-engram-lead               WORTH ZERO      -1.04%  r5      TIES        1
  triple-engram-read-threads       WORTH ZERO      -1.39%  r4      TIES        1
  triple-engram-prestage           WORTH ZERO      -1.40%  r6      TIES        1
  triple-hippocampal-warmset       WORTH ZERO      -1.54%  r6      TIES        1
  triple-granule                   NEGATIVE        -0.04%  r5      TIES        1
  triple-hotlist                   NEGATIVE        -2.73%  r4      LOSES       1
  triple-prefill-readahead-order   NEGATIVE       -11.38%  r4      LOSES       1
  triple-draft-gamma               NOT YET         +3.83%  r5      TIES        1
  triple-spec-under-offload        NOT YET              -  -       -           0
  triple-antirez-tip-latest        CONTROL              -  -       -           0
  triple-prefix-cache              CONTROL              -  -       -           0
  triple-margin                    TOOLING         +1.17%  r5      TIES        1
  triple-engram-4bit               TOOLING              -  -       -           0
  triple-iq2-lut-fix               TOOLING              -  -       -           0
  triple-word-finisher             TOOLING              -  -       -           0
```

## POSITIVELY MEASURED

### triple-all

```
  ┌──────────────────────────────────────────────────────────────────────
  │  BRANCH   triple-all                                        POSITIVE
  │  WHAT     six levers stacked in series on the expert-cache read path
  │           pool, drain sync, cache margin, hot list, page-drop, hits-first
  │           second best tree measured, below hitsfirst alone in round 9
  │  SWITCH   DS4_CUDA_HITS_FIRST=0, plus each lever's own env switch
  │  OUTPUT   not gated
  │  BASE     triple-tip-2026-09-16 @12997e9c
  └──────────────────────────────────────────────────────────────────────

  RESULTS
  round  variant                date        arm t/s  tip t/s    delta  sign   floor  verdict  n
  r9                            2026-09-17    10.11     9.42   +7.02%   3/4   17.27  TIES     4
  r8                            2026-09-17     9.10     8.96   -0.98%   1/4   10.69  TIES     4
  r6                            2026-09-17    10.35     9.69   +6.50%   3/3    3.35  BEATS    3
  gen tokens/s at min across ctx 4096 and 6144 · DGX Spark GB10 · each repeat against its bracketing tip runs · floor = max adjacent tip pair
  prefill: reported, never a verdict - r8 85.3 vs tip 84.3
  r9  2026-09-17-R9-RESULT-the-default-holds-hitsfirst-is-the-tightest-tree-on-a-loud-instrument.md
  r8  2026-09-17-R8-RESULT-hitsfirst-alone-beats-through-the-noise-the-winners-tree-does-not.md
  r6  2026-09-17-R6-RESULT-the-superseded-six-lever-stack-beats-the-tip-three-ties.md
```

### triple-all-fastest

```
  ┌──────────────────────────────────────────────────────────────────────
  │  BRANCH   triple-all-fastest                                POSITIVE
  │  WHAT     the new tip plus hitsfirst, the measured fastest tree
  │           one lever, ON by default, and nothing else
  │           set by rule from the sealed rounds, not by claim
  │  SWITCH   DS4_CUDA_HITS_FIRST=0 restores wait-then-launch
  │  OUTPUT   not gated
  │  BASE     triple-tip-2026-09-16 @12997e9c
  └──────────────────────────────────────────────────────────────────────

  RESULTS
  round  variant                date        arm t/s  tip t/s    delta  sign   floor  verdict  n
  r12                           2026-09-18    10.29     9.54   +9.07%   4/4    3.81  BEATS    4
  r12    cache=60GB             2026-09-18     8.86     9.54   -7.14%   0/4    3.81  LOSES    4
  r10                           2026-09-18     9.70     9.73   -0.64%   2/4    2.18  TIES     4
  r9                            2026-09-17    10.41     9.42  +12.09%   4/4   17.27  TIES     4
  r8                            2026-09-17    10.34     8.96  +12.31%   4/4   10.69  BEATS    4
  r4                            2026-09-17    10.37     9.65   +6.47%   4/4    2.71  BEATS    4
  gen tokens/s at min across ctx 4096 and 6144 · DGX Spark GB10 · each repeat against its bracketing tip runs · floor = max adjacent tip pair
  r12 2026-09-18-R12-RESULT-winners-qd16-default-and-scout-all-beat-the-tip-the-60gb-cache-loses-seven.md
  r10 2026-09-18-R10-RESULT-the-combination-works-on-a-quiet-box-and-hitsfirst-alone-dipped-twice.md
  r9  2026-09-17-R9-RESULT-the-default-holds-hitsfirst-is-the-tightest-tree-on-a-loud-instrument.md
  r8  2026-09-17-R8-RESULT-hitsfirst-alone-beats-through-the-noise-the-winners-tree-does-not.md
  r4  2026-09-17-R4-RESULT-hitsfirst-pool-and-the-newtip-stack-beat-the-tip-readahead-order-loses-eleven.md
```

### triple-all-fastest-newtip

```
  ┌──────────────────────────────────────────────────────────────────────
  │  BRANCH   triple-all-fastest-newtip                         POSITIVE
  │  WHAT     the ten-lever stack rebuilt on the new tip from its manifest,
  │           one commit per lever, every conflict hunk recorded, so
  │           build_stack.sh reproduces the tree from the manifest alone
  │  SWITCH   DS4_CUDA_HITS_FIRST=1 turns the hits-first lever on
  │  OUTPUT   not gated
  │  BASE     triple-tip-2026-09-16 @12997e9c
  └──────────────────────────────────────────────────────────────────────

  RESULTS
  round  variant                date        arm t/s  tip t/s    delta  sign   floor  verdict  n
  r9                            2026-09-17     9.93     9.42   +3.77%   2/3   17.27  TIES     3
  r8     readahead ON           2026-09-17     9.80     8.96   +6.64%   3/4   10.69  TIES     4
  r8     hits ON                2026-09-17     9.36     8.96   +5.46%   4/4   10.69  TIES     4
  r8     both ON                2026-09-17     8.64     8.96   -5.95%   0/4   10.69  TIES     4
  r4                            2026-09-17    10.14     9.65   +4.65%   4/4    2.71  BEATS    4
  gen tokens/s at min across ctx 4096 and 6144 · DGX Spark GB10 · each repeat against its bracketing tip runs · floor = max adjacent tip pair
  prefill: reported, never a verdict - r4 136.3 vs tip 80.7
  r9  2026-09-17-R9-RESULT-the-default-holds-hitsfirst-is-the-tightest-tree-on-a-loud-instrument.md
  r8  2026-09-17-R8-RESULT-hitsfirst-alone-beats-through-the-noise-the-winners-tree-does-not.md
  r4  2026-09-17-R4-RESULT-hitsfirst-pool-and-the-newtip-stack-beat-the-tip-readahead-order-loses-eleven.md
```

### triple-hitsfirst

```
  ┌──────────────────────────────────────────────────────────────────────
  │  BRANCH   triple-hitsfirst                                  POSITIVE
  │  WHAT     launches gate/up/down for the experts already resident before
  │           the miss reads land, so a token does not stall on the slowest
  │           read in the batch; the parallel expert pread pool sits underneath
  │  SWITCH   DS4_CUDA_HITS_FIRST=0 turns it off (=1 is the shipped default)
  │  OUTPUT   not gated
  │  BASE     triple-tip-2026-09-16 @12997e9c
  └──────────────────────────────────────────────────────────────────────

  RESULTS
  round  variant                date        arm t/s  tip t/s    delta  sign   floor  verdict  n
  r12                           2026-09-18    10.29     9.54   +9.07%   4/4    3.81  BEATS    4
  r10                           2026-09-18     9.70     9.73   -0.64%   2/4    2.18  TIES     4
  r9                            2026-09-17    10.41     9.42  +12.09%   4/4   17.27  TIES     4
  r8                            2026-09-17    10.34     8.96  +12.31%   4/4   10.69  BEATS    4
  r4                            2026-09-17    10.37     9.65   +6.47%   4/4    2.71  BEATS    4
  gen tokens/s at min across ctx 4096 and 6144 · DGX Spark GB10 · each repeat against its bracketing tip runs · floor = max adjacent tip pair
  prefill: reported, never a verdict - r9 89.0 vs tip 85.6 (r8 86.6 vs 84.3)
  r12 2026-09-18-R12-RESULT-winners-qd16-default-and-scout-all-beat-the-tip-the-60gb-cache-loses-seven.md
  r10 2026-09-18-R10-RESULT-the-combination-works-on-a-quiet-box-and-hitsfirst-alone-dipped-twice.md
  r9  2026-09-17-R9-RESULT-the-default-holds-hitsfirst-is-the-tightest-tree-on-a-loud-instrument.md
  r8  2026-09-17-R8-RESULT-hitsfirst-alone-beats-through-the-noise-the-winners-tree-does-not.md
  r4  2026-09-17-R4-RESULT-hitsfirst-pool-and-the-newtip-stack-beat-the-tip-readahead-order-loses-eleven.md
```

### triple-pair

```
  ┌──────────────────────────────────────────────────────────────────────
  │  BRANCH   triple-pair                                       POSITIVE
  │  WHAT     pool plus hitsfirst and nothing else, hits-first ON
  │           drops prefetch-pool from the winners tree to price it
  │           measured on a quiet box: +5.58 per cent, 3 of 4 runs up
  │  SWITCH   DS4_CUDA_HITS_FIRST=0, DS4_CUDA_FETCH_URING=0, _FETCH_QD=<n>
  │  OUTPUT   not gated
  │  BASE     triple-tip-2026-09-16 @12997e9c
  └──────────────────────────────────────────────────────────────────────

  RESULTS
  round  variant                date        arm t/s  tip t/s    delta  sign   floor  verdict  n
  r10                           2026-09-18    10.27     9.73   +5.58%   3/4    2.18  BEATS    4
  gen tokens/s at min across ctx 4096 and 6144 · DGX Spark GB10 · each repeat against its bracketing tip runs · floor = max adjacent tip pair
  r10 2026-09-18-R10-RESULT-the-combination-works-on-a-quiet-box-and-hitsfirst-alone-dipped-twice.md
```

### triple-pool

```
  ┌──────────────────────────────────────────────────────────────────────
  │  BRANCH   triple-pool                                       POSITIVE
  │  WHAT     serves expert reads from a parallel SSD pool, not one serial
  │           reader, with an io_uring O_DIRECT ring in front of the pool
  │           so queue depth is a switch and not the worker count
  │  SWITCH   DS4_CUDA_STREAMING_EXPERT_PREAD_POOL=0 restores serial
  │  OUTPUT   not gated
  │  BASE     triple-tip-2026-09-16 @12997e9c
  └──────────────────────────────────────────────────────────────────────

  RESULTS
  round  variant                date        arm t/s  tip t/s    delta  sign   floor  verdict  n
  r9                            2026-09-17    10.35     9.42   +9.59%   4/4   17.27  TIES     4
  r8                            2026-09-17    10.00     8.96   +8.56%   3/4   10.69  TIES     4
  r4                            2026-09-17    10.39     9.65   +7.27%   3/4    2.71  BEATS    4
  gen tokens/s at min across ctx 4096 and 6144 · DGX Spark GB10 · each repeat against its bracketing tip runs · floor = max adjacent tip pair
  prefill: reported, never a verdict - r4 87.3 vs tip 80.7
  r9  2026-09-17-R9-RESULT-the-default-holds-hitsfirst-is-the-tightest-tree-on-a-loud-instrument.md
  r8  2026-09-17-R8-RESULT-hitsfirst-alone-beats-through-the-noise-the-winners-tree-does-not.md
  r4  2026-09-17-R4-RESULT-hitsfirst-pool-and-the-newtip-stack-beat-the-tip-readahead-order-loses-eleven.md
```

### triple-prefetch-pool

```
  ┌──────────────────────────────────────────────────────────────────────
  │  BRANCH   triple-prefetch-pool                              POSITIVE
  │  WHAT     cuts the prefill read-ahead into chunked tasks for a pool of
  │           readers, each with its own pinned buffer and upload stream;
  │           the single reader stays as the fallback when the pool declines
  │  SWITCH   DS4_CUDA_SSD_PREFETCH_POOL=0 reverts the whole pool
  │  OUTPUT   not gated
  │  BASE     triple-tip-2026-09-16 @12997e9c
  └──────────────────────────────────────────────────────────────────────

  RESULTS
  round  variant                date        arm t/s  tip t/s    delta  sign   floor  verdict  n
  r4                            2026-09-17     9.87     9.65   +2.28%   3/3    2.71  PARTIAL  3
  gen tokens/s at min across ctx 4096 and 6144 · DGX Spark GB10 · each repeat against its bracketing tip runs · floor = max adjacent tip pair
  prefill: reported, never a verdict - r4 138.7 vs tip 80.7
  r4  2026-09-17-R4-RESULT-hitsfirst-pool-and-the-newtip-stack-beat-the-tip-readahead-order-loses-eleven.md
```

### triple-winners

```
  ┌──────────────────────────────────────────────────────────────────────
  │  BRANCH   triple-winners                                    POSITIVE
  │  WHAT     pool, hitsfirst and prefetch-pool stacked, hits-first ON
  │           the three levers that won round 4 solo, and nothing else
  │           on a quiet box it beats the tip by 6.5 to 6.9 per cent
  │  SWITCH   DS4_CUDA_HITS_FIRST=0, _FETCH_URING=0, _SSD_PREFETCH_POOL=0
  │  OUTPUT   greedy-identical to the tip: G1 short 5ed3e6dfe2177eca · long c5cb82566c628bed
  │  BASE     triple-tip-2026-09-16 @12997e9c
  └──────────────────────────────────────────────────────────────────────

  RESULTS
  round  variant                date        arm t/s  tip t/s    delta  sign   floor  verdict  n
  r12    QD=16                  2026-09-18    10.35     9.54  +10.25%   4/4    3.81  BEATS    4
  r10                           2026-09-18    10.40     9.73   +6.53%   3/4    2.18  BEATS    4
  r10    QD=16                  2026-09-18    10.40     9.73   +6.86%   4/4    2.18  BEATS    4
  r9                            2026-09-17    10.20     9.42   +5.03%   3/3   17.27  TIES     3
  r8                            2026-09-17     9.18     8.96   +1.54%   2/3   10.69  TIES     3
  gen tokens/s at min across ctx 4096 and 6144 · DGX Spark GB10 · each repeat against its bracketing tip runs · floor = max adjacent tip pair
  prefill: reported, never a verdict - r8 135.1 vs tip 84.3
  r12 2026-09-18-R12-RESULT-winners-qd16-default-and-scout-all-beat-the-tip-the-60gb-cache-loses-seven.md
  r10 2026-09-18-R10-RESULT-the-combination-works-on-a-quiet-box-and-hitsfirst-alone-dipped-twice.md
  r9  2026-09-17-R9-RESULT-the-default-holds-hitsfirst-is-the-tightest-tree-on-a-loud-instrument.md
  r8  2026-09-17-R8-RESULT-hitsfirst-alone-beats-through-the-noise-the-winners-tree-does-not.md
```

## MEASURED CORRECT, WORTH ZERO ON THE CLOCK

### triple-climbingfibre

```
  ┌──────────────────────────────────────────────────────────────────────
  │  BRANCH   triple-climbingfibre                              WORTH ZERO
  │  WHAT     a front cache in front of the frozen model that learns
  │           from the speculative verifier's own rejections, online,
  │           with a retention horizon. It proposes one draft token
  │  SWITCH   DS4_CLIMBINGFIBRE=1, default 0; _HORIZON, _DECAY, _NGRAM, _CAP
  │  OUTPUT   not gated
  │  BASE     triple-tip-2026-09-16 @12997e9c
  └──────────────────────────────────────────────────────────────────────

  RESULTS
  round  variant                date        arm t/s  tip t/s    delta  sign   floor  verdict  n
  r5                            2026-09-17     9.58     9.60   +1.28%   4/4    5.69  TIES     4
  gen tokens/s at min across ctx 4096 and 6144 · DGX Spark GB10 · each repeat against its bracketing tip runs · floor = max adjacent tip pair
  prefill: reported, never a verdict - r5 83.1 vs tip 84.6
  r5  2026-09-17-R5-RESULT-seven-ties-under-a-noisy-floor.md
```

### triple-draincut

```
  ┌──────────────────────────────────────────────────────────────────────
  │  BRANCH   triple-draincut                                   WORTH ZERO
  │  WHAT     reads the router's selected expert ids back on an event,
  │           not a blocking cudaMemcpy, so the host stops waiting on
  │           the shared expert: 40 device drains per token removed
  │  SWITCH   ON by default; DS4_CUDA_SELECTED_DRAIN_SYNC=1 restores blocking
  │  OUTPUT   not gated
  │  BASE     triple-tip-2026-09-16 @12997e9c
  └──────────────────────────────────────────────────────────────────────

  RESULTS
  round  variant                date        arm t/s  tip t/s    delta  sign   floor  verdict  n
  r5                            2026-09-17     9.62     9.60   +1.48%   3/4    5.69  TIES     4
  gen tokens/s at min across ctx 4096 and 6144 · DGX Spark GB10 · each repeat against its bracketing tip runs · floor = max adjacent tip pair
  prefill: reported, never a verdict - r5 87.5 vs tip 84.6
  r5  2026-09-17-R5-RESULT-seven-ties-under-a-noisy-floor.md
```

### triple-engram-lead

```
  ┌──────────────────────────────────────────────────────────────────────
  │  BRANCH   triple-engram-lead                                WORTH ZERO
  │  WHAT     starts the Engram table read one token ahead of first use,
  │           speculating on the current step's own argmax, so the read
  │           overlaps the step instead of landing exposed inside it
  │  SWITCH   DS4_V41_ENGRAM_LEAD_OFF turns it off; the lead is ON by default
  │  OUTPUT   not gated
  │  BASE     triple-tip-2026-09-16 @12997e9c
  └──────────────────────────────────────────────────────────────────────

  RESULTS
  round  variant                date        arm t/s  tip t/s    delta  sign   floor  verdict  n
  r5                            2026-09-17     9.43     9.60   -1.04%   1/4    5.69  TIES     4
  gen tokens/s at min across ctx 4096 and 6144 · DGX Spark GB10 · each repeat against its bracketing tip runs · floor = max adjacent tip pair
  prefill: reported, never a verdict - r5 86.9 vs tip 84.6
  r5  2026-09-17-R5-RESULT-seven-ties-under-a-noisy-floor.md
```

### triple-engram-prestage

```
  ┌──────────────────────────────────────────────────────────────────────
  │  BRANCH   triple-engram-prestage                            WORTH ZERO
  │  WHAT     moves the host-side Engram row lookup out of the forward
  │           pass into a prepare phase that runs before it, so the
  │           forward carries no disk read and no hash of its own
  │  SWITCH   DS4_ENGRAM_PRESTAGE=1; unset or 0 is the shipped inline path
  │  OUTPUT   not gated
  │  BASE     triple-tip-2026-09-16 @12997e9c
  └──────────────────────────────────────────────────────────────────────

  RESULTS
  round  variant                date        arm t/s  tip t/s    delta  sign   floor  verdict  n
  r6                            2026-09-17     9.54     9.69   -1.40%   0/3    3.35  TIES     3
  gen tokens/s at min across ctx 4096 and 6144 · DGX Spark GB10 · each repeat against its bracketing tip runs · floor = max adjacent tip pair
  prefill: reported, never a verdict - 87.18 t/s median against the tip's 89.16, n=4; no prefill verdict in round 6
  r6  2026-09-17-R6-RESULT-the-superseded-six-lever-stack-beats-the-tip-three-ties.md
```

### triple-engram-read-threads

```
  ┌──────────────────────────────────────────────────────────────────────
  │  BRANCH   triple-engram-read-threads                        WORTH ZERO
  │  WHAT     takes the decode step's Engram read off the batch machinery:
  │           one token goes straight to the serial reader, with no malloc,
  │           no qsort and no pool dispatch. The prefill read is unchanged
  │  SWITCH   DS4_ENGRAM_READ_THREADS=1 clamps the prefill read to serial
  │  OUTPUT   not gated
  │  BASE     triple-tip-2026-09-16 @12997e9c
  └──────────────────────────────────────────────────────────────────────

  RESULTS
  round  variant                date        arm t/s  tip t/s    delta  sign   floor  verdict  n
  r4                            2026-09-17     9.60     9.65   -1.39%   1/4    2.71  TIES     4
  gen tokens/s at min across ctx 4096 and 6144 · DGX Spark GB10 · each repeat against its bracketing tip runs · floor = max adjacent tip pair
  prefill: reported, never a verdict - r4 82.6 vs tip 80.7
  r4  2026-09-17-R4-RESULT-hitsfirst-pool-and-the-newtip-stack-beat-the-tip-readahead-order-loses-eleven.md
```

### triple-hippocampal-warmset

```
  ┌──────────────────────────────────────────────────────────────────────
  │  BRANCH   triple-hippocampal-warmset                        WORTH ZERO
  │  WHAT     persisted expert hot list gets a longer horizon: carried
  │           free inside a day, halved at a day boundary, durable once
  │           a (layer, expert) pair is demanded on three distinct days
  │  SWITCH   DS4_WARMSET_TIER=session|day, default session, today's path
  │  OUTPUT   not gated
  │  BASE     triple-tip-2026-09-16 @12997e9c
  └──────────────────────────────────────────────────────────────────────

  RESULTS
  round  variant                date        arm t/s  tip t/s    delta  sign   floor  verdict  n
  r6                            2026-09-17     9.57     9.69   -1.54%   0/3    3.35  TIES     3
  gen tokens/s at min across ctx 4096 and 6144 · DGX Spark GB10 · each repeat against its bracketing tip runs · floor = max adjacent tip pair
  prefill: reported, never a verdict - 83.97 t/s median against the tip's 89.16, n=4; no prefill verdict in round 6
  r6  2026-09-17-R6-RESULT-the-superseded-six-lever-stack-beats-the-tip-three-ties.md
```

### triple-lfu-offsetkey

```
  ┌──────────────────────────────────────────────────────────────────────
  │  BRANCH   triple-lfu-offsetkey                              WORTH ZERO
  │  WHAT     ports upstream's host expert cache: byte ranges keyed by
  │           file offset, a 16-way table, a per-slot uses counter halved
  │           every 4096 inserts. Three arms isolate the eviction policy
  │  SWITCH   DS4_EXPERT_CACHE_MODE=hotlist|offsetkey|offsetkey-lru
  │  OUTPUT   not gated
  │  BASE     triple-tip-2026-09-16 @12997e9c
  └──────────────────────────────────────────────────────────────────────

  RESULTS
  round  variant                date        arm t/s  tip t/s    delta  sign   floor  verdict  n
  r6                            2026-09-17     9.71     9.69   -0.20%   1/3    3.35  TIES     3
  gen tokens/s at min across ctx 4096 and 6144 · DGX Spark GB10 · each repeat against its bracketing tip runs · floor = max adjacent tip pair
  prefill: reported, never a verdict - 85.90 t/s median against the tip's 89.16, n=4; no prefill verdict in round 6
  r6  2026-09-17-R6-RESULT-the-superseded-six-lever-stack-beats-the-tip-three-ties.md
```

### triple-pagecache

```
  ┌──────────────────────────────────────────────────────────────────────
  │  BRANCH   triple-pagecache                                  WORTH ZERO
  │  WHAT     releases staged model pages madvise-then-fadvise so the
  │           drop actually lands, and hints WILLNEED over a layer's
  │           miss ranges on the buffered read path
  │  SWITCH   DS4_CUDA_KEEP_MODEL_PAGES=1 and DS4_CUDA_NO_EXPERT_READAHEAD=1
  │  OUTPUT   greedy-identical to the tip: G1 short bb06e711bc498bb9 · long d3355c94c70a4bb1
  │  BASE     triple-tip-2026-09-16 @12997e9c
  └──────────────────────────────────────────────────────────────────────

  RESULTS
  round  variant                date        arm t/s  tip t/s    delta  sign   floor  verdict  n
  r5                            2026-09-17     9.73     9.60   +1.47%   4/4    5.69  TIES     4
  gen tokens/s at min across ctx 4096 and 6144 · DGX Spark GB10 · each repeat against its bracketing tip runs · floor = max adjacent tip pair
  prefill: reported, never a verdict - r5 83.6 vs tip 84.6
  r5  2026-09-17-R5-RESULT-seven-ties-under-a-noisy-floor.md
```

### triple-scout

```
  ┌──────────────────────────────────────────────────────────────────────
  │  BRANCH   triple-scout                                      WORTH ZERO
  │  WHAT     at layer L, also read the experts token t-1 used at L+1
  │           reads only, never math; rides the hits-first batch
  │           on top of triple-all-fastest, the measured default
  │  SWITCH   DS4_CUDA_SCOUT=0 restores triple-all-fastest exactly
  │  OUTPUT   greedy-identical to the tip: G1 short 5ed3e6dfe2177eca · long c5cb82566c628bed
  │  BASE     triple-all-fastest @28f6f102
  └──────────────────────────────────────────────────────────────────────

  RESULTS
  round  variant                date        arm t/s  tip t/s    delta  sign   floor  verdict  n
  r12                           2026-09-18    10.26     9.54   +7.30%   4/4    3.81  BEATS    4
  r12    cache=60GB             2026-09-18     8.87     9.54   -6.19%   0/4    3.81  LOSES    4
  gen tokens/s at min across ctx 4096 and 6144 · DGX Spark GB10 · each repeat against its bracketing tip runs · floor = max adjacent tip pair
  r12 2026-09-18-R12-RESULT-winners-qd16-default-and-scout-all-beat-the-tip-the-60gb-cache-loses-seven.md
```

## NEGATIVELY MEASURED OR KILLED

### triple-granule

```
  ┌──────────────────────────────────────────────────────────────────────
  │  BRANCH   triple-granule                                    NEGATIVE
  │  WHAT     offline, CPU only: does a sparse-distributed expansion
  │           make a cheap front-cache lookup more discriminative than
  │           a plain hash at the same address budget?
  │  SWITCH   none
  │  OUTPUT   not gated
  │  BASE     triple-tip-2026-09-16 @12997e9c
  └──────────────────────────────────────────────────────────────────────

  RESULTS
  round  variant                date        arm t/s  tip t/s    delta  sign   floor  verdict  n
  r5                            2026-09-17     9.51     9.60   -0.04%   2/4    5.69  TIES     4
  gen tokens/s at min across ctx 4096 and 6144 · DGX Spark GB10 · each repeat against its bracketing tip runs · floor = max adjacent tip pair
  prefill: reported, never a verdict - r5 86.0 vs tip 84.6
  r5  2026-09-17-R5-RESULT-seven-ties-under-a-noisy-floor.md
```

### triple-hotlist

```
  ┌──────────────────────────────────────────────────────────────────────
  │  BRANCH   triple-hotlist                                    NEGATIVE
  │  WHAT     seeds the next session's SSD expert cache from the previous
  │           run's demand, so the opening tokens start warm. A warm-up
  │           device, not a selection device
  │  SWITCH   DS4_CUDA_EXPERT_HOTLIST_WRITE=0 turns the writer off
  │  OUTPUT   not gated
  │  BASE     triple-tip-2026-09-16 @12997e9c
  └──────────────────────────────────────────────────────────────────────

  RESULTS
  round  variant                date        arm t/s  tip t/s    delta  sign   floor  verdict  n
  r4                            2026-09-17     9.42     9.65   -2.73%   0/4    2.71  LOSES    4
  gen tokens/s at min across ctx 4096 and 6144 · DGX Spark GB10 · each repeat against its bracketing tip runs · floor = max adjacent tip pair
  prefill: reported, never a verdict - r4 82.3 vs tip 80.7
  r4  2026-09-17-R4-RESULT-hitsfirst-pool-and-the-newtip-stack-beat-the-tip-readahead-order-loses-eleven.md
```

### triple-prefill-readahead-order

```
  ┌──────────────────────────────────────────────────────────────────────
  │  BRANCH   triple-prefill-readahead-order                    NEGATIVE
  │  WHAT     reorders the prefill read-ahead's victim choice and holds the
  │           earliest layers of the scan, so the experts decode needs first
  │           are the last evicted. Residency policy, no cache byte added
  │  SWITCH   DS4_PREFILL_READAHEAD_HOLD=0 restores upstream exactly
  │  OUTPUT   not gated
  │  BASE     triple-tip-2026-09-16 @12997e9c
  └──────────────────────────────────────────────────────────────────────

  RESULTS
  round  variant                date        arm t/s  tip t/s    delta  sign   floor  verdict  n
  r4                            2026-09-17     8.55     9.65  -11.38%   0/4    2.71  LOSES    4
  gen tokens/s at min across ctx 4096 and 6144 · DGX Spark GB10 · each repeat against its bracketing tip runs · floor = max adjacent tip pair
  prefill: reported, never a verdict - r4 86.3 vs tip 80.7
  r4  2026-09-17-R4-RESULT-hitsfirst-pool-and-the-newtip-stack-beat-the-tip-readahead-order-loses-eleven.md
```

## NOT YET MEASURED

### triple-draft-gamma

```
  ┌──────────────────────────────────────────────────────────────────────
  │  BRANCH   triple-draft-gamma                                NOT YET
  │  WHAT     an adaptive draft-length (gamma) controller for DSpark:
  │           one EMA slot per active-lane bucket, gamma from 0 to 16,
  │           so drafting switches itself off where it is not paying
  │  SWITCH   DS4_DRAFT_GAMMA_MODE=fixed|adaptive, default fixed
  │  OUTPUT   not gated
  │  BASE     triple-tip-2026-09-16 @12997e9c
  └──────────────────────────────────────────────────────────────────────

  RESULTS
  round  variant                date        arm t/s  tip t/s    delta  sign   floor  verdict  n
  r5                            2026-09-17     9.75     9.60   +3.83%   4/4    5.69  TIES     4
  gen tokens/s at min across ctx 4096 and 6144 · DGX Spark GB10 · each repeat against its bracketing tip runs · floor = max adjacent tip pair
  prefill: reported, never a verdict - r5 88.6 vs tip 84.6
  r5  2026-09-17-R5-RESULT-seven-ties-under-a-noisy-floor.md
```

### triple-spec-under-offload

```
  ┌──────────────────────────────────────────────────────────────────────
  │  BRANCH   triple-spec-under-offload                         NOT YET
  │  WHAT     does DSpark speculative decoding pay on a box whose weights
  │           stream from disk? A residency axis crossed with a spec axis:
  │           cold and warm working set, four legs, no engine code added
  │  SWITCH   --dspark --mtp-model FILE; the off leg is DS4_MTP_SPEC_DISABLE=1
  │  OUTPUT   not gated
  │  BASE     triple-tip-2026-09-16 @12997e9c
  └──────────────────────────────────────────────────────────────────────

  RESULTS  none yet - never in a sealed round
```

## THE CONTROL

### triple-antirez-tip-latest

```
  ┌──────────────────────────────────────────────────────────────────────
  │  BRANCH   triple-antirez-tip-latest                         CONTROL
  │  WHAT     upstream main 9139e2ae5 with PRs 1034 and 1035 folded in
  │           no lever; every other branch was measured against this
  │           about 9.6 to 9.8 gen t/s at min across frontiers
  │  SWITCH   none
  │  OUTPUT   not gated
  │  BASE     triple-antirez-tip-latest @e6d9d3b8
  └──────────────────────────────────────────────────────────────────────

  RESULTS  none yet - never in a sealed round
```

### triple-prefix-cache

```
  ┌──────────────────────────────────────────────────────────────────────
  │  BRANCH   triple-prefix-cache                               CONTROL
  │  WHAT     the prefix cache the engine already ships in ds4_kvstore.c,
  │           read out of the code, plus the one thing it lacked: an off
  │           path. --prefix-cache off rewinds the session every request
  │  SWITCH   --prefix-cache on|off on ds4-server, default on
  │  OUTPUT   not gated
  │  BASE     triple-tip-2026-09-16 @12997e9c
  └──────────────────────────────────────────────────────────────────────

  RESULTS  none yet - never in a sealed round
```

## TOOLING, NOT A LEVER

### triple-engram-4bit

```
  ┌──────────────────────────────────────────────────────────────────────
  │  BRANCH   triple-engram-4bit                                TOOLING
  │  WHAT     a design record for re-encoding only the two Engram hash
  │           tables from FP8 at 264 B a row to 4-bit at 132 B. Ships the
  │           arithmetic, the four costs and the plan, and no engine code
  │  SWITCH   none
  │  OUTPUT   not gated
  │  BASE     triple-tip-2026-09-16 @12997e9c
  └──────────────────────────────────────────────────────────────────────

  RESULTS  none yet - never in a sealed round
```

### triple-iq2-lut-fix

```
  ┌──────────────────────────────────────────────────────────────────────
  │  BRANCH   triple-iq2-lut-fix                                TOOLING
  │  WHAT     a source check, not a lever: does our tree carry upstream's
  │           IQ2 dequant-LUT defect, where the codebook was staged only
  │           for n_embd <= 4096 and then read below the guard anyway
  │  SWITCH   none
  │  OUTPUT   not gated
  │  BASE     triple-tip-2026-09-16 @12997e9c
  └──────────────────────────────────────────────────────────────────────

  RESULTS  none yet - never in a sealed round
```

### triple-margin

```
  ┌──────────────────────────────────────────────────────────────────────
  │  BRANCH   triple-margin                                     TOOLING
  │  WHAT     makes the memory reserve a live, checked, recorded value
  │           and expresses the cache budget as a split of one RAM pool
  │           across three consumers: an instrument and a bound
  │  SWITCH   DS4_MEM_RESERVE_MIB=N default 512; _ENFORCE=1; _RECORD=FILE
  │  OUTPUT   not gated
  │  BASE     triple-tip-2026-09-16 @12997e9c
  └──────────────────────────────────────────────────────────────────────

  RESULTS
  round  variant                date        arm t/s  tip t/s    delta  sign   floor  verdict  n
  r5                            2026-09-17     9.66     9.60   +1.17%   3/4    5.69  TIES     4
  gen tokens/s at min across ctx 4096 and 6144 · DGX Spark GB10 · each repeat against its bracketing tip runs · floor = max adjacent tip pair
  prefill: reported, never a verdict - r5 89.5 vs tip 84.6
  r5  2026-09-17-R5-RESULT-seven-ties-under-a-noisy-floor.md
```

### triple-word-finisher

```
  ┌──────────────────────────────────────────────────────────────────────
  │  BRANCH   triple-word-finisher                              TOOLING
  │  WHAT     a suffix-lookup drafter, measured offline and closed: draft
  │           k tokens from the continuation stored against the last n,
  │           verify in one batched pass, commit the longest match
  │  SWITCH   none
  │  OUTPUT   not gated
  │  BASE     triple-tip-2026-09-16 @12997e9c
  └──────────────────────────────────────────────────────────────────────

  RESULTS  none yet - never in a sealed round
```

