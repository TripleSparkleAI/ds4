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

**✦ ✧ ✦ ✧ ✦ ✧ ✦ ✧ ✦ ✧ ✦ ✧ ✦   T R I P L E S P A R K L E   ✦ ✧ ✦ ✧ ✦ ✧ ✦ ✧ ✦ ✧ ✦ ✧ ✦**

**✦ above: the upstream README, unchanged · below: this branch's card and numbers**

Card format: `CARD_STANDARD_v2.md` at the repository root (v2, 2026-09-17).

```
  ┌──────────────────────────────────────────────────────────────────────
  │
  │  BRANCH     triple-all-fastest                                POSITIVE
  │
  │  WHAT       the new tip plus hitsfirst, the measured fastest tree, set
  │             by rule from rounds 4 and 8 and confirmed by round 9. One
  │             lever, ON by default, and nothing else.
  │
  │  LATEST     round 9 · 2026-09-17 · THE DEFAULT HOLDS · hitsfirst highest
  │             median 10.41 (tip 9.42) ·
  │             2026-09-17-R9-RESULT-the-default-holds-hitsfirst-is-the-tightest-tree-on-a-loud-instrument.md
  │             round 8 · 2026-09-17 · BEATS · +12.31 % (no-tip_1 +11.54 %) ·
  │             2026-09-17-R8-RESULT-hitsfirst-alone-beats-through-the-noise-the-winners-tree-does-not.md
  │             round 4 · 2026-09-17 · BEATS · +6.47 % (kept +7.93 %) ·
  │             2026-09-17-R4-RESULT-hitsfirst-pool-and-the-newtip-stack-beat-the-tip-readahead-order-loses-eleven.md
  │
  │  GEN        +12.31 %  floor 10.69 % (sealed, max)  sign 4/4  n=4
  │             raw: arm 10.49 · 9.43 · 10.38 · 10.29 (median 10.34 t/s) vs
  │             tip 8.61 · 9.53 · 8.87 · 9.49 · 8.96 (median 8.96 t/s),
  │             gen_steady at min across 4096/6144; every arm run above every
  │             control run; the delta is each run against the mean of its
  │             bracketing TIP runs. Cold-first-run sensitivity: +11.54 %
  │             against floor 7.17 %, still BEATS (post hoc, labelled).
  │             control = triple-tip-2026-09-16 @12997e9c, interleaved, one TIP
  │             bracket per cycle
  │             session = 2026-09-17 13:24-14:36Z · DGX Spark GB10 · native
  │             24.89 MB .text (r8-hitsfirst, 24,886,127) · box NOT quiet:
  │             four contention drops, END load 3.2 to 4.0
  │             round 4 beside it: +6.47 % (kept +7.93 %)  floor 2.71 %
  │             sign 4/4  n=4  arm median 10.37 vs tip 9.65 t/s
  │             (2026-09-17 09:50-10:57Z, same control, same box)
  │             ─── round 9, TIES on the rule, and it holds the default:
  │             +12.09 %  floor 17.27 %  sign 4/4  n=4
  │             raw: arm min-across median 10.41 t/s (10.32 to 10.51, spread
  │             0.19) vs the tip's 9.42 t/s (8.51 to 9.98), every arm run above
  │             every control run. Highest median and tightest tree of the five
  │             arms: pool 10.35, winners 10.20, triple-all 10.11, and the
  │             ten-lever stack carrying this same lever 9.93
  │             ⚠ the instrument sets that floor, not the arm: the five control
  │             runs span 8.51 to 9.98 t/s because the page cache holds 2.9 GB of
  │             an 81 GB model and the bench streams 842 MB/s from the NVMe, so
  │             identical binaries swing about 8 % run to run
  │             control = triple-tip-2026-09-16 @12997e9c, interleaved
  │             session = 2026-09-17 14:41-15:36Z · DGX Spark GB10 · native
  │             24.89 MB .text (r9-hitsfirst, 24,886,127; tip 24.86 MB) · lean
  │             regime 4096/6144 · quiet box, warm-up run discarded
  │  PREFILL    not measured
  │
  │  VERDICT    POSITIVE - three sealed rounds, two wins and one tie, and no
  │             sealed loss: round 4 +6.47 % through a 2.71 floor, round 8
  │             +12.31 % through a 10.69 floor, round 9 +12.09 % and the highest
  │             median of five arms on a quiet box. By the navigator's standing
  │             rule the all-fastest default follows the measured number, so
  │             this branch IS that tree, and ROUND 9 CONFIRMS THE DEFAULT: the
  │             word provisional is dropped. Round 9 ranked every other arm
  │             below this one lever, including the ten-lever tree that carries
  │             it (9.93), and no arm cleared the round's 17.27 floor.
  │
  │  SWITCH     DS4_CUDA_HITS_FIRST=0 turns the lever off (ON here, the lever's
  │             own default; =0 restores wait-then-launch)
  │             DS4_CUDA_HITS_FIRST_STAGED=0 keeps hits-first, single-stage read order
  │  OUTPUT     greedy-identical sha256 bb06e711bc498bb9 (triple-hitsfirst's
  │             run; this tree's code is byte-identical to that branch)
  │
  └──────────────────────────────────────────────────────────────────────
```

```
   the three sealed rounds of the one lever this tree carries   (gen vs the new tip, min-across)

              0        +2        +4        +6        +8        +10       +12
   round 4    ┃━━━━━━━━━┫━━━━━━━━━━━━━━━━━━●  +6.47   4/4   floor 2.71 (the ┫)
   round 8    ┃━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫━━━━━━━●  +12.31  4/4   floor 10.69 (the ┫)
   round 9    ┃━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━●  +12.09  4/4
              the floor is 17.27 and sits RIGHT of the delta, so TIES on the rule while the
              ranking is the sharpest yet: median 10.41, spread 0.19, every run above every
              control run. the default holds
```

### How this tree was built

- Base `triple-tip-2026-09-16` at `12997e9c8`. **One lever**: `triple-hitsfirst` at its code sha
  `3e4773671` (the card commit `40cd5b944` on top of it is README-only). `triple-hitsfirst` is a
  linear descendant of `12997e9c8`, so the lever's diff applies with **zero conflict hunks** and
  the manifest carries no `resolve` line.
- `allfastest.manifest` is the definition; `build_stack.sh` reproduces it into a fresh detached
  worktree and `--check <sha>` asserts byte identity on the build surface:
  `bash build_stack.sh allfastest.manifest --check <HEAD>`.
- The engine files of this branch are byte-identical to `triple-hitsfirst`'s:
  `git diff triple-hitsfirst -- '*.c' '*.h' '*.cu' '*.cuh' Makefile tests/` is EMPTY.
- hitsfirst's own default is ON (`cuda_hits_first_enabled`: `DS4_CUDA_HITS_FIRST=0` disables it),
  so no default flip and no hand patch were needed. Nothing else was done.
- Tests here: `tests/test_hitsfirst_logic` 142 checks, 0 failed (run 2026-09-17 on a Mac). Metal
  `make` rc 0 on a Mac. `ds4_cuda.cu` cannot compile on this machine; the Spark built this exact
  lever tree on 2026-09-17 for round 8 (`r8-hitsfirst`, `.text` 24,886,127) and round 9 runs it now.

### What it used to be

The name `triple-all-fastest` has carried three trees. None is lost; each has a sha or a tag.

- **The ten-lever stack on the old base** - tag `sealed/all-fastest-tenlever-oldbase`
  (`342f17f80`): base tip-latest `9139e2ae5` + #1034 + #1035, ten levers compiled in, nine on and
  hits-first off, victim order least-recently-routed at `dd189b2d7`, defined by `stack.manifest`
  at that tag. **Never measured as a ten-lever tree.** Its nine-lever siblings off the line of
  history were: `dd82361a` (the stack as shipped), `8aa40a52` (bisect slice NOS1, the victim
  comparator reverted), `d1dd5ec2` (bisect slice RT).
- **The ten-lever stack rebuilt on the new tip** - branch `triple-all-fastest-newtip`
  (`725d9bb90`, `stack-on-newtip.manifest`), hits-first OFF by default.
- **This tree** - the new tip plus hitsfirst, from the round 8 ruling.

Every sealed number for the stacks, in order, so the history reads whole:

| round | date | tree | gen vs tip | sign | floor | verdict | file |
| --- | --- | --- | ---: | ---: | ---: | --- | --- |
| attribution | 2026-09-16 | nine-lever `dd82361a` as shipped | -4.51 % | 0/7 | 2.2-4.1 max | LOSES | `2026-09-16-triple-all-fastest-attrib-SUMMARY-levers-attribution.txt` |
| bisect r1 | 2026-09-16 | nine-lever `dd82361a` | -4.35 % | 0/4 | 4.13 | LOSES, one comparator | `2026-09-16-BISECT-RESULT-one-comparator-carries-the-whole-loss.md` |
| bisect r1 | 2026-09-16 | NOS1 `8aa40a52`, comparator reverted | +3.69 % | 4/4 | 4.13 | inside the floor | same file |
| round 2 | 2026-09-17 | NOS1 `8aa40a52` | +3.46 % | 8/8 | 1.37 | BEATS (revert) | `2026-09-17-ROUND2-RESULT-the-revert-beats-the-tip.md` |
| round 4 | 2026-09-17 | newtip stack `triple-all-fastest-newtip`, hits OFF | +4.65 % (kept +6.87) | 4/4 | 2.71 | BEATS | `2026-09-17-R4-RESULT-hitsfirst-pool-and-the-newtip-stack-beat-the-tip-readahead-order-loses-eleven.md` |
| round 6 | 2026-09-17 | six-lever `triple-all` | +6.50 % | 3/3 | 3.35 | BEATS | `2026-09-17-R6-RESULT-the-superseded-six-lever-stack-beats-the-tip-three-ties.md` |
| round 8 | 2026-09-17 | newtip stack + readahead ON | +6.64 % | 3/4 | 10.69 | TIES | `2026-09-17-R8-RESULT-hitsfirst-alone-beats-through-the-noise-the-winners-tree-does-not.md` |
| round 8 | 2026-09-17 | newtip stack + hitsfirst ON | +5.46 % | 4/4 | 10.69 | TIES | same file |
| round 8 | 2026-09-17 | newtip stack + both ON | -5.95 % | 0/4 | 10.69 | TIES, slowest tree measured | same file |
| round 9 | 2026-09-17 | six-lever `triple-all` | +7.02 % | 3/4 | 17.27 | TIES, reads with round 6 | `2026-09-17-R9-RESULT-the-default-holds-hitsfirst-is-the-tightest-tree-on-a-loud-instrument.md` |
| round 9 | 2026-09-17 | newtip stack + hitsfirst ON | +3.77 % | 2/3 | 17.27 | TIES, lowest median of the round | same file |
| round 8 | 2026-09-17 | `triple-winners` (pool + hits + prefetch-pool) | +1.54 % | 2/3 | 10.69 | TIES | same file |
| round 8 | 2026-09-17 | six-lever `triple-all` | -0.98 % | 1/4 | 10.69 | TIES | same file |
| round 8 | 2026-09-17 | **hitsfirst alone, this tree** | **+12.31 %** | 4/4 | 10.69 | **BEATS** | same file |
| round 3 | 2026-09-17 | ten-lever FAST arm `b17231fd` | sealed, never run | - | - | no number | `2026-09-17-ROUND3-PREREGISTERED-RULE.txt` |

The old tree's full record and its lever table stay readable at the tag:
`git show sealed/all-fastest-tenlever-oldbase:README.md`.

---

### Round 9, 2026-09-17: the default holds

- **Confirmed, not provisional.** Round 9 ran the quiet box with the discarded warm-up and put
  this one lever first again: median **10.41 t/s** against the tip's 9.42, +12.09 %, sign 4 of 4.
  Three sealed rounds, two wins and one tie, no sealed loss.
- **It TIES only because the floor is 17.27.** The five control runs span 8.51 to 9.98, so the
  round confirms a ranking rather than a magnitude, and the magnitude still stands from round 8.
- **Every rival ranked below it**, including the ten-lever tree that carries this same lever:
  pool 10.35, winners 10.20, triple-all 10.11, stack + hits 9.93. Its spread of 0.19 is five times
  tighter than the next arm's.

## The index: every branch card

One card per branch, the box copied from that branch's own README at its HEAD by
`rebuild_index.py`, grouped by the status the card itself declares. Regenerate,
never hand-edit: a hand-edited row drifts from the branch it describes.
Regenerated 2026-09-17 14:45Z, 26 cards.

## POSITIVELY MEASURED

### triple-all

```
  ┌──────────────────────────────────────────────────────────────────────
  │
  │  BRANCH     triple-all                                          POSITIVE
  │
  │  WHAT       six lever branches stacked in series: the pread pool, the
  │             event-gated selected-expert readback, the cache reserve knob,
  │             the hot list seed, the staged page-drop order and hits-first.
  │             Measured second-best of all trees today at +6.50 %, and an arm
  │             of the amended round 7
  │
  │  LATEST     round 6 · 2026-09-17 · BEATS · +6.50 % ·
  │             2026-09-17-R6-RESULT-the-superseded-six-lever-stack-beats-the-tip-three-ties.md
  │
  │  GEN        +6.50 %  floor 3.35 %  sign 3/3  n=3
  │             raw: arm min-across median 10.35 t/s (10.31 · 10.24 · 10.38 ·
  │             10.39) vs the tip's 9.69 t/s (9.67 · 9.56 · 9.88 · 9.70 kept,
  │             9.45 contention-dropped; the result file's figure line reads
  │             9.67). gen_steady at min across 4096/6144. The delta is each
  │             run against its bracketing TIP runs, not against that median
  │             control = triple-tip-2026-09-16 @12997e9c, interleaved
  │             session = 2026-09-17 11:12-11:58Z · DGX Spark GB10 ·
  │             native 24.90 MB .text (tip 24.86 MB) · lean regime 4096/6144
  │  PREFILL    reported, not a verdict: 86.30 t/s min-across median vs the
  │             tip's 89.16 t/s, n=4. Round 6 makes no prefill verdict.
  │
  │  VERDICT    POSITIVE - the only arm of round 6 to clear the floor, and it
  │             clears it at every repeat (+6.1 to +6.8 %). Both the sealed
  │             legacy rule and the new rule give this verdict, so nothing
  │             turns on the straggler rule
  │
  │  SWITCH     the six levers' own switches, verified present in this tree's
  │             diff: DS4_CUDA_STREAMING_EXPERT_PREAD_POOL and _PREAD_THREADS,
  │             DS4_CUDA_SELECTED_DRAIN_SYNC, DS4_CUDA_EXPERT_CACHE_MARGIN_GB,
  │             DS4_CUDA_EXPERT_HOTLIST_WRITE, DS4_CUDA_NO_EXPERT_READAHEAD,
  │             DS4_CUDA_HITS_FIRST and _HITS_FIRST_STAGED
  │  OUTPUT     not re-run. A greedy sha was recorded on a prior card and is
  │             withheld here: it was taken before this rebase, on a tree this
  │             branch is no longer on
  │
  └──────────────────────────────────────────────────────────────────────
```

### triple-all-fastest

```
  ┌──────────────────────────────────────────────────────────────────────
  │
  │  BRANCH     triple-all-fastest                                POSITIVE
  │
  │  WHAT       the new tip plus hitsfirst, the measured fastest tree, set
  │             by rule from rounds 4 and 8, provisional on round 9. One
  │             lever, ON by default, and nothing else.
  │
  │  LATEST     round 8 · 2026-09-17 · BEATS · +12.31 % (no-tip_1 +11.54 %) ·
  │             2026-09-17-R8-RESULT-hitsfirst-alone-beats-through-the-noise-the-winners-tree-does-not.md
  │
  │  GEN        +12.31 %  floor 10.69 % (sealed, max)  sign 4/4  n=4
  │             raw: arm 10.49 · 9.43 · 10.38 · 10.29 (median 10.34 t/s) vs
  │             tip 8.61 · 9.53 · 8.87 · 9.49 · 8.96 (median 8.96 t/s),
  │             gen_steady at min across 4096/6144; every arm run above every
  │             control run; the delta is each run against the mean of its
  │             bracketing TIP runs. Cold-first-run sensitivity: +11.54 %
  │             against floor 7.17 %, still BEATS (post hoc, labelled).
  │             control = triple-tip-2026-09-16 @12997e9c, interleaved, one TIP
  │             bracket per cycle
  │             session = 2026-09-17 13:24-14:36Z · DGX Spark GB10 · native
  │             24.89 MB .text (r8-hitsfirst, 24,886,127) · box NOT quiet:
  │             four contention drops, END load 3.2 to 4.0
  │             round 4 beside it: +6.47 % (kept +7.93 %)  floor 2.71 %
  │             sign 4/4  n=4  arm median 10.37 vs tip 9.65 t/s
  │             (2026-09-17 09:50-10:57Z, same control, same box)
  │  PREFILL    not measured
  │
  │  VERDICT    POSITIVE - two sealed wins, no sealed loss: round 4 +6.47 %
  │             through a 2.71 floor, round 8 +12.31 % through a 10.69 floor,
  │             the only arm of seven that cleared round 8's floor. By the
  │             navigator's standing rule the all-fastest default follows the
  │             measured number, so this branch IS that tree. Provisional:
  │             round 9 (sealed 2026-09-17-R9-PREREGISTERED-RULE.txt, quiet
  │             box, with the warm-up run) re-measures hitsfirst, pool,
  │             winners, stack+hits and triple-all; if it ranks another tree
  │             above hitsfirst on a clean floor, the default moves again.
  │
  │  SWITCH     DS4_CUDA_HITS_FIRST=0 turns the lever off (ON here, the lever's
  │             own default; =0 restores wait-then-launch)
  │             DS4_CUDA_HITS_FIRST_STAGED=0 keeps hits-first, single-stage read order
  │  OUTPUT     greedy-identical sha256 bb06e711bc498bb9 (triple-hitsfirst's
  │             run; this tree's code is byte-identical to that branch)
  │
  └──────────────────────────────────────────────────────────────────────
```

### triple-all-fastest-newtip

```
  ┌──────────────────────────────────────────────────────────────────────
  │
  │  BRANCH     triple-all-fastest-newtip                     POSITIVE
  │
  │  WHAT       the ten-lever stack REBUILT ON THE NEW TIP from its manifest:
  │             base triple-tip-2026-09-16 (12997e9c8), one commit per lever
  │             in stack-on-newtip.manifest order, every conflict hunk decided
  │             on purpose and recorded, so build_stack.sh reproduces this
  │             tree from the manifest alone. Nine levers on, hits-first off,
  │             victim order used-ascending (the measured default).
  │
  │  LATEST     round 4 · 2026-09-17 · BEATS · +4.65 % (kept +6.87 %) ·
  │             2026-09-17-R4-RESULT-hitsfirst-pool-and-the-newtip-stack-beat-the-tip-readahead-order-loses-eleven.md
  │
  │  GEN        +4.65 % (kept +6.87 %)  floor 2.71 % (no-drop)  sign 4/4  n=4
  │             raw: arm median 10.14 t/s vs tip median 9.65 t/s, gen_steady
  │             at min across 4096/6144; the delta is each run against the
  │             mean of its bracketing TIP runs, not against that median
  │             control = triple-tip-2026-09-16 @12997e9c, interleaved, one TIP
  │             bracket per cycle
  │             session = 2026-09-17 09:50-10:57Z · DGX Spark GB10 ·
  │             native 24.94 MB .text · lean regime 4096/6144
  │  PREFILL    reported, not a verdict: 136.25 t/s min-across median vs the
  │             tip's 80.73 t/s, n=4. Round 4 makes no prefill verdict.
  │
  │  VERDICT    POSITIVE - beats the new tip on both readings, +4.65 % no-drop and
  │             +6.87 % kept, sign 4 of 4. First measurement of this tree, and it
  │             compiled first time at 24.94 MB .text, which refuted prediction P4.
  │             ⚠ Both levers round 4 measured solo are OFF in this tree by default:
  │             hits-first (+6.47 % solo, the wrong side) and the readahead arm
  │             (-11.38 % solo, the right side; corrected 11:25Z from a source read
  │             at the sealed sha, the card had said ON). Nothing here attributes the
  │             stack's number to any lever; the seal forbids it. ROUND 7 flips both
  │             switches on this same binary.
  │
  │  SWITCH     ten levers, see the table below; plus
  │             DS4_CUDA_PREFETCH_SWEEP_ORDER=1 restores the sweep-aware order
  │  OUTPUT     not re-run
  │
  └──────────────────────────────────────────────────────────────────────
```

### triple-hitsfirst

```
  ┌──────────────────────────────────────────────────────────────────────
  │
  │  BRANCH     triple-hitsfirst                              POSITIVE
  │
  │  WHAT       launches gate/up/down for the experts already resident
  │             before the miss reads land, so a token does not stall on
  │             the slowest read in the batch
  │
  │  LATEST     round 8 · 2026-09-17 · BEATS · +12.31 % 4/4 (floor 10.69;
  │             +11.54 with the cold first run excluded) ·
  │             2026-09-17-R8-RESULT-hitsfirst-alone-beats-through-the-noise-the-winners-tree-does-not.md
  │             round 4 · 2026-09-17 · BEATS · +6.47 % (kept +7.93 %) ·
  │             2026-09-17-R4-RESULT-hitsfirst-pool-and-the-newtip-stack-beat-the-tip-readahead-order-loses-eleven.md
  │
  │  GEN        +12.31 %  floor 10.69 %  sign 4/4  n=4
  │             sensitivity, cold first control run excluded: +11.54 % against a
  │             7.17 % floor, still BEATS (post hoc, labelled as such)
  │             raw: arm min-across median 10.34 t/s (10.49 · 9.43 · 10.38 ·
  │             10.29) vs the tip's 8.96 t/s (8.61 cold · 9.53 · 8.87 · 9.49 ·
  │             8.96). gen_steady at min across 4096/6144. The delta is each run
  │             against its bracketing TIP runs, not against that median
  │             ⚠ the box was not quiet: the five tip runs spanned 8.61 to 9.53
  │             and four arm runs ended at load 3.2 to 4.0 with no vitest alive,
  │             which is what widened the floor from round 4's 2.71 to 10.69
  │             control = triple-tip-2026-09-16 @12997e9c, interleaved
  │             session = 2026-09-17 13:24-14:37Z · DGX Spark GB10 ·
  │             native 24.89 MB .text (tip 24.86 MB) · lean regime 4096/6144
  │  PREFILL    reported, not a verdict: 86.55 t/s min-across median vs the
  │             tip's 84.34 t/s, n=4. Round 8 makes no prefill verdict.
  │
  │  VERDICT    POSITIVE - the only arm of round 8 to clear the floor, and the
  │             only tree with two sealed wins and no sealed loss. Every one of
  │             its four runs read above every one of the five control runs, so
  │             the win does not depend on which control run is kept. By the
  │             standing rule that the default follows the measured number, the
  │             shipped default is now the new tip plus this one lever
  │             (round 8 section 5), provisional on a quiet round 9
  │
  │  SWITCH     DS4_CUDA_HITS_FIRST=1 (default OFF in the stack; =0 is wait-then-launch)
  │             DS4_CUDA_HITS_FIRST_STAGED=0 keeps hits-first, single-stage read order
  │  OUTPUT     greedy-identical sha256 bb06e711bc498bb9
  │
  └──────────────────────────────────────────────────────────────────────
```

### triple-pool

```
  ┌──────────────────────────────────────────────────────────────────────
  │
  │  BRANCH     triple-pool                                   POSITIVE
  │
  │  WHAT       serves expert reads from a parallel SSD pool instead of one
  │             serial reader, with an io_uring O_DIRECT ring in front of the
  │             pool so queue depth is a switch, not the worker count.
  │
  │  LATEST     round 4 · 2026-09-17 · BEATS · +7.27 % (kept +7.75 %) ·
  │             2026-09-17-R4-RESULT-hitsfirst-pool-and-the-newtip-stack-beat-the-tip-readahead-order-loses-eleven.md
  │             round 8 · 2026-09-17 · TIES · +8.56 % 3/4 under a 10.69 floor,
  │             two runs in a loaded window - round 4's +7.27 stands ·
  │             2026-09-17-R8-RESULT-hitsfirst-alone-beats-through-the-noise-the-winners-tree-does-not.md
  │
  │  GEN        round 4, the branch's number: +7.27 % (kept +7.75 %)
  │             floor 2.71 % (no-drop)  sign 3/4  n=4
  │             raw: arm median 10.39 t/s vs tip median 9.65 t/s, gen_steady
  │             at min across 4096/6144; the delta is each run against the
  │             mean of its bracketing TIP runs, not against that median
  │             control = triple-tip-2026-09-16 @12997e9c, interleaved, one TIP
  │             bracket per cycle
  │             session = 2026-09-17 09:50-10:57Z · DGX Spark GB10 ·
  │             native 24.89 MB .text · lean regime 4096/6144
  │             ─── round 8, TIES, neither confirms nor retracts the above:
  │             +8.56 %  floor 10.69 %  sign 3/4  n=4 (sensitivity, cold first
  │             control run excluded: +3.31 % against a 7.17 % floor, TIES)
  │             raw: arm min-across median 10.00 t/s (10.65 · 10.47 · 8.98 ·
  │             9.53) vs the tip's 8.96 t/s (8.61 cold · 9.53 · 8.87 · 9.49 ·
  │             8.96). Two runs read as round 4 did, two fell in the loaded
  │             window. ⚠ the box was not quiet: the five tip runs spanned
  │             8.61 to 9.53 and four arm runs ended at load 3.2 to 4.0 with no
  │             vitest alive, which widened the floor from 2.71 to 10.69
  │             control = triple-tip-2026-09-16 @12997e9c, interleaved
  │             session = 2026-09-17 13:24-14:37Z · DGX Spark GB10 ·
  │             native 24.89 MB .text (tip 24.86 MB) · lean regime 4096/6144
  │  PREFILL    reported, not a verdict: round 4, 87.25 t/s min-across median
  │             vs the tip's 80.73 t/s, n=4; round 8, 86.47 t/s vs 84.34 t/s,
  │             n=4. Neither round makes a prefill verdict.
  │
  │  VERDICT    POSITIVE on round 4, +7.27 % no-drop and +7.75 % kept, and that
  │             stays the branch's number. ⚠ Round 4's first run read 9.24 t/s,
  │             below the tip, and the next three read 10.34 to 10.47; a cold
  │             io_uring ring and an ordinary straggler look identical there and
  │             round 4 could not tell them apart. Round 8 ran the branch warm
  │             and read +8.56 % at 3 of 4 under a 10.69 floor: two runs at 10.47
  │             to 10.65, as round 4, and two at 8.98 to 9.53 in the loaded
  │             window. That TIES, so it neither confirms nor retracts round 4.
  │             Round 9 re-measures it on a quiet box
  │
  │  SWITCH     DS4_CUDA_FETCH_QD=<n> ring queue depth, default 64, clamped 8-512
  │             DS4_CUDA_FETCH_URING=0 falls back to the pread pool
  │             DS4_CUDA_FETCH_BUFFERED=1 forces buffered reads
  │             DS4_CUDA_STREAMING_EXPERT_PREAD_THREADS=<n> pool workers, default 8, cap 16
  │             DS4_CUDA_STREAMING_EXPERT_PREAD_POOL=0 restores the serial path
  │  OUTPUT     not re-run
  │
  └──────────────────────────────────────────────────────────────────────
```

### triple-prefetch-pool

```
  ┌──────────────────────────────────────────────────────────────────────
  │
  │  BRANCH     triple-prefetch-pool                          POSITIVE
  │
  │  WHAT       cuts the prefill read-ahead into chunked tasks for a pool of
  │             readers with their own pinned buffers and upload streams;
  │             the single reader stays as the fallback when the pool declines
  │
  │  LATEST     round 4 · 2026-09-17 · PARTIAL · +2.28 % (kept +2.28 %) ·
  │             2026-09-17-R4-RESULT-hitsfirst-pool-and-the-newtip-stack-beat-the-tip-readahead-order-loses-eleven.md
  │
  │  GEN        +2.28 % (kept +2.28 %)  floor 2.71 % (no-drop)  sign 3/3  n=3
  │             raw: arm median 9.87 t/s vs tip median 9.65 t/s, gen_steady
  │             at min across 4096/6144; the delta is each run against the
  │             mean of its bracketing TIP runs, not against that median
  │             control = triple-tip-2026-09-16 @12997e9c, interleaved, one TIP
  │             bracket per cycle
  │             session = 2026-09-17 09:50-10:57Z · DGX Spark GB10 ·
  │             native 24.87 MB .text · lean regime 4096/6144
  │  PREFILL    reported, not a verdict: 138.67 t/s min-across median vs the
  │             tip's 80.73 t/s, n=3. Round 4 makes no prefill verdict.
  │
  │  VERDICT    POSITIVE, PARTIAL - the kept reading calls it BEATS on a single
  │             survivor against a 0.21 % floor; the no-drop reading's +2.28 % TIES
  │             a 2.71 % floor. Sign 3 of 3, one run dropped for box contention
  │             (load 3.28 at the end of r4_prefetchpool_2, not the straggler rule).
  │             Read it as a small real positive that THIS round cannot clear its
  │             floor with. It needs repeats, not a new design.
  │
  │  SWITCH     DS4_CUDA_SSD_PREFETCH_CHUNK_MB, default 8, cap 64; 0 is ignored,
  │             so the chunking has no off arm. DS4_CUDA_SSD_PREFETCH_POOL=0 or
  │             DS4_CUDA_SSD_PREFETCH_THREADS=1 reverts the whole pool
  │  OUTPUT     greedy-identical sha256 2f2dd7f89d107bbc (33,527 bytes, 12 runs)
  │
  └──────────────────────────────────────────────────────────────────────
```

## MEASURED CORRECT, WORTH ZERO ON THE CLOCK

### triple-climbingfibre

```
  ┌──────────────────────────────────────────────────────────────────────
  │
  │  BRANCH     triple-climbingfibre                              WORTH ZERO
  │
  │  WHAT       a front cache in front of the frozen model that learns from the
  │             speculative verifier's own REJECTIONS, online, with a retention
  │             horizon. It proposes one draft token; it never injects anything
  │
  │  LATEST     round 5 - 2026-09-17 - TIES - +1.28 % 4/4 under a 5.69 floor -
  │             2026-09-17-R5-RESULT-seven-ties-under-a-noisy-floor.md
  │
  │  GEN        +1.28 %  floor 5.69 %  sign 4/4  n=4
  │             raw: arm min-across median 9.58 t/s (9.11 / 9.83 / 9.52 / 9.64)
  │             vs the tip's 9.60 t/s (9.79 / 9.42 / 9.14 / 9.66 / 9.60). gen_steady at min
  │             across 4096/6144, each run against its bracketing TIP runs
  │             control = triple-tip-2026-09-16 @12997e9c, interleaved
  │             session = 2026-09-17 11:58-13:10Z - DGX Spark GB10 -
  │             native 24.86 MB .text (tip 24.86 MB) - lean regime 4096/6144
  │             the floor is 5.69 % because the five TIP controls swung 9.14 to
  │             9.79 t/s, and both the sealed legacy rule and the new rule agree
  │             that nothing clears it
  │  PREFILL    reported, not a verdict: 83.08 t/s min-across median vs the
  │             tip's 84.61 t/s, n=4. Round 5 makes no prefill verdict.
  │
  │  VERDICT    WORTH ZERO - +1.28 % at 4 of 4 inside a 5.69 % floor. The first
  │             real A/B this branch has ever had against a control, and it says the
  │             rejection-learned front cache costs nothing and buys nothing
  │             measurable at this context and this batch shape
  │
  │  SWITCH     DS4_CLIMBINGFIBRE=1, default 0. Off allocates nothing, never
  │             learns and never proposes
  │             DS4_CLIMBINGFIBRE_HORIZON=session|day|week|forever|<steps>
  │             _DECAY=hard|half|none · _NGRAM 4 · _CAP 16384
  │  OUTPUT     not re-run
  │
  └──────────────────────────────────────────────────────────────────────
```

### triple-draincut

```
  ┌──────────────────────────────────────────────────────────────────────
  │
  │  BRANCH     triple-draincut                                   WORTH ZERO
  │
  │  WHAT       reads the router's selected expert ids back on an event
  │             instead of a blocking cudaMemcpy, so the host no longer
  │             waits on the shared expert - 40 device drains per token
  │
  │  LATEST     round 5 - 2026-09-17 - TIES - +1.48 % 3/4 under a 5.69 floor -
  │             2026-09-17-R5-RESULT-seven-ties-under-a-noisy-floor.md
  │
  │  GEN        +1.48 %  floor 5.69 %  sign 3/4  n=4
  │             raw: arm min-across median 9.62 t/s (9.88 / 9.59 / 9.33 / 9.64)
  │             vs the tip's 9.60 t/s (9.79 / 9.42 / 9.14 / 9.66 / 9.60). gen_steady at min
  │             across 4096/6144, each run against its bracketing TIP runs
  │             control = triple-tip-2026-09-16 @12997e9c, interleaved
  │             session = 2026-09-17 11:58-13:10Z - DGX Spark GB10 -
  │             native 24.87 MB .text (tip 24.86 MB) - lean regime 4096/6144
  │             the floor is 5.69 % because the five TIP controls swung 9.14 to
  │             9.79 t/s, and both the sealed legacy rule and the new rule agree
  │             that nothing clears it
  │  PREFILL    reported, not a verdict: 87.52 t/s min-across median vs the
  │             tip's 84.61 t/s, n=4. Round 5 makes no prefill verdict.
  │
  │  VERDICT    WORTH ZERO - +1.48 % at 3 of 4 inside a 5.69 % floor. The lever
  │             is correct and the delta buys nothing measurable. It would sit inside
  │             round 4's tighter 2.71 % floor as well, so a quieter box is unlikely
  │             to change the word
  │
  │  SWITCH     ON by default. DS4_CUDA_SELECTED_DRAIN_SYNC=1 restores the
  │             blocking read - that is the OFF arm
  │  OUTPUT     greedy-identical sha256 bb06e711bc498bb9 (2026-09-15, before the rebase)
  │
  └──────────────────────────────────────────────────────────────────────
```

### triple-engram-lead

```
  ┌──────────────────────────────────────────────────────────────────────
  │
  │  BRANCH     triple-engram-lead                                WORTH ZERO
  │
  │  WHAT       starts the Engram table read one token ahead of first use,
  │             speculating on the current step's own argmax, so the read
  │             overlaps the step instead of landing exposed inside it
  │
  │  LATEST     round 5 - 2026-09-17 - TIES - -1.04 % 1/4 under a 5.69 floor -
  │             2026-09-17-R5-RESULT-seven-ties-under-a-noisy-floor.md
  │
  │  GEN        -1.04 %  floor 5.69 %  sign 1/4  n=4
  │             raw: arm min-across median 9.43 t/s (9.45 / 9.42 / 9.35 / 9.48)
  │             vs the tip's 9.60 t/s (9.79 / 9.42 / 9.14 / 9.66 / 9.60). gen_steady at min
  │             across 4096/6144, each run against its bracketing TIP runs
  │             control = triple-tip-2026-09-16 @12997e9c, interleaved
  │             session = 2026-09-17 11:58-13:10Z - DGX Spark GB10 -
  │             native 24.86 MB .text (tip 24.86 MB) - lean regime 4096/6144
  │             the floor is 5.69 % because the five TIP controls swung 9.14 to
  │             9.79 t/s, and both the sealed legacy rule and the new rule agree
  │             that nothing clears it
  │  PREFILL    reported, not a verdict: 86.94 t/s min-across median vs the
  │             tip's 84.61 t/s, n=4. Round 5 makes no prefill verdict.
  │
  │  VERDICT    WORTH ZERO - -1.04 % at 1 of 4 positive, the lowest delta and the
  │             weakest sign of the round, and still inside a 5.69 % floor. Measured
  │             correct and worth nothing. It is not carded NEGATIVE, because -1.04 %
  │             does not clear the floor in either direction
  │
  │  SWITCH     DS4_V41_ENGRAM_LEAD_OFF - the lead is ON by default; set the
  │             variable to anything to turn it off. Read once per process,
  │             all-or-nothing
  │  OUTPUT     not re-run
  │
  └──────────────────────────────────────────────────────────────────────
```

### triple-engram-prestage

```
  ┌──────────────────────────────────────────────────────────────────────
  │
  │  BRANCH     triple-engram-prestage                            WORTH ZERO
  │
  │  WHAT       moves the host-side Engram row lookup out of the forward pass
  │             into a prepare phase that runs before it, so the forward carries
  │             no disk read and no hash of its own
  │
  │  LATEST     round 6 · 2026-09-17 · TIES · -1.40 % ·
  │             2026-09-17-R6-RESULT-the-superseded-six-lever-stack-beats-the-tip-three-ties.md
  │
  │  GEN        -1.40 %  floor 3.35 %  sign 0/3  n=3
  │             raw: arm min-across median 9.54 t/s (9.62 · 9.50 · 9.57 · 9.52)
  │             vs the tip's 9.69 t/s (9.67 · 9.56 · 9.88 · 9.70 kept, 9.45
  │             contention-dropped; the result file's figure line reads 9.67).
  │             gen_steady at min across 4096/6144. The delta is each run
  │             against its bracketing TIP runs, not against that median
  │             control = triple-tip-2026-09-16 @12997e9c, interleaved
  │             session = 2026-09-17 11:12-11:58Z · DGX Spark GB10 ·
  │             native 24.85 MB .text (tip 24.86 MB) · lean regime 4096/6144
  │  PREFILL    reported, not a verdict: 87.18 t/s min-across median vs the
  │             tip's 89.16 t/s, n=4. Round 6 makes no prefill verdict.
  │
  │  VERDICT    WORTH ZERO - inside the floor at -1.40 %, repeat range -2.8 to
  │             -1.2 %, every repeat negative. That is what the card predicted
  │             for a precondition that takes host I/O out of the forward
  │             without hiding it behind anything. Both rules agree
  │
  │  SWITCH     DS4_ENGRAM_PRESTAGE=1 enables the prepare; unset or 0 is the
  │             shipped inline path. DS4_ENGRAM_PRESTAGE_DEBUG=1 names every
  │             demand-read miss. Read per call, not latched
  │  OUTPUT     not re-run
  │
  └──────────────────────────────────────────────────────────────────────
```

### triple-engram-read-threads

```
  ┌──────────────────────────────────────────────────────────────────────
  │
  │  BRANCH     triple-engram-read-threads                  WORTH ZERO
  │
  │  WHAT       takes the DECODE step's Engram read off the batch machinery:
  │             one token now goes straight to the serial reader, with no
  │             malloc, no qsort and no pool dispatch. The wide PREFILL read
  │             keeps the one-wave rule the branch was named for.
  │
  │  LATEST     round 4 · 2026-09-17 · TIES · -1.39 % (kept +0.31 %) ·
  │             2026-09-17-R4-RESULT-hitsfirst-pool-and-the-newtip-stack-beat-the-tip-readahead-order-loses-eleven.md
  │
  │  GEN        -1.39 % (kept +0.31 %)  floor 2.71 % (no-drop)  sign 1/4  n=4
  │             raw: arm median 9.60 t/s vs tip median 9.65 t/s, gen_steady
  │             at min across 4096/6144; the delta is each run against the
  │             mean of its bracketing TIP runs, not against that median
  │             control = triple-tip-2026-09-16 @12997e9c, interleaved, one TIP
  │             bracket per cycle
  │             session = 2026-09-17 09:50-10:57Z · DGX Spark GB10 ·
  │             native 24.86 MB .text · lean regime 4096/6144
  │  PREFILL    reported, not a verdict: 82.62 t/s min-across median vs the
  │             tip's 80.73 t/s, n=4. Round 4 makes no prefill verdict.
  │
  │  VERDICT    WORTH ZERO - confirmed on the new tip, with its own CUDA build at
  │             last. The no-drop reading is -1.39 % at sign 1 of 4, inside a 2.71 %
  │             floor: TIES. ⚠ The kept reading says +0.31 % BEATS on one surviving
  │             run against a floor that two dropped TIP controls shrank to
  │             0.21 %, and where the two readings disagree the no-drop column
  │             is the one to quote (result section 3).
  │             NOT IT at round 1, not it here. OWED: the one-wave PREFILL read.
  │
  │  SWITCH     ⚠ NEITHER KNOB REACHES THE DECODE READ ANY MORE. Both govern
  │             the wide prefill read only, where the defaults are unchanged.
  │             DS4_ENGRAM_ROWS_PER_READER=1|2, default 1 (one wave); 2 is the
  │             old two-round divisor, the control arm in the same binary.
  │             DS4_ENGRAM_READ_THREADS=<n> overrides the reader count;
  │             0 clamps to one reader, the serial path.
  │  OUTPUT     not re-run (the previous revision's gates bb06e711bc498bb9 /
  │             2f2dd7f89d107bbc / 652dcda32c176cab belong to that revision)
  │
  └──────────────────────────────────────────────────────────────────────
```

### triple-hippocampal-warmset

```
  ┌──────────────────────────────────────────────────────────────────────
  │
  │  BRANCH     triple-hippocampal-warmset                        WORTH ZERO
  │
  │  WHAT       gives the persisted expert hot list a longer horizon: carried
  │             free inside a day, halved once per day boundary, promoted to a
  │             durable set when a (layer, expert) pair is demanded on three
  │             distinct days
  │
  │  LATEST     round 6 · 2026-09-17 · TIES · -1.54 % ·
  │             2026-09-17-R6-RESULT-the-superseded-six-lever-stack-beats-the-tip-three-ties.md
  │
  │  GEN        -1.54 %  floor 3.35 %  sign 0/3  n=3
  │             raw: arm min-across median 9.57 t/s (9.76 · 9.48 · 9.38 · 9.66)
  │             vs the tip's 9.69 t/s (9.67 · 9.56 · 9.88 · 9.70 kept, 9.45
  │             contention-dropped; the result file's figure line reads 9.67).
  │             gen_steady at min across 4096/6144. The delta is each run
  │             against its bracketing TIP runs, not against that median
  │             control = triple-tip-2026-09-16 @12997e9c, interleaved
  │             session = 2026-09-17 11:12-11:58Z · DGX Spark GB10 ·
  │             native 24.89 MB .text (tip 24.86 MB) · lean regime 4096/6144
  │  PREFILL    reported, not a verdict: 83.97 t/s min-across median vs the
  │             tip's 89.16 t/s, n=4. Round 6 makes no prefill verdict.
  │
  │  VERDICT    WORTH ZERO - inside the floor at -1.54 %, repeat range -3.5 to
  │             -1.3 %, every repeat negative. The largest negative of the three
  │             tied arms, and still inside the floor. Both rules agree
  │
  │  SWITCH     DS4_WARMSET_TIER=session|day, default session, which IS
  │             today's behaviour: one file, halved at every load
  │             DS4_WARMSET_DAYS 3 · _MAX <n> · _DIR <dir> · _PROFILE=1
  │  OUTPUT     not re-run
  │
  └──────────────────────────────────────────────────────────────────────
```

### triple-lfu-offsetkey

```
  ┌──────────────────────────────────────────────────────────────────────
  │
  │  BRANCH     triple-lfu-offsetkey                              WORTH ZERO
  │
  │  WHAT       ports upstream NeutronStar's host expert cache: byte ranges
  │             keyed by FILE OFFSET, a 16-way table, a per-slot uses counter
  │             halved every 4096 inserts. Three arms out of one binary isolate
  │             the eviction POLICY on one structure
  │
  │  LATEST     round 6 · 2026-09-17 · TIES · -0.20 % ·
  │             2026-09-17-R6-RESULT-the-superseded-six-lever-stack-beats-the-tip-three-ties.md
  │
  │  GEN        -0.20 %  floor 3.35 %  sign 1/3  n=3
  │             raw: arm min-across median 9.71 t/s (9.52 · 9.85 · 9.64 · 9.77)
  │             vs the tip's 9.69 t/s (9.67 · 9.56 · 9.88 · 9.70 kept, 9.45
  │             contention-dropped; the result file's figure line reads 9.67).
  │             gen_steady at min across 4096/6144. The delta is each run
  │             against its bracketing TIP runs, not against that median
  │             control = triple-tip-2026-09-16 @12997e9c, interleaved
  │             session = 2026-09-17 11:12-11:58Z · DGX Spark GB10 ·
  │             native 24.87 MB .text (tip 24.86 MB) · lean regime 4096/6144
  │  PREFILL    reported, not a verdict: 85.90 t/s min-across median vs the
  │             tip's 89.16 t/s, n=4. Round 6 makes no prefill verdict.
  │
  │  VERDICT    WORTH ZERO - inside the floor, with a repeat range of -0.8 to
  │             +2.4 % and one repeat of three positive. The CUDA build that
  │             was owed here exists and ran; the policy costs nothing and
  │             buys nothing on this tip. Both rules give this verdict
  │
  │  SWITCH     DS4_EXPERT_CACHE_MODE=hotlist|offsetkey|offsetkey-lru, default
  │             hotlist, in which the cache is inert: it never allocates, probes
  │             or inserts. DS4_CUDA_HOST_EXPERT_CACHE_GB=<GiB> sizes it; unset,
  │             it takes the configured expert-cache byte size
  │  OUTPUT     not re-run. Residency changes WHEN a byte arrives, never WHICH
  │
  └──────────────────────────────────────────────────────────────────────
```

### triple-pagecache

```
  ┌──────────────────────────────────────────────────────────────────────
  │
  │  BRANCH     triple-pagecache                                  WORTH ZERO
  │
  │  WHAT       releases staged model pages madvise-then-fadvise so the drop
  │             actually lands, and hints WILLNEED over a layer's miss ranges
  │             on the buffered read path
  │
  │  LATEST     round 5 - 2026-09-17 - TIES - +1.47 % 4/4 under a 5.69 floor -
  │             2026-09-17-R5-RESULT-seven-ties-under-a-noisy-floor.md
  │
  │  GEN        +1.47 %  floor 5.69 %  sign 4/4  n=4
  │             raw: arm min-across median 9.73 t/s (9.66 / 9.85 / 9.51 / 9.80)
  │             vs the tip's 9.60 t/s (9.79 / 9.42 / 9.14 / 9.66 / 9.60). gen_steady at min
  │             across 4096/6144, each run against its bracketing TIP runs
  │             control = triple-tip-2026-09-16 @12997e9c, interleaved
  │             session = 2026-09-17 11:58-13:10Z - DGX Spark GB10 -
  │             native 24.86 MB .text (tip 24.86 MB) - lean regime 4096/6144
  │             the floor is 5.69 % because the five TIP controls swung 9.14 to
  │             9.79 t/s, and both the sealed legacy rule and the new rule agree
  │             that nothing clears it
  │  PREFILL    reported, not a verdict: 83.55 t/s min-across median vs the
  │             tip's 84.61 t/s, n=4. Round 5 makes no prefill verdict.
  │
  │  VERDICT    WORTH ZERO - +1.47 % at 4 of 4 inside a 5.69 % floor, which is the
  │             same word this card already carried from a different pass. The memory
  │             argument is unchanged and is still an argument about the resident set,
  │             not a measured speed
  │
  │  SWITCH     DS4_CUDA_KEEP_MODEL_PAGES=1 disables both drops
  │             DS4_CUDA_NO_EXPERT_READAHEAD=1 disables the hint
  │  OUTPUT     greedy-identical sha256 bb06e711bc498bb9 (long d3355c94c70a4bb1,
  │             2026-09-15, before the rebase)
  │
  └──────────────────────────────────────────────────────────────────────
```

### triple-winners

```
  ┌──────────────────────────────────────────────────────────────────────
  │
  │  BRANCH     triple-winners                                WORTH ZERO
  │
  │  WHAT       the three levers round 4 measured as winners solo on the
  │             new tip, stacked and nothing else: pool (+7.27 %), hitsfirst
  │             (+6.47 %) and prefetch-pool (+2.28 %), with HITSFIRST ON.
  │             Round 8 measured it level with the tip: the combination costs
  │             what each part wins
  │
  │  LATEST     round 8 · 2026-09-17 · TIES · +1.54 % (-0.54 sensitivity) ·
  │             level with the tip while its parts read +8 to +12 in the same
  │             session ·
  │             2026-09-17-R8-RESULT-hitsfirst-alone-beats-through-the-noise-the-winners-tree-does-not.md
  │
  │  GEN        +1.54 %  floor 10.69 %  sign 2/3  n=3 of 4 paired
  │             sensitivity, cold first control run excluded: -0.54 % against a
  │             7.17 % floor, TIES either way (post hoc, labelled as such)
  │             raw: arm min-across median 9.18 t/s (9.21 · 9.42 · 9.13 · 9.15)
  │             vs the tip's 8.96 t/s (8.61 cold · 9.53 · 8.87 · 9.49 · 8.96).
  │             gen_steady at min across 4096/6144. The delta is each run
  │             against its bracketing TIP runs, not against that median
  │             ⚠ the box was not quiet: the five tip runs spanned 8.61 to 9.53
  │             and four arm runs ended at load 3.2 to 4.0 with no vitest alive,
  │             which is what widened the floor to 10.69. But this arm's own
  │             four runs span only 9.13 to 9.42, so its LEVEL reading is not a
  │             noise artifact; the noise bounds how strongly it can be called
  │             a loss, not whether it won
  │             control = triple-tip-2026-09-16 @12997e9c, interleaved
  │             session = 2026-09-17 13:24-14:37Z · DGX Spark GB10 ·
  │             native 24.90 MB .text (tip 24.86 MB) · lean regime 4096/6144
  │  PREFILL    reported, not a verdict: 135.09 t/s min-across median vs the
  │             tip's 84.34 t/s, n=4. Round 8 makes no prefill verdict.
  │
  │  VERDICT    WORTH ZERO - built correct, compiled, measured, and level with
  │             the tip. The combination costs what each part wins: pool read
  │             10.47 to 10.65 on its clean runs and hitsfirst read 10.29 to
  │             10.49, all four of them above every control run, in this same
  │             session, while this tree that carries both read 9.13 to 9.42.
  │             So the answer to "why not combine pool and hitsfirst and
  │             prefetch-pool" is measured, and it is no. The two pread-path
  │             levers fight; the claim-ledger interaction the build lane
  │             carried is the first suspect and round 8 does not prove it
  │
  │  SWITCH     DS4_CUDA_HITS_FIRST=0 turns hits-first off (ON here, the
  │             lever's own default); DS4_CUDA_HITS_FIRST_STAGED=0 keeps it
  │             single-stage
  │             DS4_CUDA_FETCH_URING=0 falls back to the pread pool;
  │             DS4_CUDA_STREAMING_EXPERT_PREAD_POOL=0 restores the serial path;
  │             DS4_CUDA_FETCH_QD=<n> ring depth, default 64, clamped 8-512
  │             DS4_CUDA_SSD_PREFETCH_POOL=0 or DS4_CUDA_SSD_PREFETCH_THREADS=1
  │             reverts the prefetch pool; DS4_CUDA_SSD_PREFETCH_CHUNK_MB,
  │             default 8, cap 64
  │  OUTPUT     not re-run
  │
  └──────────────────────────────────────────────────────────────────────
```

## NEGATIVELY MEASURED OR KILLED

### triple-granule

```
  ┌──────────────────────────────────────────────────────────────────────
  │
  │  BRANCH     triple-granule                                      NEGATIVE
  │
  │  WHAT       offline, CPU only: does a sparse-distributed (SDR) expansion
  │             make a cheap front-cache lookup more discriminative than a
  │             plain hash at the same address budget? Run on the committed
  │             track-3 token streams
  │
  │  LATEST     round 5 - 2026-09-17 - TIES - -0.04 % 2/4 under a 5.69 floor -
  │             2026-09-17-R5-RESULT-seven-ties-under-a-noisy-floor.md
  │             (the NEGATIVE word above stays, and is the branch's own offline
  │             run of record on a different quantity: hash-vs-SDR discrimination)
  │
  │  GEN        -0.04 %  floor 5.69 %  sign 2/4  n=4  (legacy rule +1.06)
  │             raw: arm min-across median 9.51 t/s (9.74 / 9.01 / 9.50 / 9.52)
  │             vs the tip's 9.60 t/s (9.79 / 9.42 / 9.14 / 9.66 / 9.60). gen_steady at min
  │             across 4096/6144, each run against its bracketing TIP runs
  │             control = triple-tip-2026-09-16 @12997e9c, interleaved
  │             session = 2026-09-17 11:58-13:10Z - DGX Spark GB10 -
  │             native 24.86 MB .text (tip 24.86 MB) - lean regime 4096/6144
  │             the floor is 5.69 % because the five TIP controls swung 9.14 to
  │             9.79 t/s, and both the sealed legacy rule and the new rule agree
  │             that nothing clears it
  │  PREFILL    reported, not a verdict: 85.98 t/s min-across median vs the
  │             tip's 84.61 t/s, n=4. Round 5 makes no prefill verdict.
  │
  │  VERDICT    NEGATIVE - unchanged, and killed by its own offline measurement of
  │             discrimination, not by round 5. The round's bench delta TIES: -0.04 %
  │             at 2 of 4 inside a 5.69 % floor, which is the reading a branch whose
  │             engine files match the tip should give
  │
  │  SWITCH     NONE - an offline experiment under experiments/granule/, no
  │             DS4_* knob exists and none is proposed
  │  OUTPUT     not re-run - the engine files are byte-identical to the tip
  │
  └──────────────────────────────────────────────────────────────────────
```

### triple-hotlist

```
  ┌──────────────────────────────────────────────────────────────────────
  │
  │  BRANCH     triple-hotlist                                NEGATIVE
  │
  │  WHAT       seeds the next session's SSD expert cache from the previous
  │             run's demand, so the opening tokens start WARM. A warm-up
  │             device, not a selection device.
  │
  │  LATEST     round 4 · 2026-09-17 · LOSES · -2.73 % (kept -3.34 %) ·
  │             2026-09-17-R4-RESULT-hitsfirst-pool-and-the-newtip-stack-beat-the-tip-readahead-order-loses-eleven.md
  │
  │  GEN        -2.73 % (kept -3.34 %)  floor 2.71 % (no-drop)  sign 0/4  n=4
  │             raw: arm median 9.42 t/s vs tip median 9.65 t/s, gen_steady
  │             at min across 4096/6144; the delta is each run against the
  │             mean of its bracketing TIP runs, not against that median
  │             control = triple-tip-2026-09-16 @12997e9c, interleaved, one TIP
  │             bracket per cycle
  │             session = 2026-09-17 09:50-10:57Z · DGX Spark GB10 ·
  │             native 24.90 MB .text · lean regime 4096/6144
  │  PREFILL    reported, not a verdict: 82.25 t/s min-across median vs the
  │             tip's 80.73 t/s, n=4. Round 4 makes no prefill verdict.
  │
  │  VERDICT    NEGATIVE - loses on both readings, -2.73 % no-drop and -3.34 % kept,
  │             sign 0 of 4: not one repeat read above the tip. The victim-walk
  │             repair costs on this tip. The two readings agree, so nothing here
  │             turns on the straggler rule. The 2026-09-15 solo -1.9 %, which sat
  │             inside its own floor and decided nothing, is superseded by this.
  │
  │  SWITCH     DS4_CUDA_EXPERT_HOTLIST_WRITE=<file> (0 disables the writer)
  │             DS4_CUDA_EXPERT_HOTLIST_DECAY=none|halve|quarter|eighth|shift:<n>, default halve
  │             DS4_CUDA_EXPERT_HOTLIST_ROLLING=1 runs the mine/score measurement
  │             DS4_METAL_DISABLE_STREAMING_EXPERT_HOTLIST=1 disables seeding
  │  OUTPUT     greedy-identical sha256 bb06e711bc498bb9 (three gate runs, one sha)
  │
  └──────────────────────────────────────────────────────────────────────
```

### triple-prefill-readahead-order

```
  ┌──────────────────────────────────────────────────────────────────────
  │
  │  BRANCH     triple-prefill-readahead-order                NEGATIVE
  │
  │  WHAT       reorders the prefill read-ahead's victim choice and holds the
  │             earliest layers of the scan, so the experts decode needs first
  │             are the last ones evicted: the prefill hit list survives into
  │             decode. Residency policy only, no byte of cache added.
  │
  │  LATEST     round 4 · 2026-09-17 · LOSES · -11.38 % (kept -11.38 %) ·
  │             2026-09-17-R4-RESULT-hitsfirst-pool-and-the-newtip-stack-beat-the-tip-readahead-order-loses-eleven.md
  │
  │  GEN        -11.38 % (kept -11.38 %)  floor 2.71 % (no-drop)  sign 0/4  n=4
  │             raw: arm median 8.55 t/s vs tip median 9.65 t/s, gen_steady
  │             at min across 4096/6144; the delta is each run against the
  │             mean of its bracketing TIP runs, not against that median
  │             control = triple-tip-2026-09-16 @12997e9c, interleaved, one TIP
  │             bracket per cycle
  │             session = 2026-09-17 09:50-10:57Z · DGX Spark GB10 ·
  │             native 24.88 MB .text · lean regime 4096/6144
  │  PREFILL    reported, not a verdict: 86.34 t/s min-across median vs the
  │             tip's 80.73 t/s, n=4. Round 4 makes no prefill verdict.
  │
  │  VERDICT    NEGATIVE - loses by eleven percent, -11.38 % on BOTH readings, sign
  │             0 of 4, every repeat between -9.7 and -13.3. THE LARGEST LOSS THIS
  │             PROJECT HAS MEASURED ON ANY LEVER. The two readings are identical,
  │             so the straggler rule does not enter into it. Holding the earliest
  │             layers of the prefill scan buys residency and pays for it in decode
  │             throughput, and the price is four times the floor.
  │
  │  SWITCH     DS4_PREFILL_READAHEAD_HOLD, default on; =0 (off/no/false)
  │             restores upstream exactly. DS4_CUDA_SSD_PREFETCH_STATS=1 counts
  │             only. The off switch is new: the previous revision reordered
  │             unconditionally and could not be measured against itself.
  │  OUTPUT     not re-run; expected output-invariant, residency changes when a
  │             byte arrives and never which byte
  │
  └──────────────────────────────────────────────────────────────────────
```

## NOT YET MEASURED

### triple-draft-gamma

```
  ┌──────────────────────────────────────────────────────────────────────
  │
  │  BRANCH     triple-draft-gamma                                   NOT YET
  │
  │  WHAT       an adaptive draft-length (gamma) controller for DSpark: one EMA
  │             slot per active-lane bucket, choosing gamma from {0..16}. Zero
  │             is a candidate, so drafting switches itself off where it is not
  │             paying
  │
  │  LATEST     round 5 - 2026-09-17 - TIES - +3.83 % 4/4 under a 5.69 floor,
  │             re-measure on a quiet box - 2026-09-17-R5-RESULT-seven-ties-under-a-noisy-floor.md
  │
  │  GEN        +3.83 %  floor 5.69 %  sign 4/4  n=4  (legacy rule +4.39, n=2)
  │             raw: arm min-across median 9.75 t/s (9.88 / 9.74 / 9.76 / 9.65)
  │             vs the tip's 9.60 t/s (9.79 / 9.42 / 9.14 / 9.66 / 9.60). gen_steady at min
  │             across 4096/6144, each run against its bracketing TIP runs
  │             control = triple-tip-2026-09-16 @12997e9c, interleaved
  │             session = 2026-09-17 11:58-13:10Z - DGX Spark GB10 -
  │             native 24.88 MB .text (tip 24.86 MB) - lean regime 4096/6144
  │             the floor is 5.69 % because the five TIP controls swung 9.14 to
  │             9.79 t/s, and both the sealed legacy rule and the new rule agree
  │             that nothing clears it
  │  PREFILL    reported, not a verdict: 88.63 t/s min-across median vs the
  │             tip's 84.61 t/s, n=4. Round 5 makes no prefill verdict.
  │
  │  VERDICT    NOT YET - +3.83 % is the largest delta of the round and 4 of 4
  │             positive, and it still does not clear this round's 5.69 % floor.
  │             It would have cleared round 4's 2.71 % floor, so this is the one
  │             arm round 5 leaves genuinely open. A re-run on a quiet box decides it
  │
  │  SWITCH     DS4_DRAFT_GAMMA_MODE=fixed|adaptive, default fixed. On fixed no
  │             controller is allocated and today's constant is returned
  │             Knobs: _ALPHA 0.2 · _UPDATE_INTERVAL 5 · _WARMUP 10 ·
  │             _DOWN_HYST -0.25 · _UP_HYST 0.0 · _CEILING 1.5 ·
  │             _MAX_STEPS 16 · _LOG=1
  │  OUTPUT     not re-run
  │
  └──────────────────────────────────────────────────────────────────────
```

### triple-spec-under-offload

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

## THE CONTROL

### triple-antirez-tip-latest

```
  ┌──────────────────────────────────────────────────────────────────────
  │
  │  BRANCH     triple-antirez-tip-latest                          CONTROL
  │
  │  WHAT       upstream main 9139e2ae5 with PRs 1034 and 1035 folded in,
  │             squashed to one commit. No lever. Every other branch is
  │             measured against this one.
  │
  │  RESULT     gen t/s   THREE READINGS OF ONE OBJECT, NOT A DISAGREEMENT
  │             9.70 median (min 9.44, max 9.92, MAD 0.11)  n=16 of 18
  │                the dedicated baseline session, 18 runs, sealed rule
  │                (2026-09-16-triple-antirez-tip-latest-BASELINE.txt)
  │             9.65 median   n=4 kept bracket runs
  │                the bisect round 1 brackets, min-across-frontiers
  │                (2026-09-16-BISECT-RESULT-one-comparator-carries-the-
  │                 whole-loss.md)
  │             9.57 median (band 9.47 - 9.62)  n=8 kept bracket runs
  │                the round 2 brackets, min-across-frontiers
  │                (2026-09-17-ROUND2-RESULT-the-revert-beats-the-tip.md)
  │             ★ ALL THREE MEASURE THE SAME SHA, e6d9d3b8, on the same box
  │               with the same argv. The spread 9.57 - 9.70 is 1.3 %, inside
  │               this configuration's own floor, and the baseline session
  │               already recorded a ~2.2 % drift between its own two passes.
  │               ⇒ quote the level as ~9.6 - 9.8 and never treat a bracket
  │               median as a re-baselining of the control.
  │             ⚠ A DELTA IS ONLY EVER TAKEN AGAINST ITS OWN ROUND'S
  │               BRACKETS, never against another round's absolute t/s.
  │             prefill   84.06 median (79.13 - 87.19)   n=16 of 18
  │             control = this branch IS the control; nothing to compare against
  │             session  baseline 2026-09-16 04:27-05:38Z; round 1 2026-09-16;
  │                      round 2 2026-09-17 15:06-16:30Z. DGX Spark GB10,
  │                      native .text 23,118,434 B, box quiet by the harness
  │                      gate (load < 2.0, memavail 117 GB, gpu < 50 %),
  │                      ctx 2048 discarded
  │
  │  VERDICT    CONTROL
  │             the zero point: ~9.6 - 9.8 gen t/s at min-across-frontiers,
  │             in-run floor median 1.0 - 1.5 %, max 2.2 - 4.1 %.
  │             ⚠ THIS IS NO LONGER THE NEWEST TIP. triple-tip-2026-09-16
  │             (upstream f39675195 + our PR carry) is the control for round 3
  │             and after; this branch stays the control for every number
  │             already taken against it.
  │
  │  SWITCH     NONE - no lever in this tree
  │  OUTPUT     not re-run
  │
  └──────────────────────────────────────────────────────────────────────
```

### triple-prefix-cache

```
  ┌──────────────────────────────────────────────────────────────────────
  │
  │  BRANCH     triple-prefix-cache                                  CONTROL
  │
  │  WHAT       the prefix cache the engine already ships (ds4_kvstore.c), read
  │             out of the code, plus the one thing it lacked: an off path.
  │             --prefix-cache off rewinds the live session at every request
  │
  │  LATEST     never in a sealed round, and the standing bench cannot put it in
  │             one. The arm lists of 2026-09-17-ROUND3-, -R4- and
  │             -R5-PREREGISTERED-RULE.txt name this branch zero times, and
  │             every round's argv is ds4-bench (R4 rule, THE DESIGN), a
  │             different binary from the ds4-server this switch lives on
  │
  │  GEN        not measured. The sealed rounds drive ds4-bench and this switch
  │             is a ds4-server flag, so no round exercises it
  │  PREFILL    not measured, same reason. Prefill is also the half a prefix hit
  │             actually moves, so a ds4-bench row would miss the effect twice
  │
  │  VERDICT    CONTROL - a duplicate of the shipped prefix cache, re-scoped as
  │             its control arm. The on-versus-off sweep needs a server harness
  │             that does not exist yet, not a slot in the current round
  │
  │  SWITCH     --prefix-cache on|off on ds4-server, default on. off means
  │             rewind to token zero, nothing read from disk, no checkpoint
  │             written
  │  OUTPUT     not re-run
  │
  └──────────────────────────────────────────────────────────────────────
```

## TOOLING, NOT A LEVER

### triple-engram-4bit

```
  ┌──────────────────────────────────────────────────────────────────────
  │
  │  BRANCH     triple-engram-4bit                                   TOOLING
  │
  │  WHAT       a design record for re-encoding ONLY the two Engram hash tables
  │             from FP8 at 264 B a row to 4-bit at 132 B a row. Ships the
  │             arithmetic, the four costs, the plan, and ZERO engine code
  │
  │  LATEST     never in a sealed round. The arm lists of 2026-09-17-ROUND3-,
  │             -R4- and -R5-PREREGISTERED-RULE.txt name this branch zero times,
  │             and there is nothing to enter one with. Its own newest figures
  │             are the arithmetic re-run here 2026-09-17, below
  │
  │  GEN        not measured - no engine code, nothing for the bench to run
  │  PREFILL    not measured - no engine code, nothing for the bench to run
  │
  │  VERDICT    TOOLING, NOT A LEVER - halving a row cannot halve a round trip.
  │             The measured 23.9 ms Engram read moves 12,672 B, so it runs at
  │             0.53 MB/s and the axis is IOPS (triple-engram-read-threads),
  │             not bytes. The re-encode earns its place as a CAPACITY fact
  │
  │  SWITCH     NONE - nothing to switch; the re-encode changes the ARTIFACT
  │  OUTPUT     not re-run - nothing runs. The quantizer skeleton encodes zero
  │             rows by design: five NotImplementedError entry points, and
  │             --check-format prints "rows encoded by this file today: 0"
  │
  └──────────────────────────────────────────────────────────────────────
```

### triple-iq2-lut-fix

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

### triple-margin

```
  ┌──────────────────────────────────────────────────────────────────────
  │
  │  BRANCH     triple-margin                                        TOOLING
  │
  │  WHAT       makes the memory reserve a live, checked, recorded value, and
  │             expresses the cache budget as a split of one RAM pool across
  │             three consumers. An instrument and a bound, not a speed lever
  │
  │  LATEST     round 5 - 2026-09-17 - TIES - +1.17 % 3/4 under a 5.69 floor -
  │             2026-09-17-R5-RESULT-seven-ties-under-a-noisy-floor.md
  │
  │  GEN        +1.17 %  floor 5.69 %  sign 3/4  n=4  (legacy rule +1.20)
  │             raw: arm min-across median 9.66 t/s (9.72 / 9.60 / 9.40 / 9.74)
  │             vs the tip's 9.60 t/s (9.79 / 9.42 / 9.14 / 9.66 / 9.60). gen_steady at min
  │             across 4096/6144, each run against its bracketing TIP runs
  │             control = triple-tip-2026-09-16 @12997e9c, interleaved
  │             session = 2026-09-17 11:58-13:10Z - DGX Spark GB10 -
  │             native 24.89 MB .text (tip 24.86 MB) - lean regime 4096/6144
  │             the floor is 5.69 % because the five TIP controls swung 9.14 to
  │             9.79 t/s, and both the sealed legacy rule and the new rule agree
  │             that nothing clears it
  │  PREFILL    reported, not a verdict: 89.49 t/s min-across median vs the
  │             tip's 84.61 t/s, n=4. Round 5 makes no prefill verdict.
  │
  │  VERDICT    TOOLING - still tooling, and now with a number that agrees. Round 5
  │             reads +1.17 % at 3 of 4 inside a 5.69 % floor, which is exactly what an
  │             instrument that changes no arithmetic on the default path should read.
  │             Nothing about t/s is claimed or owed here
  │
  │  SWITCH     DS4_MEM_RESERVE_MIB=N default 512 · DS4_MEM_RESERVE_ENFORCE=1|0
  │             default 1 (a breach stops) · DS4_MEM_RESERVE_RECORD=FILE (one JSON
  │             line per run) · DS4_MEM_RESERVE_FLOOR_HISTORY_MIB (ladder, record only)
  │             DS4_SSD_PAGECACHE_FLOOR_PCT default 10 (range 1-40)
  │             DS4_SSD_AUTO_CACHE_PCT default 80, now a SHARE, not a maximum
  │  OUTPUT     not re-run
  │
  └──────────────────────────────────────────────────────────────────────
```

### triple-word-finisher

```
  ┌──────────────────────────────────────────────────────────────────────
  │
  │  BRANCH     triple-word-finisher                                 TOOLING
  │
  │  WHAT       a suffix-lookup drafter, measured offline and closed: draft k
  │             tokens from the continuation stored against the last n tokens,
  │             verify in one batched pass, commit the longest matching prefix
  │
  │  LATEST     never in a sealed round, and it never can be. The arm lists of
  │             2026-09-17-ROUND3-, -R4- and -R5-PREREGISTERED-RULE.txt name
  │             this branch zero times. Its newest measurement is its own,
  │             MEASURED_WORDFINISHER_lookup-drafter-acceptance-offline_2026-09-15.md
  │             on this branch, verdict CLOSED, deciding number miss reuse
  │             0 of 1,219
  │
  │  GEN        not measured - no engine code, nothing for the bench to run
  │  PREFILL    not measured - no engine code, nothing for the bench to run
  │
  │  VERDICT    TOOLING - two offline probes and a closing record, not a lever.
  │             The idea is DEAD on its own numbers: a k-token verify pass reads
  │             the same NVMe bytes as k single passes, because the union of k
  │             steps' misses equals their sum, 0 of 1,219 reused
  │
  │  SWITCH     NONE - a measurement branch, no code lever
  │  OUTPUT     not re-run - no patch exists to gate
  │
  └──────────────────────────────────────────────────────────────────────
```

