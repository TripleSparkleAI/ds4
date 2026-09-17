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
  │             round 8 · +DS4_CUDA_HITS_FIRST=1 · TIES · +5.46 % (+0.33 sens)
  │             round 8 · +DS4_PREFILL_READAHEAD_HOLD=1 · TIES · +6.64 % (+5.54)
  │             round 8 · +both switches ON · TIES · -5.95 % (-6.20), the
  │             slowest tree measured, all three under a 10.69 floor ·
  │             2026-09-17-R8-RESULT-hitsfirst-alone-beats-through-the-noise-the-winners-tree-does-not.md
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
  │             ─── round 8, three switch variants on THIS binary, all TIES
  │             under a 10.69 % floor (7.17 % on the cold-run sensitivity):
  │             +hits   +5.46 %  sign 4/4  n=4  (sens +0.33 %)  arm 9.36 t/s
  │                     (10.05 · 9.21 · 9.21 · 9.51)
  │             +hold   +6.64 %  sign 3/4  n=4  (sens +5.54 %)  arm 9.80 t/s
  │                     (9.92 · 9.71 · 9.89 · 9.06)
  │             +both   -5.95 %  sign 0/4  n=4  (sens -6.20 %)  arm 8.64 t/s
  │                     (8.53 · 8.63 · 8.75 · 8.64), below every control run
  │             vs the tip's 8.96 t/s (8.61 cold · 9.53 · 8.87 · 9.49 · 8.96),
  │             gen_steady at min across 4096/6144, each run against its
  │             bracketing TIP runs
  │             ⚠ the box was not quiet: the five tip runs spanned 8.61 to 9.53
  │             and four arm runs ended at load 3.2 to 4.0 with no vitest
  │             alive, which is what widened the floor to 10.69
  │             control = triple-tip-2026-09-16 @12997e9c, interleaved
  │             session = 2026-09-17 13:24-14:37Z · DGX Spark GB10 ·
  │             native 24.94 MB .text (tip 24.86 MB) · lean regime 4096/6144
  │             prefill round 8, reported not a verdict, min-across medians vs
  │             the tip's 84.34 t/s, n=4 each: +hits 137.00, +hold 147.16,
  │             +both 144.86 t/s
  │
  │  VERDICT    POSITIVE - beats the new tip on both readings, +4.65 % no-drop and
  │             +6.87 % kept, sign 4 of 4. First measurement of this tree, and it
  │             compiled first time at 24.94 MB .text, which refuted prediction P4.
  │             ⚠ Both levers round 4 measured solo are OFF in this tree by default:
  │             hits-first (+6.47 % solo, the wrong side) and the readahead arm
  │             (-11.38 % solo, the right side; corrected 11:25Z from a source read
  │             at the sealed sha, the card had said ON). Nothing here attributes the
  │             stack's number to any lever; the seal forbids it.
  │             ⚠⚠ ROUND 8 FLIPPED BOTH SWITCHES ON THIS BINARY AND ALL THREE
  │             VARIANTS TIE: +hits +5.46 %, +hold +6.64 %, +both -5.95 %. The
  │             two switches FIGHT - with both on this is the slowest tree
  │             measured, below every control run - and the readahead arm read
  │             ABOVE hits-first here, against its -11.38 % solo, so prediction
  │             P2 missed in the opposite direction.
  │             ★ the ten-lever stack is no longer the all-fastest default; the
  │             default is hitsfirst alone as of round 8 (round 8 section 5:
  │             `triple-all-fastest` becomes the new tip plus the hitsfirst
  │             lever, from a one-lever manifest), provisional on round 9
  │
  │  SWITCH     ten levers, see the table below; plus
  │             DS4_CUDA_PREFETCH_SWEEP_ORDER=1 restores the sweep-aware order
  │  OUTPUT     not re-run
  │
  └──────────────────────────────────────────────────────────────────────
```

**How this tree was built.** `bash build_stack.sh stack-on-newtip.manifest --check <HEAD>` is
the derivation and the proof: base `triple-tip-2026-09-16`, then each lever's code-only diff
against the base it was written on, applied `--3way`, every conflict hunk resolved by the
manifest's own `resolve` lines, then the stack-local victim-order commit. The manifest header
names every judgement call. The branch's history mirrors the manifest: one commit per lever.

```
   triple-tip-2026-09-16  12997e9c8   (Qwen tip, UP)
        +-- pool · margin · draincut · pagecache · hotlist        [on]
        +-- engram-lead · engram-read-threads                     [on]
        +-- readahead-order                                       [OFF by default; DS4_PREFILL_READAHEAD_HOLD=1 turns the arm on]
        +-- hits-first                                            [OFF - measured null solo]
        +-- prefetch-pool                                         [on]
        +-- victim order used-ascending                           [the measured default]
        v
   triple-all-fastest-newtip   ten levers in, nine live, NEVER COMPILED FOR CUDA
```

| lever | switch | default | off |
| --- | --- | --- | --- |
| pool | `DS4_CUDA_STREAMING_EXPERT_PREAD_POOL` | on | `=0` |
| margin | `DS4_CUDA_EXPERT_CACHE_MARGIN_GB` | 8 | knob, no Boolean |
| draincut | `DS4_CUDA_SELECTED_DRAIN_SYNC` | on | `=1` restores blocking |
| pagecache | `DS4_CUDA_KEEP_MODEL_PAGES` | new order | `=1` restores old |
| hotlist | `DS4_CUDA_EXPERT_HOTLIST_WRITE` | on | `=0` |
| engram-lead | `DS4_V41_ENGRAM_LEAD_OFF` | on | set |
| engram-read-threads | `DS4_ENGRAM_READ_THREADS` | request count | `=1` |
| readahead-order | `DS4_PREFILL_READAHEAD_HOLD` | **off** (also on with `DS4_CUDA_PREFETCH_SWEEP_ORDER=1`) | `=1` turns the arm on |
| hits-first | `DS4_CUDA_HITS_FIRST` | **off** | `=1` turns it on |
| prefetch-pool | `DS4_CUDA_SSD_PREFETCH_CHUNK_MB` | 8 MB | knob |
| victim order | `DS4_CUDA_PREFETCH_SWEEP_ORDER` | used-ascending | `=1` sweep-aware |

⚠ What this card does NOT say: any speed. The old-base stack's numbers live on
`triple-all-fastest`'s card and describe a different tree. Read them there, never here.

## Round 4, 2026-09-17: on the new tip, in a sealed round

- **First measurement of this tree, and it beats the new tip.** +4.65 % no-drop, +6.87 % kept, sign 4 of 4, no-drop floor 2.71 %. Raw min-across medians 10.14 t/s arm against 9.65 t/s tip. It compiled on the first attempt at 24.94 MB .text, which scored prediction P4 wrong.
- ⚠ **It ships both measured levers the wrong way round.** `DS4_PREFILL_READAHEAD_HOLD` is ON by default and that lever measured **-11.38 %** solo; `DS4_CUDA_HITS_FIRST` is OFF by default and that lever measured **+6.47 %** solo. The stack still beats the tip by +4.65 % while carrying both.
- Round 4 attributes NOTHING of this number to any lever, by its own seal. The measurable question is round 7: this stack with hits-first ON, with the readahead hold OFF, with both, and hitsfirst plus pool as a two-lever tree. `spark_arms.py` needs per-arm environment switches first.

Sealed rule `2026-09-17-R4-PREREGISTERED-RULE.txt` @`3914a3226`, result `2026-09-17-R4-RESULT-hitsfirst-pool-and-the-newtip-stack-beat-the-tip-readahead-order-loses-eleven.md`, raw CSVs and runlog in `sweeps/r4/`.

## Round 8, 2026-09-17: both switches flipped on this binary, and they fight

- **All three variants TIE under a 10.69 % floor**, so round 4's +4.65 % is not retracted:
  `+DS4_CUDA_HITS_FIRST=1` reads **+5.46 %** (4/4, sensitivity +0.33 %),
  `+DS4_PREFILL_READAHEAD_HOLD=1` reads **+6.64 %** (3/4, sensitivity +5.54 %), and both on reads
  **-5.95 %** (0/4, sensitivity -6.20 %), which is the slowest tree measured all day, below every
  one of the five control runs.
- **Every stack prediction missed.** P1 (stack+hits beats the plain stack's +4.65 by 1.5) missed;
  P2 (stack+readahead at least 5 below stack+hits) missed **in the opposite direction**, because
  the readahead arm read ABOVE hits-first here despite its -11.38 % solo in round 4; P3
  (stack+hits beats the tip) missed. The round scored 1 of 8.
- ★ **The default moved off this tree.** `triple-all-fastest` is now the new tip plus the hitsfirst
  lever alone, built from a one-lever manifest (round 8 section 5), because hitsfirst solo is the
  only arm with two sealed wins and no sealed loss. That is provisional on round 9, which
  re-measures hitsfirst, pool, winners, stack+hits and triple-all on a quiet box.

```
   B O T H   S W I T C H E S   O N   I S   T H E   S L O W E S T   T H I N G
                                     (round 8, min-across gen, this binary, raw t/s)

   control tip      8.61 ↑cold   8.87   8.96        9.49  9.53
                                                      ╵
   +hold                              9.06      9.71 9.89 9.92        +6.64 %  3/4
   +hits                                   9.21 9.21  9.51      10.05  +5.46 %  4/4
   +both      8.53 8.63 8.75 8.64                                      -5.95 %  0/4
              ▲ every run below every control run

   hits-first ON and the readahead arm ON pull against each other; neither clears the floor alone
```

Sealed rule `2026-09-17-R8-PREREGISTERED-RULE.txt` @`c173ad6d7`, result
`2026-09-17-R8-RESULT-hitsfirst-alone-beats-through-the-noise-the-winners-tree-does-not.md`,
raw CSVs and runlog in `sweeps/r8/`.
