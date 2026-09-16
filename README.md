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
**DeepSeek V4 PRO**, and **Qwen3.8 Flash Next** (Metal). The code is self-contained and
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

```
  ┌──────────────────────────────────────────────────────────────────────
  │
  │  BRANCH     triple-all-fastest                                NEGATIVE
  │
  │  WHAT       the stacked tree: ten levers compiled into one binary, nine
  │             on by default and hits-first off, so one build exposes many
  │             arms through environment switches
  │
  │  RESULT     gen t/s   as shipped vs clean tip  -4.51 % min-across   floor 1.0-1.5 % med
  │                       (-8.63 / -3.75 / -5.38 at ctx 4096/6144/8192)   sign 0/7
  │                       all seven switches OFF vs tip  -12.13 %  sign 0/8
  │                       L = all-off minus as-shipped = -7.62 pp, 3.5x the floor
  │                       + DS4_CUDA_HITS_FIRST=1 vs tip  -0.95 % min-across, sign 1/6
  │                       (-4.23 / -0.57 / -0.76; only 4096 clears the floor)
  │             prefill   as shipped vs tip +6.41 % min-across (5/7); +2.66 / +16.82 /
  │                       +1.33 at 4096/6144/8192; secondary, does not offset gen
  │             control = clean tip triple-antirez-tip-latest @e6d9d3b8, 9.70 gen t/s
  │                       median, bracketing every arm repeat on both sides
  │             session  2026-09-16 06:41-09:08Z, DGX Spark GB10, native .text
  │                      23,148,652 B (tip 23,118,434 B), load 0.13-1.85, gpu 0-16 %,
  │                      memavail 117 GB, 50 runs all rc=0, ctx 2048 discarded
  │             earlier  +7.1 % over all-off (9.18 vs 8.57), floor 7.29 %: the noisy-box
  │                      series before the 2026-09-16 runs, its own off state as control
  │
  │  VERDICT    MEASURED LOSS to the tip
  │             -4.51 %, 0 of 7 repeats at every frontier, 2.1x the in-run floor; the
  │             seven switchable levers are EXONERATED (all-off loses -12.13 %, so they
  │             are worth about +7.6 pp); the cause sits in the non-switchable diff
  │             over the tip (ds4_cuda.cu +1256, ds4.c +174, ds4_engram.c +29,
  │             ds4_gpu.h +9 lines) - the bisect is OWED and not run
  │
  │  SWITCH     ten levers, see the table; six real off switches, two knobs, two none
  │  OUTPUT     not re-run
  │
  └──────────────────────────────────────────────────────────────────────
```

Sources: `2026-09-16-triple-all-fastest-attrib-SUMMARY-levers-attribution.txt` (the three
arms, the floor, L), `-attrib-as-shipped.txt`, `-attrib-all-off.txt`, `-attrib-hits-first-on.txt`
(phase 2, sealed exploratory as amendment A1, excluded from the exoneration verdict),
`2026-09-16-INDEX-attrib-stack-alloff-vs-tip.txt`, tip baseline
`2026-09-16-triple-antirez-tip-latest-BASELINE.txt`. The -4.51 % reproduces the earlier
same-day TIPVSTACK lane's -4.11 % (0/7). Min-vs-min at min-across: tip 9.54, as shipped 8.95,
all-off 8.32; the bands do not overlap.

**The levers, one line each** (all in `ds4_cuda.cu` unless said):
- **pool** - a private read pool for the expert pread, so a batch in flight does not force the
  demand path back to the serial staged copy.
- **margin** - how much page cache the streaming reads keep after the cache slots take theirs;
  8 GiB by default.
- **draincut** - queues the selected-expert readback before the shared expert runs. ON by
  default; `=1` restores the blocking read. Re-measured both ways on the GB10: inside the floor.
- **pagecache** - drops staged expert pages in the order the kernel honours, plus a read-ahead
  hint that pre-passes a layer's missing experts.
- **hotlist** - seeds the expert cache from a learned hot list written after a run.
- **engram-lead** - gives the Engram table read a token of lead (`ds4.c`).
- **engram-read-threads** - scales the Engram batch reader to the request count (`ds4_engram.c`).
- **readahead-order** - stops the prefill read-ahead evicting what decode needs. No off switch.
- **hits-first** - compiled in, OFF by default; measured a null solo (the decode path's
  shared-expert kernels already fill its window). On this stack, ON sits inside the floor
  against the tip at 6144/8192 and recovers most of the deficit (above).
- **prefetch-pool** - parallel SSD read-ahead during prefill on a private-fd pread pool, 8 MB
  chunks. Collided with pool on 30 hunks, all resolved by hand (8 ours, 16 theirs, 6 combined).
  Solo: prefill +72 to +81 %, blocking wait 398.0 -> 196.5 ms/layer (-50.6 %), its own card.

| lever | switch | default | off |
| --- | --- | --- | --- |
| pool | `DS4_CUDA_STREAMING_EXPERT_PREAD_POOL` | on | `=0` |
| hotlist | `DS4_CUDA_EXPERT_HOTLIST_WRITE` | on | `=0` |
| hits-first | `DS4_CUDA_HITS_FIRST` | **off** | (default); `=1` turns it on |
| pagecache | `DS4_CUDA_KEEP_MODEL_PAGES` | new order | `=1` restores old |
| engram-lead | `DS4_V41_ENGRAM_LEAD_OFF` | on | set |
| draincut | `DS4_CUDA_SELECTED_DRAIN_SYNC` | **on** | `=1` restores blocking |
| engram-read-threads | `DS4_ENGRAM_READ_THREADS` | request count | `=1` |
| margin | `DS4_CUDA_EXPERT_CACHE_MARGIN_GB` | 8 | knob, no Boolean |
| prefetch-pool | `DS4_CUDA_SSD_PREFETCH_CHUNK_MB` | 8 MB | knob; `DS4_CUDA_SSD_PREFETCH_POOL` is read NOWHERE in this tree (inert, amendment A2) |
| readahead-order | `DS4_CUDA_SSD_PREFETCH_STATS` | on | stats only, not the reorder |

The all-off arm set the first seven rows at once: pool, hotlist, pagecache, engram-lead,
draincut, engram-read-threads and the inert prefetch-pool name; five of the six live switches
emit no log line, so their application rests on source greps plus the harness's verbatim env
pass. Two levers need their own build for a true A/B: readahead-order and prefetch-pool.

## Interactions - the stack is not the sum of its parts

All ten levers pull on one NVMe, and pool and prefetch-pool edit the same expert-pread code,
so summed solo deltas count the same saved bytes twice. Each solo delta was taken against
unpatched code, not on top of nine others. The 2026-09-16 figures show it directly: the seven
switches are worth +7.6 pp against their own off state and the stack still loses to the tip,
so the levers and the non-lever diff move the number in opposite directions inside one binary.

| combination | what is on | gen vs tip, min-across | clears floor? |
| --- | --- | ---: | --- |
| stack | as shipped: nine levers at default, hits-first off | -4.51 % (0/7) | yes, 2.1x, as a loss |
| all-off | the seven switches off; knob-only and switchless levers stay engaged | -12.13 % (0/8) | yes, 5.6x, as a loss |
| stack vs all-off | derived: L = -7.62 pp (-7.51 / -6.73 / -1.28 at 4096/6144/8192) | levers +7.6 pp | yes, 3.5x (8192 inside) |
| hits-first on | stack + `DS4_CUDA_HITS_FIRST=1`, exploratory | -0.95 % (1/6) | no at 6144/8192/min; 4096 -4.23 % yes |

The six random on/off draws planned for this table were never run; nothing is written for them.
Corroboration from the quiet series on the stack's own control (`out_sweeps/2026-09-16-triple-
all-fastest-stack-vs-off-quiet-SUMMARY.txt`): all-off paired median -6.73 / -5.74 / -1.18 %
(8/8, 8/8, 5/8), min-vs-min stack +9.39 % at 4096; hits-first ON +3.72 / +4.33 / +4.08 %
(8/8 each), same-config floor 6.50-8.69 %.

## Methodology

- Like binaries only: native `make cuda-spark` gives ~23.1 MB of `ds4-server` text; ~29.6 MB is
  the JIT-fallback vintage and never compares.
- The control is interleaved: every arm repeat sits between two tip runs, and both arms share
  the same tip brackets, which is what makes their deltas subtractable.
- ctx 2048 is warmup and is discarded, never reported.
- The deciding statistic is the paired median of each repeat against the mean of its two
  bracketing controls, with its sign count; the max-min band is extreme-value and wide by
  construction (the quiet series measured 6.78 % on an idle box, the same order as 7.29 % on a
  loaded one), so more repeats buy sign consistency, not precision.
- The floor is printed beside every delta: in-run tip adjacent-pair median 1.0-1.5 %, max
  2.2-4.1 %; the earlier solo-session native floors were 3.3-4.9 % gen, 9.7-15.8 % prefill.
- Drop rules are sealed before the first run (`2026-09-16-ATTRIB-PREREGISTERED-RULE.txt`,
  amendments A1-A3) and applied as sealed; the no-drop sensitivity is printed beside them.
- The instrument is `try.sh`: one binary, arms made of environment switches, ten fixed tests
  over ssh against the Spark, one result file per run plus an index row.

**How this tree was built.** Base `triple-antirez-tip-latest`; the four missing levers applied
as code-only patches onto the six already stacked (`README.md` excluded from every patch); one
`ds4.c` conflict between the Engram lead read and the batch reader resolved by keeping both;
hits-first's default flipped on -> off with the reason beside it; `make cuda-spark -j12` on the
Spark, 0 errors.

```
   tip-latest  9139e2ae5 + #1034 + #1035
        +-- pool · margin · draincut · pagecache · hotlist        [on]
        +-- engram-lead · engram-read-threads                     [on]
        +-- readahead-order                                       [on, no off switch]
        +-- hits-first                                            [OFF - measured null solo]
        +-- prefetch-pool                                         [in, 30 hunks resolved]
        v
   triple-all-fastest  dd82361a   builds clean, 10 levers in, 9 live
```

---

## The index: every branch card, in full

Every branch on the fork, one card each, taken verbatim from that branch's
own README and grouped by what its own card says about it. A delta is the
generation percentage the card states against that branch's own control.

Placement rule for a branch that carries no throughput claim: it goes in the
group its own card's verdict belongs to. The IQ2 LUT check and the 4-bit
re-encode are closed questions - the first found our tree already clean, the
second argues against itself - so they sit with the other closed ideas, where
a null is still a result. The prepared arms of triple-spec-under-offload have
had no run, so they sit in NOT YET MEASURED.

Two branches withdrew their own number and moved with it: triple-pool and
triple-margin each carry a blank result and an OWED verdict saying the old
figure no longer describes the code, so both are placed as not yet measured
rather than at the withdrawn value.

No fourth group was needed; every branch fitted one of the three honestly.

---

## POSITIVELY MEASURED

Most of these deltas sit inside their session noise floor, so read this group
as a DIRECTION and not as a proven win. The cards were measured in DIFFERENT
sessions whose floors differ, so the order is indicative rather than a strict
ranking.

### triple-hitsfirst

*+8.3 % - resident experts served first, misses deferred*

```
  ┌──────────────────────────────────────────────────────────────────────
  │
  │  BRANCH    triple-hitsfirst
  │
  │  WHAT           serves the experts already resident first and defers
  │                 the misses, so the token does not stall on the slowest
  │                 read in the batch
  │
  │  RESULTS              tokens/s        tip      change       floor
  │    generation            10.35       9.56      +8.3 %       4.9 %
  │    prefill                ____       ____        ____      15.8 %
  │
  │  ENV    GB10 128GB · native 23.1MB · load ____ · gpu ____ · mem ____
  │
  │  VERDICT        OUTSIDE the floor: a real improvement at this
  │                 measurement
  │
  │  SWITCH         DS4_CUDA_HITS_FIRST=1 (default OFF in the stack)
  │  HEADLINE       +8.3% control vs patch, native build, 2 repeats
  │  OUTPUT         greedy-identical, sha256 bb06e711bc498bb9
  │
  └──────────────────────────────────────────────────────────────────────
```

### triple-all-fastest

*+7.1 % - the stacked tree, every lever in one binary*

```
  ┌──────────────────────────────────────────────────────────────────────
  │
  │  BRANCH    triple-all-fastest
  │
  │  WHAT           the stacked tree: every lever compiled into one
  │                 binary, nine on by default and hits-first off, so one
  │                 build exposes many arms
  │
  │  RESULTS              tokens/s        tip      change       floor
  │    generation             9.18       8.57      +7.1 %      7.29 %
  │    prefill                ____       ____        ____      15.8 %
  │
  │  ENV    GB10 128GB · native 23.1MB · load ____ · gpu ____ · mem ____
  │
  │  VERDICT        MEASURED, SIGN-CONSISTENT, NOT FLOOR-CLEARING
  │
  │  REFERENCE      the tip column is ALL-OFF (every kill-switch off), not
  │                 the
  │                 plain tip: for the stack the meaningful control is its
  │                 own off state
  │  FLOOR          this session's floor was 7.29 % gen, twice the 3.3-4.9
  │                 % of the
  │                 solo native run - the edge is the SAME ORDER as the
  │                 floor
  │  LEVERS         10 compiled in, 9 on by default, hits-first 1 of 10
  │                 OFF
  │  EXCLUDED       0 - nothing is left out of this tree
  │  SWITCHES       10 - one binary, many arms
  │  BUILD          clean: make cuda-spark -j12, 0 errors
  │  STACK NUMBER   MEASURED - +7.1 % over all-off, sign-consistent
  │                 under all three references and NOT floor-clearing
  │
  └──────────────────────────────────────────────────────────────────────
```

### triple-pagecache

*+0.6 % - pages dropped so the working set stays resident*

```
  ┌──────────────────────────────────────────────────────────────────────
  │
  │  BRANCH    triple-pagecache
  │
  │  WHAT           drops staged pages in an order that keeps the working
  │                 set resident, instead of dropping them in arrival
  │                 order
  │
  │  RESULTS              tokens/s        tip      change       floor
  │    generation             9.62       9.56      +0.6 %       4.9 %
  │    prefill                ____       ____        ____      15.8 %
  │
  │  ENV    GB10 128GB · native 23.1MB · load ____ · gpu ____ · mem ____
  │
  │  VERDICT        inside the floor: this measurement does not resolve it
  │
  │  SWITCH         the drop order is staged; the arms are in the section
  │                 below
  │  HEADLINE       measured: awaiting the clean sweep
  │  OUTPUT         greedy-identical, sha256 bb06e711bc498bb9
  │
  └──────────────────────────────────────────────────────────────────────
```

---

## NEGATIVELY MEASURED

A null is a result. A closed branch is kept here on purpose, so nobody
rebuilds it.

### triple-draincut

*-0.9 % - selected reads evented, 40 device drains per token*

```
  ┌──────────────────────────────────────────────────────────────────────
  │
  │  BRANCH    triple-draincut
  │
  │  WHAT           reads selected experts back on an event instead of
  │                 blocking, removing 40 device drains per token from the
  │                 host's wait
  │
  │  RESULTS              tokens/s        tip      change       floor
  │    generation             9.09       9.17      -0.9 %       4.9 %
  │    prefill                ____       ____        ____      15.8 %
  │
  │  ENV    GB10 128GB · native 23.1MB · load ____ · gpu ____ · mem ____
  │
  │  VERDICT        inside the floor: this measurement does not resolve it
  │
  │  POLARITY       the lever is ON by default;
  │                 DS4_CUDA_SELECTED_DRAIN_SYNC=1
  │                 restores the blocking read, which is the OFF arm
  │  HEADLINE       40 device drains per token removed from the host's
  │                 wait
  │  OUTPUT         greedy-identical, sha256 bb06e711bc498bb9
  │
  └──────────────────────────────────────────────────────────────────────
```

### triple-hotlist

*-1.9 % - the seeding arm measured small; the re-scope is owed*

```
  ┌──────────────────────────────────────────────────────────────────────
  │
  │  BRANCH    triple-hotlist
  │
  │  WHAT           seeds the next session's SSD expert cache from the
  │                 previous run's demand, so the opening tokens start
  │                 WARM instead of cold.  A WARM-UP device, not a
  │                 selection device.
  │
  │  RESULTS              tokens/s        tip      change       floor
  │    generation             ____        ____        ____       4.9 %
  │    prefill                ____        ____        ____      15.8 %
  │
  │  ENV    GB10 128GB · native 23.1MB · load ____ · gpu ____ · mem ____
  │
  │  VERDICT        OWED.  The seeding arm this branch shipped measured
  │                 -1.9 %, which is INSIDE the floor.  The re-scoped
  │                 lever, a warm cache, is not measured yet.
  │
  │  SWITCH         DS4_CUDA_EXPERT_HOTLIST_WRITE=<file>, 0 disables
  │                 DS4_CUDA_EXPERT_HOTLIST_DECAY=none|halve|quarter|
  │                 eighth|shift:<n>, DEFAULT halve, UNCHANGED
  │                 DS4_CUDA_EXPERT_HOTLIST_ROLLING=1 turns the
  │                 retention rule into a measurement
  │                 DS4_METAL_DISABLE_STREAMING_EXPERT_HOTLIST=1
  │                 disables seeding
  │  HEADLINE       presence 27x to 45x the content effect
  │                 (WIKI/theory/167, lines 57 and 68-70); warm resident
  │                 35.14 vs cold 15.87 tokens/s, 2.2x
  │                 (WIKI/theory/48, section 4, lines 148-152)
  │  OUTPUT         greedy-identical, sha256 bb06e711bc498bb9
  │
  └──────────────────────────────────────────────────────────────────────
```

### triple-iq2-lut-fix

*CLOSED, no patch - the IQ2 LUT check; our tree is clean*

```
  ┌──────────────────────────────────────────────────────────────────────
  │
  │  BRANCH    triple-iq2-lut-fix
  │
  │  WHAT           a CHECK first: does our tree carry upstream's IQ2
  │                 dequant-LUT defect, where the codebook staged in
  │                 shared memory for n_embd <= 4096 was consumed
  │                 unconditionally? VERDICT: no, we are clean
  │
  │  RESULTS        (none - see below)
  │
  │  ENV    GB10 128GB · native 23.1MB · load ____ · gpu ____ · mem ____
  │
  │  VERDICT        OUR TREE IS NOT AFFECTED. The defect was present in
  │                 our lineage and was already removed by upstream's own
  │                 correction, commit a04f46fa42. No patch is owed
  │
  │  FINDING        1 of 1 checked site set: CLEAN
  │  FILE           ds4_cuda.cu, IQ2 LUT shared-memory staging
  │  LINES          20611-20623, 20679-20689, 21194-21219, 21284-21309,
  │                 21378-21403 (5 kernels, 5 loads, 10 consumers)
  │  BUG SHAPE      xq_blocks <= 16 (n_embd 4096) guarded the LOAD while
  │                 the dot products read the tables UNCONDITIONALLY
  │  WOULD HAVE BEEN  fluent garbage at full speed, on a MULTI-TOKEN
  │                 batch of any n_embd > 4096 model, no crash
  │  AT RISK        DeepSeek V4.1 Flash (5120) and V4 Pro (7168), on the
  │                 CUDA IQ2 routed-MoE expert-tile batch path
  │
  └──────────────────────────────────────────────────────────────────────
```

### triple-engram-4bit

*CLOSED, no lever - the 4-bit re-encode; its arithmetic says no*

```
  ┌──────────────────────────────────────────────────────────────────────
  │
  │  BRANCH    triple-engram-4bit
  │
  │  WHAT           a DESIGN RECORD, not a lever: what re-encoding ONLY
  │                 the two Engram hash tables from FP8 at 264 bytes a
  │                 row to 4-bit at 132 bytes a row would save US, and
  │                 why that work is tooling rather than an engine branch
  │
  │  RESULTS              tokens/s        tip      change       floor
  │    generation             ____        ____        ____       4.9 %
  │    prefill                ____        ____        ____      15.8 %
  │
  │  ENV    GB10 128GB · native 23.1MB · load ____ · gpu ____ · mem ____
  │
  │  VERDICT        NOT AN ENGINE BRANCH, and there is no lever here to
  │                 measure: the re-encode changes the ARTIFACT. This
  │                 branch ships the arithmetic, the four costs and the
  │                 plan, and ZERO engine code, because there is nothing
  │                 to switch
  │
  │  SWITCH         none, deliberately. No switch for a plan
  │
  │  HEADLINE       94.417 GiB off the file (202.758 GB to 101.379 GB),
  │                 and 6,336 BYTES a token, which is 0.0000655 percent
  │                 of the 9.670 GB/token census floor. The table is
  │                 55.4 percent of the FILE and 27.5 NANOSECONDS of a
  │                 token
  │  RECORD         gguf-tools/engram-4bit/PLAN.md
  │  OUTPUT         nothing runs. The quantizer skeleton encodes zero
  │                 rows by design, and the build gate is OWED
  │
  └──────────────────────────────────────────────────────────────────────
```

### triple-granule

*CLOSED - the sparse expansion lost to the plain hash, 10 of 10*

```
  ┌──────────────────────────────────────────────────────────────────────
  │
  │  BRANCH    triple-granule
  │
  │  WHAT           asks GRANULE, offline and CPU only: does a
  │                 sparse-distributed expansion make a cheap front-cache
  │                 lookup MORE DISCRIMINATIVE than a plain hash, at the
  │                 same address budget? Run on the committed track-3
  │                 token streams, no GPU and no model
  │
  │  RESULTS        discrimination, SDR minus plain hash, same readout
  │                   matched entropy     -0.0414 .. -0.2629
  │                   hash 10 of 10
  │                   matched bytes       -0.0294 .. -1.6357
  │                   hash 16 of 16
  │
  │  ENV    GB10 128GB · native 23.1MB · load ____ · gpu ____ · mem ____
  │
  │  VERDICT        THE PLAIN HASH WINS, in both readings
  │                 of "same budget", on both corpora, at every budget.
  │                 The expansion's
  │                 effective address space PLATEAUS near 2^15 while the
  │                 hash's tracks its width
  │
  │  PREDICTION     sealed before the run and WRONG: P1 predicted a NULL
  │                 (within +/-0.010) and measured a hash win at 10 of 10.
  │                 P2 HELD, P3 HELD, P4 FAILED in both directions
  │  PRIOR          against this family, and it was right: presence
  │                 is 27-45x content (WIKI/theory/167), the encoder
  │                 was never the axis (T3-69), a counted shortlist
  │                 held while a cost claim did not (T3-58)
  │  THE ONE WIN    the expansion's false routings are STRUCTURED: 1-4%
  │                 land on the perturbed-from entry against the hash's
  │                 0-0.1%
  │  SPACE REALISED 14.96 effective bits against a nominal 144.23 at 1024
  │                 bits, where the hash still reads zero collisions
  │  DUPLICATE KEYS 61 of 2048 keys share an address at EVERY width tried,
  │                 m of 467 and m of 1174 alike
  │  OWED           nothing: the run happened, 10.0 s wall, selftest 21/21
  │  SWITCH         none - an offline experiment, no engine code, no knob
  │
  └──────────────────────────────────────────────────────────────────────
```

### triple-word-finisher

*CLOSED, no patch - the lookup drafter, tested and killed*

```
  ┌──────────────────────────────────────────────────────────────────────
  │
  │  BRANCH    triple-word-finisher
  │
  │  WHAT           the lookup drafter, tested and CLOSED on measurement:
  │                 a k-token verify pass reads the SAME NVMe bytes as k
  │                 single passes, so it saves nothing
  │
  │  RESULTS              tokens/s        tip      change       floor
  │    no patch   a measurement branch, see VERDICT
  │
  │  ENV    GB10 128GB · native 23.1MB · load ____ · gpu ____ · mem ____
  │
  │  VERDICT        CLOSED: the measurement killed the idea, and that is a
  │                 result
  │
  │  ACCEPTANCE     2.37x code / 1.17x prose: the speedup is real, and not
  │                 the lever
  │  BYTES SAVED    0.0000 at k of 2, 4 and 8
  │  ROUTING OVERLAP 38.90% between adjacent decode tokens
  │  MISS REUSE     0 of 1,219: no next miss was selected by the previous
  │                 token
  │  OUTPUT         not run - there is no patch; this branch is a
  │                 measurement
  │
  └──────────────────────────────────────────────────────────────────────
```

---

## NOT YET MEASURED

Nothing here has a number yet. A blank cell awaits the measurement; it is
not a zero.

### triple-climbingfibre

*not yet measured - a front cache learned from the drafter's rejects*

```
  ┌──────────────────────────────────────────────────────────────────────
  │
  │  BRANCH    triple-climbingfibre
  │
  │  WHAT           a front cache in front of a FROZEN model that learns
  │                 from the speculative decoder's own REJECTIONS, online,
  │                 and never modifies the model. At a refusal the engine
  │                 already holds (context, what was drafted, what the
  │                 target actually produced) and throws it away
  │
  │  RESULTS              tokens/s        tip      change       floor
  │    generation             ____       ____        ____       4.9 %
  │    prefill                ____       ____        ____      15.8 %
  │
  │  ENV    GB10 128GB · native 23.1MB · load ____ · gpu ____ · mem ____
  │
  │  VERDICT        AWAITING THE SWEEP
  │
  │  SWITCH         DS4_CLIMBINGFIBRE=1|0, DEFAULT 0. With 0 the shortcut
  │                 NEVER LEARNS and NEVER PROPOSES, and it allocates
  │                 nothing at all
  │  RETENTION      DS4_CLIMBINGFIBRE_HORIZON=
  │                 session|day|week|forever|<steps>
  │                 DS4_CLIMBINGFIBRE_DECAY=hard|half|none
  │                 the horizon is a PARAMETER, not a constant
  │  CACHE LAWS     never modify the deep path, only skip it / a miss is
  │                 free / correctness never depends on the cache / learn
  │                 from the misses. All four are enforced in code, and
  │                 the first is enforced by the module having no handle
  │                 on the model at all
  │  PRIOR ART      retrieval drafting is OCCUPIED (REST n-gram,
  │                 DReSD hidden-state, RASD, ReSpec) and the
  │                 drafting SUBSTRATE is
  │                 NOT the novel part. What is claimed here is narrower
  │  OUTPUT         unchanged by construction. NOT YET RUN ON THE BOX
  │
  └──────────────────────────────────────────────────────────────────────
```

### triple-draft-gamma

*not yet measured - an adaptive draft length that can choose zero*

```
  ┌──────────────────────────────────────────────────────────────────────
  │
  │  BRANCH    triple-draft-gamma
  │
  │  WHAT           an adaptive draft-length (gamma) controller for DSpark
  │                 speculative decoding: one EMA slot per active-lane
  │                 bucket, each choosing gamma from a candidate set that
  │                 INCLUDES ZERO, so the machine stops drafting when a
  │                 draft is not paying for the verify bandwidth it costs
  │
  │  RESULTS              tokens/s        tip      change       floor
  │    generation             ____       ____        ____       4.9 %
  │    prefill                ____       ____        ____      15.8 %
  │
  │  ENV    GB10 128GB · native 23.1MB · load ____ · gpu ____ · mem ____
  │
  │  VERDICT        AWAITING THE SWEEP: no CUDA build and no measurement
  │                 yet, so this branch claims no number. The build gate
  │                 is OWED
  │
  │  SWITCH         DS4_DRAFT_GAMMA_MODE=fixed|adaptive. Default `fixed`,
  │                 and `fixed` is today's behaviour exactly
  │  KNOBS          DS4_DRAFT_GAMMA_ALPHA 0.2 · UPDATE_INTERVAL 5 ·
  │                 WARMUP 10 · DOWN_HYST -0.25 · UP_HYST 0.0 ·
  │                 CEILING 1.5 · MAX_STEPS 16
  │  GAMMA TODAY    the support model's GGUF `dspark.block_size`, a
  │                 hand-chosen constant (ds4.c:76368), clamped at 16 by
  │                 DS4_DSPARK_MAX_BLOCK_SIZE (ds4.c:2907) and only
  │                 overridable by the static DS4_DSPARK_VERIFY_CAP
  │  GAMMA 0        a first-class candidate: drafting is DISABLED for that
  │                 lane bucket, rested, then re-probed at gamma 1
  │  FED BY         the accept length the DSpark loop already computes
  │                 (ds4.c:60251), not a new measurement path
  │  TEST           `make test-draft-gamma`: standalone, no GPU, no model,
  │                 no Spark, and it passes
  │  OUTPUT         not run. There is no build on this branch yet
  │
  └──────────────────────────────────────────────────────────────────────
```

### triple-engram-lead

*not yet measured - the Engram read started a token ahead of use*

```
  ┌──────────────────────────────────────────────────────────────────────
  │
  │  BRANCH    triple-engram-lead
  │
  │  WHAT           starts the Engram table read a token of lead ahead of
  │                 first use, so the read overlaps the step instead of
  │                 landing exposed inside it
  │
  │  RESULTS              tokens/s        tip      change       floor
  │    generation             ____       ____        ____       4.9 %
  │    prefill                ____       ____        ____      15.8 %
  │
  │  ENV    GB10 128GB · native 23.1MB · load ____ · gpu ____ · mem ____
  │
  │  VERDICT        AWAITING THE SWEEP: a blank cell is not a zero
  │
  │  HEADLINE       the read is 23.9 ms of a 198.6 ms step, 12.04%, fully
  │                 exposed
  │  RECORD         speed-bench/v41_engram_lead_gb10.md
  │  OUTPUT         ____  gates and the end-to-end A/B are OWED here
  │
  └──────────────────────────────────────────────────────────────────────
```

### triple-engram-prestage

*not yet measured - the Engram row lookup moved out of the forward*

```
  ┌──────────────────────────────────────────────────────────────────────
  │
  │  BRANCH    triple-engram-prestage
  │
  │  WHAT           moves the host-side Engram row lookup out of the
  │                 forward pass into a prepare phase that runs before it,
  │                 so the decode step stops waiting on the disk and the
  │                 forward is left with no host I/O of its own
  │
  │  RESULTS              tokens/s        tip      change       floor
  │    generation             ____       ____        ____       4.9 %
  │    prefill                ____       ____        ____      15.8 %
  │
  │  ENV    GB10 128GB · native 23.1MB · load ____ · gpu ____ · mem ____
  │
  │  VERDICT        AWAITING THE SWEEP: a blank cell is not a zero
  │
  │  SWITCH         DS4_ENGRAM_PRESTAGE=1 enables the prepare phase;
  │                 unset or 0 keeps the lookup inline in the forward,
  │                 which is the shipped path
  │                 DS4_ENGRAM_PRESTAGE_DEBUG=1 names every
  │                 demand-read miss
  │  HEADLINE       the read is 23.9 ms of a 198.6 ms step, 12.04%, fully
  │                 exposed
  │  RECORD         speed-bench/v41_engram_lead_gb10.md
  │  CONFLICT       collides with triple-engram-lead and
  │                 triple-engram-read-threads: they touch the same read
  │                 path, and the lead read lives in ds4.c, where we have
  │                 hand-resolved a merge before
  │  OUTPUT         ____  the build gate, the A/B and a full-step capture
  │                 are OWED
  │
  └──────────────────────────────────────────────────────────────────────
```

### triple-engram-read-threads

*not yet measured - one wave of readers, not two serial rounds*

```
  ┌──────────────────────────────────────────────────────────────────────
  │
  │  BRANCH    triple-engram-read-threads
  │
  │  WHAT           issues a decode step's Engram read as ONE WAVE: one
  │                 row per reader on a single persistent pool shared by
  │                 both tables, instead of two serial rounds of twelve
  │                 readers
  │
  │  RESULTS              tokens/s        tip      change       floor
  │    generation             ____       ____        ____       4.9 %
  │    prefill                ____       ____        ____      15.8 %
  │
  │  ENV    GB10 128GB · native 23.1MB · load ____ · gpu ____ · mem ____
  │
  │  VERDICT        AWAITING THE SWEEP: a blank cell is not a zero
  │
  │  SWITCH         DS4_ENGRAM_ROWS_PER_READER=1|2, default 1
  │                 2 restores the old two-round divisor as the control
  │                 arm
  │                 DS4_ENGRAM_READ_THREADS=<n> still overrides the count
  │  HEADLINE       48 serial 264-byte preads, 12,672 bytes, 12.04% of the
  │                 step; the one-wave fix itself is OWED
  │  RECORD         speed-bench/v41_engram_read_threads_gb10.md
  │  OUTPUT         the reader count is proven not to change a byte
  │                 (tests/test_engram.c); the release gates and the
  │                 end-to-end A/B are OWED
  │
  └──────────────────────────────────────────────────────────────────────
```

### triple-hippocampal-warmset

*not yet measured - a longer horizon for the learned expert set*

```
  ┌──────────────────────────────────────────────────────────────────────
  │
  │  BRANCH    triple-hippocampal-warmset
  │
  │  WHAT           takes the learned expert set the hot list already
  │                 persists and gives it a longer horizon: held FREE
  │                 inside a day, halved once per day boundary, and
  │                 promoted to a durable set when a pair is demanded
  │                 on three distinct days
  │
  │  RESULTS              tokens/s        tip      change       floor
  │    generation             ____       ____        ____        ____
  │    prefill                ____       ____        ____        ____
  │
  │  ENV    GB10 128GB · native 23.1MB · load ____ · gpu ____ · mem ____
  │
  │  VERDICT        AWAITING THE SWEEP
  │
  │  SWITCH         DS4_WARMSET_TIER=session|day   default session =
  │                 today's behaviour, one file halved at every load
  │                 DS4_WARMSET_DAYS=<n>   consolidation threshold (3)
  │                 DS4_WARMSET_MAX=<n>    cap on carried entries
  │                 DS4_WARMSET_DIR=<dir>   where the tier lives
  │                 DS4_WARMSET_PROFILE=1   trace what the tier did
  │
  │  RETENTION      session  one run, halved at every load
  │                 day      free inside the day, halved once per day
  │                          boundary
  │                 durable  demanded on three or more distinct days
  │                 FORGOTTEN  seen on one day only, decayed to zero,
  │                          recorded against another model size, and
  │                          every seeded slot
  │
  │  SAFETY         a warm-set entry is (layer, expert) plus three
  │                 integer evidence counts.  No value, weight, scale
  │                 or gate exists in its vocabulary, so it cannot be
  │                 injected into anything.  A wrong entry costs one
  │                 wasted slot and nothing else
  │
  └──────────────────────────────────────────────────────────────────────
```

### triple-lfu-offsetkey

*not yet measured - upstream's offset-keyed LFU host cache*

```
  ┌──────────────────────────────────────────────────────────────────────
  │
  │  BRANCH    triple-lfu-offsetkey
  │
  │  WHAT           adds upstream NeutronStar's host expert cache: byte
  │                 ranges keyed by FILE OFFSET in a 16-way table whose
  │                 per-slot uses counter halves every 4096 inserts, so
  │                 ONE eviction POLICY can be measured against our
  │                 expert-ID hot list from ONE binary
  │
  │  RESULTS              tokens/s        tip      change       floor
  │    generation             ____       ____        ____       4.9 %
  │    prefill                ____       ____        ____      15.8 %
  │
  │  ENV    GB10 128GB · native 23.1MB · load ____ · gpu ____ · mem ____
  │
  │  VERDICT        AWAITING THE SWEEP: a blank cell is not a zero
  │
  │  SWITCH         DS4_EXPERT_CACHE_MODE=hotlist|offsetkey|
  │                 offsetkey-lru, DEFAULT hotlist (this branch inert)
  │                 DS4_CUDA_HOST_EXPERT_CACHE_GB=<GiB> sizes the host
  │                 cache; unset, it takes the configured expert-cache
  │                 byte size so the arms are size-matched
  │  HEADLINE       the argument is about POLICY, not SIZE: upstream says
  │                 LFU beats LRU, we measured hotness-with-decay LOSING
  │                 to plain LRU, and our own roadmap already has a knee
  │                 that says cache SIZE is not the lever
  │  OUTPUT         not claimed: residency changes WHEN a byte arrives,
  │                 never which byte, and no arithmetic is touched
  │
  └──────────────────────────────────────────────────────────────────────
```

### triple-margin

*not yet measured - the reserve made live; its old number is withdrawn*

```
  ┌──────────────────────────────────────────────────────────────────────
  │
  │  BRANCH    triple-margin
  │
  │  WHAT           turns the memory reserve into a LIVE, CHECKED,
  │                 RECORDED value, and expresses the cache budget as a
  │                 SPLIT of one pool of RAM across three consumers.  An
  │                 INSTRUMENT and a bound, not a speed lever.
  │
  │  RESULTS              tokens/s        tip      change       floor
  │    generation             ____        ____        ____       4.9 %
  │    prefill                ____        ____        ____      15.8 %
  │
  │  ENV    GB10 128GB · native 23.1MB · load ____ · gpu ____ · mem ____
  │
  │  VERDICT        OWED.  Neither change has been measured.  The cells
  │                 above are blank on purpose.  The numbers this file
  │                 used to carry (8.68 against 9.56, -9.2 %) belong to
  │                 the OLD lever, the expert-cache margin knob, which
  │                 this branch no longer claims.  A blank cell is
  │                 AWAITING THE MEASUREMENT, not a zero.
  │
  │  SWITCH         DS4_MEM_RESERVE_MIB=N        default 512 (the
  │                                              per-device safety margin,
  │                                              now named)
  │                 DS4_MEM_RESERVE_ENFORCE=1|0  default 1 (a breach
  │                 stops)
  │                 DS4_SSD_PAGECACHE_FLOOR_PCT  default 10
  │                 DS4_SSD_AUTO_CACHE_PCT       default 80, UNCHANGED in
  │                                              value, changed in MEANING
  │                                              (a share, not a maximum)
  │  HEADLINE       no headline number, and that is deliberate.  The claim
  │                 SHRANK: WIKI/theory/48-decode-speedup-levers.md
  │                 section 8 (lines 368-372) says that on the Spark the
  │                 lever is hot-expert prediction, "NOT BIGGER CACHE (THE
  │                 KNEE PROVED SIZE ISN'T IT)".  This branch says STOP
  │                 OVER-ALLOCATING, not allocate more.
  │  OUTPUT         ____ (OWED: not run.  The default path is arithmetic-
  │                 identical, but that is an argument, not a measurement)
  │  BUILD          OWED.  A CUDA build needs the Spark and the Spark is
  │                 running an unrelated series, so nothing here was built
  │                 or benchmarked.  The code is committed unbuilt.
  │
  └──────────────────────────────────────────────────────────────────────
```

### triple-pool

*not yet measured - the io_uring read ring; its old number is withdrawn*

```
  ┌──────────────────────────────────────────────────────────────────────
  │
  │  BRANCH    triple-pool
  │
  │  WHAT           serves expert reads from a shared parallel SSD pool
  │                 instead of one serial reader, and now puts an io_uring
  │                 O_DIRECT ring in front of that pool so queue depth
  │                 is a switch rather than the worker count
  │
  │  RESULTS              tokens/s        tip      change       floor
  │    generation             ____       ____        ____        ____
  │    prefill                ____       ____        ____        ____
  │
  │  ENV    GB10 128GB · native 23.1MB · load ____ · gpu ____ · mem ____
  │
  │  VERDICT        OWED: the fetch engine changed after the last
  │                 measurement, so the old +4.0 % no longer describes it
  │
  │  SWITCH         DS4_CUDA_FETCH_QD=<n>; io_uring queue depth,
  │                 default 64, clamped 8-512
  │                 DS4_CUDA_FETCH_URING=0 falls back to the pread pool;
  │                 DS4_CUDA_FETCH_BUFFERED=1 forces buffered reads
  │                 DS4_CUDA_STREAMING_EXPERT_PREAD_THREADS=<n>; pool
  │                 workers when the ring is off, default 8, cap 16
  │  HEADLINE       the ring is ported; device concurrency is now a
  │                 knob, and no number here has been re-measured since
  │  OUTPUT         not re-run on this engine: sha256 ____
  │  BUILD          OWED: a CUDA build needs the DGX Spark, which is busy
  │
  └──────────────────────────────────────────────────────────────────────
```

### triple-prefill-readahead-order

*not yet measured - eviction reordered, the earliest layers held*

```
  ┌──────────────────────────────────────────────────────────────────────
  │
  │  BRANCH    triple-prefill-readahead-order
  │
  │  WHAT           reorders prefill victim eviction and holds the
  │                 earliest layers of the scan, so the experts decode
  │                 will need are not the ones dropped: the prefill hit
  │                 list survives into decode
  │
  │  RESULTS              tokens/s        tip      change       floor
  │    generation             ____       ____        ____       4.9 %
  │    prefill                ____       ____        ____      15.8 %
  │
  │  ENV    GB10 128GB · native 23.1MB · load ____ · gpu ____ · mem ____
  │
  │  VERDICT        OWED: no clean interleaved A/B exists yet, so a
  │                 blank cell is not a zero. The CUDA build gate is
  │                 OWED as well.
  │
  │  SWITCH         DS4_PREFILL_READAHEAD_HOLD=0 restores upstream
  │                 exactly: the plain used-ascending victim order, no
  │                 held set. Unset or 1 keeps the reorder plus the held
  │                 band. This is the A/B switch the branch did not have
  │                 before.
  │  STATS          DS4_CUDA_SSD_PREFETCH_STATS=1 counts only, it does not
  │                 gate
  │  GATE           expected output-invariant: residency changes WHEN a
  │                 byte arrives, never which byte
  │
  └──────────────────────────────────────────────────────────────────────
```

### triple-prefix-cache

*not yet measured - the shipped prefix cache, given an off arm*

```
  ┌──────────────────────────────────────────────────────────────────────
  │
  │  BRANCH    triple-prefix-cache
  │
  │  WHAT           the prefix cache our engine already ships, checked
  │                 against the code rather than assumed, plus the one
  │                 thing it did not have: an off path. --prefix-cache
  │                 off rewinds the live session at every request, so the
  │                 cache can be measured against its own control
  │
  │  RESULTS              tokens/s        tip      change       floor
  │    generation             ____       ____        ____       4.9 %
  │    prefill                ____       ____        ____      15.8 %
  │
  │  ENV    GB10 128GB · native 23.1MB · load ____ · gpu ____ · mem ____
  │
  │  VERDICT        AWAITING THE SWEEP
  │
  │  SWITCH         --prefix-cache on|off   (default: on)
  │                 off = control arm: rewind to token zero, nothing read
  │                 from disk, no checkpoint written
  │  REFERENCE      prefix cache 1.86x on repeated context, against
  │                 prefill work 1.25x and DSpark 0.93x: the largest
  │                 measured win in our record
  │                 source: experiments/track3-semiotic-codebook/
  │                 NEW_IDEA_THE_CEREBELLAR_FRONT_CACHE_2026-09-01.md
  │                 section 4, line 106 (tracked at 408695d0b; this
  │                 branch does not carry the experiments/ tree)
  │  BUILD GATE     OWED. No CUDA build and no run. The Spark is busy with
  │                 an unrelated series. The only thing executed here is a
  │                 local clang -fsyntax-only parse of the two touched
  │                 files, and it is clean
  │
  └──────────────────────────────────────────────────────────────────────
```

### triple-prefetch-pool

*not yet measured - parallel chunked prefill read-ahead*

```
  ┌──────────────────────────────────────────────────────────────────────
  │
  │  BRANCH    triple-prefetch-pool
  │
  │  WHAT           a parallel chunked read-ahead for prefill, with the
  │                 single reader kept as the fallback whenever the pool
  │                 declines the work
  │
  │  RESULTS              tokens/s        tip      change       floor
  │    generation             ____       ____        ____       4.9 %
  │    prefill                ____       ____        ____      15.8 %
  │
  │  ENV    GB10 128GB · native 23.1MB · load ____ · gpu ____ · mem ____
  │
  │  VERDICT        AWAITING THE SWEEP: a blank cell is not a zero
  │
  │  SWITCH         DS4_CUDA_SSD_PREFETCH_CHUNK_MB, default 8 MiB, cap 64;
  │                 a
  │                 value of 0 is IGNORED, so there is no clean off arm
  │  FALLBACK       pool declines -> the single reader, unchanged
  │  GATE           expected output-invariant: WHEN bytes arrive changes,
  │                 never
  │                 how many
  │
  └──────────────────────────────────────────────────────────────────────
```

### triple-spec-under-offload

*not yet measured - prepared arms only, every cell blank*

```
  ┌──────────────────────────────────────────────────────────────────────
  │
  │  BRANCH    triple-spec-under-offload
  │
  │  WHAT           ____
  │                 ____
  │
  │  RESULTS              tokens/s        tip      change       floor
  │    generation             ____       ____        ____        ____
  │    prefill                ____       ____        ____        ____
  │
  │  ENV    GB10 128GB · native 23.1MB · load ____ · gpu ____ · mem ____
  │
  │  VERDICT        AWAITING THE SWEEP
  │
  │  ARM 1          RAW (cold): streaming box, page cache evicted, the
  │                 DSpark pair run against its own interleaved control
  │  ARM 2          WARM WORKING SET: the same pair again with the working
  │                 set ALREADY resident at a fixed context. This is the
  │                 arm nobody has run, ours or anyone else's
  │  SWITCH         --dspark --mtp-model <file>   (ds4_help.c:190)
  │                 DS4_MTP_SPEC_DISABLE=1        (ds4_server.c:13794)
  │                 --ssd-streaming-cold          (ds4_help.c:174)
  │  STATUS         OWED: the CUDA build gate and all four legs need the
  │                 Spark, and the Spark is serialized to an unrelated
  │                 series. No number in this file has been run
  │  OUTPUT         ____
  │
  └──────────────────────────────────────────────────────────────────────
```

---

## THE CONTROL

Not a lever. The rolled tip carries no result of its own: it is the zero
point the other branches are measured against.

### triple-antirez-tip-latest

*reference, no delta - the rolled tip every branch is cut from*

```
  ┌──────────────────────────────────────────────────────────────────────
  │
  │  BRANCH    triple-antirez-tip-latest
  │
  │  WHAT           the rolled tip: upstream main left untouched with PRs
  │                 1034 and 1035 folded in. This is the CONTROL every
  │                 other branch is cut from
  │
  │  RESULTS              tokens/s        tip      change       floor
  │    generation             9.56       9.56   reference       4.9 %
  │    prefill                ____       ____        ____      15.8 %
  │
  │  ENV    GB10 128GB · native 23.1MB · load ____ · gpu ____ · mem ____
  │
  │  VERDICT        THIS IS THE TIP: it carries no lever and is the zero
  │                 point
  │
  │  UPSTREAM       9139e2ae5 untouched
  │  + PR 1034      folded: Metal decode-queue sync, with its test and
  │                 bench
  │  + PR 1035      folded: macOS Engram parallel reads, +5.2% by its
  │                 author
  │  = BRANCH       8ee53cb8b, tree-identical to 84ba6ef1b
  │  PR 1031 / 1030 not taken: superseded by evolution / moot
  │
  └──────────────────────────────────────────────────────────────────────
```
