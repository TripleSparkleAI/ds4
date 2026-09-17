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
  │  RESULT     gen t/s   THREE READINGS OF THE STACK, ALL ON NINE-LEVER SHAS
  │             1. attrib series, as shipped vs clean tip  -4.51 % min-across
  │                (-8.63 / -3.75 / -5.38 at ctx 4096/6144/8192)  sign 0/7
  │                floor med 1.0-1.5 %, max 2.2-4.1 %
  │                (2026-09-16-triple-all-fastest-attrib-SUMMARY-levers-
  │                 attribution.txt)
  │             2. bisect round 1, the same tree re-measured at n=4
  │                -4.35 % min-across, sign 0/4, floor med 0.52 / max 4.13
  │                ⇒ the loss REPRODUCES within 0.16 points across sessions
  │                (2026-09-16-BISECT-RESULT-one-comparator-carries-the-
  │                 whole-loss.md)
  │             3. round 1 + round 2, ONE COMPARATOR REVERTED (arm NOS1)
  │                round 1  +3.69 % n=4, sign 4/4, floor max 4.13
  │                round 2  +3.46 % n=8, sign 8/8, floor max 1.37, cleared
  │                         2.5x; bands 9.83-10.04 vs tip 9.47-9.62, DISJOINT
  │                ⇒ the revert BEATS the clean tip by the sealed rule
  │                (2026-09-17-ROUND2-RESULT-the-revert-beats-the-tip.md)
  │             all seven switches OFF vs tip  -12.13 % sign 0/8 (attrib)
  │                and -13.02 % sign 0/4 (round 1) - the levers HELP either way
  │             L = all-off minus as-shipped = -7.62 pp, 3.5x the floor
  │             + DS4_CUDA_HITS_FIRST=1 vs tip  -0.95 % min-across, sign 1/6
  │                (-4.23 / -0.57 / -0.76; only 4096 clears the floor)
  │             prefill   as shipped vs tip +6.41 % min-across (5/7); +2.66 /
  │                       +16.82 / +1.33; secondary, does not offset gen
  │             control = clean tip triple-antirez-tip-latest @e6d9d3b8; its own
  │                       baseline 9.70, round 1 brackets 9.65, round 2 9.57
  │             session  attrib 2026-09-16 06:41-09:08Z, 50 runs; round 1
  │                      2026-09-16, 41 runs, 3 lost to the OOM killer; round 2
  │                      2026-09-17 15:06-16:30Z, 21 runs, zero lost.
  │                      DGX Spark GB10, native .text 23.1 MB vintage
  │             ⚠ EVERY NUMBER ABOVE DESCRIBES A NINE-LEVER BINARY that sits
  │               off this branch's line of history (dd82361a / 8aa40a52 /
  │               d1dd5ec2). THE TEN-LEVER TREE HAS NEVER BEEN MEASURED.
  │               (2026-09-16-CORRECTION-the-measured-stack-is-nine-levers-
  │                and-has-diverged.md)
  │
  │  VERDICT    THE DEFAULT WAS CHANGED TO THE MEASURED WINNER, AND THE NEW
  │             DEFAULT IS UNMEASURED.
  │             What was measured: the stack as it stood LOST to the tip by
  │             -4.35 to -4.51 %, and one std::stable_sort comparator in
  │             ds4_gpu_stream_expert_cache_prefetch carried the whole loss.
  │             Reverting it moved the tree 8.04 points, from -4.35 % to
  │             +3.69 %, and round 2 confirmed +3.46 % at n=8 with bands that
  │             do not touch - the slowest reverted run beat the fastest tip
  │             run by 0.21 t/s.
  │             What followed: least-recently-routed became the shipped victim
  │             order at dd189b2d7, under the standing rule that the default
  │             follows the number and never a card's claim. The sweep-aware
  │             reasoning is kept verbatim in the comment and
  │             DS4_CUDA_PREFETCH_SWEEP_ORDER=1 restores it, so the losing arm
  │             is still there to be re-measured.
  │             ⚠ THIS TREE, WITH THIS DEFAULT, IS NOT MEASURED. Round 3 is
  │             sealed (2026-09-17-ROUND3-PREREGISTERED-RULE.txt, arm FAST at
  │             b17231fd) and has not been run. Nothing here is a number for
  │             the ten-lever tree.
  │
  │  SWITCH     ten levers, see the table; six real off switches, two knobs,
  │             two none. Plus DS4_CUDA_PREFETCH_SWEEP_ORDER=1, which restores
  │             the measured-slower sweep-aware victim order (new at dd189b2d7)
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

The two sealed bisect rounds add their own files:
`2026-09-16-BISECT-PREREGISTERED-RULE.txt` and
`2026-09-16-BISECT-RESULT-one-comparator-carries-the-whole-loss.md` (round 1, 41 runs, four
arms), `2026-09-17-ROUND2-PREREGISTERED-RULE.txt` and
`2026-09-17-ROUND2-RESULT-the-revert-beats-the-tip.md` (round 2, 21 runs, two arms), and the
scope fence `2026-09-16-CORRECTION-the-measured-stack-is-nine-levers-and-has-diverged.md`.
Round 3 is sealed at `2026-09-17-ROUND3-PREREGISTERED-RULE.txt` and has not been run.

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

## The measured record

Every sealed round on this stack and its tip, in order. A round is listed only when its result
file exists; a sealed round that has not run carries no numbers and says so. The verdict column
quotes the deciding number at min-across-frontiers, against that round's own in-run tip floor.

| round | date | question | arms | n | verdict | file |
| --- | --- | --- | --- | ---: | --- | --- |
| tip baseline | 2026-09-16 | where is the zero point? | TIP alone | 16 of 18 | **9.70 gen t/s** median, MAD 0.11, prefill 84.06; the two passes disagree narrowly, so quote the level as 9.6-9.8 | `2026-09-16-triple-antirez-tip-latest-BASELINE.txt` |
| attribution | 2026-09-16 | do the seven switchable levers cause the stack's loss? | TIP, ALLOFF, STACK | 13 / 8 / 7 | **NO, they help.** stack -4.51 % (0/7), all-off -12.13 % (0/8), so the levers are worth **+7.6 pp**; the cause sits in the non-switchable diff | `2026-09-16-triple-all-fastest-attrib-SUMMARY-levers-attribution.txt` |
| bisect round 1 | 2026-09-16 | which slice of the non-switchable diff carries the loss? | TIP, NOS1, STACK, RT, ALLOFF2 | 4 per arm | **ONE COMPARATOR.** stack -4.35 % (0/4) reproduced the loss; reverting the victim comparator gave **+3.69 % (4/4)**, an 8.04-point move. RT -6.07 % is **NOT IT** (1.72 inside the 4.13 floor). ALLOFF2 -13.02 % (0/4) | `2026-09-16-BISECT-RESULT-one-comparator-carries-the-whole-loss.md` |
| round 2 | 2026-09-17 | does the revert tie the clean tip or beat it? | TIP, NOS1 | 8 kept of 10 | **IT BEATS IT.** **+3.46 % (8/8)** against floor max 1.37, cleared 2.5x; bands 9.83-10.04 vs 9.47-9.62, **disjoint**. ⚠ the no-drop sensitivity keeps the direction (9/10, +2.78 %) and fails the floor test (floor 4.22) | `2026-09-17-ROUND2-RESULT-the-revert-beats-the-tip.md` |
| round 3 | 2026-09-17 | are the six primaries and this stack, on the NEW tip, faster than it? | TIP3 plus HITS HOT ENG POOL READ PREF FAST | planned 4 per arm | **SEALED, NOT RUN. No result. No number from this round exists.** | `2026-09-17-ROUND3-PREREGISTERED-RULE.txt` |

**What the record adds up to, in one line.** The stack's loss to the tip is real, repeatable and
attributed to one `std::stable_sort` comparator; reverting that comparator puts the tree ahead of
the tip by about 3.5 %; the default was changed to it at `dd189b2d7`; and the tree carrying that
new default has not yet been measured.

⚠ **THE SCOPE FENCE ON EVERY ROW ABOVE.** The attribution round and both bisect rounds were
measured on binaries built from `dd82361a`, `8aa40a52` and `d1dd5ec2` - **nine-lever trees that
sit off this branch's line of history**
(`2026-09-16-CORRECTION-the-measured-stack-is-nine-levers-and-has-diverged.md`). A measurement
names a sha; a card names a branch; a branch moves and a sha does not. Round 3's FAST arm is the
first measurement of the ten-lever tree, and it has not been taken.

---

## The index: every branch card

One card per branch, the box copied from that branch's own README at its HEAD by
`rebuild_index.py`, grouped by the status the card itself declares. Regenerate,
never hand-edit: a hand-edited row drifts from the branch it describes.
Regenerated 2026-09-16 16:30Z, 24 cards.

## POSITIVELY MEASURED

### triple-hitsfirst

```
  ┌──────────────────────────────────────────────────────────────────────
  │
  │  BRANCH     triple-hitsfirst                                    POSITIVE
  │
  │  WHAT       launches gate/up/down for the experts already resident
  │             before the miss reads land, so a token does not stall on
  │             the slowest read in the batch
  │
  │  RESULT     gen t/s   10.35  vs control 9.56   +8.3 %   floor 4.9 %   n=2
  │             prefill   not measured
  │             control = unpatched tip, interleaved, same native vintage
  │             session  2026-09-15, DGX Spark GB10, native 23.1 MB .text, unstamped
  │
  │             in the stack (2026-09-16, hits-first ON vs clean tip @e6d9d3b8;
  │             that stack is the NINE-lever binary at dd82361a, not this branch):
  │             -4.23 % at 4096 (0/6, clears the 2.89 % floor)
  │             -0.57 / -0.76 / -0.95 % at 6144 / 8192 / min - INSIDE the floor
  │             min-vs-min 9.55 vs 9.55 = +0.00 %, paired n=6
  │
  │  VERDICT    MEASURED WIN solo, INSIDE THE FLOOR in the stack
  │             +8.3 % clears the 4.9 % floor at n=2; in the stack it ties the
  │             tip at three of four readings and loses -4.23 % at 4096
  │             the repair ca5232d40 has NOT been measured anywhere
  │
  │  SWITCH     DS4_CUDA_HITS_FIRST=1 (default OFF in the stack; =0 is wait-then-launch)
  │             DS4_CUDA_HITS_FIRST_STAGED=0 keeps hits-first, single-stage read order
  │  OUTPUT     greedy-identical sha256 bb06e711bc498bb9
  │
  └──────────────────────────────────────────────────────────────────────
```

## NEGATIVELY MEASURED OR KILLED

### triple-all-fastest

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

### triple-granule

```
  ┌──────────────────────────────────────────────────────────────────────
  │
  │  BRANCH     triple-granule                                     NEGATIVE
  │
  │  WHAT       offline, CPU only: does a sparse-distributed (SDR) expansion make a
  │             cheap front-cache lookup more discriminative than a plain hash at the
  │             same address budget? Run on the committed track-3 token streams
  │
  │  RESULT     gen t/s   not measured - no engine code, no switch, not applicable
  │             prefill   not measured
  │             dDISCR, SDR minus plain hash, same readout:
  │               matched entropy  -0.0414 .. -0.2629   plain hash wins 10 of 10
  │               matched bytes    -0.0294 .. -1.6357   plain hash wins 16 of 16
  │             control = the plain multiplicative-XOR hash, same readout, same corpus
  │             session  2026-09-16, laptop CPU, 10.0 s wall, run of record @466ad0ed8
  │
  │  VERDICT    KILLED
  │             the plain hash wins every one of 26 contrasts by more than the sealed
  │             SESOI 0.010; the expansion's address space plateaus near 2^15
  │
  │  SWITCH     NONE - an offline experiment under experiments/granule/, no DS4_* knob
  │  OUTPUT     not re-run - engine files byte-identical to the control
  │
  └──────────────────────────────────────────────────────────────────────
```

### triple-iq2-lut-fix

```
  ┌──────────────────────────────────────────────────────────────────────
  │
  │  BRANCH     triple-iq2-lut-fix                                  NEGATIVE
  │
  │  WHAT       a source check, no lever: does our tree carry upstream's
  │             IQ2 dequant-LUT defect, where the codebook was staged in
  │             shared memory only for n_embd <= 4096 and read unconditionally
  │
  │  RESULT     gen t/s   not measured - the branch changes no executable byte
  │             prefill   not measured
  │             control = none; a throughput row would describe a change that
  │                       does not exist
  │             session  none. Read on 2026-09-16, not run; no Spark build
  │
  │  VERDICT    KILLED - our tree is NOT AFFECTED and no patch is owed
  │             upstream's own fix, a04f46fa42 (2026-09-13), already lifts the
  │             5 staging loops out of the guard; the check found 5 of 5 clean
  │
  │  SWITCH     NONE - no code changed, no knob added
  │  OUTPUT     not re-run
  │
  └──────────────────────────────────────────────────────────────────────
```

### triple-word-finisher

```
  ┌──────────────────────────────────────────────────────────────────────
  │
  │  BRANCH     triple-word-finisher                                NEGATIVE
  │
  │  WHAT       a suffix-lookup drafter: draft k tokens from the continuation
  │             stored against the last n tokens, verify in one batched pass,
  │             commit the longest matching prefix
  │
  │  RESULT     gen t/s   not measured - there is no patch to run
  │             prefill   not measured
  │             acceptance 2.37x code / 1.17x prose (ceiling; 88 % of prose
  │             passes commit one token); NVMe bytes saved 0.0000 at k of 2, 4, 8;
  │             adjacent-token routing overlap 38.90 %; next-step misses already
  │             selected by the previous token 0 of 1,219
  │             session  offline, from a fixed token stream and a sibling lane's
  │             probe dump; zero box time
  │
  │  VERDICT    KILLED
  │             a k-token verify pass reads the same NVMe bytes as k single
  │             passes: the union of k steps' misses equals their sum (0 of 1,219)
  │
  │  SWITCH     NONE - a measurement branch, no code lever
  │  OUTPUT     not re-run - no patch exists to gate
  │
  └──────────────────────────────────────────────────────────────────────
```

## NOT YET MEASURED

### triple-all

```
  ┌──────────────────────────────────────────────────────────────────────
  │
  │  BRANCH     triple-all                                         NOT YET
  │
  │  WHAT       six lever branches stacked in series on 9139e2ae5: the
  │             pread pool, the event-gated selected-expert readback, the
  │             cache reserve knob, the hot list seed, the staged page-drop
  │             order and hits-first. A reference tree, never a submission.
  │
  │  RESULT     gen t/s   not measured
  │             prefill   not measured
  │             control = none run on this tree
  │             session  none
  │
  │  VERDICT    OWED - SUPERSEDED by triple-all-fastest
  │             that tree carries these six plus four more, on the rolled tip,
  │             and it is the one measured (-4.51 % vs the tip, 0 of 7)
  │
  │  SWITCH     the six levers' own switches; see the triple-all-fastest table
  │  OUTPUT     greedy-identical sha256 bb06e711bc498bb9 (prior card; no result file)
  │
  └──────────────────────────────────────────────────────────────────────
```

### triple-climbingfibre

```
  ┌──────────────────────────────────────────────────────────────────────
  │
  │  BRANCH     triple-climbingfibre                                NOT YET
  │
  │  WHAT       a front cache in front of the frozen model that learns from the
  │             speculative verifier's own REJECTIONS, online, with a retention
  │             horizon. It proposes one draft token; it never injects anything
  │
  │  RESULT     gen t/s   not measured
  │             prefill   not measured
  │             control = none run; no CUDA build of this branch exists
  │             session  none - host-only self test, no box, unstamped
  │
  │  VERDICT    OWED
  │             the CUDA build gate is owed; the only acceptance numbers are from
  │             a scripted 61-token target (+139 tokens cold, +0 poisoned)
  │
  │  SWITCH     DS4_CLIMBINGFIBRE=1, default 0 (off allocates nothing, never
  │             learns, never proposes). DS4_CLIMBINGFIBRE_HORIZON=session|day|
  │             week|forever|<steps>, _DECAY=hard|half|none, _NGRAM 4, _CAP 16384
  │  OUTPUT     not re-run
  │
  └──────────────────────────────────────────────────────────────────────
```

### triple-draft-gamma

```
  ┌──────────────────────────────────────────────────────────────────────
  │
  │  BRANCH     triple-draft-gamma                                  NOT YET
  │
  │  WHAT       an adaptive draft-length (gamma) controller for DSpark: one EMA
  │             slot per active-lane bucket, choosing gamma from {0..16}. Zero is
  │             a candidate, so drafting switches itself off where it is not paying
  │
  │  RESULT     gen t/s   not measured
  │             prefill   not measured
  │             control = none run; no CUDA build of this branch exists
  │             session  none - host-only state-machine test, unstamped
  │
  │  VERDICT    OWED
  │             the CUDA build gate is owed; the only run is `make test-draft-gamma`,
  │             a state trace with 0 failures and no t/s in it
  │
  │  SWITCH     DS4_DRAFT_GAMMA_MODE=fixed|adaptive, default fixed (no controller is
  │             allocated; today's constant is returned). Knobs: _ALPHA 0.2,
  │             _UPDATE_INTERVAL 5, _WARMUP 10, _DOWN_HYST -0.25, _UP_HYST 0.0,
  │             _CEILING 1.5, _MAX_STEPS 16, _LOG=1
  │  OUTPUT     not re-run
  │
  └──────────────────────────────────────────────────────────────────────
```

### triple-draincut

```
  ┌──────────────────────────────────────────────────────────────────────
  │
  │  BRANCH     triple-draincut                                     NOT YET
  │
  │  WHAT       reads the router's selected expert ids back on an event
  │             instead of a blocking cudaMemcpy, so the host no longer
  │             waits on the shared expert - 40 device drains per token
  │
  │  RESULT     gen t/s   9.09  vs control 9.17   -0.9 %   floor 4.9 %   n=2
  │             prefill   not measured
  │             control = unpatched tip, interleaved, same native vintage
  │             lever OFF arm (DS4_CUDA_SELECTED_DRAIN_SYNC=1) read 9.71, above control
  │             session  2026-09-15, DGX Spark GB10, native 23.1 MB .text, unstamped
  │
  │  VERDICT    INSIDE THE FLOOR
  │             -0.9 % engaged, 9.71 vs 9.17 dormant, both inside a 4.9 % floor;
  │             the two arms point opposite ways and neither resolves it
  │
  │  SWITCH     ON by default. DS4_CUDA_SELECTED_DRAIN_SYNC=1 restores the
  │             blocking read - that is the OFF arm
  │  OUTPUT     greedy-identical sha256 bb06e711bc498bb9
  │
  └──────────────────────────────────────────────────────────────────────
```

### triple-engram-lead

```
  ┌──────────────────────────────────────────────────────────────────────
  │
  │  BRANCH     triple-engram-lead                                   NOT YET
  │
  │  WHAT       starts the Engram table read one token ahead of first use,
  │             speculating on the current step's own argmax, so the read
  │             overlaps the step instead of landing exposed inside it
  │
  │  RESULT     gen t/s   not measured (this lever alone)
  │             prefill   not measured
  │             control = none run for this lever alone. This switch is one
  │             of the SEVEN in the 2026-09-16 all-off block: stack as
  │             shipped -4.51 % vs tip (0/7), all seven off -12.13 % (0/8),
  │             L = -7.62 pp joint, 3.5x the floor. Not attributable.
  │             session  unstamped
  │
  │  VERDICT    OWED - the lead-on vs lead-off A/B has never been run with a
  │             sound control; the seven levers together are net +7.6 pp
  │
  │  SWITCH     DS4_V41_ENGRAM_LEAD_OFF - on by default; set to anything to
  │             turn the lead off (read once per process, all-or-nothing)
  │  OUTPUT     not re-run
  │
  └──────────────────────────────────────────────────────────────────────
```

### triple-engram-prestage

```
  ┌──────────────────────────────────────────────────────────────────────
  │
  │  BRANCH     triple-engram-prestage                               NOT YET
  │
  │  WHAT       moves the host-side Engram row lookup out of the forward
  │             pass into a prepare phase that runs before it, so the
  │             forward carries no disk read and no hash of its own
  │
  │  RESULT     gen t/s   not measured
  │             prefill   not measured
  │             control = none run. This switch is NOT in the stack and was
  │             in no 2026-09-16 series.
  │             session  unstamped - never built against CUDA
  │
  │  VERDICT    OWED - a precondition, not a payoff: the prepare removes host
  │             I/O from the forward but does not yet hide it behind anything
  │
  │  SWITCH     DS4_ENGRAM_PRESTAGE=1 enables the prepare; unset or 0 is the
  │             shipped inline path. DS4_ENGRAM_PRESTAGE_DEBUG=1 names every
  │             demand-read miss. Read per call, not latched.
  │  OUTPUT     not re-run
  │
  └──────────────────────────────────────────────────────────────────────
```

### triple-hippocampal-warmset

```
  ┌──────────────────────────────────────────────────────────────────────
  │
  │  BRANCH     triple-hippocampal-warmset                          NOT YET
  │
  │  WHAT       gives the persisted expert hot list a longer horizon: carried free
  │             inside a day, halved once per day boundary, promoted to a durable set
  │             when a (layer, expert) pair is demanded on three distinct days
  │
  │  RESULT     gen t/s   not measured
  │             prefill   not measured
  │             control = none run; no CUDA build of this branch exists
  │             session  none - host-only tests, no box, unstamped
  │
  │  VERDICT    OWED
  │             the CUDA build gate is owed; the only runs are `make test-warmset`
  │             (92 checks, 0 failures) and an uncommitted persistence harness (14, 0)
  │
  │  SWITCH     DS4_WARMSET_TIER=session|day, default session = today's behaviour
  │             (one file, halved at every load). DS4_WARMSET_DAYS 3, _MAX <n>,
  │             _DIR <dir>, _PROFILE=1
  │  OUTPUT     not re-run
  │
  └──────────────────────────────────────────────────────────────────────
```

### triple-hotlist

```
  ┌──────────────────────────────────────────────────────────────────────
  │
  │  BRANCH     triple-hotlist                                        NOT YET
  │
  │  WHAT       seeds the next session's SSD expert cache from the previous
  │             run's demand, so the opening tokens start WARM. A warm-up
  │             device, not a selection device.
  │
  │  RESULT     gen t/s   9.38  vs control 9.56   -1.9 %   floor 3.3-4.9 %   n=2
  │             prefill   not measured (floor on that set 9.7-15.8 %)
  │             control = this branch's writer off, interleaved, same native build
  │             session  2026-09-16, DGX Spark, native 23.1 MB .text, load/gpu unstamped
  │
  │  VERDICT    OWED - the seeding arm's -1.9 % is INSIDE THE FLOOR (3.3-4.9 %);
  │             the re-scoped lever, a warm cache at the first 64 tokens, is not measured
  │             the repair 44bc5398c is unbuilt against CUDA and unmeasured
  │
  │  SWITCH     DS4_CUDA_EXPERT_HOTLIST_WRITE=<file> (0 disables the writer)
  │             DS4_CUDA_EXPERT_HOTLIST_DECAY=none|halve|quarter|eighth|shift:<n>, default halve
  │             DS4_CUDA_EXPERT_HOTLIST_ROLLING=1 runs the mine/score measurement
  │             DS4_METAL_DISABLE_STREAMING_EXPERT_HOTLIST=1 disables seeding
  │  OUTPUT     greedy-identical sha256 bb06e711bc498bb9 (three gate runs, one sha)
  │
  └──────────────────────────────────────────────────────────────────────
```

### triple-lfu-offsetkey

```
  ┌──────────────────────────────────────────────────────────────────────
  │
  │  BRANCH     triple-lfu-offsetkey                                 NOT YET
  │
  │  WHAT       ports upstream NeutronStar's host expert cache: byte ranges
  │             keyed by FILE OFFSET, 16-way table, per-slot uses counter
  │             halved every 4096 inserts. Three arms from one binary isolate
  │             the eviction POLICY on one structure
  │
  │  RESULT     gen t/s   not measured
  │             prefill   not measured
  │             control = none run yet; the plan is three arms interleaved
  │                       against the unpatched tip on one vintage
  │             session  none. No Spark build, no run
  │
  │  VERDICT    OWED - a CUDA build and a three-arm sweep on the Spark
  │             the only numbers held are host-side: a 17-check harness of the
  │             eviction block, all passing, on which LFU and LRU disagree
  │
  │  SWITCH     DS4_EXPERT_CACHE_MODE=hotlist|offsetkey|offsetkey-lru
  │             default hotlist, in which the cache is inert (never allocates,
  │             probes or inserts). DS4_CUDA_HOST_EXPERT_CACHE_GB=<GiB> sizes
  │             it; unset, it takes the configured expert-cache byte size
  │  OUTPUT     not re-run; residency changes when a byte arrives, never which
  │
  └──────────────────────────────────────────────────────────────────────
```

### triple-pagecache

```
  ┌──────────────────────────────────────────────────────────────────────
  │
  │  BRANCH     triple-pagecache                                    NOT YET
  │
  │  WHAT       releases staged model pages madvise-then-fadvise so the
  │             drop lands, and hints WILLNEED over a layer's miss ranges
  │             on the buffered read path
  │
  │  RESULT     gen t/s   9.62  vs control 9.56   +0.6 %   floor 4.9 %   n=2
  │             prefill   not measured
  │             control = unpatched tip, interleaved, same native vintage
  │             session  2026-09-15, DGX Spark GB10, native 23.1 MB .text, unstamped
  │
  │  VERDICT    INSIDE THE FLOOR
  │             +0.6 % against a 4.9 % floor; the ordering fix rests on a
  │             reading of the Linux semantics, not on a measurement of its own
  │
  │  SWITCH     DS4_CUDA_KEEP_MODEL_PAGES=1 disables both drops
  │             DS4_CUDA_NO_EXPERT_READAHEAD=1 disables the hint
  │  OUTPUT     greedy-identical sha256 bb06e711bc498bb9 (long d3355c94c70a4bb1)
  │
  └──────────────────────────────────────────────────────────────────────
```

### triple-pool

```
  ┌──────────────────────────────────────────────────────────────────────
  │
  │  BRANCH     triple-pool                                           NOT YET
  │
  │  WHAT       serves expert reads from a parallel SSD pool instead of one
  │             serial reader, with an io_uring O_DIRECT ring in front of the
  │             pool so queue depth is a switch, not the worker count.
  │
  │  RESULT     gen t/s   not measured on this engine
  │             prefill   not measured
  │             control = none run since the io_uring port; the last A/B (9.94 vs
  │                       9.56, +4.0 %; device probe 7.6 -> 10.1 GB/s single vs 8
  │                       readers) was the pthread-only pool and is superseded
  │             session  none - this branch has NEVER been CUDA-compiled, on any
  │                       box, at any sha
  │
  │  VERDICT    OWED, and further back than most: the io_uring engine is
  │             unproven even as a BUILD, so the +4.0 % describes a fetch engine
  │             this tree no longer has
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
  │  BRANCH     triple-prefetch-pool                                NOT YET
  │
  │  WHAT       cuts the prefill read-ahead into chunked tasks for a pool of
  │             readers with their own pinned buffers and upload streams;
  │             the single reader stays as the fallback when the pool declines
  │
  │  RESULT     gen t/s   not measured
  │             prefill   wait per layer 398.0 -> 196.5 ms at four readers;
  │                       no tokens/s or percent figure for prefill is held
  │             control = none with a sound control; the 2026-09-15 pass is withheld
  │             session  2026-09-15, DGX Spark GB10, native 23.1 MB .text, unstamped
  │
  │  VERDICT    OWED
  │             in the 2026-09-15 pass every arm, including the OFF arms, read
  │             above its control (9.17-9.46) - a bad control, not a lever
  │
  │  SWITCH     DS4_CUDA_SSD_PREFETCH_CHUNK_MB, default 8, cap 64; 0 is ignored,
  │             so the chunking has no off arm. DS4_CUDA_SSD_PREFETCH_POOL=0 or
  │             DS4_CUDA_SSD_PREFETCH_THREADS=1 reverts the whole pool
  │  OUTPUT     greedy-identical sha256 2f2dd7f89d107bbc (33,527 bytes, 12 runs)
  │
  └──────────────────────────────────────────────────────────────────────
```

### triple-prefill-readahead-order

```
  ┌──────────────────────────────────────────────────────────────────────
  │
  │  BRANCH     triple-prefill-readahead-order                        NOT YET
  │
  │  WHAT       reorders the prefill read-ahead's victim choice and holds the
  │             earliest layers of the scan, so the experts decode needs first
  │             are the last ones evicted: the prefill hit list survives into
  │             decode. Residency policy only, no byte of cache added
  │
  │  RESULT     gen t/s   not measured - a 2026-09-15 native pass exists on the
  │                       Spark and is WITHHELD: its off arms beat their own
  │                       control, and an off arm is the control
  │             prefill   not measured
  │             control = none sound yet; the 2026-09-15 control read 9.17 to
  │                       9.46 tokens/s against 9.56 in a later mixed pass
  │             session  2026-09-15, GB10, native, load/gpu unstamped here
  │
  │  VERDICT    OWED - a clean interleaved A/B of both arms from a Spark build
  │             the number it has to move: the first 32 decode tokens miss
  │             32.22 experts/token at hit rate 0.8658 on the unpatched tip
  │
  │  SWITCH     DS4_PREFILL_READAHEAD_HOLD, default on; =0 (off/no/false)
  │             restores upstream exactly. DS4_CUDA_SSD_PREFETCH_STATS=1 counts
  │             only. The off switch is new: the previous revision reordered
  │             unconditionally and could not be measured against itself
  │  OUTPUT     not re-run; expected output-invariant, residency changes when a
  │             byte arrives and never which byte
  │
  └──────────────────────────────────────────────────────────────────────
```

### triple-spec-under-offload

```
  ┌──────────────────────────────────────────────────────────────────────
  │
  │  BRANCH     triple-spec-under-offload                             NOT YET
  │
  │  WHAT       does DSpark speculative decoding PAY on a box whose weights
  │             stream from disk? Two arms crossed with a spec axis: RAW (cold,
  │             page cache evicted) and WARM WORKING SET (resident before the
  │             measured block). Four legs, one binary, no engine code added
  │
  │  RESULT     gen t/s   not measured, all four legs
  │             prefill   not measured
  │             control = per arm, the same argv with DS4_MTP_SPEC_DISABLE=1,
  │                       drafter loaded on both legs, interleaved on one vintage
  │             session  none. No Spark build, no run, no lock taken
  │
  │  VERDICT    OWED - the CUDA build and all four legs need the Spark
  │             the warm arm is the one nobody has run: every DSpark A/B we hold
  │             ran WITHOUT residency, because the F16 drafter at 117.73 GiB
  │             does not fit beside the 115 GiB the GB10 offers
  │
  │  SWITCH     --dspark --mtp-model <file>; off leg DS4_MTP_SPEC_DISABLE=1;
  │             --ssd-streaming, --ssd-streaming-cold, --ssd-streaming-cache-
  │             experts N|NGB (all pre-existing, ds4_help.c / ds4_server.c)
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
  │  RESULT     gen t/s   9.70 median (min 9.44, max 9.92, MAD 0.11)   n=16 of 18
  │             prefill   84.06 median (79.13 - 87.19)                  n=16 of 18
  │             control = this branch IS the control; nothing to compare against
  │             session  2026-09-16 04:27-05:38Z, DGX Spark GB10, native .text
  │                      23,118,434 B, box quiet by the harness gate (load < 2.0,
  │                      memavail 117 GB, gpu < 50 %), ctx 2048 discarded
  │
  │  VERDICT    CONTROL
  │             the zero point: 9.70 gen t/s, in-run floor median 1.0-1.5 %
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
  │  BRANCH     triple-prefix-cache                                   CONTROL
  │
  │  WHAT       the prefix cache the engine already ships (ds4_kvstore.c), read
  │             out of the code, plus the one thing it lacked: an off path.
  │             --prefix-cache off rewinds the live session at every request.
  │
  │  RESULT     gen t/s   not measured
  │             prefill   not measured
  │             control = this branch off (--prefix-cache off), same binary; not run
  │             session  none - no CUDA build, no run (the Spark was busy)
  │
  │  VERDICT    CONTROL - a duplicate of the shipped prefix cache re-scoped as its
  │             control arm; the sweep against --prefix-cache off is owed
  │
  │  SWITCH     --prefix-cache on|off, default on; off = rewind to token zero,
  │             nothing read from disk, no checkpoint written
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
  │  WHAT       a design record for re-encoding ONLY the two Engram hash
  │             tables from FP8 at 264 B a row to 4-bit at 132 B a row.
  │             Ships the arithmetic, the four costs, the plan, and ZERO
  │             engine code.
  │
  │  RESULT     gen t/s   not measured - there is no arm: no quantizer, no
  │                       re-encoded artifact, no 132-byte dequant path
  │             prefill   not measured
  │             control = none. Bytes are arithmetic on our own record:
  │             94.417 GiB off the file (202.758 GB to 101.379 GB), and
  │             6,336 B a token = 0.0000655 % of the 9.670 GB/token floor,
  │             27.5 ns at 230.5 GB/s.
  │             session  n/a
  │
  │  VERDICT    TOOLING, NOT A LEVER - halving a row cannot halve a round
  │             trip: the measured 23.9 ms Engram read moves 0.53 MB/s, so
  │             the axis is IOPS (triple-engram-read-threads), not bytes
  │
  │  SWITCH     NONE - nothing to switch; the re-encode changes the artifact
  │  OUTPUT     not re-run - nothing runs; the quantizer skeleton encodes
  │             zero rows by design
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
  │             three consumers. An instrument and a bound, not a speed lever.
  │
  │  RESULT     gen t/s   not measured (the branch withdrew its number)
  │             prefill   not measured
  │             control = none run; the old 8.68 vs 9.56 (-9.2 %) belonged to the
  │                       retired expert-cache-margin knob on a different tree
  │             session  none - no CUDA build (the Spark was busy); macOS host
  │                       binary and two CPU-only test binaries built and pass
  │
  │  VERDICT    OWED - nothing run; the default path is arithmetic-identical to
  │             before, which is an argument and not a measurement
  │
  │  SWITCH     DS4_MEM_RESERVE_MIB=N default 512 · DS4_MEM_RESERVE_ENFORCE=1|0
  │             default 1 (a breach stops) · DS4_MEM_RESERVE_RECORD=FILE (JSON line
  │             per run) · DS4_MEM_RESERVE_FLOOR_HISTORY_MIB (ladder, record only)
  │             DS4_SSD_PAGECACHE_FLOOR_PCT default 10 (range 1-40)
  │             DS4_SSD_AUTO_CACHE_PCT default 80, now a SHARE, not a maximum
  │  OUTPUT     not re-run
  │
  └──────────────────────────────────────────────────────────────────────
```

## UNTAGGED - the card carries no status

### triple-engram-read-threads

```
  ┌──────────────────────────────────────────────────────────────────────
  │
  │  BRANCH     triple-engram-read-threads                 CORRECT, WORTH ZERO
  │
  │  WHAT       takes the DECODE step's Engram read off the batch machinery:
  │             one token now goes straight to the serial reader, with no
  │             malloc, no qsort and no pool dispatch. The wide PREFILL read
  │             keeps the one-wave rule the branch was named for.
  │
  │  RESULT     gen t/s   MEASURED, AND IT BUYS NOTHING
  │             the bisect's RT arm is this same decode revert, made on the
  │             nine-lever stack at dd82361a: -6.07 % vs the clean tip, where
  │             the stack as shipped read -4.35 %. |D_RT - D_STACK| = 1.72,
  │             inside the 4.13 % floor, so NOT IT by the sealed rule and no
  │             direction is reported
  │             (2026-09-16-BISECT-RESULT-one-comparator-carries-the-whole-loss.md)
  │             prefill   not measured. The one-wave PREFILL read has never run
  │             session  the RT binary is tb-bis-rt @d1dd5ec2, native 23,147,372
  │             .text, 2026-09-16. THIS revision has never been built for CUDA
  │
  │  VERDICT    CORRECT, TESTED, AND WORTH NOTHING ON THE CLOCK.
  │             The decode fast path is right: it takes a malloc, a qsort and
  │             a broadcast to 32 parked workers off the critical path, and a
  │             counter proves the pool never wakes. The bisect measured that
  │             same revert at 1.72 points, inside the floor.
  │             A fix can be right, well tested, and buy nothing.
  │             OWED: the one-wave PREFILL read, which is the untested half
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

