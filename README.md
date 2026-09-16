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





**✦   ✧   ✦   ✧   ✦   ✧   ✦   ✧   ✦**


**✦ ✧ ✦ ✧ ✦ ✧ ✦ ✧ ✦ ✧ ✦ ✧ ✦   T R I P L E S P A R K L E   ✦ ✧ ✦ ✧ ✦ ✧ ✦ ✧ ✦ ✧ ✦ ✧ ✦**

**✦ above: the upstream README, unchanged · below: this branch's card and numbers**

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

## What the branch does

- The LUT decode kernel takes a `pair_mask` and returns early for any pair the launch does not cover.
- Gate/up, and a per-slot down partial, run for the experts already resident, while the miss reads are still in flight.
- After the wait, the same kernels run again for the miss slots.
- The six partials are summed in slot order, from `0.0f`. That is the float sequence
  `moe_down_sum_qwarp32_kernel<6>` already ran, which is why the output is bit-identical.
- A second stage reorders the pool's pick-up order: every miss's gate/up first, then the downs.
- The branch carries the parallel expert pread pool underneath, because it calls the pool's own
  entry points. The pool is what makes the misses concurrent.

```
   ONE TOKEN, hits-first ON: who drains, who launches, who waits

   as shipped   wait for EVERY miss read · launch all six · sum slots 0..5

   this branch
     1  drain the batch still in flight?     ds4_hitsfirst_must_wait, asks the quant
           Q4_K    YES    was NO DRAIN         launched gate/up over victim slots the
           MXFP4   YES    was NEVER REACHED    pool was still uploading: a real race
           IQ2     NO     unchanged, on purpose - the overlap IS the branch
     2  dispatch the miss reads
     3  launch the HIT mask now               pair_mask, resident experts only
     4  wait_stage
     5  launch the MISS mask
     6  sum_partials6, slots 0..5, from 0.0f  the same float sequence as the fused
                                              kernel, so the sha does not move
```

## The measurements, kept apart

**Solo, 2026-09-15 native pass. The win.**

- 10.35 vs 9.56 gen t/s, **+8.3 %**.
- Two repeats per arm, interleaved control, minimum across the three stable frontiers.
  ctx 2048 discarded as warmup.
- The control-to-control floor of that native set is 3.3-4.9 % gen and 9.7-15.8 % prefill.
- This is the only generation delta in the five-branch series that cleared its floor.

**In the stack, 2026-09-16 phase 2. Inside the floor at three readings of four.**

- File: `2026-09-16-triple-all-fastest-attrib-hits-first-on.txt`.
- Arm: `triple-all-fastest` @dd82361a with `DS4_CUDA_HITS_FIRST=1`, against nine fresh
  clean-tip brackets @e6d9d3b8.
- 17 runs, all rc=0, 0 contention drops, box load under 2.0 and gpu 0-21 % at every stamp.
- Paired medians **-4.23 / -0.57 / -0.76 / -0.95 %** at 4096 / 6144 / 8192 / min.
  Signs 0/6, 1/6, 1/6, 1/6.
- The n=8 no-straggler-drop sensitivity gives the same medians.
- Prefill paired median, secondary: +1.60 / +15.29 / +4.33 / +4.13 %.
- The as-shipped stack, hits-first OFF, lost -8.63 / -3.75 / -5.38 / -4.51 % against the same
  tip design in phase 1. Turning this one lever on closes most of that deficit at three of
  four readings.
- The arm was sealed as exploratory (amendment A1). It is not part of the sealed verdict on
  the seven kill switches.
- ⚠ **The subject is narrower than the word "stack".** That binary carries NINE levers, sits at
  `dd82361a`, and is not on `triple-all-fastest`'s line of history: the two are siblings off
  `84ba6ef1b` (`2026-09-16-CORRECTION-the-measured-stack-is-nine-levers-and-has-diverged.md`).
  The ten-lever tree has never been measured.

**Superseded, not refuted.**

- An earlier on/off A/B on JIT-fallback binaries (29.6 MB .text, no `-gencode`) read
  8.31 vs 8.61 t/s, a null.
- JIT and native vintages are not comparable: prefill is about 37 t/s against about 86 t/s.
- The structural explanation once given for that null, that the shared expert's gate/up/down
  already fill the interval, is unproven.

**Elsewhere.** The same change measured about +5 % on our own CUDA backend, a different tree.
Cited only to say where the idea pays.

## The repair carried on this branch (`ca5232d40`)

Four findings from R1 and from HUNTTHECOST C2, fixed on the branch that introduced them. The
decision logic moved into `ds4_hitsfirst_logic.h` so a host with no CUDA toolkit can test it.

- **A real race, on two paths.** `begin_load` leaves a hits-first batch in flight whenever
  `slot_count <= 8`, and it does not know the expert quant. The only pre-launch wait asked
  `g_hits_first.active && !(n_tokens == 1u && use_decode_lut_gate)`, which does not ask about
  the quant either.
  - Q4_K experts skipped the wait, then launched gate/up over victim slots the pool's workers
    were still uploading on their own `cudaStreamNonBlocking` streams. Nothing ordered those
    streams before the kernel.
  - MXFP4 is worse in shape and easier to miss: it returns from `routed_moe_launch` well before
    the wait line, so it never drained at all. It now drains at the top of its own block.
  - The predicate is `ds4_hitsfirst_must_wait`, and it asks the quant.
  - The IQ2 one-token decode-LUT path still does NOT drain. Keeping that overlap is the whole
    point of the branch, and a test case is the control for it.
  - Latent only because the IQ2 daily driver routes to the LUT branch.
- **A leak.** `g_hf_partials` was never freed. `cuda_hf_partials_release()` now runs inside
  `ds4_gpu_stream_expert_cache_release_resident`, after the hits-first wait that function
  already does, so no kernel can be reading the partials.
- **A latent correctness bug.** `moe_down_slot_partial_qwarp32_kernel` mapped a negative slot
  onto expert 0 and relied on the LUT kernel having zeroed that pair's `mid_out`. The fused
  kernel it claims bit-identity with, `moe_down_sum_qwarp32_kernel`, skips the slot instead.
  Both now share one rule, `ds4_hitsfirst_slot_contributes`: the partial kernel writes `0.0f`
  and returns, which is what makes `moe_down_sum_partials6_kernel`'s sum equal the fused skip.
- **Host time spent in a GPU-idle window.** `begin_load` ran a `getenv`, a heap allocation and
  a clock on every call, hit or miss, 40 layers per token. By HUNTTHECOST 0.1 that function
  sits between a `cudaStreamSynchronize` of the decode stream and the routed launch, so host
  time there is GPU idle time. Removed:
  - the `getenv("DS4_CUDA_EXPERT_CACHE_STATS")` linear scan of environ, now read once via
    `ds4_env_gate`;
  - the `std::vector<char> was_miss(unique.size())` heap allocation, now a uint64 mask, which
    the hits-first cap of `slot_count <= 8` makes exact;
  - `cuda_wall_sec()`, now taken only when stats are on.
  - ⚠ **No number is claimed for any of this.** What was removed is named; the number is the
    bench's.
- **Behaviour note.** `DS4_CUDA_EXPERT_CACHE_STATS` is now sampled once, at the first
  `begin_load` of the process, and cannot be flipped mid-run. It is a diagnostic switch in a
  per-layer hot path, and this is the shape `cuda_hits_first_enabled` already has.
  `g_stream_expert_sec_read` likewise accumulates only when stats are on, and its only reader
  is the stats line itself.
- No default, no switch direction and no output byte changed.

## The test for the repair

`tests/test_hitsfirst_logic.c` is pure C99, with no CUDA and no model. 15 cases, one
independent control each.

- The drain predicate is swept over the whole (`n_tokens`, lut_gate, quant) space, so exactly
  one path consumes its own batch and none reads without draining.
- The negative-slot rule is EXERCISED, not asserted: host models of both reductions run over
  the same selection and must agree.
- The getenv saving is asserted on a read counter through a substituted reader, so it is a
  counted fact.
- RED, with all four pre-fix rules restored in the header: 13 of 142 checks fail, at
  `paths: 3 consume their own batch` and `env reads: 4000 over 4000 calls`.
- GREEN: `paths: 1 consume their own batch, 0 read without draining`,
  `env reads: 1 over 4000 calls`, 142 checks, 0 failed.
- Re-run on this Mac 2026-09-17 (`make tests/test_hitsfirst_logic && ./tests/test_hitsfirst_logic`):
  **142 checks, 0 failed, exit 0.**
- ⚠ NOT exercised: no Q4_K-expert model, no MXFP4 model, no GPU, and `ds4_cuda.cu` does not
  compile on this macOS host. The race is asserted on the predicate, never observed.

## Still open

- A native solo on/off repeat. n=2 is the whole solo sample.
- The 4096 deficit that survives hits-first in the stack. It points at the non-switchable
  branch diff, or at an untested switch.
- A CUDA build, and a measurement on the new tip `triple-tip-2026-09-16`. Nothing on this
  branch has been measured since the repair landed, so the card's numbers all pre-date it.
