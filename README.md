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

Rebased onto triple-tip-2026-09-16 (12997e9c8) on 2026-09-17; tests make test 29 verdicts all pass (it stops at ds4_test, whose model is absent in a worktree, identically before and after), make test-draft-gamma ok 0 failures, cc -fsyntax-only clean on 3 touched C files. ⚠ Re-counted on this Mac 2026-09-17: **26** verdicts, all pass, stopping at ds4_test as before. Counting the run's `PASS` and `ok` verdict lines does not reproduce 29, on this branch or on any of the eight, so 26 is the current figure and 29 is superseded. `make test-draft-gamma` re-run the same day: **ok, 0 failures**.

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

## The mechanism

- Today gamma is the support model's GGUF `dspark.block_size`, read once by
  `ds4_session_dspark_fixed_gamma`, clamped at 16 by `DS4_DSPARK_MAX_BLOCK_SIZE`, and never
  revisited for the rest of the process.
- `ds4_draft_gamma.c/.h` keeps, per lane bucket (1, 2, 4, 8, 16, 32+), an EMA of the accept length
  the DSpark loop already computes. No new counter and no new timing path is added.
- The EMA is updated every 5 batches after a 10-batch warmup.
- It probes one step past observed acceptance: `target = clamp(round(ema) + 1, 0, max_steps)`.
- A hysteresis band makes it hold rather than oscillate; the ceiling caps downward only.
- At gamma 0 drafting is disabled for that bucket, rested one interval, then re-probed at gamma 1,
  so zero is not absorbing.

```
   THE CONTROLLER WALKING DOWN AND BACK UP, from the state trace

   accept length 0 every batch
     gamma  4 ─▶ 3 ─▶ 2 ─▶ 1 ─▶ 0 ·rest· 1 ─▶ 0 ·rest· 1 ─▶ 0
                                  ╰── 13 zero intervals, 12 re-probes
                                      in 38 decisions: zero is NOT absorbing

   every draft lands
     gamma  2 ─▶ 4 ─▶ 8 ─▶ 16   and stops at _MAX_STEPS
   ceiling 1.5 against an EMA of 4
     gamma  2 ─▶ 2              held DOWN, never held up

   the engine's own cap of 5 was never exceeded in any trace
```

## What is measured, and what is not

- `make test-draft-gamma`, standalone, no GPU and no model: **ok, 0 failures**, re-run 2026-09-17.
- Traces in that run: gamma walks 4 to 0 on accept length 0 (13 zero intervals, 12 re-probes in 38
  decisions), climbs 2 to 16 when every draft lands, holds inside a widened band, the ceiling
  holds gamma at 2 against an EMA of 4, bucket 0 collapses while bucket 5 holds 6, and an engine
  cap of 5 is never exceeded.
- Those are STATE TRACES. None of them is a speedup, and none of them ran a model.
- A known narrow band: with integer accept lengths the hold band is
  `(prev_step - 0.75, current_step - 0.5]`, so in steady state the controller cycles between gamma
  and gamma + 1. `DS4_DRAFT_GAMMA_UP_HYST` widens it, and the sweep is what decides the value.

## Why the claim is narrow

- `WIKI/theory/48-decode-speedup-levers.md` section 8 rules more spec-decode "settled, do not",
  because it adds compute to a bandwidth-bound step. This branch REMOVES compute.
- `WIKI/theory/50-moe-verify-budgeting-evict.md` section 1 puts the real bottleneck at the verify
  slope kappa of about 0.32, not at acceptance. This branch does not pull that lever.
- `WIKI/theory/34-synthesis-dspark-implications.md` section 2's action item 1, the
  confidence-head prefix cut, already exists in the tree as `dspark_confident_prefix_len` in
  `ds4.c`. What is left there is sizing theta, not building a component.
- `MEASURED_OPTK_*_2026-09-01.md` (track 3): on code the acceptance-optimal k is 16 while the
  throughput-optimal k is 4, and shipping k=16 would have cost 31 %; on prose k=2 is optimal and
  k=16 runs at 0.8476x plain decode. Acceptance alone picks wrong on both, and that file is
  batch 1 only. A controller reading what the loaded box achieves is the answer to that fence, and
  not a substitute for measuring.
- The closed `triple-word-finisher` finding (a k-token verify reads the same bytes as k single
  passes) is about BYTES and does not kill this branch, which is about not wasting verify passes.

## Owed

- A CUDA build on the Spark, then `fixed` against `adaptive` interleaved at 1 and 32 lanes with
  `DS4_DRAFT_GAMMA_LOG=1` and `DS4_DSPARK_STATS=1`, then the greedy-identical check.
- The expected visible effect is a SMALLER LOSS where fixed gamma overspent, not a win. If the
  A/B reads a win, that is a surprise and wants explaining before it is believed.
- Files: `ds4.c`, `ds4_draft_gamma.c`, `ds4_draft_gamma.h`, `tests/test_draft_gamma.c`, `Makefile`,
  `.gitignore`. Pre-rebase base `21d323dfe`, pre-rebase code commit `ef6889465`.

## Round 5, and the number this card now carries

- **Round 5 built and ran this branch for the first time**: +3.83 % gen, 4 of 4 positive, and a TIES verdict only because the box was noisy. The `NOT YET` above is therefore about the instrument, not about the lever.
- The new rule reads +3.83 at n=4; the sealed legacy rule reads +4.39 at n=2 and keeps that n where the result file says so. Both give TIES under the 5.69 % floor.
- Its cycle-1 run waited 210 s at the load gate and started at load 1.61, which is one of the two contention events the result file names for this round.
