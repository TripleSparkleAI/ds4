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
  │  BRANCH     triple-hotlist                                        NOT YET
  │
  │  WHAT       seeds the next session's SSD expert cache from the previous
  │             run's demand, so the opening tokens start WARM. A warm-up
  │             device, not a selection device.
  │
  │  LATEST     never in a sealed round
  │             round 3 is measuring this branch now: in flight, unresolved
  │
  │  GEN        9.38 t/s  vs control 9.56  -1.9 %  floor 3.3-4.9 %  n=2
  │             control = this branch's writer off, interleaved, same native build
  │             session = 2026-09-16 · DGX Spark · native 23.1 MB .text · load/gpu unstamped
  │  PREFILL    not measured
  │
  │  VERDICT    NOT YET - the seeding arm's -1.9 % sits inside its own floor, so
  │             it decides nothing, and the lever as re-scoped, a warm cache over
  │             the first 64 tokens, has never been measured at all. The repair
  │             44bc5398c has no CUDA build behind it.
  │
  │  SWITCH     DS4_CUDA_EXPERT_HOTLIST_WRITE=<file> (0 disables the writer)
  │             DS4_CUDA_EXPERT_HOTLIST_DECAY=none|halve|quarter|eighth|shift:<n>, default halve
  │             DS4_CUDA_EXPERT_HOTLIST_ROLLING=1 runs the mine/score measurement
  │             DS4_METAL_DISABLE_STREAMING_EXPERT_HOTLIST=1 disables seeding
  │  OUTPUT     greedy-identical sha256 bb06e711bc498bb9 (three gate runs, one sha)
  │
  └──────────────────────────────────────────────────────────────────────
```

## What the branch does

- The CUDA session counts every unique `(layer, expert)` the streaming cache is asked for,
  once per batch.
- At exit it writes them hits-descending to `~/.cache/ds4/cuda_expert_hotlist.txt`, under a
  `# model_size` header, through a temp file renamed into place.
- `ds4_session_create` loads that list, ages each count by the decay rule, and seeds EMPTY
  slots only, before the first prefill, through `ds4_gpu_stream_expert_cache_seed_experts`.
- Seeded slots are not counted as demand, so seeding cannot flatter itself.
- A missing, unreadable or wrong-model list warns and starts cold.
- Four files: `ds4.c`, `ds4_gpu.h`, `ds4_cuda.cu` (its module comment is the map), this README.
  The repair adds `ds4_hotcache.h` and `tests/test_hotcache.c`.

```
   THE VICTIM SCAN, 8 misses against a 4096-slot cache
   the test's own numbers, tests/test_hotcache.c

   BEFORE   miss 1    ████████████████  4096 slots read
            miss 2    ████████████████  4096
             ...      the early exit fires only on an EMPTY slot, and
            miss 8    ████████████████  seeding had filled every one at startup
                      ────────────────
                      32768 slot reads   per missed expert, per routed layer,
                                         per token

   AFTER    one pass  ████████████████  4096 slot reads
                      ────────────────
                      4096              the SAME 8 victims, in the SAME order

   COLD CACHE, the early exit still intact
            one pass  ███               stops at the 4th empty slot: 10 of 64 read
```

## Why the claim shrank to warm-up

- `WIKI/theory/167`, the content control, measured a memory table's PRESENCE effect at
  z = +7.14, sd 0.00 across seeds.
- Its CONTENT effect had a sign that flipped between seeds.
- Presence is 27x to 45x the content effect.
- A slot-occupancy cache is exactly that regime: a wrong seed wastes one slot and corrupts
  nothing.
- So the honest claim is "the cache is warm at the first token", and the axis to measure is
  warmth, not ranking.
- `WIKI/theory/48` section 4 puts the warm-up effect at cold resident 15.87 vs warm resident
  35.14 t/s, a factor of 2.2.
- Its section 4a puts the `--warm-weights` flag at +4.4 % on the M5 and -5.4 % on the
  streaming Spark. The flag is the small one.
- Section 8 ranks hot-expert prefetch (Lever C) as the Spark lever, and says "not bigger cache".

## Measured so far

**The seeding arm, native (`make cuda-spark`, sm_121a).**

- 9.38 vs 9.56 gen t/s, **-1.9 %**, inside the 3.3-4.9 % floor.
- Two repeats per arm, first frontier discarded, minimum across repeats.
- Writer output: 5,177 experts after a short run, 8,679 after a long one.
- Startup seeded 1,278 / 1,606 / 1,450 slots across three gate runs, with one identical sha.
- Earlier JIT-vintage figures (29.6 MB .text) are superseded and not carried here.

**This branch's writer switch inside the 2026-09-16 bisect.**

- `DS4_CUDA_EXPERT_HOTLIST_WRITE=0` is one of the switches the bisect's ALLOFF2 arm turned off.
  That arm turns off the WRITER, not the seeding.
- ALLOFF2 measured **-13.02 %** against the tip, where the stack as shipped measured **-4.35 %**.
  The gap is 8.67 points, twice the 4.13 % floor
  (`2026-09-16-BISECT-RESULT-one-comparator-carries-the-whole-loss.md`).
- ⇒ **the switchable levers HELP, jointly.** Nothing in that arm attributes any part of the
  8.67 points to this lever alone.
- ⚠ That arm is the NINE-lever binary at `dd82361a`, a sibling of `triple-all-fastest` off
  `84ba6ef1b` (`2026-09-16-CORRECTION-the-measured-stack-is-nine-levers-and-has-diverged.md`).
  The ten-lever tree has never been measured.

## Open question: LFU-with-aging vs LRU

- Upstream NeutronStar (`WIKI/research/neutronstar/NEUTRONSTAR_02_expert-streaming.md`
  section 1) evicts by LFU with aging, `uses >>= 1` every 4096 inserts, and says pure LRU
  thrashes.
- Our own `WIKI/theory/177`, the sparkport day, measured hotness-with-decay WORSE than plain
  LRU: hit rate 0.851 vs 0.868, 4.95 vs 5.28 t/s. It was reverted.
- Unresolved whether our implementation or their keying explains the opposite sign.

## ROLLINGSPLIT, the decay rule as a measurement

- `halve` was invented here. It was never measured.
- `DS4_CUDA_EXPERT_HOTLIST_ROLLING=1` keeps a ring of demand windows, mines
  `[t-k-gap, t-gap)`, cuts the result to the slots the cache actually has, and scores its
  coverage of window `t` under each decay rule.
- `_K`, `_GAP`, `_RULES`, `_STRIDE` and `_WINDOWS` are runtime ladders.
- `_MODE=report` is the default: it writes a coverage grid and a CSV and changes nothing.
  `_MODE=act` writes the mined window as the list.
- Off the box, on synthetic Zipf demand:
  - `k=512` covered **71.3 %** of the next window, against **15.1 %** at `k=8`.
  - At `k=8` the decay rule cost coverage monotonically: `none` 15.07 %, `halve` 3.30 %,
    `quarter` 0.05 %.
  - After a pool replacement, coverage fell from 25.0 % at `gap=0` to 12.5 % at `gap=256`.
- ⚠ Those are harness numbers. They are not Spark evidence.

## The repair carried on this branch (`44bc5398c`)

Four defects from REVIEW-R4, smallest correct change each. The two things `ds4_cuda.cu`
DECIDES moved into `ds4_hotcache.h`, a pure header with no I/O, no CUDA and no allocation,
so a C test can reach them on a laptop. Same shape as `ds4_warmset.h` on
`triple-hippocampal-warmset`.

- **H3, a silent wrong value.** `DS4_CUDA_EXPERT_HOTLIST_DECAY=shift:<junk>` selected "none".
  `strtol(env + 6, NULL, 10)` with no end pointer returns 0 on `"shift:x"`, and 0 is the valid
  shift "none", meaning carry the whole history forward. That is the opposite end of the axis
  from the default. The value was wrong, the run looked fine, and nothing said anything.
  - The rule is now read whole or refused. A refusal keeps `halve`, and the message names the
    syntax.
  - The rolling-rules parser two functions down already read the same syntax this way. The two
    now agree.
- **H2, a full victim walk per miss.** The scan exits early only on an EMPTY slot. Seeding
  fills every free slot at startup, so from token one there are none, and a full cache cost a
  full walk per missed expert, per routed layer, per token.
  - **The policy is unchanged**: least-recently-used first, ties to the lowest slot index,
    never a prefetch-held slot, never a slot this batch has already stamped.
  - **The seeding is unchanged too**, because presence is 27x to 45x the effect of content
    (`WIKI/theory/167`) and a changed eviction policy has already measured worse once
    (`WIKI/theory/177`).
  - What changes: ONE pass answers a whole batch of misses instead of one pass per miss. The
    old loop's only mutation between passes was stamping the victim it had just taken, which
    the next pass would have skipped anyway (`used < stamp` is the candidate test).
  - The early exit survives whole. A cold cache still stops at the first M empty slots.
  - The protection mask is built only while a look-ahead reader is in flight.
- **H1 (part), a getenv on every demand record.** `getenv("DS4_CUDA_EXPERT_HOTLIST_ROLLING")`
  ran on every routed layer of every token, because `g_rolling.initialized` is only ever set
  inside the call the guard protects, so with rolling off it never became set. The switch is
  now read once, in the same block as every other one-time read. No behaviour change: with the
  variable unset or `=0` the window was built and discarded, and now is not built.
- **H5, a truncated list could replace a good one.** Both writers tested `fclose` and not
  `ferror`, and `fclose` reports only the LAST flush, so a write error on an earlier flush (a
  disk that filled mid-list) could be followed by a successful empty flush and a rename. On
  any write error the temp file now goes and the previous list stays. An old list is a worse
  seed; a truncated one is a lie about what the run demanded.
- No default changed, no switch direction changed, no output byte changed.
- ⚠ **No performance claim is made for the repair.** What was removed is named above. The
  number is the bench's.

## The test for the repair

`tests/test_hotcache.c`, 46 checks, `make test-hotcache`. No GPU, no model, no CUDA.

- The victim test carries a transcription of the pre-fix loop and asserts both that the
  victims are identical over 400 random caches and that a batch costs one pass.
- RED for H3, with the shipped parse restored: 16 of 46 checks fail, headline
  `"shift:x" was refused, so the shift must stay at the default halve, not 0`.
- RED for H2, with the per-miss shape restored: 3 of 46 fail,
  `a full cache must cost ONE pass over 4096 slots, cost 32768` and
  `the early exit must stop the pass at the 4th empty slot, examined 10 of 64`.
- Re-run on this Mac 2026-09-17 (`make test-hotcache`):
  **46 checks passed (no GPU, no model, no CUDA).**
- ⚠ `ds4_cuda.cu` is an nvcc translation unit and does not build on this host at all. That is
  why the two decisions live in a header a C test can reach.

## Owed

- The warm-cache A/B on the Spark: the first 64 tokens reported apart from a 256-token run,
  warm against cold.
- The ROLLINGSPLIT curve on a real demand stream, rather than synthetic Zipf.
- The CUDA build gate. Both the ROLLINGSPLIT change and the repair `44bc5398c` are committed
  unbuilt.
- A measurement on the new tip `triple-tip-2026-09-16`. Every number on this card pre-dates
  the repair.
