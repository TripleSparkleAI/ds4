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

---





**✦   ✧   ✦   ✧   ✦   ✧   ✦   ✧   ✦**


**✦   ✧   ✦   ✧   ✦   ✧   ✦   ✧   ✦**

**✦✦✦  ✧  T R I P L E S P A R K L E  ✧  ✦✦✦**

**✦ above: the README, unchanged**

**✦ below: our modifications and numbers for this branch**

```
  ┌──────────────────────────────────────────────────────────────────────
  │
  │  BRANCH    triple-margin
  │
  │  WHAT           turns the expert-cache size into a knob, trading
  │                 resident experts against the RAM left to everything
  │                 else on the box
  │
  │  RESULTS              tokens/s        tip      change       floor
  │    generation             8.68       9.56      -9.2 %       4.9 %
  │    prefill                ____       ____        ____      15.8 %
  │
  │  VERDICT        OUTSIDE the floor: a real regression at this
  │                 measurement
  │
  │  SWITCH         is the reserve itself; the size goes up, the margin
  │                 goes down
  │  HEADLINE       5,677 slots @0.918 hit vs 5,234 @0.901 (probe, other
  │                 tree)
  │  OUTPUT         greedy-identical, sha256 bb06e711bc498bb9
  │
  └──────────────────────────────────────────────────────────────────────
```

**Standard results table**

| arm / measurement | value | change |
| --- | ---: | ---: |
| tokens/s - control (tip, unpatched) | 9.56 | - |
| tokens/s - this branch | 8.68 | -9.2% |
| slots / hit rate at margin 8 vs 12 | 5,677 @0.918 / 5,234 @0.901 | trade |

*Measured on the DGX Spark, one arm against its own interleaved control, minimum
across three stable frontiers (the first frontier of each run is discarded as warmup).
The change is stated beside the noise floor because that is the only honest way to read
it: a delta smaller than the floor is not a win.*

- **Where it lives**: one block in `cuda_stream_selected_cache_begin_load` in `ds4_cuda.cu`, +17/-1 outside the stats print. No data path, no read ordering, no arithmetic change.
- **The sizing chain, in order**: the cache starts from `ds4_gpu_stream_expert_cache_budget_for_expert_size`; that budget is then clamped by free memory, where `cudaMemGetInfo` is replaced by `min(host nonmovable, total device bytes)` when the device reports itself integrated and `ds4_linux_nonmovable_memory` can answer, because on unified memory the device free figure and real host headroom diverge; then `available = free - reserve`, zero if the reserve exceeds free; then `capacity = min(capacity, available / expert_bytes)`.
- **The reserve parse is strict**: `DS4_CUDA_EXPERT_CACHE_MARGIN_GB` is read with `strtoul`, taken only when the whole string parses and the value is at most 120, otherwise left at 8. With the variable unset the reserve is exactly 8 GiB, so a tree that does not set it behaves as before, byte for byte.
- **Why a knob instead of a better constant**: `cudaMalloc` overcommits, so an arena sized past what the host can really give is accepted and then killed on first touch. That surfaces as an OOM kill during warm-up rather than a clean allocation failure, which means the reserve is the only dial between arena size and being the OOM killer's first pick on a shared box.
- **Who still needs the reserve**: the KV cache, the staging buffers, and the page cache of the streaming reads are all allocated after the slots take their share.
- **Sizing is readable now**: under `DS4_CUDA_EXPERT_CACHE_STATS=1` the branch prints free GiB, the integrated flag, host nonmovable GiB, the reserve GiB and the resulting slot budget. The line prints after the `min()` clamp, so the budget it reports is the one the arena is being asked for.
- **One caveat on that print**: it is emitted before the `while (capacity >= unique.size())` backoff loop that shrinks the arena by 3/4 steps if the allocation cannot actually be served, so on a box where the arena is refused the printed budget is not the final slot count. The `ds4: CUDA SSD expert cache: N slots` line printed afterwards is the one that reports what was really taken.
- **No throughput claim at the default**: with the variable unset the executed path is identical to the control, so the -9.2 percent recorded above is a reading outside the 3.3 to 4.9 percent noise floor on an UNCHANGED data path, which makes it more likely to be drift or a repeat artefact than a property of the knob. What the knob changes is the price of the reserve, not the arithmetic.
- **The trade, quoted as a shape**: an equivalent probe on our own CUDA backend gave 5,677 slots at 0.918 hit rate at margin 8 against 5,234 slots at 0.901 at margin 12, about 0.08 t/s per GiB of arena at that point on the curve. That probe belongs to a DIFFERENT tree and is quoted only to show the direction of the trade, so the slot counts and hit rates are not this branch's numbers.
- **Output identity**: greedy output is identical at both margins, `bb06e711bc498bb9` short and `d3355c94c70a4bb1` long, because a different arena size is still the same arithmetic.

```
   cuda_stream_selected_cache_begin_load
        |
        |  cudaMemGetInfo -> free, total
        |  integrated device? -> free = min(host nonmovable, total)
        v
   free GiB
        |
        +-- minus reserve_gb << 30 ------------------------------+
        |        reserve_gb = DS4_CUDA_EXPERT_CACHE_MARGIN_GB   |
        |                    default 8 GiB, unset = unchanged   |
        |                    ignored unless the whole string    |
        |                    parses and v <= 120                |
        v                                                       |
   available  ->  capacity = min(budget_for_expert_size, available / expert_bytes)
        |
        |  capacity < unique experts? -> refuse, stay unstreamed
        v
   arena: the slots the cache will actually use (min after the clamp)

        |                             arena is a PROMISE, not a reservation:
        |                             cudaMalloc overcommits and the host
        |                             settles it on first touch
        v
   +---------------------+   +------------------------------------------+
   | slots, LRU, by_gate |   | KV cache, staging, page cache of reads   |
   | hit rate follows    |   | <-- too small a reserve dies HERE, during |
   +---------------------+   |     warm-up, as an OOM kill               |
                             +------------------------------------------+

   reserve too large -> fewer slots, more misses, slower
   reserve too small -> more slots, then the OOM killer
```

**Provenance, 2026-09-16, native re-measure.** Every number above comes from a native run:
all six binaries rebuilt with `make cuda-spark -j12` (nvcc `-gencode arch=compute_121a,code=sm_121a`,
`ds4-server` text 23.11 to 23.13 MB), interleaved control then arm, two clean repeats per arm,
the 2048 frontier discarded as warmup, minimum across repeats reported. The control-to-control
floor on that native set is **3.3 to 4.9 percent on generation** and 9.7 to 15.8 percent on
prefill, so a generation delta under 4.9 percent is not resolved by this measurement.

The numbers this file carried before were measured on **JIT-fallback binaries** (17:12 builds
with no `-gencode`, `ds4-server` text 29.6 MB). Those runs were internally consistent - control
and arm shared a vintage - but they are not comparable with native figures: the control read
about 37 prefill tokens per second on JIT against about 86 native. They are superseded here.

**Read the direction, not just the size.** The minimum across repeats is negative, -9.2 percent,
and at ctx 8192 one repeat read -10.4 percent which is outside the floor. The two repeats
disagree at that frontier (margin_1 read 9.71), so this is a warning that needs a repeat rather
than a verdict. The prefill side of the same runs is positive at every frontier (+4.8 to +10.5
percent), which is at least consistent with the mechanism being about cache reserve rather than
about decode.
