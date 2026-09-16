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


**✦   ✧   ✦   ✧   ✦   ✧   ✦   ✧   ✦**

**✦✦✦  ✧  T R I P L E S P A R K L E  ✧  ✦✦✦**

**✦ above: the README, unchanged**

**✦ below: our modifications and numbers for this branch**

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

**Standard results table**

| arm / measurement | value | change | what the number is |
| --- | ---: | ---: | --- |
| tokens/s - control (tip, unpatched) | 9.56 | - | the control arm |
| tokens/s - this branch, SEEDING arm | 9.38 | -1.9% | BELOW THE FLOOR (4.9 %), so it resolves nothing |
| tokens/s - this branch, WARM-CACHE arm | ____ | ____ | OWED: not measured on the Spark |
| experts written by the hot-list recorder | 5,177 short / 8,679 long | cold start | mechanism, not a speed claim |

*Measured on the DGX Spark, one arm against its own interleaved control, minimum
across three stable frontiers (the first frontier of each run is discarded as warmup).
The change is stated beside the noise floor because that is the only honest way to read
it: a delta smaller than the floor is not a win. The -1.9 % row was taken while this
branch still claimed to be picking the RIGHT experts. It is kept here as the seeding
arm's number, not as the warm-up lever's. The lever this file now claims is unmeasured,
and the cells that describe it are blank on purpose rather than back-filled.*

## The claim, restated: this is a WARM-UP device, not a selection device

The branch previously sold itself on ranking the hot set correctly, which is a claim
about the CONTENT of the list. Our own measurement says the content is the small axis.

- `WIKI/theory/167-the-content-control-a-frozen-random-table-performs-like-a-real-one-2026-08-31.md`
  ran the control nobody else ran: on text, with a frozen backbone, a memory table whose
  VALUES WERE RANDOM performed statistically indistinguishably from a real learned table.
- The PRESENCE effect was z = +7.14 and EXACTLY reproducible, sd 0.00 across seeds
  (that file, line 57, and line 64). The CONTENT effect's sign FLIPPED between seeds and
  its seed-to-seed sd EXCEEDED its own mean (lines 61, 68-70). **The presence effect is
  27x to 45x the content effect** (line 70).
- `experiments/track3-semiotic-codebook/NEW_IDEA_THE_CEREBELLAR_FRONT_CACHE_2026-09-01.md`
  names WHY, in four cache laws (section 3, lines 65-78): never modify the deep path, only
  skip it; a miss must be free; correctness never depends on the cache; learn from the
  misses. A memory arm that CORRUPTS on a wrong entry breaks all four. A cache whose wrong
  entry merely wastes a slot does not.

**Our hot list is a slot-occupancy cache.** A wrong seed does not corrupt anything: it
wastes one slot, and the slot is refilled by demand. That is exactly the regime where
content is invisible and PRESENCE is what pays. So the honest claim for this branch is
**the cache is warm at the first token**, and the experiment that fits that claim is the
mine/score split below, not an expert-ranking study.

**The saliency question is the smaller axis.** Whether the list is ranked by hits, by
frequency, by recency, or by anything else is the CONTENT axis that `WIKI/theory/167`
measures at 27x to 45x smaller than presence. Warmth is the axis this branch can move;
ranking is the axis it cannot.

## Our own roadmap already ranks this lever

`WIKI/theory/48-decode-speedup-levers.md` section 8 (lines 351-372) ranks the decode
levers, and its one-line recommendation (lines 368-370) reads: on the **Spark**, where
residency is impossible, the lever is **prediction/prefetch of hot experts (Lever C)**,
and it says so explicitly against the alternative: *"not bigger cache (the knee proved
size isn't it)"*. Lever B (warm-up pre-pay) is ranked 2 for the M5 and measured at only
+4.4 % (line 358). This branch is Lever C, and the ranking belongs here as independent
corroboration from our own theory file, not as a result.

## The two numbers, side by side

| effect | measurement | source |
| --- | --- | --- |
| WARM-UP (resident working set warming as decode proceeds) | cold resident **15.87 t/s** against warm resident **35.14 t/s** = **2.2x** | `WIKI/theory/48`, section 4, lines 148-152 |
| the `--warm-weights` pre-fault flag at LOAD, same power | **+4.4 %** on M5; **-5.4 %** on the streaming Spark | `WIKI/theory/48`, section 4a lines 184-197 and 216-225 |
| CONTENT of a memory table against its presence | **27x to 45x SMALLER** than presence; presence exactly reproducible (sd 0.00), content sign flips | `WIKI/theory/167`, lines 57, 68-70 |

The 2.2x is **NOT** the flag: `WIKI/theory/48` separates the two deliberately (lines
165-172), and the flag is the small one. A warm-up effect of 2.2x, set beside a content
effect that our own control puts 27x to 45x below presence, is the clearest statement of
why "warm the cache" is the right claim here and "pick the right experts" is the weaker
one.

## OPEN QUESTION: LFU with aging against LRU, where our own verdict is the opposite

Both sides are cited. This is NOT resolved here, and it decides whether our verdict is
about the IDEA or about OUR CODE.

- **Upstream, for LFU.** `WIKI/research/neutronstar/NEUTRONSTAR_02_expert-streaming.md`
  section 1 (lines 13-36) describes a host expert cache keyed by FILE OFFSET, so it is
  model-agnostic, 16-way set-associative, evicting by **LFU WITH AGING**: every 4096
  inserts all `uses >>= 1` (lines 24-25). Their stated rationale, in their own code
  (lines 28-30): *"Frequency-weighted eviction protects repeatedly hit experts from being
  flushed by one-shot streams (pure LRU thrashes at these cache sizes)."*
- **Us, against it.** `WIKI/theory/177-the-sparkport-day-a-cache-that-was-a-stub-and-a-token-that-is-latency-not-bytes-2026-09-13.md`
  lines 33-36: a hotness-with-decay eviction measured **WORSE** than plain LRU
  (**0.851 against 0.868** hit rate, **4.95 against 5.28** t/s) and was reverted, recorded
  and not kept as dead code.

Two readings, neither tested: (a) frequency weighting is right and our implementation of
it was wrong, or (b) at GB10 streaming speed with our slot sizes, recency wins and their
result is about their media and their keying. The decay rule switch below exists so this
can be measured rather than argued.

## The invented decay becomes a measured rule: ROLLINGSPLIT

Loading halves every count. That rule was invented here; nothing measured it. It is now
switchable, and the branch carries the experiment that can price it.

**The switch.** `DS4_CUDA_EXPERT_HOTLIST_DECAY=none|halve|quarter|eighth|shift:<n>`,
parsed at `ds4_cuda.cu:27238`, applied to every loaded count
at `ds4_cuda.cu:27324`. **The default is `halve` and it does not change until a
measurement says it should.** The old path is therefore still the control arm, which is
the only way the new measurement can be read against it.

**The measurement.** `DS4_CUDA_EXPERT_HOTLIST_ROLLING=1` turns on the sliding mine/score
split described in `NEW_IDEA_THE_CEREBELLAR_FRONT_CACHE_2026-09-01.md` section 6, item 1
(lines 145-148): mine on the window `[t-k, t)`, score on window `t`, sliding. The demand
stream is held as a ring of windows (`ds4_cuda.cu:27330-27761`); every stride-th window it
mines `[t-k-gap, t-gap)`, cuts the mined counts to the slots the cache actually has, and
scores how much of window `t` was already resident.

- **k is how much history the list uses** and **gap is how stale the list is when it is
  used**. Both are runtime ladders, so the curve is ASKED FOR rather than assumed:
  `_ROLLING_K=0,8,32,128,512[,...]` and `_ROLLING_GAP=0,16,64,256[,...]`, with `all` for
  every window the ring still holds. `k=0` is the cold anchor and reads 0 by construction.
- **The mode REPORTS.** In the default `_ROLLING_MODE=report` it changes nothing about
  what the cache does: it prints one coverage grid per decay rule to stderr and writes one
  CSV row per scored `(window, k, gap, rule)` (`ds4_cuda.cu:27670-27751`). In `act` mode
  the mined window, not the whole run, is what gets written, which is how a measured
  retention rule replaces the halve.
- **The decay rule is an axis of the curve** (`_ROLLING_RULES=none,halve,...`), so the
  cost of our invented rule is read off the same run as the alternatives. The office
  harness (below) already shows the direction for short histories: with counts of one or
  two per key, `>>1` deletes them, and a `k=8` list loses most of its coverage.

**Off the box, the policy was exercised.** The block is host-only, so a harness compiled
it verbatim (between the `ROLLINGSPLIT BEGIN` and `END` markers) off the Spark and fed it
synthetic Zipf demand. On a stationary pool, `k=512` covered 71.3 % of the next window
against 15.1 % at `k=8`, and the decay rule cost coverage monotonically: `none` 15.07 %,
`halve` 3.30 %, `quarter` 0.05 % at `k=8`. When the pool was replaced 300 windows before
the end, the last window's coverage fell from 25.0 % at `gap=0` to 12.5 % at `gap=256`.
These are harness numbers on a synthetic stream: they exercise the policy, they are not
evidence about the Spark.

```
   RUN N                                     RUN N+1
   -----                                     -------
   cache starts EMPTY (first run)            ds4_session_create
       |                                         |
       v                                         v
   each unique requested expert              load list, age each count by the
   counted where the cache is                DECAY RULE (default halve; none |
   asked for bytes                           quarter | eighth | shift:n), drop
       |                                     any entry that ages to 0
       v                                         |
   atexit: sort hits descending                  v
   write tmp, rename into place              seed EMPTY slots only, via
       |                                     seed_experts (not counted as demand)
       v                                         |
   ~/.cache/ds4/cuda_expert_hotlist.txt --->     v
   # model_size <n>  layer expert hits       opening tokens start with the
                                             hot set already resident

   with ROLLING=1 the same windows also feed the mine/score split:
   mine [t-k-gap, t-gap) -> score window t -> coverage curve over k and gap
   (report mode writes the curve and nothing else; act mode writes the mined list)
```

**What this branch does**

- Counts every unique `(layer, expert)` the CUDA session is asked for, once per batch, taken where the cache is asked for bytes; at exit it writes them hits-descending in the format the existing loader already reads.
- `ds4_session_create` seeds a streaming V4.1 engine from that file, once per engine and before the first prefill, through `ds4_gpu_stream_expert_cache_seed_experts`.
- Guards that stop it doing harm: a `# model_size` header so a list never crosses models; seeded slots are NOT counted as demand, so seeding cannot flatter itself; the load-side decay rule (`DS4_CUDA_EXPERT_HOTLIST_DECAY`, default `halve`) so hotness cannot freeze; seeds fill EMPTY slots only, so a small cache never reads gigabytes it cannot hold; and the key budget on the rolling ring (`ds4_cuda.cu:27644`).
- The write goes to a temp file and is renamed into place, so a crash mid-write never leaves half a list.
- A missing, unreadable or mismatched list never fails a session - it warns and the run starts cold.
- `DS4_METAL_DISABLE_STREAMING_EXPERT_HOTLIST=1` disables seeding; `DS4_CUDA_EXPERT_HOTLIST_WRITE=<file>` names the writer, `0` disables it; `DS4_CUDA_EXPERT_HOTLIST_ROLLING=1` with `_MODE`, `_K`, `_GAP`, `_RULES`, `_STRIDE`, `_WINDOWS`, `_ACT_K`, `_ACT_GAP`, `_REPORT` runs the measurement.
- Four files: `ds4.c`, `ds4_gpu.h`, `ds4_cuda.cu` (the module comment at `ds4_cuda.cu:27140-27184` is the map), plus the root `README.md` you are reading. Code: `UP` +262/-1 then +577/-42, all in `ds4_cuda.cu` for the second.

**What is observable now, and what is still owed**

- The mechanism is observable INDEPENDENTLY of any t/s: the writer records **5,177 experts after a short run and 8,679 after a long one** on this box. Both were measured on the seeding arm.
- The startup stats line reads **1,278, 1,606 and 1,450 seeded slots** across three gate runs - **three different seed counts giving ONE identical sha**, which is the point of the test.
- **OWED, and it is the headline**: the early-token t/s A/B on the Spark. It needs the first 64 generated tokens reported separately from a 256-token run, on a warm cache against a cold one. The 2.2x in `WIKI/theory/48` is what that measurement is aimed at.
- **OWED: the ROLLINGSPLIT curve on a real run.** The harness above validates the policy; it says nothing about the k and gap our own demand stream actually wants. Until that curve exists the default decay stays `halve`.
- **OWED: the CUDA build gate.** The Spark is running an unrelated series, so nothing here was rebuilt or benchmarked after the ROLLINGSPLIT change. The code is committed unbuilt.
- Hand-tested: a missing file, a file recorded against a different model size, and a cache too small to hold the list - all three start cold without failing the session.

**Provenance, 2026-09-16, native re-measure (the seeding arm).** The -1.9 % and the floor
below come from a native run of the SEEDING arm: all six binaries rebuilt with
`make cuda-spark -j12` (nvcc `-gencode arch=compute_121a,code=sm_121a`, `ds4-server` text
23.11 to 23.13 MB), interleaved control then arm, two clean repeats per arm, the 2048
frontier discarded as warmup, minimum across repeats reported. The control-to-control
floor on that native set is **3.3 to 4.9 percent on generation** and 9.7 to 15.8 percent on
prefill, so a generation delta under 4.9 percent is not resolved by this measurement. That
floor is the reason the warm-up lever is OWED rather than claimed: the seeding arm's own
delta sits inside it.

The numbers this file carried before were measured on **JIT-fallback binaries** (17:12
builds with no `-gencode`, `ds4-server` text 29.6 MB). Those runs were internally
consistent - control and arm shared a vintage - but they are not comparable with native
figures: the control read about 37 prefill tokens per second on JIT against about 86
native. They are superseded here, and they too were seeding-arm numbers.
