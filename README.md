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


**✦   ✧   ✦   ✧   ✦   ✧   ✦   ✧   ✦**

**✦✦✦  ✧  T R I P L E S P A R K L E  ✧  ✦✦✦**

**✦ above: the README, unchanged**

**✦ below: our modifications and numbers for this branch**

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

| arm / measurement | value | change |
| --- | ---: | ---: |
| tokens/s - this branch, as one tree | owed | must not be summed |
| the levers' own native deltas | -9.2% to +8.3% | floor 3.3-4.9%; only hits-first clears it |
| the prefill read-ahead's blocking wait | 398.0 -> 196.5 ms/layer | -50.6% (the one solid win) |

*The stack number is a measurement or it is nothing. It must not be assembled by
adding the branches' own deltas: they contend for one NVMe, and each was measured
against unpatched code, not against the other levers.*

**What is actually in this tree, lever by lever.**

- **`pool`** - a private read pool for the expert pread, so a batch in flight does not
  force the demand path back to the serial staged copy.
- **`margin`** - a knob for how much page cache the streaming reads keep after the
  cache slots take theirs. 8 GiB by default.
- **`draincut`** - queues the selected-expert readback before the shared expert runs, so
  the drain overlaps instead of blocking. **On by default**; `DS4_CUDA_SELECTED_DRAIN_SYNC=1`
  restores the blocking read, which is the A/B arm. Note the direction, because it is easy
  to get backwards: the default is the lever engaged. It was re-measured both ways on the
  GB10 and the engaged form does not beat the blocking one at any frontier, so there is no
  evidence the lever helps here; both readings sit inside the noise floor.
- **`pagecache`** - drops staged expert pages in the order the kernel honours, and adds
  a read-ahead hint that pre-passes a layer's missing experts.
- **`hotlist`** - seeds the expert cache from a learned hot list written after a short
  or long run.
- **`engram-lead`** - gives the Engram table read a token of lead, so the step that
  needs it finds it already in flight.
- **`engram-read-threads`** - scales the Engram batch reader to the request count.
- **`readahead-order`** - stops the prefill read-ahead evicting what decode needs.
  **No off switch**: it ships a stats flag, the reorder itself is unconditional, so a
  true A/B of this one needs the branch's own binary.
- **`hits-first`** - **compiled in, OFF by default.** It measured a **null** here, and
  the reason is structural: the decode path's shared-expert kernels already occupy the
  window it wanted to fill. It is kept in the tree so the A/B stays reproducible;
  `DS4_CUDA_HITS_FIRST=1` turns it on for a run.
- **`prefetch-pool`** - **IN this tree.** It runs a parallel SSD read-ahead during prefill,
  copying in chunks (8 MB default, `DS4_CUDA_SSD_PREFETCH_CHUNK_MB`) on a private-fd pread
  pool. It and the `pool` lever modify the same expert-pread code and collided across **30
  hunks** in `ds4_cuda.cu`; all 30 were resolved by hand, 8 keeping ours, 16 taking theirs and
  6 combining both, and the result **built clean natively**. This is the one lever that clears
  the noise floor: prefill **+72 to +81%**, with its revert arm back at baseline.

**The switches - one binary, many arms.**

| lever | switch | default | off |
| --- | --- | --- | --- |
| pool | `DS4_CUDA_STREAMING_EXPERT_PREAD_POOL` | on | `=0` |
| hotlist | `DS4_CUDA_EXPERT_HOTLIST_WRITE` | on | `=0` |
| hits-first | `DS4_CUDA_HITS_FIRST` | **off** | (default) |
| pagecache | `DS4_CUDA_KEEP_MODEL_PAGES` | new order | `=1` restores old |
| engram-lead | `DS4_V41_ENGRAM_LEAD_OFF` | on | set |
| draincut | `DS4_CUDA_SELECTED_DRAIN_SYNC` | **on** | `=1` restores blocking |
| margin | `DS4_CUDA_EXPERT_CACHE_MARGIN_GB` | 8 | knob, no Boolean |
| prefetch-pool | `DS4_CUDA_SSD_PREFETCH_CHUNK_MB` | 8 MB | **`0` is ignored**; `DS4_CUDA_SSD_PREFETCH_POOL=0` restores the serial reader, but that is a revert of the whole path, not an isolated chunk-size A/B |
| readahead-order | `DS4_CUDA_SSD_PREFETCH_STATS` | on | **stats only, not the reorder** |
| engram-read-threads | `DS4_ENGRAM_READ_THREADS` | request count | **not in `ds4_cuda.cu`** |

*Read this table honestly: six levers have a real off switch, which is what makes one
binary serve several arms. Two have only a knob, and two have no switch at all - those
arms still need their own build.*

## INTERACTIONS - why the stack is not the sum of its parts

- All ten levers pull on ONE NVMe. The Engram table read, the staged expert
  pages, the streaming expert preads, the prefetch pool, the hot list seed -
  all of it is I/O against the same device, much of it at the same moment.
- Each lever's own delta was measured against UNPATCHED code, not against the
  other levers. A solo delta is that lever's value on top of nothing; it says
  nothing about the same lever on top of nine others.
- Two levers, `pool` and `prefetch-pool`, modify the same expert-pread code.
  Summed, their solo deltas would count the same saved bytes twice.
- So the stack's gain is not the sum of the parts. Solo deltas can shrink,
  vanish, or invert once several levers are engaged together; only a
  measurement of the combination says anything about the combination.

```
        ten levers, one NVMe

   pool -----------\
   pagecache ------\        every lever's I/O
   prefetch-pool ---+---->  [ the one NVMe ]  <----+-- engram-lead
   hotlist ---------/                               \-- engram-read-threads
   draincut -------->   its readback rides the same bus
   margin ---------->   it decides what the device may evict

   how the solo numbers were taken - and why they do not add:

     unpatched --+ lever A only -->  delta A
                 + lever B only -->  delta B
                 + A and B on  -->  ???    not A + B - one device, one queue
```

| combination | what is on | result | clears floor? |
| --- | --- | --- | --- |
| all-off | the control: every lever with a real off switch, off (the four knob-only or switchless levers stay engaged and are recorded as such) | | |
| all-on | all ten levers forced on, `hits-first` included via `DS4_CUDA_HITS_FIRST=1` | | |
| stack | this branch as shipped: every lever at its default (`hits-first` off) | | |
| random 1 | one of six random on/off combinations; the exact env is recorded in the sweep file | | |
| random 2 | second draw of the same six-combination series | | |
| random 3 | third draw of the same six-combination series | | |
| random 4 | fourth draw of the same six-combination series | | |
| random 5 | fifth draw of the same six-combination series | | |
| random 6 | sixth draw of the same six-combination series | | |
| stack vs all-off | derived row: the stack against the control from the same session | | |

**Every blank cell above is awaiting the 2026-09-16 Spark run. A blank cell is
awaiting measurement - it is NOT a zero, and it must not be filled in from a
solo delta.** The results land as `~/sweeps/2026-09-16-*.txt`, one file per arm,
with a MATRIX file beside them; this table is filled only from those files.
The floor each row must clear is the native one, 3.3-4.9% generation and
9.7-15.8% prefill; so far only `prefetch-pool` has a measured delta that clears
it (+72 to +81% prefill, measured alone).

## METHODOLOGY - how a measurement is taken here

- **Like binaries only - the vintage trap.** A `make cuda-spark` build has
  about 23.1 MB of `ds4-server` text. A binary at about 29.6 MB of text is the
  JIT-fallback build; it is not comparable to the native one, and a delta taken
  across that line measures the build, not the levers. Check the text size
  before trusting any pair of numbers.
- **The control is interleaved.** The all-off control does not run once at the
  start or the end of a session; it runs interleaved between the arms. Page
  cache state and thermals drift across a session, and a control at one end
  cannot see that drift.
- **The first frontier is discarded.** Every run throws its first frontier
  away: caches, readahead state and the page cache warm up on it, so it reads
  fast for the wrong reason.
- **Minimum across repeats, not the mean.** The recorded number is the minimum
  of the repeats. The mean lets one disturbed repeat bend every number; the
  minimum is the reading that survives a loaded machine - conservative for
  tokens/s, and the undisturbed reading for time-per-unit metrics.
- **The noise floor beside every delta.** No delta is quoted without the floor
  it must clear: 3.3-4.9% generation, 9.7-15.8% prefill, native. A delta inside
  the floor is not a gain - it is a measurement of the noise.
- **The instrument is `try.sh`.** One binary, many arms, the arms made of
  environment switches. It runs ten fixed tests over ssh against the Spark and
  writes a markdown-table result file per run plus an index row; its output is
  paste-ready, which is what the INTERACTIONS table above waits to receive.

**How this tree was built.**

- Base: `triple-antirez-tip-latest`, which is upstream `main` at `9139e2ae5` plus
  upstream PR `#1034` and PR `#1035`, squashed to one commit.
- The four levers that were missing were applied as **code-only** patches onto the six
  already stacked. `README.md` was excluded from every patch on purpose: every branch
  appends its note to the same place, so merging those is noise, not content.
- One conflict, in `ds4.c`, between the Engram lead read and the Engram batch reader.
  Resolved by keeping **both**: the lead read stays, and the miss path goes through the
  batch reader.
- `hits-first`'s default was flipped from on to off, with the reason written into the
  code beside it.
- **Verified by building it**: `make cuda-spark -j12` on the DGX Spark, **0 errors**,
  `ds4-server` and `ds4-bench` produced.

```
   tip-latest  9139e2ae5 + #1034 + #1035
        |
        +-- pool            [on]
        +-- margin          [on]
        +-- draincut        [ON by default; =1 restores blocking]
        +-- pagecache       [on]
        +-- hotlist         [on]
        +-- engram-lead     [on]
        +-- engram-read-thr [on]
        +-- readahead-order [on, no off switch]
        +-- hits-first      [OFF - measured null]
        |
        +-- prefetch-pool   [in - 30 hunks resolved, built native]
        |
        v
   triple-all-fastest  -  builds clean: 10 levers compiled in, 9 live, hits-first off
```


---

## The index: every branch card, in full

Every branch in the fork, its card as the branch itself carries it, grouped by what
its own measurement said. The separators are real rules and the order is the point:
what helped, then what did not, then what nobody has run yet.

---

## POSITIVELY MEASURED

Ordered by the scale of the effect, LARGEST FIRST. Most of these deltas sit inside their session noise floor, so read the group as a DIRECTION and not as a proven win - and note the cards were measured in DIFFERENT sessions, whose floors differ, so the order is indicative rather than a strict ranking.

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

*+7.1 % - the prime configuration - every lever in one binary*

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

### triple-pool

*+4.0 % - parallel expert reads through a shared SSD pool*

```
  ┌──────────────────────────────────────────────────────────────────────
  │
  │  BRANCH    triple-pool
  │
  │  WHAT           serves expert reads from a shared parallel SSD pool
  │                 instead of one serial reader, and now puts an io_uring
  │                 O_DIRECT ring in front of that pool so queue depth is a
  │                 switch rather than the worker count
  │
  │  RESULTS              tokens/s        tip      change       floor
  │    generation             ____       ____        ____        ____
  │    prefill                ____       ____        ____        ____
  │
  │  VERDICT        OWED: the fetch engine changed after the last
  │                 measurement, so the old +4.0 % no longer describes it
  │
  │  SWITCH         DS4_CUDA_FETCH_QD=<n>; io_uring queue depth, default 64,
  │                 clamped 8-512
  │                 DS4_CUDA_FETCH_URING=0 falls back to the pread pool;
  │                 DS4_CUDA_FETCH_BUFFERED=1 forces buffered reads
  │                 DS4_CUDA_STREAMING_EXPERT_PREAD_THREADS=<n>; pool workers
  │                 when the ring is off, default 8, cap 16
  │  HEADLINE       the ring is ported; device concurrency is now a knob, and
  │                 no number here has been re-measured since
  │  OUTPUT         not re-run on this engine: sha256 ____
  │  BUILD          OWED: a CUDA build needs the DGX Spark, which is busy
  │
  └──────────────────────────────────────────────────────────────────────
```

### triple-pagecache

*+0.6 % - staged page-drop order*

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

Ordered by the scale of the effect, LEAST HARMFUL FIRST and WORST LAST. A null is a result: the closed branch sits at the bottom because it has no patch at all, and it is kept precisely so nobody rebuilds it.

### triple-draincut

*-0.9 % - event-gated selected-expert readback*

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

*-1.9 % - seed the cache from the previous run's demand*

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

### triple-margin

*-9.2 % - the expert-cache reserve and the budget split*

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

### triple-word-finisher

*CLOSED, no patch - CLOSED - the measurement killed the idea*

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

Prepared, committed, and carrying no throughput number yet. Every cell in their cards is blank on purpose, with the reason stated on the card itself.

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

### triple-prefill-readahead-order

*not yet measured - prefill eviction that spares what decode needs*

```
  ┌──────────────────────────────────────────────────────────────────────
  │
  │  BRANCH    triple-prefill-readahead-order
  │
  │  WHAT           reorders prefill victim eviction and holds the earliest
  │                 layers of the scan, so the experts decode will need are
  │                 not the ones dropped: the prefill hit list survives into
  │                 decode
  │
  │  RESULTS              tokens/s        tip      change       floor
  │    generation             ____       ____        ____       4.9 %
  │    prefill                ____       ____        ____      15.8 %
  │
  │  VERDICT        OWED: no clean interleaved A/B exists yet, so a blank
  │                 cell is not a zero. The CUDA build gate is OWED as well.
  │
  │  SWITCH         DS4_PREFILL_READAHEAD_HOLD=0 restores upstream exactly:
  │                 the plain used-ascending victim order, no held set.
  │                 Unset or 1 keeps the reorder plus the held band. This is
  │                 the A/B switch the branch did not have before.
  │  STATS          DS4_CUDA_SSD_PREFETCH_STATS=1 counts only, it does not
  │                 gate
  │  GATE           expected output-invariant: residency changes WHEN a
  │                 byte arrives, never which byte
  │
  └──────────────────────────────────────────────────────────────────────
```

### triple-engram-lead

*not yet measured - Engram read, a token of lead*

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

### triple-engram-read-threads

*not yet measured - Engram readers scaled to the rows needed*

```
  ┌──────────────────────────────────────────────────────────────────────
  │
  │  BRANCH    triple-engram-read-threads
  │
  │  WHAT           issues a decode step's Engram read as ONE WAVE: one row
  │                 per reader on a single persistent pool shared by both
  │                 tables, instead of two serial rounds of twelve readers
  │
  │  RESULTS              tokens/s        tip      change       floor
  │    generation             ____       ____        ____       4.9 %
  │    prefill                ____       ____        ____      15.8 %
  │
  │  VERDICT        AWAITING THE SWEEP: a blank cell is not a zero
  │
  │  SWITCH         DS4_ENGRAM_ROWS_PER_READER=1|2, default 1
  │                 2 restores the old two-round divisor as the control arm
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

---

## THE CONTROL

Not a lever and never grouped with one: this is the tip every other branch is cut
from, and the zero point every number above is measured against.

### triple-antirez-tip-latest

*the zero point - the control: upstream main, PRs 1034 and 1035 folded*

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
