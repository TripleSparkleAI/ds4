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

Rebased onto triple-tip-2026-09-16 (12997e9c8) on 2026-09-17; tests make test 42 pass lines, 5 pre-existing failures (the Qwen3.8 and GLM 5.3 model-absent skips, counted as failures by ds4_test and identical on every branch); make test-host-range-cache 1 PASS.

```
  ┌──────────────────────────────────────────────────────────────────────
  │
  │  BRANCH     triple-lfu-offsetkey                                 NOT YET
  │
  │  WHAT       ports upstream NeutronStar's host expert cache: byte ranges
  │             keyed by FILE OFFSET, a 16-way table, a per-slot uses counter
  │             halved every 4096 inserts. Three arms out of one binary isolate
  │             the eviction POLICY on one structure
  │
  │  LATEST     never in a sealed round. The arm lists of
  │             2026-09-17-ROUND3-, -R4- and -R5-PREREGISTERED-RULE.txt name
  │             this branch zero times, and no 2026-09-16 series switch block
  │             contains DS4_EXPERT_CACHE_MODE. The only numbers this branch
  │             holds are its own host-side harness, below
  │
  │  GEN        not measured
  │  PREFILL    not measured
  │
  │  VERDICT    NOT YET - a CUDA build and the three-arm sweep are owed. What
  │             exists is a 36-assertion host harness of the eviction block,
  │             all passing, including one window where LFU and LRU disagree
  │
  │  SWITCH     DS4_EXPERT_CACHE_MODE=hotlist|offsetkey|offsetkey-lru, default
  │             hotlist, in which the cache is inert: it never allocates, probes
  │             or inserts. DS4_CUDA_HOST_EXPERT_CACHE_GB=<GiB> sizes it; unset,
  │             it takes the configured expert-cache byte size
  │  OUTPUT     not re-run. Residency changes WHEN a byte arrives, never WHICH
  │
  └──────────────────────────────────────────────────────────────────────
```

## The question

- Upstream and we disagree about POLICY, not size. NeutronStar's code says "Frequency-weighted
  eviction protects repeatedly hit experts from being flushed by one-shot streams (pure LRU
  thrashes at these cache sizes)".
- Our own record measured a hotness-with-decay rule LOSING to plain LRU: hit rate 0.851 against
  0.868, and 4.95 against 5.28 t/s, and it was reverted
  (`WIKI/theory/177-the-sparkport-day-a-cache-that-was-a-stub-and-a-token-that-is-latency-not-bytes-2026-09-13.md`
  section 3, re-read 2026-09-17).
- Our decay was not their decay: ours was a route-hotness table, theirs is a per-slot uses counter
  halved on an insert count. That is the whole reason this branch exists.
- `WIKI/theory/48-decode-speedup-levers.md` sections 5 and 8 record the knee from
  `ds4-why --sweep-cache`: cache SIZE is not the lever, so only a policy difference earns a branch.
- ⚠ Those `WIKI/` paths are not in this branch's tree. The branch is rooted on the upstream tip,
  which carries no `WIKI/`, so they resolve in the dwarfstar working branch.

## Why a host cache exists at all

- The Linux path reads experts with `O_DIRECT`, so the page cache never retains them.
- We measured the other side: `O_DIRECT` off costs about 40 percent, 5.80 and 6.09 t/s against
  9.72 and 9.70, with expert read time 27.5 s against 5.7 s per 256 tokens (`WIKI/theory/177`
  section 16, re-read 2026-09-17).

## The mechanism, as ported

- The cache sits in front of `pread` in `cuda_model_copy_to_device_streamed`, the only integration
  point.
- A hit uploads the same bytes from pageable host RAM and skips the disk read. A miss takes the
  existing path byte for byte, then captures a pageable copy and inserts it.
- It can only SKIP a read, so a wrong entry costs a fetch and never a wrong output.
- The four cache laws of
  `experiments/track3-semiotic-codebook/NEW_IDEA_THE_CEREBELLAR_FRONT_CACHE_2026-09-01.md` section 3
  hold by construction: the deep path is untouched, a miss is free, correctness never depends on the
  cache, and the 4096-insert decay learns from the miss stream.
- Constants read from `ds4_host_range_cache.h` at HEAD: `DS4_HRC_WAYS` 16, `DS4_HRC_MAX_ENTRY`
  16 MiB, `DS4_HRC_AGING_INSERTS` 4096, `DS4_HRC_SLOT_BYTES` 1 MiB, `DS4_HRC_MAX_SLOTS` 262144.

```
   read(offset, bytes)
        │
        ├── bucket = murmur(offset) % nslots ──▶ 16-slot probe window
        │
        HIT   uses++ · age = ++tick · upload from host RAM · disk SKIPPED
        MISS  pread as before · upload as before · capture a copy · INSERT
                 every 4096th insert:  all uses >>= 1
                 victim = least uses, oldest age breaks the tie,
                 taken even when an empty slot exists (the budget fills first)

   arm            table    victim rule            what it tests
   hotlist        INERT    n/a                    today's expert-ID arena  ◀ default
   offsetkey      this     least uses, age tie    upstream's policy exactly
   offsetkey-lru  this     oldest age wins        the policy control
```

## Kept from upstream, and changed

- **Kept:** 16 ways, the offset hash, the 4096-insert aging, the age tiebreak, the least-uses
  victim, the empty-slot rule, the 262,144 slot cap, one slot per MiB.
- **Changed:** we copy on insert, because our pinned four-deep staging ring cannot be donated. The
  entry cap is 16 MiB and not 64, because our expert tensors are about 3 MiB. The budget defaults to
  our configured expert-cache byte size so the arms are size-matched. Their quiet `contains()` probe
  is omitted, because our look-ahead fills the VRAM arena and publishes a slot as valid before its
  copies are issued.

## Structural caveat

- On the GB10 the device arena is already host RAM, so this cache is a second tier behind the arena
  competing for the same memory. That differs from upstream's disk to host to GPU tiering and is the
  likeliest reason a result here would differ from theirs.

## What is held today

**⚠ Corrected here, twice.** The previous card said "17 checks" and "The harness is not committed".
Both are now false and were checked rather than reasoned:

- The harness IS committed, as `tests/test_host_range_cache.c`, 206 lines, with its own Makefile
  target `make test-host-range-cache` and a `clean` entry.
- Run on this Mac 2026-09-17 it prints `Host range cache: PASS`.
- It carries **36** `check()` assertions, not 17. Counted as call sites minus the one definition at
  line 22, two ways, both giving 36.
- Among them: `offsetkey` parses to `DS4_HRC_LFU` and `offsetkey-lru` to `DS4_HRC_LRU`, and one
  hand-built window where frequency and recency disagree, where LFU keeps the old hot entry and LRU
  keeps the once-recent one.

## Owed

- The CUDA build. Nothing on this branch has been compiled against CUDA.
- Three arms interleaved out of one binary, control first, the 2048 frontier discarded as warmup,
  read beside the in-run tip floor.
- The cache prints a hit rate, resident GiB, inserts and evictions every 2048 lookups
  (`ds4_cuda.cu:264`) and a `hit_rate=` final line at exit, so the two policy arms can be told apart
  from the log before any token rate is read.
