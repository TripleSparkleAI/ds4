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

Rebased onto triple-tip-2026-09-16 (12997e9c8) on 2026-09-17; tests make test 42 pass lines, 5 pre-existing failures (the Qwen3.8 and GLM 5.3 model-absent skips, counted as failures by ds4_test and identical on every branch); cc -fsyntax-only clean on ds4.c and ds4_gpu.h.

```
  ┌──────────────────────────────────────────────────────────────────────
  │
  │  BRANCH     triple-all                                          POSITIVE
  │
  │  WHAT       six lever branches stacked in series: the pread pool, the
  │             event-gated selected-expert readback, the cache reserve knob,
  │             the hot list seed, the staged page-drop order and hits-first.
  │             Measured second-best of all trees today at +6.50 %, and an arm
  │             of the amended round 7
  │
  │  LATEST     round 9 · 2026-09-17 · TIES · +7.02 % 3/4 under a 17.27 floor ·
  │             median 10.11, and it reads with round 6 rather than round 8 ·
  │             2026-09-17-R9-RESULT-the-default-holds-hitsfirst-is-the-tightest-tree-on-a-loud-instrument.md
  │             round 6 · 2026-09-17 · BEATS · +6.50 % ·
  │             2026-09-17-R6-RESULT-the-superseded-six-lever-stack-beats-the-tip-three-ties.md
  │             round 8 · 2026-09-17 · TIES · -0.98 % 1/4 under a 10.69 floor ·
  │             2026-09-17-R8-RESULT-hitsfirst-alone-beats-through-the-noise-the-winners-tree-does-not.md
  │
  │  GEN        +6.50 %  floor 3.35 %  sign 3/3  n=3
  │             raw: arm min-across median 10.35 t/s (10.31 · 10.24 · 10.38 ·
  │             10.39) vs the tip's 9.69 t/s (9.67 · 9.56 · 9.88 · 9.70 kept,
  │             9.45 contention-dropped; the result file's figure line reads
  │             9.67). gen_steady at min across 4096/6144. The delta is each
  │             run against its bracketing TIP runs, not against that median
  │             control = triple-tip-2026-09-16 @12997e9c, interleaved
  │             session = 2026-09-17 11:12-11:58Z · DGX Spark GB10 ·
  │             native 24.90 MB .text (tip 24.86 MB) · lean regime 4096/6144
  │  PREFILL    reported, not a verdict: 86.30 t/s min-across median vs the
  │             tip's 89.16 t/s, n=4. Round 6 makes no prefill verdict.
  │             ─── round 8, TIES, and the two rounds disagree:
  │             -0.98 %  floor 10.69 %  sign 1/4  n=4 (sensitivity, cold first
  │             control run excluded: -1.74 % against a 7.17 % floor, TIES)
  │             raw: arm min-across median 9.10 t/s (9.21 · 9.18 · 9.02 · 8.99)
  │             vs the tip's 8.96 t/s (8.61 cold · 9.53 · 8.87 · 9.49 · 8.96),
  │             so it read level with the tip against round 6's +6.50 %.
  │             ⚠ the box was not quiet: the five tip runs spanned 8.61 to 9.53
  │             and four arm runs ended at load 3.2 to 4.0 with no vitest alive,
  │             which widened the floor from round 6's 3.35 to 10.69. Round 6 is
  │             the quieter round and keeps LATEST
  │             control = triple-tip-2026-09-16 @12997e9c, interleaved
  │             session = 2026-09-17 13:24-14:37Z · DGX Spark GB10 ·
  │             native 24.90 MB .text (tip 24.86 MB) · lean regime 4096/6144
  │             prefill round 8, reported not a verdict: 85.28 t/s min-across
  │             median vs the tip's 84.34 t/s, n=4
  │
  │             ─── round 9, TIES, and it settles which of the two rounds to read:
  │             +7.02 %  floor 17.27 %  sign 3/4  n=4
  │             raw: arm min-across median 10.11 t/s (9.14 to 10.29, spread 1.15)
  │             vs the tip's 9.42 t/s (8.51 to 9.98). That is round 6's level, not
  │             round 8's 9.10, so the disagreement resolves toward round 6
  │             ⚠ the instrument sets that floor, not the arm: the five control
  │             runs span 8.51 to 9.98 t/s because the page cache holds 2.9 GB of
  │             an 81 GB model and the bench streams 842 MB/s from the NVMe, so
  │             identical binaries swing about 8 % run to run
  │             control = triple-tip-2026-09-16 @12997e9c, interleaved
  │             session = 2026-09-17 14:41-15:36Z · DGX Spark GB10 · native
  │             24.90 MB .text (r9-all, 24,895,653; tip 24.86 MB) · lean regime
  │             4096/6144 · quiet box, warm-up run discarded
  │
  │  VERDICT    POSITIVE - the only arm of round 6 to clear the floor, and it
  │             clears it at every repeat (+6.1 to +6.8 %). Both the sealed
  │             legacy rule and the new rule give this verdict, so nothing
  │             turns on the straggler rule. ⚠ Round 8 re-ran it and read
  │             -0.98 %, level with the tip, 1 of 4 under a 10.69 % floor. That
  │             is a TIE, so it retracts nothing, and round 6's floor was 3.35
  │             against this round's 10.69: the two rounds disagree and the
  │             noisier one is round 8. Round 9 re-measures it on a quiet box
  │
  │  SWITCH     the six levers' own switches, verified present in this tree's
  │             diff: DS4_CUDA_STREAMING_EXPERT_PREAD_POOL and _PREAD_THREADS,
  │             DS4_CUDA_SELECTED_DRAIN_SYNC, DS4_CUDA_EXPERT_CACHE_MARGIN_GB,
  │             DS4_CUDA_EXPERT_HOTLIST_WRITE, DS4_CUDA_NO_EXPERT_READAHEAD,
  │             DS4_CUDA_HITS_FIRST and _HITS_FIRST_STAGED
  │  OUTPUT     not re-run. A greedy sha was recorded on a prior card and is
  │             withheld here: it was taken before this rebase, on a tree this
  │             branch is no longer on
  │
  └──────────────────────────────────────────────────────────────────────
```

## Round 6, and the word this card used to carry

- Round 6 built and ran this tree for the first time: **+6.50 % gen, 3 of 3 above a 3.35 % floor**,
  against the ten-lever `triple-all-fastest-newtip` stack's +4.65 % in round 4. It carries pool and
  hotlist and four levers that measured zero or negative solo, and no prefetch-pool.
- **This card used to say SUPERSEDED, and the result file scores that against the orchestrator**: it
  believed the word over a measurement nobody had taken, and three of its five sealed predictions
  missed as one miss. The word is gone from the box; the supersession history stays below.
- The four extra levers of the ten-lever tree cost it about 1.9 points against these six. Not
  attributable from round 6, which is why round 7 was amended to carry this tree as an arm.

## Why it carried no number until round 6

- Until round 6 the combined number had never been measured on this tree.
- It must not be assembled from the six branches' own deltas: the levers overlap, since pool and
  prefetch-pool touch the same expert-pread code and all six pull on one NVMe, and each solo delta
  was taken against unpatched code.
- Hits-first measured a null on this tree because the decode path's shared-expert kernels already
  occupy the window it wanted, which is itself a reason a stack is not a sum.
- Files at HEAD, re-counted with `git diff --numstat 12997e9c8 HEAD`: `ds4_cuda.cu` **+1112/-41**,
  `ds4.c` **+37**, `ds4_gpu.h` **+9**, `README.md` **+42**. So it is a real tree, not a pointer, and
  that is exactly why an unmeasured one was a hazard.

## ⚠ The number this card used to carry, and why it is gone

The previous card's VERDICT read: *"SUPERSEDED by triple-all-fastest - that tree carries these six
plus four more, on the rolled tip, and it is the one measured (-4.51 % vs the tip, 0 of 7)."* The
supersession stands. **The -4.51 % is removed as a live figure, on three counts:**

- **It is not this branch's number**, and never was. It is the as-shipped arm of the attrib series
  (`2026-09-16-triple-all-fastest-attrib-SUMMARY-levers-attribution.txt`, min-across-frontiers row).
- **It was measured on a binary that is not on this line of history.**
  `2026-09-16-CORRECTION-the-measured-stack-is-nine-levers-and-has-diverged.md` records that the
  measured sha `dd82361a` carries **nine** levers, lacks the prefetch pool entirely, and is a
  SIBLING of `triple-all-fastest` off `84ba6ef1b` rather than an ancestor of it.
- **The behaviour it measured is no longer the default.** The bisect
  (`2026-09-16-BISECT-RESULT-one-comparator-carries-the-whole-loss.md`) found one
  `std::stable_sort` comparator in `ds4_gpu_stream_expert_cache_prefetch` carrying the loss, and
  reverting it moved the stack 8.04 percentage points, from -4.35 % to +3.69 %. Round 2
  (`2026-09-17-ROUND2-RESULT-the-revert-beats-the-tip.md`) re-ran it at n=8 on a cleared box:
  **+3.46 % against the clean tip, sign 8 of 8, floor max 1.37, cleared by 2.5x, bands that do not
  touch.** Under the standing rule that the default follows the measured number, the revert became
  the stack's shipped behaviour, and `triple-all-fastest-newtip` (`b8f8a5a06`) carries it as
  "the victim order defaults to used-ascending".

⇒ **a stack figure of -4.51 % now describes neither the tree it named nor the behaviour that
ships.** It is kept above as a dated record of one arm of one series, and nowhere else.

```
   W H E R E   T H E   S T A C K   W E N T ───────────────────────────

   triple-all              six levers, never built, never run      ◀ this card
        │
        ▼
   triple-all-fastest      +prefetch-pool +engram-lead
                           +engram-read-threads +readahead-order
        │                  measured as the nine-lever dd82361a, a SIBLING
        │                  as-shipped  -4.51 %  (attrib, 2026-09-16)
        │                  as-shipped  -4.35 %  (bisect, 2026-09-16)
        ▼
   BISECT ─── one std::stable_sort comparator carries the whole loss
        │     revert it:  -4.35 %  ──▶  +3.69 %        8.04 pp
        ▼
   ROUND 2 ── +3.46 % vs the clean tip · 8 of 8 · floor 1.37 · cleared 2.5x
        │     the revert becomes the DEFAULT
        ▼
   triple-all-fastest-newtip @b8f8a5a06   ◀ round 4's ALLFASTESTNE arm

   this branch's six are the first two rungs of that ladder and nothing else
```

## Where the branch now sits

- It was rebased onto `triple-tip-2026-09-16` (`12997e9c8`) on 2026-09-17, so **`12997e9c8` is its
  base today**.
- ⚠ The previous card said the six were "stacked in series on `9139e2ae5`" and that the branch
  "stays as the record of the first stack at `84ba6ef1b`". Both are dated statements about a line of
  history the branch has left. Measured: `9139e2ae5` IS an ancestor of the new tip;
  `84ba6ef1b` is **not** an ancestor of the new tip and is **no longer an ancestor of this branch**.
  The record of the first stack is the commit `84ba6ef1b` itself, reachable by sha, not by this
  branch's ancestry.

## Read instead

- `triple-all-fastest-newtip` for the measured stack on this tip.
- The six levers' own cards for what each one is, and `triple-draincut`'s for the
  `DS4_CUDA_SELECTED_DRAIN_SYNC` arm in particular.
- `2026-09-16-CORRECTION-the-measured-stack-is-nine-levers-and-has-diverged.md` before quoting any
  2026-09-16 stack figure, whichever branch it is attached to.

## Round 8, 2026-09-17: the second reading disagrees with the first

- **It read level with the tip: -0.98 %, 1 of 4, under a 10.69 % floor** (9.21 · 9.18 · 9.02 ·
  8.99 t/s against a control median of 8.96), and -1.74 % on the cold-run sensitivity. That is a
  TIE, so round 6's **+6.50 %** is not retracted and stays this card's LATEST.
- **The two rounds disagree and the noisier one is round 8.** Round 6's control-to-control floor
  was 3.35 %; round 8's was 10.69 %, because the five tip runs spanned 8.61 to 9.53 t/s and four
  arm runs ended at load 3.2 to 4.0 with no vitest alive. Round 9 re-measures this tree on a quiet
  box with the discarded warm-up run, and that reading decides between them.
- ⚠ **Its raw t/s fell too, which the floor does not explain by itself.** Round 6 read 10.24 to
  10.39 and round 8 read 8.99 to 9.21 on the same binary vintage, 24.90 MB .text, two hours apart.
  The control fell with it (9.69 to 8.96), so the ratio moved less than either arm, and nothing
  here attributes the drop to the box rather than to the tree.

Sealed rule `2026-09-17-R8-PREREGISTERED-RULE.txt` @`c173ad6d7`, result
`2026-09-17-R8-RESULT-hitsfirst-alone-beats-through-the-noise-the-winners-tree-does-not.md`,
raw CSVs and runlog in `sweeps/r8/`.

## Round 9, 2026-09-17: the quiet round reads with round 6

- **The round-6-versus-round-8 disagreement resolves toward round 6.** On a quiet box with the
  discarded warm-up run this tree reads **10.11 t/s, +7.02 %, sign 3 of 4**, beside round 6's
  10.35 t/s and +6.50 % and against round 8's 9.10 t/s and -0.98 %. Round 8 was the noisier round
  and this is the third reading.
- **The raw-t/s drop of round 8 is now attributable to the box.** Round 6 read 10.24 to 10.39,
  round 8 read 8.99 to 9.21, round 9 reads 9.14 to 10.29 on the same 24.90 MB binary vintage. The
  tree did not change between them; the instrument did.
- **It is still below the one-lever tree.** 10.11 against hitsfirst's 10.41, fourth of five arms,
  with a spread of 1.15 against hitsfirst's 0.19. Six levers do not beat the one that wins.

Sealed rule `2026-09-17-R9-PREREGISTERED-RULE.txt` @`60d1bd06a`, result
`2026-09-17-R9-RESULT-the-default-holds-hitsfirst-is-the-tightest-tree-on-a-loud-instrument.md`,
raw CSVs and runlog in `sweeps/r9/`.
