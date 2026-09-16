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

**Standard results table**

| arm / measurement | value | change |
| --- | ---: | ---: |
| tokens/s - control (tip, unpatched) | ____ | - |
| tokens/s - this branch | ____ | ____ % |
| misses/token, first 32 decode tokens - this branch | ____ | ____ % |

*Cells left blank are AWAITING THE SWEEP, not zero. Nothing here is estimated.*

- Changes one `stable_sort` comparator in `ds4_gpu_stream_expert_cache_prefetch`, adds a held subset in front of it, and makes both conditional on one switch. The victim SET is identical before and after, so the starvation path (`if (p.slots.size() >= victims.size()) throw 0;`) is reached exactly as often as upstream: the hazard that silently switches the read-ahead off is structurally unreachable by this change. That equality is not asserted, it is exercised: a host-side harness splices the patched code out of `ds4_cuda.cu` and replays 20,000 randomized cache states against upstream's own comparator and victim choice, checking the order with the switch off, the set with it on, held-last placement, and identical starvation counts in all three arms.
- **The off switch exists now, and it was the blocking defect.** `DS4_PREFILL_READAHEAD_HOLD=0` restores the tip exactly: upstream's plain `used`-ascending victim sort, no held set, and the demand path's original single-pass victim choice. Unset or `1` keeps the reorder and the held band, so a default run is unchanged from the previous revision of this branch. `off`, `no` and `false` (any case) are accepted as well, and `00` reads as zero, because a control arm that silently lands in the arm it was meant to control is the failure this switch exists to remove. The variable is read once, on first use, so it must be set before the process starts. Before this, the only variable the diff shipped was `DS4_CUDA_SSD_PREFETCH_STATS`, which counts and prints and changes no decision, and the reorder was unconditional, so the branch could not be measured against its own control at all.
- **What "held" means here.** A slot holding an expert from the earliest layers of the scan is a victim only after every unheld candidate: the read-ahead appends the held slots behind the sorted victims, and the reserve loop takes from the front, so a held slot is touched only once nothing else is left. The demand path does the same with a two-pass choice. Held is a preference and never a guarantee, and the candidate SET is unchanged, so nothing starves here that does not starve upstream. One honest consequence: when the cache sits at its minimum, the choice inside that set does differ from upstream, because an older held slot is kept in place of a newer unheld one. That is the trade this change makes and the thing the owed sweep has to price. The band is anchored on the lowest layer the read-ahead is shown (layer 0 for a prefill that starts at the top of the model) and sized to a quarter of the cache, at least one layer: that is a chosen constant, not a tuned one, and its size is part of what the sweep should attack.
- The figures this patch has to beat, measured on the unpatched tip and recorded in `plans/prs/PR_PREFILLHANDOFF.md` on branch `lane-pr-docs` at `cc9bcac8f`: the first 32 decode tokens miss **32.22 experts per token at a 0.8658 hit rate** against a steady state of **9.16 at 0.9618**; a second run reproduced it at 33.09 / 0.8621 against 9.19 / 0.9617.
- The same record shows the prefill takes 36,022 expert lookups, hits 35,377 and evicts **zero** through the demand path, with the arena full at 8,548 of 8,548. The eviction that matters is the read-ahead's own reservation loop, which a demand-only counter cannot see: `res=` climbing 384 a layer to the cap and then sitting flat is eviction running at exactly the rate of admission.
- The probe lane's independent measurement, recorded in `plans/prs/MEASURED_PREFETCH_PROBE.md` on the same branch: **93.8 percent of the first decode step's misses were experts the prefill had selected and the cache no longer holds**, and 65.9 percent of the last prefill batch's experts are still resident at that layer's first decode step.
- The opening is layer-shaped: **98.4 percent of the opening's misses live in layers 0-19**, while layers 20-39 miss 0.47 percent of their lookups (`plans/channel/CUDA_LANES_CHANNEL_snapshot.md`, `lane-pr-docs`).
- **Caveat on the three citations above:** they name files on branch `lane-pr-docs`, which is
  **not published on this fork** because it carries the internal coordination log. A reader
  working from the fork alone cannot fetch them, so treat those three figures as attributable
  but not independently checkable here.
- An earlier pair of figures from this lane, 4.7 percent surviving and 92.7 percent loaded-then-evicted, was **withdrawn**: the offline replay reconstructed residency from the demand path alone and read 645 where the engine's own `res=` printed 8,548. They are not used here and are recorded so nobody re-derives them.
- Counters ship separately so a silent self-disable cannot masquerade as a win: `DS4_CUDA_SSD_PREFETCH_STATS=1` prints asked / started / refused_early / no_victims / cancelled / published / dropped / slots.

```
   VICTIM ORDER ACROSS ONE PREFILL SWEEP

   layer       0    1    2   ...   20   ...   39
               |    |    |          |          |
   sweep  -->------------------------------------------->
   used stamp   1    2    3   ...   21   ...   40
                (rises with the layer: an expert tensor's offset rises
                 with its layer, 40 of 40 strictly, so the gate offset
                 orders layers without carrying a layer on the slot)

   OLD - sort victims by used ASCENDING
     evict layer 0 first .................... layer 39 last
     the arena ends the prefill holding the LATE layers
     decode restarts at layer 0  ==>  the opening is cold

   NEW - this branch
     1. slots AHEAD of the sweep go first: the sweep re-reads those
        layers on the way past anyway, so losing one costs one
        prefetch, not the opening
     2. among slots BEHIND the sweep, the ones CLOSEST to the sweep
        go first: decode reaches layer 0 first, so the earliest
        layers are the last thing worth surrendering
     3. used stays the tie-break within one layer, so the least
        recently routed expert in a layer still goes first
     4. and the HELD band (the earliest layers of the scan, a quarter
        of the cache, anchored on the lowest layer the read-ahead is
        shown) is appended AFTER all of that: the reserve loop takes
        from the front, so a held slot is only ever released when no
        unheld candidate is left
     evict layer 39 first .................... layer 0 last
     decode restarts at layer 0  ==>  still resident
```

**Two independent lines of support, and why this branch moves policy rather than capacity.**

- **(a) The traversal order, from the source.** A decode token walks all layers in the same order every token, so under LRU each admission is followed by an eviction of exactly the entry the next token will want, because the reuse distance equals the whole scan. Their reader states the response in one line: "Stable admission avoids thrashing during the full layer scan each token" (`research/v41-flash-landscape/09-macs-and-dwarfstar/sources/atbender-mac-mini/deepseek_v41_mlx/native_stream.py:107`), and it never evicts (`:108-110`). The reading of that comment with our own numbers attached is in the same folder, `research/v41-flash-landscape/09-macs-and-dwarfstar/HOW-WE-USE-IT.md:21-41`, which is where the 93.8 percent figure above is tied to their sentence.
- **(b) A measured knee of our own, recorded independently of this branch.** Our roadmap's one-line recommendation says the Spark lever is prediction/prefetch of hot experts "**not** bigger cache (the knee proved size isn't it)" (`WIKI/theory/48-decode-speedup-levers.md:369-370`). The knee is a measured thing, not a hypothesis: `ds4-why --sweep-cache` sweeps the streaming expert-cache size and locates the point beyond which each extra GiB buys less than half the steepest t/s-per-GiB gain, and the roadmap reads that as "raw cache *size* is not the lever ... the remaining lever is therefore prediction/prefetch quality, not more cache" (`WIKI/theory/48-decode-speedup-levers.md:267-271`; the sweep and the half-gain criterion are defined in `WIKI/theory/47-profiling-methodology.md:118-120`).
- Together those two say the same thing from opposite ends: capacity is not where the win is on this host, and the reason is the scan order rather than the size of the arena. Both point at ADMISSION and VICTIM policy, which is exactly what this branch changes. The reorder and the held band are residency policy; neither adds a byte of cache, and neither changes which byte the model reads or what it computes.

**The currency lesson, recorded rather than fixed.**

- Their cache budget is denominated in the wrong unit and under-delivers by the dequantization ratio. `cached_bytes` sums `w.nbytes` of the **dequantized** tensor (`research/v41-flash-landscape/09-macs-and-dwarfstar/sources/atbender-mac-mini/deepseek_v41_mlx/native_stream.py:108-110`), and FP8 dequantizes to bf16, so a 4 GiB budget holds about 2 GiB of on-disk FP8 bytes. Recomputed from their own logs: the 4 GiB cache removes **2.035 GiB** of disk reads per token and the 6 GiB cache removes **3.051 GiB**, about half the declared budget each time (`research/v41-flash-landscape/09-macs-and-dwarfstar/CODE.md:150-158`). They measured it and did not assert it, and nobody corrected the accounting.
- **Our arena is denominated in stored GGUF expert bytes on the device, which is the same currency as the disk read it removes, so this trap does not apply to us.** The byte budget is converted to slots by dividing by the stored per-expert size: `ds4_streaming_cache_experts_for_byte_budget` divides a byte figure by `per_expert_bytes` (`ds4.c:5205-5216`), and `per_expert_bytes` comes from `routed_expert_row_bytes`, which multiplies the quant block size and never a dequantized size (`ds4.c:5004-5009`, `ds4.c:5054-5082`). On the CUDA side the same unit is used to size and fill the arena: `bytes = 2 * gate_expert_bytes + down_expert_bytes` and the slot count is `budget / bytes` (`ds4_cuda.cu:33771-33780`), the three arena arrays are sized `capacity * gate_expert_bytes` / `capacity * down_expert_bytes` (`ds4_cuda.cu:27271-27277`), and each miss copies that many bytes out of the model mapping at the stored offset (`ds4_cuda.cu:27349-27361`). Since `--ssd-streaming-cache-experts NGB` is a byte target and one eviction removes exactly one stored expert, a nominal budget of N GiB buys N GiB of arena and removes that many bytes per token. The warning belongs to any future **dense** weight cache on this path, where GGUF bytes go in and dequantized bytes come out: such a budget must be divided by the dequantization ratio before anyone claims the disk saving, or it will under-deliver by exactly that factor and look like a wall it is not.
- Both of this section's sources are workspace-local and **not tracked in this fork's tree**: `research/v41-flash-landscape/09-macs-and-dwarfstar/` (the Mac-mini study and its cloned reader) and `WIKI/theory/48-decode-speedup-levers.md` with `WIKI/theory/47-profiling-methodology.md` (our roadmap and the knee's method). They are cited by path and line so a reader with the workspaces can check them, and flagged so nobody hunts for them in a clone.

**Why these cells are still blank.** A native pass was run on 2026-09-15 and its result files
exist on the Spark, but its control is not sound, so the numbers are WITHHELD rather than
published. In that pass every arm read above its control - including the arms that switch this
lever **off**, and an off arm is the control by construction, so it cannot beat it by several
percent. Two of the three levers measured better switched off than switched on. That is the
signature of a bad control rather than of a lever, and the control in that pass read 9.17 to
9.46 tokens/s against 9.56 in a later pass that produced mixed, believable results.

So the honest statement is that this measurement is **owed** with a clean interleaved control,
and these cells stay blank until it exists. Publishing the 2026-09-15 figures would put a
number in a table that the run behind it does not support.

**Where the clean numbers will arrive.** A clean series is being run on the Spark with the
harness `try.sh` on branch `triple-all-fastest`. Its results land under `~/sweeps/` as dated
markdown-table files matching `~/sweeps/2026-09-16-*.txt`, plus one MATRIX file beside them,
and every file states what was ON, the build vintage, the interleaved control and the noise
floor - so any figure there can be traced back to its run without trusting this README.

**Build status of this revision.** The CUDA build gate is **OWED**: a CUDA build needs the DGX
Spark, which is running an unrelated series, and there is no CUDA toolchain on this box. So the
code in this branch has been **reviewed and logic-checked, not compiled**, and that is not a claim of
working. What was actually run is the host-side harness described above (`/tmp/hold_harness.cpp`,
generated by `/tmp/mk_harness.py`, both outside the repo and so not committed: this branch's commit
shape is fixed at two commits). It splices the patched
regions out of `ds4_cuda.cu` verbatim, compiles them with `clang++`, and replays 20,000 randomized
cache states plus a tight-cache phase against upstream's own comparator and victim choice. That
checks ordering, set membership, held-last placement and starvation parity. It does NOT check that
the file compiles, that the held band behaves as intended under a real prefill, or anything about
tokens/s. The first honest number for this branch requires a Spark build of both arms.

Status: **OWED** - until a file in `~/sweeps/` carries this lever's clean interleaved A/B,
the cells above stay blank.
