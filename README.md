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
  │  BRANCH     triple-engram-read-threads                         WORTH ZERO
  │
  │  WHAT       takes the DECODE step's Engram read off the batch machinery:
  │             one token now goes straight to the serial reader, with no
  │             malloc, no qsort and no pool dispatch. The wide PREFILL read
  │             keeps the one-wave rule the branch was named for.
  │
  │  LATEST     round 1 bisect · 2026-09-16 · NOT IT: the RT arm is this same
  │             decode revert, made on the nine-lever stack at dd82361a, and it
  │             read -6.07 % against the clean tip where the stack as shipped
  │             read -4.35 %, so |D_RT - D_STACK| = 1.72, inside the 4.13 %
  │             floor, and the sealed rule reports no direction
  │             (2026-09-16-BISECT-RESULT-one-comparator-carries-the-whole-loss.md)
  │             round 3 is measuring this branch now: in flight, unresolved
  │
  │  GEN        not measured on this revision, which has never been built for
  │             CUDA. The RT arm above measured the revert on another tree.
  │  PREFILL    not measured. The one-wave prefill read has never run.
  │
  │  VERDICT    WORTH ZERO - correct, tested, and worth nothing on the clock.
  │             The decode fast path is right: it takes a malloc, a qsort and a
  │             broadcast to 32 parked workers off the critical path, and a
  │             counter proves the pool never wakes. The bisect put that same
  │             revert 1.72 points from the stack against the one comparator's
  │             8.04, a factor of 4.7. A fix can be right, well tested, and buy
  │             nothing. OWED: the one-wave PREFILL read, the untested half.
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

## The cost it attacks

- A decode step's Engram read is 48 serial 264-byte `pread` calls at queue depth one, moving
  12,672 bytes, all complete before the first GPU command. That is 24 rows per table, two
  tables, per token.
- Measured on the GB10: engram mean **23.922 ms** of a **198.636 ms** step, **12.04 %**
  (`speed-bench/v41_engram_lead_gb10.md`).
- ⚠ That is the cost the branch attacks. It is not a measurement of the branch.

## The two defects the branch found

**One. Sixteen concurrent readers were already in the tree and decode could not reach them.**

- The old rule engaged one reader per two rows, capped at 16.
- A 24-request table therefore got 12 readers and ran two serial rounds.

**Two. The batch machinery was charging the decode path for work it could not use.**

- `ds4_engram_read_batch` routed every one-token read through a malloc, a qsort, a pool
  dispatch and a broadcast to 32 parked workers, per table, per token, on the critical path.
- The sort cannot reorder work that is already one token wide, and the dedup it buys is a
  memcpy against a 264-byte `pread`.
- No value of `DS4_ENGRAM_READ_THREADS` or `DS4_ENGRAM_ROWS_PER_READER` restored the plain
  `ds4_engram_read` loop the engine used at `9139e2ae`, so the branch had no OFF form for its
  own experiment.
- Replaying the same 48-read pattern with no model and no GPU: serial p50 **9.600 ms** against
  the pre-fix batch path's p50 **10.054 ms** (`speed-bench/v41_engram_read_threads_gb10.md`),
  because at one token that batch reader took the serial path anyway, plus a sort.

```
   ds4_engram_read_batch(table, rows, tokens, stride, out)

   tokens == 1 · THE DEMAND PATH, decode
       no malloc · no qsort · no pool dispatch
       ds4_engram_read(t, rows, 24, out)      24 serial 264-byte preads
       ds4_engram_pool_dispatches() must not move  <- the asserted invariant
       NEITHER KNOB IS READ ON THIS PATH

   tokens >= 2 · THE WIDE PATH, prefill, untouched by the fix
       malloc · qsort once · slice the sorted array across the parts
       engram_reader_count(count, pool) = count / ROWS_PER_READER
           capped by the REAL pool size, 32 parked workers + the caller = 33
           and capped by count, floor 1
       ROWS_PER_READER=1  default   24 ids  ->  24 parts, ONE wave
       ROWS_PER_READER=2  control   24 ids  ->  12 parts, two serial rounds
       the caller claims parts too, so progress never waits on a wakeup
```

## The mechanism as it stands

- `ENGRAM_ROWS_PER_READER` is 1. `engram_reader_count(count, pool)` is
  `count / rows_per_reader`, capped by the real pool size and by `count`, floor 1.
- The pool is one process-wide instance, created on first use and shared by both tables: 32
  parked workers plus the caller, so the cap is 33 and widening the pool widens the read with
  no second constant.
- The per-call `dispatch_apply_f` / `pthread_create` batch is gone. The caller claims parts
  too, so progress never depends on a wakeup.
- No backend symbol is used, so Metal, ROCm and CPU all get the same pool.
- Both decode call sites, single-sequence and batched, use the batch entry point, and the
  `tokens == 1` branch inside it hands them to the serial reader.
- The override is read per call rather than cached. It runs once per token per table,
  immediately before dozens of disk reads, so the lookup is free and no first caller wins the
  setting for the lifetime of the process.
- `ds4_engram_pool_dispatches()` counts the runs that woke the pool, so the decode claim is
  observable rather than argued.

## What the bisect measured, and why it is worth writing plainly

- The bisect's **RT** arm, sealed in `2026-09-16-BISECT-PREREGISTERED-RULE.txt`, is
  *"STACK with ONLY the two decode-path `ds4_engram_read_batch` calls reverted to
  `ds4_engram_read`"*. That is this branch's decode change, reached at the call sites instead
  of from inside the function. The effect on the decode path is the same: no malloc, no qsort,
  no pool.
- Result: RT **-6.07 %** against the clean tip, where the stack as shipped read **-4.35 %**.
  Per frontier, RT read -9.25 / -6.06 / -5.73 % at 4096 / 6144 / 8192, sign 0 of 4.
- The sealed rule: `|D_RT - D_STACK| = 1.72 <= 4.13`, the in-run tip floor. **RT is NOT IT.**
  Its sign is slightly worse than the stack's, that sign is inside the floor, and it is
  therefore not reported as a direction.
- ⇒ **The fix is correct, it is tested, and it is measured worth nothing on the clock.**
  Both halves of that sentence are true at once, and neither cancels the other. The malloc,
  the qsort and the 32-worker broadcast really are off the critical path; the clock did not
  notice.
- ⚠ The arm ran on the NINE-lever binary at `dd82361a`, a sibling of `triple-all-fastest` off
  `84ba6ef1b` (`2026-09-16-CORRECTION-the-measured-stack-is-nine-levers-and-has-diverged.md`).
  The ten-lever tree has never been measured.
- For scale, from the same round: the one thing that DID move the stack was a single eviction
  comparator in `ds4_gpu_stream_expert_cache_prefetch`, worth **8.04 points**, -4.35 % to
  +3.69 %, sign 4 of 4. That is 4.7x this change's 1.72.

## A different regime, not to be read against this one

- `WIKI/theory/177` section 16 finds a few-MiB expert tensor read is bandwidth-bound: chunking
  is slower, and 8 / 16 / 32 whole reads in flight give 10.1 / 9.9 / 9.7 GB/s.
- This read is 264 bytes a row and IOPS-bound: 48 rows in 9.6 ms is 5.0k IOPS, against a drive
  at about 3.5k at QD=1 and about 112k at QD=64.
- The row store this design follows
  (`research/v41-flash-landscape/03-triple-spark-mxfp4-engram/`) uses chunk 1 and one shared
  pool, and reports its exposed read going from about 11 ms to 1-3 ms a step.
- ⚠ That is their number on their shape.

## The tests

`tests/test_engram.c`. Pure C, no GPU, no model.

- It sweeps both knobs at 1, 2 and 40 tokens against the serial per-row reader: the reader
  count changes how rows are fetched, never which rows or where they land.
- `test_decode_read_never_dispatches` asserts `ds4_engram_pool_dispatches()` is zero on entry
  and does not move across seven settings of the reader knob, with the bytes compared against
  the serial reader each time.
- `test_wide_read_still_dispatches` is its control, proving the counter CAN move.
- RED before the fix, at the first unset pass:
  `Assertion failed: (ds4_engram_pool_dispatches() == before), function
  test_decode_read_never_dispatches, line 391`.
- Re-run on this Mac 2026-09-17 (`make tests/test_engram && ./tests/test_engram`):
  **PASS, exit 0** - hashes, history, bounded disk rows, reader counts and a serial demand read.

## Caution for re-measuring

- `F_NOCACHE` and `F_RDAHEAD(0)` are set on macOS only.
- So on Linux a repeated n-gram costs about 0.15 ms against 20 to 25 ms for a novel one.
- ⇒ two runs of one prompt are not two samples, and a baseline prompt is single-use.

## Owed

- The CUDA build, and this revision's release gates. This revision has never been compiled
  against CUDA anywhere.
- ⚠ The `ROWS_PER_READER=1` vs `2` A/B has CHANGED MEANING and must be re-scoped. It was the
  decode experiment's control arm. Since the fix, neither knob reaches the decode read, so
  that A/B now measures the **wide prefill read** and nothing else. The wide-decode experiment
  it used to be needs the `tokens == 1` fast path removed before it can run at all.
- That prefill A/B, interleaved on one host against the tip: 9.70 gen t/s median (n=16 of 18,
  2026-09-16), whose round-1 band was 9.44 to 9.83 with a median of 9.65.
- The 2026-09-15 native pass is withheld: its control read 9.17 to 9.46 t/s and every arm, the
  off arms included, beat it.
