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
  │  BRANCH     triple-pool                                   POSITIVE
  │
  │  WHAT       serves expert reads from a parallel SSD pool instead of one
  │             serial reader, with an io_uring O_DIRECT ring in front of the
  │             pool so queue depth is a switch, not the worker count.
  │
  │  LATEST     round 4 · 2026-09-17 · BEATS · +7.27 % (kept +7.75 %) ·
  │             2026-09-17-R4-RESULT-hitsfirst-pool-and-the-newtip-stack-beat-the-tip-readahead-order-loses-eleven.md
  │             round 8 · 2026-09-17 · TIES · +8.56 % 3/4 under a 10.69 floor,
  │             two runs in a loaded window - round 4's +7.27 stands ·
  │             2026-09-17-R8-RESULT-hitsfirst-alone-beats-through-the-noise-the-winners-tree-does-not.md
  │
  │  GEN        round 4, the branch's number: +7.27 % (kept +7.75 %)
  │             floor 2.71 % (no-drop)  sign 3/4  n=4
  │             raw: arm median 10.39 t/s vs tip median 9.65 t/s, gen_steady
  │             at min across 4096/6144; the delta is each run against the
  │             mean of its bracketing TIP runs, not against that median
  │             control = triple-tip-2026-09-16 @12997e9c, interleaved, one TIP
  │             bracket per cycle
  │             session = 2026-09-17 09:50-10:57Z · DGX Spark GB10 ·
  │             native 24.89 MB .text · lean regime 4096/6144
  │             ─── round 8, TIES, neither confirms nor retracts the above:
  │             +8.56 %  floor 10.69 %  sign 3/4  n=4 (sensitivity, cold first
  │             control run excluded: +3.31 % against a 7.17 % floor, TIES)
  │             raw: arm min-across median 10.00 t/s (10.65 · 10.47 · 8.98 ·
  │             9.53) vs the tip's 8.96 t/s (8.61 cold · 9.53 · 8.87 · 9.49 ·
  │             8.96). Two runs read as round 4 did, two fell in the loaded
  │             window. ⚠ the box was not quiet: the five tip runs spanned
  │             8.61 to 9.53 and four arm runs ended at load 3.2 to 4.0 with no
  │             vitest alive, which widened the floor from 2.71 to 10.69
  │             control = triple-tip-2026-09-16 @12997e9c, interleaved
  │             session = 2026-09-17 13:24-14:37Z · DGX Spark GB10 ·
  │             native 24.89 MB .text (tip 24.86 MB) · lean regime 4096/6144
  │  PREFILL    reported, not a verdict: round 4, 87.25 t/s min-across median
  │             vs the tip's 80.73 t/s, n=4; round 8, 86.47 t/s vs 84.34 t/s,
  │             n=4. Neither round makes a prefill verdict.
  │
  │  VERDICT    POSITIVE on round 4, +7.27 % no-drop and +7.75 % kept, and that
  │             stays the branch's number. ⚠ Round 4's first run read 9.24 t/s,
  │             below the tip, and the next three read 10.34 to 10.47; a cold
  │             io_uring ring and an ordinary straggler look identical there and
  │             round 4 could not tell them apart. Round 8 ran the branch warm
  │             and read +8.56 % at 3 of 4 under a 10.69 floor: two runs at 10.47
  │             to 10.65, as round 4, and two at 8.98 to 9.53 in the loaded
  │             window. That TIES, so it neither confirms nor retracts round 4.
  │             Round 9 re-measures it on a quiet box
  │
  │  SWITCH     DS4_CUDA_FETCH_QD=<n> ring queue depth, default 64, clamped 8-512
  │             DS4_CUDA_FETCH_URING=0 falls back to the pread pool
  │             DS4_CUDA_FETCH_BUFFERED=1 forces buffered reads
  │             DS4_CUDA_STREAMING_EXPERT_PREAD_THREADS=<n> pool workers, default 8, cap 16
  │             DS4_CUDA_STREAMING_EXPERT_PREAD_POOL=0 restores the serial path
  │  OUTPUT     not re-run
  │
  └──────────────────────────────────────────────────────────────────────
```

### The repair: one missing store hung the whole run

`cuda_uring_prep_read` advanced a private tail counter. It never stored to `*r->sq_tail`, the
only tail the kernel reads. So:

```
   THE DEFECT, and the single store that closes it

   prep_read fills SQE 0..4         PRIVATE tail          SHARED *r->sq_tail
                                    ▔▔▔▔▔▔▔▔▔▔▔▔          ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔
   before the fix                   5                     0    never stored
                                                          ▲
       io_uring_enter(to_submit=5) ──────────────────────  the kernel reads THIS one
       kernel admits what the SHARED tail admits  =  0 SQEs
       returns 0, waits for nothing, nothing ever drains
       ⇒ the FIRST cache miss of the run busy-spins forever

   after the fix, in ds4_uring_sq.h
       ds4_uring_sq_publish:  __atomic_store_n(sq_tail, tail, RELEASE)
       called before EVERY enter, so the SQE writes are visible first
       the loop is BOUNDED at sq.entries + 8 rounds
       a ring that accepts nothing is DECLINED, and the pread pool takes the batch
```

Three defects were fixed in one commit, all from review R2:

- **The ring never submitted.** The tail fix above. The SQ accounting moved into
  `ds4_uring_sq.h`, which publishes the tail with release ordering, fills the SQ index array per
  SQE, and bounds the submit loop. The drained test is wrap-safe. `cuda_uring_peek_cqe` now loads
  the CQ tail with acquire ordering before reading the CQE it admits.
- **A slot was claimed before its read.** `cuda_stream_selected_cache_begin_load` published
  gate -> victim before the bytes arrived. Its mid-loop `victim == UINT32_MAX` return left every
  earlier claim published over device memory nothing had written, and the next call routing to one
  of those experts took the hit path and ran the layer on stale bytes. Claims now go through
  `ds4_expert_claim_ledger`: provisional until `commit()`, withdrawn on every exit. Withdrawal is
  exact - a claim erases its own gate, and only while that gate still names the slot it claimed.
- **The O_DIRECT rejection was retried per chunk.** A worker cannot close the global direct fd, so
  the disable waited for the join, and until then every 8 MiB chunk of every task paid its own
  failing `pread`. The rejection is now recorded once against the fd that took it
  (`g_model_direct_rejected_fd`), read by both the pool's chunk loop and the ring's job prepare,
  and cleared by `ds4_gpu_set_model_fd` when a new fd is opened.

Tests are host-side, because this engine only builds under `nvcc` on Linux. Both are in `make test`:

- `tests/test_uring_sq.cpp` - 7 cases, driving the submit loop against a fake kernel that reads
  the shared tail as the real one does. Planting the unpublished tail back fails 9 verdicts;
  planting the unfilled index array fails 1.
- `tests/test_expert_claims.cpp` - 6 cases on the ledger. Planting a no-op withdrawal back fails 7.

### ⚠ What the repair makes reachable

The bounded loop is correct and it changes which failure you get. The lane that wrote it said so
in its own review, and the card carries the warning rather than burying it:

- Before the bound, a ring that accepted nothing **hung** in `cuda_uring_enter`. The process
  stopped there and never reached the code below it.
- With the bound, that same ring **declines**. The decline path in `cuda_expert_uring_dispatch`
  returns every job's host buffer to the freelist with `cuda_fetch_buf_put` and closes the ring.
- Those buffers can still carry `inflight = 1` from SQEs already handed to the kernel. So the
  freelist may take back a buffer the kernel could still complete into, and hand it to the next
  task.
- **Neither state has ever been observed.** The branch had no CUDA build at all until round 3
  compiled it on 2026-09-17, and its runs in that round are still in flight. The hang is gone;
  whether the decline is clean is an open question and the first thing that build's runs answer.

**Re-check it by name, not by line.** This was read-verified in `ds4_cuda.cu`, in
`cuda_expert_uring_dispatch`: the queueing loop sets `j->inflight = 1` before `cuda_uring_enter`,
and the `if (!touched)` branch taken when that enter fails calls `cuda_fetch_buf_put` on every
`jobs[i].host_buf` and then `cuda_uring_ring_close`, with no wait for the inflight completions.
The bound that makes the branch reachable is `max_rounds` in `ds4_uring_sq_submit`
(`ds4_uring_sq.h`), passed as `r->sq.entries + 8u`. Four names, all unique, all greppable.

### The three engines, in order

`cuda_expert_pread_pool_dispatch` tries them on every miss batch:

- **The io_uring ring** (`cuda_expert_uring_dispatch`). It speaks the kernel UAPI directly -
  `io_uring_setup` / `io_uring_enter`, mmap'd SQ and CQ. No liburing, no Makefile change. It
  compiles out unless `<linux/io_uring.h>` and both syscall numbers exist.
  `DS4_CUDA_NO_IO_URING` forces it out; `DS4_CUDA_HAVE_IO_URING=1` forces it in.
- **The pthread pread pool**, N workers, each doing `pread` -> H2D -> sync -> drop pages.
- **The serial ring**, unchanged from upstream: one tensor at a time, 4-chunk staging.

Per task the ring's geometry matches `cuda_expert_stage_read_mt`: an aligned O_DIRECT bracket on
the direct fd when the window fits the file, otherwise a buffered read. A rejected direct read
(`EINVAL`, `EFAULT`, `ENOTSUP`, `EOPNOTSUPP`) is re-read buffered once in batch. Aligned buffers
recycle through a process-wide freelist (`cuda_fetch_buf_get` / `cuda_fetch_buf_put`, cap 128
buffers, 512 MiB).

The pool declines a batch when the lever is off, the batch holds one task or none, an allocation
fails, or a batch is already in flight. Then the serial path takes it. When all three tensors
land the layer remaps; when they do not, the slot is rolled back and the load fails.

### Failure is a decline, never a fault

- A ring that cannot init, a batch of one task, or a batch it cannot stage all decline **before a
  byte is read**, and the pool takes the work.
- A ring that breaks mid-batch reports the failed tasks through their `ok` flags. That is the
  contract the pool already uses.
- Unchanged from upstream: the resident slot cache, gate-indexed lookup, LFU-with-stamp eviction,
  look-ahead protection, per-tensor error reporting.
- `ds4_gpu_stream_expert_cache_release_resident` joins the pool, destroys the ring and the upload
  stream, and drains the freelist.

### Two notes on what is NOT here

- **Not ported, on purpose.** Upstream donates a completed read buffer to a disk-to-host-RAM
  expert cache and reclaims it on eviction. This tree's cache is the VRAM slot table keyed by gate
  offset, so there is nothing to donate to.
- **A dead switch name.** An earlier card named `DS4_SSD_READERS=<n>`. No source file reads that
  name. The worker count is `DS4_CUDA_STREAMING_EXPERT_PREAD_THREADS`.

### Owed, in order

1. `make cuda-spark` on the Spark. This branch has never compiled anywhere, so the build is the
   first fact, not the last.
2. One interleaved A/B of the ring against the pool, `DS4_CUDA_EXPERT_CACHE_STATS=1` on both arms.
3. A greedy-identity check.
4. Point the first run at the decline path above and find out whether it is clean.

The line to clear: the clean tip reads **9.65 t/s median**
(`2026-09-16-BISECT-RESULT-one-comparator-carries-the-whole-loss.md`).

## Round 4, 2026-09-17: on the new tip, in a sealed round

- **First CUDA build this branch has ever had, and it beats the tip.** +7.27 % no-drop, +7.75 % kept, sign 3 of 4, no-drop floor 2.71 %. Raw min-across medians 10.39 t/s arm against 9.65 t/s tip.
- ⚠ The dip is on the round's FIRST run: 9.24 t/s, then 10.47, 10.45, 10.34. A cold io_uring ring and an ordinary straggler produce the same shape, and round 4 has no arm that separates them. The card claims the win and keeps the caution.
- The two readings differ only in whether that first run is kept, and both say BEATS, so the straggler-rule problem in section 3 of the result does not decide this card.

Sealed rule `2026-09-17-R4-PREREGISTERED-RULE.txt` @`3914a3226`, result `2026-09-17-R4-RESULT-hitsfirst-pool-and-the-newtip-stack-beat-the-tip-readahead-order-loses-eleven.md`, raw CSVs and runlog in `sweeps/r4/`.

## Round 8, 2026-09-17: warm, and still not settled

- **Round 4's +7.27 % stands as the branch's number.** Round 8 read **+8.56 %, 3 of 4, under a
  10.69 % floor**, which is a TIE and decides nothing either way. Two of its runs read 10.47 and
  10.65 t/s, exactly where round 4's three good runs sat; the other two read 8.98 and 9.53 in the
  loaded window that widened the round's floor.
- **The cold-ring question from round 4 is answered in one direction only.** This round started
  warm (the tool now discards a warm-up run, `c8699505b`) and the first run was one of the two
  high ones, so no cold-start dip appeared. That is consistent with the cold-ring reading and does
  not exclude an ordinary straggler at n=4.
- ⚠ **Stacking this lever with hitsfirst loses both wins.** `triple-winners` (pool + hitsfirst +
  prefetch-pool) read 9.13 to 9.42 in this same session, level with the tip, while this branch's
  clean runs read 10.5 to 10.65 and hitsfirst alone read 10.3 to 10.5. Both levers touch the same
  expert-pread path; the claim ledger is the first suspect and round 8 does not prove it.

Sealed rule `2026-09-17-R8-PREREGISTERED-RULE.txt` @`c173ad6d7`, result
`2026-09-17-R8-RESULT-hitsfirst-alone-beats-through-the-noise-the-winners-tree-does-not.md`,
raw CSVs and runlog in `sweeps/r8/`.
