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
  │  BRANCH     triple-winners                                WORTH ZERO
  │
  │  WHAT       the three levers round 4 measured as winners solo on the
  │             new tip, stacked and nothing else: pool (+7.27 %), hitsfirst
  │             (+6.47 %) and prefetch-pool (+2.28 %), with HITSFIRST ON.
  │             Round 8 measured it level with the tip: the combination costs
  │             what each part wins
  │
  │  LATEST     round 9 · 2026-09-17 · TIES · +5.03 % 3/3 under a 17.27 floor ·
  │             median 10.20, between its parts ·
  │             2026-09-17-R9-RESULT-the-default-holds-hitsfirst-is-the-tightest-tree-on-a-loud-instrument.md
  │             round 8 · 2026-09-17 · TIES · +1.54 % (-0.54 sensitivity) ·
  │             level with the tip while its parts read +8 to +12 in the same
  │             session ·
  │             2026-09-17-R8-RESULT-hitsfirst-alone-beats-through-the-noise-the-winners-tree-does-not.md
  │
  │  GEN        +1.54 %  floor 10.69 %  sign 2/3  n=3 of 4 paired
  │             sensitivity, cold first control run excluded: -0.54 % against a
  │             7.17 % floor, TIES either way (post hoc, labelled as such)
  │             raw: arm min-across median 9.18 t/s (9.21 · 9.42 · 9.13 · 9.15)
  │             vs the tip's 8.96 t/s (8.61 cold · 9.53 · 8.87 · 9.49 · 8.96).
  │             gen_steady at min across 4096/6144. The delta is each run
  │             against its bracketing TIP runs, not against that median
  │             ⚠ the box was not quiet: the five tip runs spanned 8.61 to 9.53
  │             and four arm runs ended at load 3.2 to 4.0 with no vitest alive,
  │             which is what widened the floor to 10.69. But this arm's own
  │             four runs span only 9.13 to 9.42, so its LEVEL reading is not a
  │             noise artifact; the noise bounds how strongly it can be called
  │             a loss, not whether it won
  │             control = triple-tip-2026-09-16 @12997e9c, interleaved
  │             session = 2026-09-17 13:24-14:37Z · DGX Spark GB10 ·
  │             native 24.90 MB .text (tip 24.86 MB) · lean regime 4096/6144
  │             ─── round 9, TIES, and it reads higher than round 8 did:
  │             +5.03 %  floor 17.27 %  sign 3/3  n=3
  │             raw: arm min-across median 10.20 t/s (9.69 to 10.45, spread 0.76)
  │             vs the tip's 9.42 t/s (8.51 to 9.98). That sits BETWEEN its parts,
  │             below hitsfirst's 10.41 and pool's 10.35, which is the second
  │             round to rank it under hitsfirst alone
  │             ⚠ the instrument sets that floor, not the arm: the five control
  │             runs span 8.51 to 9.98 t/s because the page cache holds 2.9 GB of
  │             an 81 GB model and the bench streams 842 MB/s from the NVMe, so
  │             identical binaries swing about 8 % run to run
  │             control = triple-tip-2026-09-16 @12997e9c, interleaved
  │             session = 2026-09-17 14:41-15:36Z · DGX Spark GB10 · native
  │             24.90 MB .text (r9-winners, 24,900,766; tip 24.86 MB) · lean
  │             regime 4096/6144 · quiet box, warm-up run discarded
  │  PREFILL    reported, not a verdict: 135.09 t/s min-across median vs the
  │             tip's 84.34 t/s, n=4. Round 8 makes no prefill verdict.
  │
  │  VERDICT    WORTH ZERO - built correct, compiled, measured, and level with
  │             the tip. The combination costs what each part wins: pool read
  │             10.47 to 10.65 on its clean runs and hitsfirst read 10.29 to
  │             10.49, all four of them above every control run, in this same
  │             session, while this tree that carries both read 9.13 to 9.42.
  │             So the answer to "why not combine pool and hitsfirst and
  │             prefetch-pool" is measured, and it is no. The two pread-path
  │             levers fight; the claim-ledger interaction the build lane
  │             carried is the first suspect and round 8 does not prove it
  │
  │  SWITCH     DS4_CUDA_HITS_FIRST=0 turns hits-first off (ON here, the
  │             lever's own default); DS4_CUDA_HITS_FIRST_STAGED=0 keeps it
  │             single-stage
  │             DS4_CUDA_FETCH_URING=0 falls back to the pread pool;
  │             DS4_CUDA_STREAMING_EXPERT_PREAD_POOL=0 restores the serial path;
  │             DS4_CUDA_FETCH_QD=<n> ring depth, default 64, clamped 8-512
  │             DS4_CUDA_SSD_PREFETCH_POOL=0 or DS4_CUDA_SSD_PREFETCH_THREADS=1
  │             reverts the prefetch pool; DS4_CUDA_SSD_PREFETCH_CHUNK_MB,
  │             default 8, cap 64
  │  OUTPUT     not re-run
  │
  └──────────────────────────────────────────────────────────────────────
```

### Why this tree exists

Round 4 (`2026-09-17-R4-RESULT-hitsfirst-pool-and-the-newtip-stack-beat-the-tip-readahead-order-loses-eleven.md`)
measured seven arms solo on the new tip. Three won. The ten-lever stack that carries all of them
measured +4.65 % with hitsfirst OFF, and round 6 found the six-lever `triple-all`, which carries
none of hitsfirst or prefetch-pool, at +6.50 %. So the open question is not "more levers" but
"the winners, alone, with the one that was shipped off turned on". The navigator asked it
directly: *"why not combine pool and hitsfirst and prefetch-pool?"* This branch is that tree.

```
   the arms round 4 measured solo, and the one this tree stacks   (gen, no-drop, vs the tip)

   readahead-order   -11.38   ●━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
   hotlist            -2.73   ●━━━━━━━━━━━━┫
   engram-read        -1.39   ●━━━━━━┫
                                            ╵ tip
   prefetch-pool      +2.28                 ┣━━━━━━━━━●     ✦ carried
   newtip stack       +4.65                 ┣━━━━━━━━━━━━━━━━━━━●     (hitsfirst OFF)
   triple-all         +6.50                 ┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━●     round 6
   hitsfirst          +6.47                 ┣━━━━━━━━━━━━━━━━━━━━━━━━━━━●       ✦ carried, ON
   pool               +7.27                 ┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━●   ✦ carried

   triple-winners     +1.54                 ┣━━━━━●     ROUND 8, level with the tip
```

### How this tree was built

- Base `triple-tip-2026-09-16` at `12997e9c8`. One commit per lever, in manifest order, each
  the lever's live rebased branch at its code sha: `triple-pool` `6371428b0` · `triple-hitsfirst`
  `3e4773671` · `triple-prefetch-pool` `ed11cb474`. Every lever branch is already rebased onto
  this tip, so each diff carries only its own code.
- `winners.manifest` is the definition; `build_stack.sh` reproduces it into a fresh detached
  worktree and `--check <sha>` asserts byte identity on the build surface. 49 conflict hunks,
  every one decided and recorded: ours 14 · theirs 19 · both 3 · hand 13. The hand merges are
  data under `stack.patches/winners/` and the manifest header says why each is a third text.
- The claim-ledger hunk (hitsfirst `ds4_cuda.cu 12`) IS needed with hits-first on: when
  hits-first takes the batch the pool owns the reads and `cuda_hits_first_wait` drops any slot
  whose bytes do not land, so every claim is committed there, or pool's RAII ledger would
  withdraw the slots on return while the reads are still in flight.
- The prefetch victim order is the tip's own, `used` ascending. The sweep-aware comparator and
  `DS4_CUDA_PREFETCH_SWEEP_ORDER` belong to prefill-readahead-order, which this tree does not
  carry; there is no region here to gate.
- Not carried, on purpose: margin, draincut, pagecache, hotlist, engram-lead,
  engram-read-threads, prefill-readahead-order. Each has its own branch and its own number.
- Tests here: `tests/test_hitsfirst_logic` 142 checks · `tests/test_pread_pool_config` 59
  checks · `tests/test_uring_sq` PASS · `tests/test_expert_claims` PASS. Metal `make` rc 0.
  `ds4_cuda.cu` cannot compile on this machine.

## Round 8, 2026-09-17: the question is answered, and the answer is no

- **It reads level with the tip: +1.54 %, 2 of 3, under a 10.69 % floor**, and -0.54 % on the
  cold-run sensitivity against a 7.17 % floor. Its four runs span 9.13 to 9.42 t/s against a
  control median of 8.96, so the LEVEL reading is its own, not the round's noise.
- **Its parts beat, in the same session, on the same box.** hitsfirst alone read 10.29 to 10.49
  (**+12.31 %, 4/4, BEATS**) and pool's two clean runs read 10.47 and 10.65. A tree carrying both
  read below both. Prediction P7 ("winners highest delta") and P8 ("winners beats triple-all's
  +6.50") were sealed at 0.5 each and both missed.
- ⚠ **Not attributed.** pool and prefetch-pool touch the same expert-pread code and hitsfirst
  calls the pool's own entry points, so three candidates remain open: the claim ledger, queue-depth
  contention on one NVMe, and the hits-first drain interacting with the prefetch pool's own
  in-flight batch. Round 8 separates none of them.

```
   T H E   P A R T S   B E A T ,   T H E   W H O L E   D O E S   N O T
                                        (round 8, min-across gen, raw t/s)

   control tip   8.61 ↑cold  8.87   8.96   9.49   9.53
                              ╵      ╵      ╵      ╵
   winners                 9.13  9.15 9.21 9.42        ◀ inside the control span
   hitsfirst                                    9.43        10.29 10.38 10.49  ✦ above all five
   pool                                   8.98      9.53          10.47 10.65

   the two levers that clear the tip alone do not clear it together
```

Sealed rule `2026-09-17-R8-PREREGISTERED-RULE.txt` @`c173ad6d7`, result
`2026-09-17-R8-RESULT-hitsfirst-alone-beats-through-the-noise-the-winners-tree-does-not.md`,
raw CSVs and runlog in `sweeps/r8/`.

## Round 9, 2026-09-17: it reads between its parts, and the answer is still no

- **Round 8's level-with-the-tip reading was the loaded window, and round 9 says so.** On a quiet
  box this tree reads 10.20 t/s, +5.03 %, sign 3 of 3, against round 8's +1.54 %. Two of round 8's
  four winners runs sat in that loaded window.
- **The combination is still not better than hitsfirst alone, in either round.** 10.20 against
  hitsfirst's 10.41 and pool's 10.35: it lands between its own parts. How much worse it is depends
  on the day, and that is the honest width of the claim.
- **What it is NOT is a loss.** Both readings TIE against their floors, so nothing here retracts
  the tree's correctness or kills a lever; what is measured is a ranking, twice.

Sealed rule `2026-09-17-R9-PREREGISTERED-RULE.txt` @`60d1bd06a`, result
`2026-09-17-R9-RESULT-the-default-holds-hitsfirst-is-the-tightest-tree-on-a-loud-instrument.md`,
raw CSVs and runlog in `sweeps/r9/`.
