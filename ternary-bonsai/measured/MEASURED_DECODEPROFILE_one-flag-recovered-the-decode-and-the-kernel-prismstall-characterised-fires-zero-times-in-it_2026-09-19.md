# MEASURED · DECODEPROFILE - one flag recovered the decode, and the kernel PRISMSTALL characterised fires ZERO times in it

**Lane DECODEPROFILE, branch `autonomous/dspark-concurrency-and-spark-queue`. Sealed as
`PREREG_DECODEPROFILE_01_*` at commit `cb2877b38` before anything ran. 2026-09-19 (UTC); the spark is
UTC+10 and its `ls` prints 2026-09-20 AEST for the same instant.**

Row taken: PRISMSTALL's owed item 1, verbatim - *"re-run `~/qr_prism_nsys.sh` with
`--cuda-graph-trace=node`, ~15 min with the lock. Until then no claim about decode's kernel mix is
supported by anything, mine included."* Seal `270488ba5`, result `8e1a51616`, both on `lane/prismjev`.

**No t/s claim is made here for anything. Nothing is called lossless. No code was changed, no token
was generated for a gate, and no A/B was bracketed.**

---

## ⓪ THE HEADLINE, in four parts

**(1) THE FLAG WORKS, AND THE GAP CLOSES TO ZERO.** PRISMSTALL's artifact had **2,470 launch calls
issued after its last kernel record**. Mine has **0**. The kernel channel now runs to **81.099 s**
against a runtime channel ending **81.462 s**: a residual of **+0.363 s**, where PRISMSTALL's was
**+42.814 s**. **297,561 recovered rows carry a non-NULL `graphNodeId`.**

**(2) THE ANSWER THE ROW ASKED FOR.** `mul_mat_q<(ggml_type)142, (int)64>` runs **353.00 times per
decode step**, **45,184 times over 128 steps**, which is **5.515625 launches per GENERATED token**.
The whole `<142>` family runs **561 per decode step = 8.765625 per generated token.**

**(3) ★★★ THE KERNEL PRISMSTALL CHARACTERISED RUNS ZERO TIMES IN THE DECODE.**
`mul_mat_q<(ggml_type)142, (int)128>` fires **25,600 times in the prefill and 0 times in the decode**.
Its 255 registers, its 102,400 B of shared memory and its 4,051 SASS instructions are a description of
**the prefill tile width.** The decode runs the **J=64** instantiation instead.

**(4) THE DECODE IS NOT MATMUL-DOMINATED THE WAY THE PREFILL IS.** The `<142>` family is **59.003 %**
of prefill kernel time and **32.580 %** of decode kernel time. The two largest decode kernels are
neither: `k_get_rows_float_vec` at **27.86 %** and `gated_delta_net_cuda` at **27.71 %**.

⇒ **THE CALL-PATH GATE IS NOW SATISFIED FOR THE DECODE - for `mul_mat_q<142,64>`, not for the kernel
the PUSH-branch row was written about.**

---

## ① DID THE FLAG RECOVER THE CHANNEL, AND HOW DO I KNOW

The before column is PRISMSTALL's artifact, **re-measured by this lane's own script rather than quoted**
(see §② for why that distinction is load-bearing).

    quantity                              BEFORE (no flag)      AFTER (--cuda-graph-trace=node)
    kernel rows                                  150,106                 447,667   (+297,561)
    rows with graphNodeId NOT NULL                     0                 297,561
    launchType histogram                     {1: 150106}             {1: 447667}
    kernel channel ends                         34.748 s                81.099 s
    runtime channel ends                        77.562 s                81.462 s
    RESIDUAL GAP                               +42.814 s                +0.363 s
    LAUNCH CALLS AFTER LAST KERNEL RECORD          2,470                       0
    cudaGraphLaunch                            127 calls               127 calls
    total kernel time                  32,686,641,781 ns       78,887,667,406 ns

**The 2,470 is the witness that makes this attributable rather than merely bigger.** PRISMSTALL's
finding was that 2,343 explicit launches plus 127 graph launches were issued into a window carrying no
kernel record. That number is now **zero**: every launch the CPU issued has a kernel record against it.

⚠ **ONE HALF OF MY OWN PREDICTION WAS WRONG IN A WAY A QUERY-WRITER NEEDS: `launchType` DID NOT MOVE.**
All 447,667 rows still read `launchType = 1 (REGULAR)`, in both arms. **The discriminator is
`graphNodeId` alone.** A guard written as `WHERE launchType != 1` finds nothing on this box and reads
as a clean absence. I sealed the prediction as "`graphNodeId` not NULL **or** `launchType` not
REGULAR", so it scores a hit, and the disjunction is doing all the work.

---

## ② THE INSTRUMENT WAS CALIBRATED AGAINST PRISMSTALL'S ARTIFACT BEFORE IT TOUCHED MINE

My analysis script was run on `nsys_batched_B64_20260918T134318Z.sqlite` first. It reproduces
PRISMSTALL's published figures **to the digit**: 150,106 kernel rows, total kernel time
**32,686,641,781 ns**, all eight `<142>` family members at their published counts
(25,600 / 10,240 / 353 / 400 / 48 / 16 / 160 / 208), `graphNodeId` 0 of 150,106, one
`cudaGraphInstantiate`, 127 `cudaGraphLaunch`, one `cudaGraphExecDestroy`.

★ My launch-gap figure prints **2,470** where PRISMSTALL wrote **2,343 explicit plus 127 graph**.
**2,343 + 127 = 2,470.** Same measurement, one number against two, and I say which I counted.

⚠ My timestamps sit about **0.096 s** below PRISMSTALL's throughout, because my `t0` is the earliest
RUNTIME row and theirs was the session start. A constant offset that changes no relation between events.

---

## ③ ARM D - THE CALIBRATION ON MY OWN RUN, WHICH IS WHERE THE FLOOR COMES FROM

The floor is PRISMSTALL's measured prefill figure. ARM A had to reproduce it before any decode number
could be set beside that lane's work:

    <142,128> in the prefill    25,600 instances   400.0000 per ubatch   0.781250 per prompt token

**Every one of those three matches PRISMSTALL exactly.** V1 does not fire and the two artifacts are
comparable. The model is the same object too: sha256 prefix **`3907dc1658db1f78`**, **7,206,168,928 B**,
fork `5d80cff0b8cb9f2bf823cfc4e71e3abb97f290d6`, so V6 does not fire either.

Whole-phase family share: **59.003 %** here against PRISMSTALL's **59.160 %**. ⚠ Those are not the same
denominator - theirs is a share of total kernel time in an artifact that was almost entirely prefill,
mine is a share of the prefill phase proper - so the 0.157-point difference is the denominator, not a
disagreement.

---

## ④ THE ANSWER, WITH ITS DENOMINATORS NAMED

**Denominators, from the invocation's own flags** (`-npp 512 -ntg 128 -npl 64 -ub 512`): 64 sequences x
512 = **32,768 prompt tokens** in **64 prefill ubatches**; **128 decode steps** emitting 64 tokens each
= **8,192 generated tokens**.

    kernel                                 PREFILL    DECODE   per decode step   per generated token
    mul_mat_q<142,128>                      25,600         0           0.0000            0.000000
    mul_mat_q<142,64>                            0    45,184         353.0000            5.515625
    mul_mat_vec_q<142,1>                         0     6,144          48.0000            0.750000
    mul_mat_q_stream_k_fixup<142,64>             0    20,480         160.0000            2.500000
    mul_mat_q_stream_k_fixup<142,128>       10,240         0                -                   -
    mul_mat_q<142,16>                          400         0                -                   -
    mul_mat_q_stream_k_fixup<142,16>           208         0                -                   -
    mul_mat_vec_q<142,4>                        16         0                -                   -
    ------------------------------------------------------------------------------------------------
    THE <142> FAMILY                        36,464    71,808         561.0000            8.765625

★ **Every per-step figure is an exact integer over 128 steps**, which is how a forward pass should
behave and is itself a check that the split is clean.

⚠ **STATE WHICH QUANTITY YOU MEAN.** *"`mul_mat_q<142>` launches per generated token"* is **5.515625**
if it means the matmul proper, and **8.765625** if it means the whole family including the stream-K
fixup and the vector path. PRISMSTALL's prefill figure of **0.781250** is `<142,128>` **alone**, so the
like-for-like decode partner of that number is **5.515625**.

### ★ The premise behind P3, checked rather than assumed, and it holds to one launch

I sealed the claim that a forward pass costs the same number of matmul launches however many tokens
ride in it, marked its decode half UNCHECKED, and said the difference would be the finding. Measured:

    one PREFILL ubatch (512 tokens)   400 x <142,128>  + 160 x fixup<142,128>          = 560
    one DECODE step    (64 tokens)    353 x <142,64>   + 160 x fixup<142,64>
                                                       +  48 x mul_mat_vec_q<142,1>    = 561

**560 against 561.** The premise holds. ⚠ **The COMPOSITION does not** - the tile width changes and a
48-launch vector path appears - which is exactly what part (3) of the headline is about.

⚠ My stated reasoning that decode would be *"exactly 8x the prompt-token figure"* (512/64) was
**wrong**: the real ratio is **5.515625 / 0.781250 = 7.06**, because the decode pass issues 353 matmul
launches rather than 400. The band held; the arithmetic behind the point estimate did not.

---

## ⑤ ★★★ THE 353 PRISMSTALL COUNTED AS PREFILL WERE ONE DECODE STEP

PRISMSTALL's artifact showed 353 `<142,64>`, 48 `mul_mat_vec_q<142,1>` and 160 `fixup<142,64>` and
placed all of them in the prefill, correctly by the only rule available to it - they precede the first
graph launch. **They are one decode step, executed directly before the graph was captured.**

Timestamps from ARM A, all in one contiguous band between the prefill's end and the graph's first replay:

    last prefill <142,128>     35.739 s
    <142,64>       n=353       35.744 s .. 36.096 s
    fixup<142,64>  n=160       35.745 s .. 36.096 s
    vec_q<142,1>   n=48        35.749 s .. 36.092 s
    cudaGraphInstantiate       36.108 s
    first cudaGraphLaunch      36.117 s

And the arithmetic closes from the other side: **44,831 graph-replayed `<142,64>` launches / 127
replays = 353.000 exactly**, matching the one direct step. Same for 6,096 / 127 = 48 and 20,320 / 127
= 160.

⇒ **PRISMSTALL's artifact did contain exactly one decode step's kernels all along**, and nothing in it
could have said so. ★ **A single instance of a repeating thing, binned by a rule that cannot see the
repetition, is indistinguishable from a stray.** ⚠ The neighbouring `<142,16>` n=400 is **not** a
second decode step - checked by the same timestamps, it sits in the prefill band and stays there.

---

## ⑥ THE DECODE KERNEL MIX, MEASURED TWICE BY TWO INSTRUMENTS THAT SHARE NO MECHANISM

**ARM A** changed the profiler (`--cuda-graph-trace=node`, graph ON) and reads only rows with a
non-NULL `graphNodeId`, which is unambiguously the 127 replayed steps. **ARM B** changed the
application (`GGML_CUDA_DISABLE_GRAPHS=1`, no nsys flag at all) so there is no graph and the ordinary
channel records everything; it splits at the last prefill `<142,128>` and covers 128 steps.

    share   per step   kernel                                   ARM A      ARM B
    27.86 %    97.00   k_get_rows_float_vec<float>              27.86 %    27.75 %
    27.71 %    48.00   gated_delta_net_cuda<128,...>            27.71 %    27.49 %
    18.44 %   353.00   mul_mat_q<(ggml_type)142, (int)64>       18.44 %    18.44 %
    13.90 %    48.00   mul_mat_vec_q<(ggml_type)142, (int)1>    13.90 %    13.92 %
     5.16 %    16.00   flash_attn_ext_f16<256,256,1,8,...>       5.16 %     5.18 %
     1.26 %    48.00   ssm_conv_f32<1,128,4>                     1.26 %     1.27 %
     1.00 %   112.00   cpy_scalar<cpy_1_scalar<float,float>>     1.00 %     1.01 %
     0.89 %    48.00   concat_cont<unsigned int, 0>              0.89 %     0.87 %
     0.77 %   257.00   fwht_cuda_block<1024,256,float,true>      0.77 %     0.91 %
     0.48 %   112.00   unary_gated_op_kernel<op_silu,float>      0.48 %     0.49 %

    THE <142> FAMILY   32.580 % of decode kernel time           32.580 %   32.605 %
    per decode step                                             561.0000   561.0156
    per generated token                                         8.765625   8.765869

★ **The two instruments agree to 0.003 % on the per-step count and 0.08 % relative on the time share**,
and every per-step kernel count above is identical between them. P6 asked for agreement within 10 %.

⚠ **The small ARM B excess is real and is explained rather than smoothed:** ARM B's timestamp split
catches **2 extra family rows** (71,810 against 128 x 561 = 71,808) and **31 distinct kernels against
ARM A's 28**, because a timestamp rule admits work that happens between steps and outside the graph,
which a `graphNodeId` rule excludes by construction. Neither is wrong; they answer slightly different
questions and I say which.

⚠ **THE MIX IS SHARES OF KERNEL TIME, NOT OF WALL TIME**, and it is a share of the DECODE phase's
kernel time only. Nothing here says what fraction of the decode is spent on the GPU at all.

---

## ⑦ WHAT PART (3) DOES TO PRISMSTALL'S ARM B AND ARM C

PRISMSTALL measured, correctly and on a real kernel, that `<142,128>` is pinned to one block per SM by
two limiters at once. **That kernel is not in the decode.** The decode's kernels, read from their own
launch records in this artifact:

    kernel                     block        thr  warps  reg  shmExec    blocks/SM        occupancy
    mul_mat_q<142,128>  PREFILL 32x8x1      256      8  255  102,400    1 (both)            16.67 %
    mul_mat_q<142,64>   DECODE  32x8x1      256      8  254   65,536    1 (both)            16.67 %
    mul_mat_vec_q<142,1> DECODE 32x6x1      192      6   40   32,768    3 (shared-bound)    37.50 %

⇒ **PRISMSTALL's CONCLUSION survives for the decode matmul - one block per SM, 16.67 % theoretical -
but its EVIDENCE does not transfer.** The striking part of that lane's finding was that
`sharedMemoryExecuted = 102,400` is `maxShmemPerSm` **exactly, with zero bytes spare**. The decode
kernel uses **65,536 of 102,400**, leaving **36,864 B unused** - still only one block, because a second
would need another 65,536, but **not at the wall.** Registers are the tighter of the two for `<142,64>`
at 65,024 of 65,536.

★ **And the decode's third kernel breaks the pattern outright:** `mul_mat_vec_q<142,1>` reaches
**37.50 %** theoretical occupancy, 3 blocks per SM, 40 registers. **13.90 % of decode kernel time runs
at more than twice the occupancy the "this kernel is starved" story was built on.**

⛔ **THEORETICAL, never achieved.** The achieved figure needs the counters PRISMSTALL found blocked by
`ERR_NVGPUCTRPERM` with `RmProfilingAdminOnly: 1`, which is an operator act and was not attempted here.

⚠ **ARM C's 4,051 SASS instructions and its twelve-scalar-FP-per-IMMA ratio are `<142,128>`'s**, and
that kernel does not run in the decode. **The decode kernel's instruction mix is NOT measured by this
cell or by that one.** It is a `cuobjdump` away and nobody has taken it.

---

## ⑧ STREAM-K IN THE DECODE, REPORTED AS COUNTS AND NOT AS A MECHANISM

Per decode step, `<142,64>` launches at four grids, identically in all 128 steps:

    gridX    per step   x128
       48         208   26,624      <- one wave on 48 SMs
      136         128   16,384
       96          16    2,048
    1,940           1      128
    fixup<142,64>  160   20,480

**160 fixups against 208 single-wave launches.** ⛔ **I did NOT run the per-launch join PRISMSTALL ran,
so I claim no pairing rate.** That lane's own warning applies and I am obeying it rather than repeating
it: a matching total is not a mechanism, and 160 against 208 does not even match. What is measured is
that the decode's single-wave launch count and its fixup count are **both exactly constant across all
128 steps**, and that the prefill's 100.0 % pairing at `gridX = 48` is a `<142,128>` result which may
not be carried across.

---

## ⑨ THE SEALED PREDICTIONS, SCORED - and why 8 of 8 is not the compliment it looks like

    1  recovery: graphNodeId or launchType moves, channel reaches the end   0.80  HIT
    2  calibration: 25,600 +/- 256 prefill <142,128>                        0.85  HIT (25,600 exact)
    3  decode mul_mat_q<142,*> per step in [300, 500], point 400            0.55  HIT (353.00)
    4  per generated token in [4.7, 7.8], point 6.25                        0.50  HIT (5.515625)
    5  decode's dominant instantiation is NOT <142,128>                     0.65  HIT (it is ZERO)
    6  ARM A and ARM B agree within 10 %                                    0.70  HIT (0.003 %)
    7  kernel rows grow by at least 40,000                                  0.60  HIT (+297,561)
    8  at least three distinct <142> instantiations in decode               0.55  HIT (exactly three)

**Eight of eight, and the honest reading is that the bands were generous.** Both point estimates
missed - P3 said 400 and measured 353, P4 said 6.25 and measured 5.515625 - and **P4's stated
reasoning was wrong even though its interval held** (I argued a factor of 8; the factor is 7.06).
**P1's `launchType` half was flatly wrong** and only the disjunction saved it. ★ A sweep on wide bands
with two bad point estimates and one wrong clause inside a disjunction is **a calibration warning, not
a triumph** - the next cell's intervals should be tighter, or they are not predictions.

★ PRISMSTALL's instructive miss was a prediction whose PREMISE was false. I marked P3's premise
UNCHECKED in the seal and named what would make it wrong; it held to one launch out of 561. **That is
what the premise line is for**, and it is the part of this seal worth copying.

**No VOID condition fired.** V1 no (25,600 exact) · V2 no (both `nsys` rc=0, both exports rc=0) · V3 no
(both instruments recovered) · V5 no (0.003 %) · V6 no (sha16 and byte size exact). **V4 needs a
paragraph of its own.**

---

## ⑩ V4, THE TENANT CENSUS - AND A FALSE PRESENCE I PRODUCED MYSELF

**No tenant was touched. The orphan at pid 1687005, state `T`, was not touched.** But my own opening
check was wrong and the correction belongs here rather than in a footnote.

At 16:56Z I ran `pgrep -f "$n"` over the five tenant names from an interactive `ssh`, and it reported
**postgres RUNNING**. The detached script's own census, one minute later, reported **postgres NOT
SEEN**. I stopped and settled it rather than picking the reading I preferred:

    pgrep -f postgres  ->  pid 3758288  comm=bash  cmd=bash -c echo "=== ... postgres ... "
                           pid 3759256  comm=bash  cmd=bash -c echo "=== ... postgres ... "

**Both hits were my own `bash -c` wrapper, whose command line quotes the word I was searching for.**
A clean census reading `/proc/*/cmdline` directly and excluding self-matches gives, at 17:03Z:

    gitea        2245                            RUNNING
    rotundus     2596 2599 2601 4578 4580        RUNNING (5 python)
    sci_compute  2364 tmux, 2370 sh, 2377 bash   RUNNING
    sunshine     3909                            RUNNING
    postgres     -                               ABSENT, and absent before I took the lock

⇒ **postgres was not running when I arrived and was not running when I left. I did not stop it, and I
must not report that I did.** The script's own census was the correct one throughout, because its
command line is `bash /home/hologram/decodeprofile_run.sh` and contains no tenant name.

★★★ **This is the brief's own warning arriving with the opposite sign.** The recorded hazard is that
`pgrep -f` yields **false presences** through wrappers that quote the pattern. Here it did exactly
that - and the danger was not that I would miss a process, it was that I would later read the script's
correct `NOT SEEN` as **a tenant I had killed**, fire V4, and report a disturbance that never happened.
⇒ **A false presence recorded at the START of a cell becomes a false ALARM at the end of it**, and the
alarm is the expensive direction, because the safe-looking response is to blame your own run.

---

## ⑪ WHAT REMAINS OWED

1. **The decode kernel's instruction mix is NOT measured.** PRISMSTALL's SASS histogram is
   `<142,128>`'s. The decode runs `<142,64>`, whose body is a different template instantiation.
   `cuobjdump --dump-sass --function` on that mangled name, offline, no lock, minutes.
2. **The stall reason proper stays BLOCKED**, unchanged by this cell: admin-only counters,
   `RmProfilingAdminOnly: 1`, an operator act on a box with four live tenants. ⛔ Not a lane's.
3. **The stream-K pairing in decode is uncounted** - 160 fixups against 208 single-wave launches needs
   the per-launch join before anything is said about it.
4. **`gridX = 384` at 41.8 %** in the prefill remains unexplained, untouched by this cell.
5. **The PUSH-branch row still does not fire, and the reason has changed.** Its gate is now open for
   the decode - but it is open on **`mul_mat_q<142,64>` at 18.44 % of decode kernel time**, not on
   `<142,128>` at 58.27 % of prefill. ★ **A kernel change aimed at the characterised kernel would touch
   the decode zero times.** The target has to be re-named before the row is costed, and a cause is
   still not a lever.
6. **No bytes-per-launch derivation** was attempted; the matrix shapes are not in the artifact and
   inventing them would be the defect. The codec is **PQ2_0 at 2.125 bpw**, never 1.58 or 1.75.

---

## ⑫ STAMPS

**Box** DGX Spark GB10, chip GB20B, compute 12.1, driver **580.159.03**, nsys **2025.3.2.474**. Load
**0.31**, GPU **1 %**, MemAvailable **116.5 GB** at lock time. **Model**
`Ternary-Bonsai-2-27B-PQ2_0.gguf`, sha256 prefix **`3907dc1658db1f78`**, **7,206,168,928 B**. **Binary**
`llama-batched-bench` mtime 2026-09-18 19:18:21 +1000. **Fork**
`5d80cff0b8cb9f2bf823cfc4e71e3abb97f290d6`. **Seal** `cb2877b38`.

**Lock** taken 17:01:30Z as DECODEPROFILE, released 17:04:32Z. **Hold 3 m 02 s**, one hold across both
arms, against a 15-minute budget. **Cont net** pid 3757625 keyed to driver pid 3757619, logged
`driver 3757619 gone, net down` at 17:04:34Z, which only a loop that actually evaluated can write.
**Zero `CONT` lines**, so nothing was ever stopped.

**Artifacts, and they are where the seal said they would be:**

    ~/prism/measured/decodeprofile_A_node_20260919T170135Z.{nsys-rep,sqlite,runlog}   19.7 MB / 53.2 MB
    ~/prism/measured/decodeprofile_B_nograph_20260919T170303Z.{nsys-rep,sqlite,runlog}
    ~/decodeprofile_run.log   ~/decodeprofile_cont.log   ~/dp_analyze.py   ~/decodeprofile_run.sh

**Reported throughput, AS STAMPS ONLY.** ARM A `S_PP 974.24, S_TG 180.60`; ARM B `S_PP 966.82,
S_TG 181.23`; PRISMSTALL's profiled run `S_PP 1,013.36, S_TG 191.25`. ⛔ **These three are NOT
comparable and no comparison is drawn.** They are unbracketed, uninterleaved, single runs on different
days under different tracing, one of them with CUDA graphs disabled outright. **This cell measures what
runs, never how fast.**
