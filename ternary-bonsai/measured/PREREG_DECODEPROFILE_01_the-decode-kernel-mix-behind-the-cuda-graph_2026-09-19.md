# PREREG · DECODEPROFILE 01 - the decode kernel mix behind the CUDA graph

**Lane DECODEPROFILE, branch `autonomous/dspark-concurrency-and-spark-queue`. Sealed BEFORE anything
ran. 2026-09-19 (UTC); the spark is UTC+10, so its `ls` prints 2026-09-20 AEST for the same instant.**

Row taken: lane PRISMSTALL's owed item 1, in its own words - *"re-run `~/qr_prism_nsys.sh` with
`--cuda-graph-trace=node`, ~15 min with the lock. Until then no claim about decode's kernel mix is
supported by anything, mine included."* Seal `270488ba5`, result `8e1a51616`, both on `lane/prismjev`.

**This cell profiles. It runs no A/B, brackets nothing, and interleaves nothing, so it makes NO SPEED
CLAIM about anything. Nothing here is called lossless. No code is changed and no identity gate is run.**

---

## ⓪ THE QUESTION, and why it is currently unanswerable

PRISMSTALL measured that all 25,600 launches of `mul_mat_q<(ggml_type)142, (int)128, (bool)0>` fall in
the PREFILL, between 1.706 s and 34.651 s of a 78.341 s session, and that the kernel channel alone
stops there while RUNTIME, MEMCPY, SYNCHRONIZATION and OSRT all run to about 77.5 s. 2,343 explicit
launch calls and 127 `cudaGraphLaunch` calls were issued in the window carrying no kernel record. The
profile was taken with `--trace=cuda,nvtx,osrt` and no `--cuda-graph-trace=node`, so the kernels
replayed inside the captured graph emit no per-kernel activity record.

⇒ **launches per GENERATED token is UNMEASURED, and unmeasured is not zero.** The CALL-PATH GATE is
satisfied for the prefill and open for the decode, and the 8.6x headroom motivating the PUSH branch is
a decode statement. This cell exists to close that, or to say plainly that one flag does not close it.

⛔ **THE TYPE, carried forward because a lane sent at the wrong one lands on a stub.** `142` is
`GGML_TYPE_PQ2_0`, 2.125 bpw, *"the same 2-bit codec as Q2_0"* by the fork's own header. The ternary
type is `143`, `PTQ1_0`, 1.75 bpw, a separate gguf on disk whose kernel body is 40 instructions of
relocatable stub in this object. **The checkpoint is ternary; the codec carrying it is not.** Every
bytes-per-weight figure for this kernel is 2.125.

---

## ① THE ARMS

Both GPU arms run the identical `llama-batched-bench` invocation PRISMSTALL profiled, from the same
binary against the same model file, inside ONE lock hold:

    -m <full path>/Ternary-Bonsai-2-27B-PQ2_0.gguf -c 81920 -b 2048 -ub 512
    -npp 512 -ntg 128 -npl 64 -fa 1 -ngl 99

**ARM A - THE SURGICAL REPAIR.** The same `nsys profile` line plus `--cuda-graph-trace=node`. The
CUDA graph stays ON, so this profiles the shipped configuration and the only change is what the
profiler records.

**ARM B - THE INDEPENDENT INSTRUMENT.** No special nsys flag at all; instead `GGML_CUDA_DISABLE_GRAPHS=1`
in the environment, which the fork reads in `ggml_cuda_graph::is_enabled` (`common.cuh`) as
`getenv(...) != nullptr`. With graphs off, every decode kernel is an ordinary launch that the plain
kernel channel records without help.

★ **ARM B SHARES NO MECHANISM WITH ARM A.** A repeats the run and changes the PROFILER; B repeats the
run and changes the APPLICATION. If they agree on decode launches per step, the agreement is
cross-instrument rather than two readings of one instrument - the discipline THE NULL MAY NOT SHARE A
MODEL WITH THE THING IT TESTS asks for, applied to instruments. If ARM A fails, ARM B still answers the
mix question under a stated caveat.

⚠ **ARM B's caveat, stated in advance: turning the graph off changes launch overhead, so ARM B's t/s is
not the shipped configuration's t/s.** ARM B is offered as a measurement of WHICH kernels run and HOW
MANY times, never of how fast. Both arms' reported t/s are recorded as STAMPS and no arm is compared to
any other on time.

**ARM C - THE RECOVERY VERIFICATION.** Analysis of ARM A's artifact, not a separate run. Three
readings, each with its before from PRISMSTALL's artifact: the last kernel record's timestamp (before:
34.651 s of 78.341 s); the count of kernel rows with `launchType != 1` or `graphNodeId IS NOT NULL`
(before: 0 of 150,106); and the launch-call gap (before: 2,343 explicit plus 127 graph launches issued
after the last kernel record).

**ARM D - THE CALIBRATION, AND WHERE THE FLOOR COMES FROM.** The floor is PRISMSTALL's own ARM D1
figure, measured on the half the old artifact COULD see: **25,600 launches of `<142,128>` in the
prefill, 400.0000 per 512-token ubatch exactly.** ARM A must reproduce that before any decode number it
carries may be set beside PRISMSTALL's. A profile that changes the prefill is not the same run.

---

## ② THE MEASURAND

Recorded launches of `mul_mat_q<(ggml_type)142, J>` for every instantiation J, split into the PREFILL
window and the DECODE window, expressed three ways:

    per decode STEP          decode launches / 128
    per GENERATED token      decode launches / 8,192
    per instantiation J      the family table, so a tile-width change is visible

**The denominators, from the invocation's own flags:** 64 sequences x 512 prompt tokens = **32,768
prompt tokens** in **64 prefill ubatches**; **128 decode steps** emitting 64 tokens each = **8,192
generated tokens**. These are the denominators PRISMSTALL used and they are not re-derived here.

**The split rule, stated before the data is seen:** a kernel row belongs to the DECODE if it carries a
non-NULL `graphNodeId` or a `launchType` other than REGULAR, or if it starts after the single
`cudaGraphInstantiate`. ARM B has no graph, so its split is by the timestamp of the last prefill ubatch
alone. **If the two rules disagree within ARM A, the disagreement is reported before any rate.**

---

## ③ PREDICTIONS, with the premise each one rests on named

★ PRISMSTALL's most instructive miss was prediction 6, whose PREMISE was false, so it returned a wrong
question rather than a wrong number. Each prediction below therefore carries its premise and whether
that premise is CHECKED.

**P1 · RECOVERY (0.80).** ARM A's artifact contains at least one kernel row with `graphNodeId` not NULL
or `launchType` not REGULAR, and the last kernel record moves from 34.651 s to within 5 s of session
end. *Premise CHECKED: `nsys profile --help` on this box, version 2025.3.2.474, documents
`--cuda-graph-trace=<granularity>` with `node` as a value and says graph granularity is the mode in
which node activities are NOT collected.*

**P2 · THE CALIBRATION (0.85).** ARM A records 25,600 +/- 256 launches of `<142,128>` in the prefill.
*Premise CHECKED: identical binary, identical model file, identical flags. A miss fires V1.*

**P3 · LAUNCHES PER DECODE STEP, band [300, 500], point 400 (0.55).** *Premise, HALF CHECKED and the
unchecked half is exactly what this cell tests:* one prefill ubatch is one forward pass and costs
400.0000 PQ2_0 matmul launches exactly, which is measured. That a DECODE step is also one forward pass
costing the same number of launches is a structural claim I have NOT checked. If the dispatch takes a
different path at `ne11 = 64` the count differs, and that difference is the finding rather than an
error.

**P4 · LAUNCHES PER GENERATED TOKEN, band [4.7, 7.8], point 6.25 (0.50).** Exactly 8x the prompt-token
figure of 0.781250, because 512/64 = 8 and a forward pass carries 8x fewer tokens in decode. ⚠ **This
is P3 divided by 64 and is NOT independent of it**; it is stated separately only because it is the
number the row was asked for.

**P5 · THE TILE WIDTH CHANGES (0.65).** The decode's dominant instantiation is `<142,64>` or `<142,16>`
rather than `<142,128>`, because `ne11 = 64` is below the 128-wide tile. *Premise PARTLY CHECKED: both
smaller instantiations already exist in PRISMSTALL's artifact, at 353 and 400 launches.* ★ **If P5
holds it has a consequence this cell must carry: PRISMSTALL's ARM B occupancy story (255 registers,
102,400 B shared, 16.67 %) and ARM C instruction mix (4,051 SASS, twelve scalar FP per IMMA) were all
measured on `<142,128>`, and would then describe a kernel the decode does not run.**

**P6 · CROSS-INSTRUMENT AGREEMENT (0.70).** ARM B's decode launches per step falls within 10 % of ARM
A's. *Premise PARTLY CHECKED: I read `ggml_backend_cuda_graph_compute` and `ggml_cuda_graph_set_enabled`
and the enable flag gates capture and replay rather than dispatch, so the same cgraph is executed either
way. I did not read every branch.* A miss fires V5.

**P7 · ROW GROWTH (0.60).** ARM A's total kernel row count exceeds PRISMSTALL's 150,106 by at least
40,000. A weak structural check that node granularity added rows rather than merely relabelling them.

**P8 · THE FAMILY IS WIDER IN DECODE THAN ONE KERNEL (0.55).** The decode window contains at least
three distinct `mul_mat*<(ggml_type)142, ...>` instantiations. *Premise PARTLY CHECKED: the prefill
already carries eight.*

---

## ④ WHAT VOIDS THIS CELL

- **V1** ARM A's prefill `<142,128>` count falls outside 25,600 +/- 1 %. The run is not comparable to
  the calibration and no decode figure may be set beside PRISMSTALL's artifact.
- **V2** `nsys` returns non-zero or produces no `.nsys-rep` for an arm. That arm measured nothing.
- **V3** BOTH arms fail to record decode kernels. Then the cell publishes NO launches-per-token number
  at all, says plainly that a second instrument also failed, and names what would work instead. **A
  second failed instrument is a result.**
- **V4** A tenant process (`sci_compute_jobs`, `rotundus`, `gitea`, `postgres`, `sunshine`) is not
  running at the closing census, or the lock is found held by another owner at any check.
- **V5** ARM A and ARM B disagree on decode launches per step by more than 10 %. The disagreement is
  reported FIRST and neither number is cited alone.
- **V6** The model file's sha256 prefix is not `3907dc1658db1f78` or its size is not 7,206,168,928 B.
  Then it is not the object PRISMSTALL profiled and nothing may be set beside that lane's figures.

---

## ⑤ PROTOCOL

**The lock is taken**, because both GPU arms load the model. One `mkdir ~/spark_model.lock` with an
owner line naming DECODEPROFILE, held across both arms, released by a trap that fires only on a lock
this lane owns. **If `mkdir` fails, nothing runs.**

**A CONT net per arm, keyed to the driving script's PID and killed by that PID.** Liveness is read from
`/proc/<pid>/stat` field 3 on a pid written down, never from `pgrep -x`, whose name match cannot exceed
`TASK_COMM_LEN` = 15 characters and which prints nothing on stdout while warning on stderr. The net's
own `driver <pid> gone, net down` line is the positive witness that it evaluated.

**Tenants are never touched.** The orphan at pid 1687005, state `T`, is left alone.

**No fetch is run from `~/dwarfstar`**; the model is named by full path throughout.

**Outputs land at, and only at:**

    ~/prism/measured/decodeprofile_A_node_<TS>.{nsys-rep,sqlite,runlog}
    ~/prism/measured/decodeprofile_B_nograph_<TS>.{nsys-rep,sqlite,runlog}
    ~/decodeprofile_run.log          the driver log
    ~/decodeprofile_cont.log         the CONT net log

---

**Stamps to be recorded at run time:** load, GPU utilisation, MemAvailable, driver version, nsys
version, model sha256 prefix and byte size, binary mtime, fork commit, lock take and release times in
UTC, and each arm's reported S_PP and S_TG as STAMPS ONLY with no comparison drawn between them.
