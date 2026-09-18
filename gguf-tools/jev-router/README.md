# jev-router - the scout's guess from the model's own routing, trained offline

Branch `triple-jev-router`. Three parts, all reads-only, greedy output byte-identical by
construction (the hook moves reads, never math):

1. **The route trace** (engine, `DS4_CUDA_ROUTE_TRACE=<file>`, in `ds4_cuda.cu` on the scout
   path). Format, one record per single-token streaming call, appended:
   ```
   # ds4 route trace 1
   t <token> <token_id>                 the TOKEN ID this decode step is embedding, once per token,
                                        immediately before that token's first layer line. Written
                                        only when a single-token embed has run, so a token with no
                                        `t` line is NOT AVAILABLE, never id 0. The absolute KV
                                        position is not reachable at the embed seam, so <token> -
                                        the trace's own counter, the key every other line uses - is
                                        the position within the traced decode stream, per process.
                                        Lane FORECASTLEARN's token-id table (shape A) reads this.
   <token> <layer> <id> <id> ...        the router's routed ids at this layer (top-6 of 384 on
                                        V4.1 Flash, measured: n_routed_experts 384,
                                        num_experts_per_tok 6, 40 layers)
   m <token> <layer> <0|1> <0|1> ...    aligned to the ids: 1 = miss, read from the SSD this call
   w <token> <layer> <wait_us> <n_miss> the hits-first pool wait for this layer's miss batch
   f <token> <layer> <source> <reads> <id>:<conf> ...
                                        the FORECAST for this layer, made at the layer before it:
                                        source = layer (the jev table, L to L+1 plus the sticky
                                        bonus) | sticky (the last token's ids, conf 1.000, an
                                        uncalibrated placeholder) | ahead (reserved for the
                                        token-ahead head, not yet in the engine); reads = the
                                        scout reads it issued (0 = all resident or no victim)
   o <token> <layer> <id>=HIT ... <id>=WASTE ... <id>=MISS ...
                                        the OUTCOME when the layer routes: HIT forecast and routed,
                                        WASTE forecast and not routed, MISS routed and not forecast
                                        (a wait). An o with no f of the same key is a forecast that
                                        found no miss batch to ride, so it spent nothing.
   h <token> <v> <v> ...                OPTIONAL, not yet logged: the token's final hidden state
   ```
   Field names are the record shape lane FORECASTLEARN's shared module and `--inspect` verb
   read; `replay_three_tier.py` prints the same per-source HIT / MISS / WASTE table and the
   worst windows from it.
   `<token>` is a counter that advances when a layer at or below the last seen layer arrives, so
   it is per process, not the sampler's position. Prefill batches (more than 8 ids) are not
   logged. About 1.3 KB per token as text on Flash (V4.1: 40 layers, top-6 of 384; one expert is 9,953,280 B = 9.49 MiB on disk, gate 3,041,280 + up 3,041,280 IQ2_XXS + down 3,870,720 Q2_K). Lane FORECASTLEARN's guesser
   reads this same format.
2. **`train_next_layer.py`** (numpy): one held-out split (the last 20 % of tokens, in order),
   one table with the hit rate per layer and averaged for three guess sources against the sticky
   baseline: (a) sticky, (b) the per-layer successor table L to L+1 from this token's ids, (c) the
   token-ahead head from token t's routing signature (or its hidden state when `h` lines exist)
   to every layer of token t+1. Exports (b) as the `ds4-jev-router 1` text table the engine loads.
   `--selftest` plants a permutation and a stickiness and asserts the instrument reads both.
3. **`replay_three_tier.py`** (numpy): replays the trace through a three-tier cache ranked by
   EXPECTED EXPERT REUSE (PINNED, STAGED, PASS-THROUGH) against today's LRU, sweeping the pinned
   fraction, the lookahead k and the admission confidence; reports hit rate today, the
   popularity curve (hot-set size covering 50/80/90 % of demand), stickiness, and whether the
   wait is latency- or bandwidth-bound from the `w` lines. The lookahead is an oracle bounded by
   the trace, an upper bound on any guesser at that tier split.

The engine hook: `DS4_CUDA_JEV_ROUTER=<table>` (default OFF). With a table loaded the scout's
guess for L+1 is `ds4_jev_router_predict` over this token's ids at L with the last token's L+1
ids as the sticky bonus; a layer with no row falls back to stickiness. The stats line
(`DS4_CUDA_EXPERT_CACHE_STATS=1`) prints `jev_hits=<hits>/<guessed>` beside the scout's own.

The spark run that produces the real trace is OWED (the box is busy; the model lock is not this
lane's). Exact command, from the branch's worktree on the spark, with the model already served
by nothing else:

```
DS4_CUDA_ROUTE_TRACE=$HOME/jev-route-flash.txt DS4_CUDA_EXPERT_CACHE_STATS=1 \
  ./ds4-bench -m ./ds4flash.gguf --ctx 4096 --gen 512      # the round's own bench argv, plus the two vars
python3 gguf-tools/jev-router/train_next_layer.py $HOME/jev-route-flash.txt --out jev-flash.table
python3 gguf-tools/jev-router/replay_three_tier.py $HOME/jev-route-flash.txt --slots 4096
```

No number from a synthetic trace is a routing fact. If the trace shows stickiness at or above
90 %, the finding is that the guess was never the problem and the table is not loaded.
