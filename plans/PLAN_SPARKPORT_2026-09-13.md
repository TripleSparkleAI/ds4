# SPARKPORT log — 2026-09-13

## ★ IT RUNS: DeepSeek V4.1 Flash generating on the DGX Spark (CUDA), coherent output
`./ds4 -m gguf/DeepSeek-V4.1-Flash-Q2.gguf --cuda --ssd-streaming --ctx 32768 -n 40 -p "The capital of France is"`
-> "The user asks a simple question ... likely wants the answer "Paris". Need to respond concisely. Paris."
prefill 1.38 t/s, generation 1.50-1.90 t/s (untuned; DS4_CUDA_DIRECT_MODEL=1 still required).

## What the port needed, in the order the failures appeared
1. ds4_cuda_dsv41.cuh - 10 kernels 1:1 from metal/dsv41.metal + wrappers (rope, quantize, engram_add,
   pool2, candidate blocks/filter, carry_copy, gather_kv, bf16_linear); unpacked indexer path.
2. ds4_cuda.cu - attention-output split into an impl with a round_low_bf16 hook.
3. ds4_gpu.h - V4.1 types moved into a DS4_V41_TYPES_DEFINED block (ds4_cuda.cu does not include it).
4. ds4.c - 33 guards flipped from __APPLE__ to !DS4_NO_GPU (3 in the graph, 30 dispatch sites);
   without the dispatch flips a V4.1 model silently ran the V4 graph and died in its FFN.
5. Makefile - ds4_engram.o added to the CUDA CORE_OBJS.
6. glm_graph_host_memory_bytes - Linux branch (sysconf), else the V4.1 memory admission sees host=0.
7. General CUDA router - the V4 router is hard-locked to 256 experts/top-6/scale 1.5; V4.1 routes 6 of
   384. Tuned path kept for the V4 shape, everything else takes dsv41_router_select_general.
8. ds41_cuda_stream_selected_load - CUDA's routed MoE requires the per-layer selected-expert streaming
   cache primed with this token's experts (the V4 decode path does this via
   metal_graph_decode_cuda_selected_load). Without it: "CUDA streaming selected experts are unavailable".

## Next
- Remove the DS4_CUDA_DIRECT_MODEL=1 requirement: accelerator_{prepare_model_tensor_spans,cache_q8_tensors}
  return false on tensors outside the map; V4.1's Engram tables are deliberately outside it -> skip, don't fail.
- Speed: ds41_cuda_stream_selected_load ends/begins the command buffer every layer (40 syncs/token).
  Batch the seeding or prime from the previous layer's selection.
- Gate: quality batch (gguf-tools/quality-testing/deepseek-v4.1-flash-20260910-batch) + M5 Metal golden
  (rsync at 41/366 GB) for greedy top-1 agreement.
