# SOURCES - prism

## The announcement (the navigator pasted the post text on 2026-09-18; X refuses fetchers)
- https://x.com/PrismML/status/2100692248480596348 - PrismML, 2026-09-18 07:03: "Today, we're
  announcing Ternary Bonsai 2 27B. Based on Qwen3.8 27B, Bonsai 2 27B is 9x smaller than its
  full-precision counterpart while retaining 98.2% of its aggregate benchmark performance. Two
  months after the first Bonsai 27B release, the biggest change is quality. The footprint remains
  5.9 GB ... particularly strong gains in agentic coding, multimodal reasoning, and long-horizon
  tool use. Ternary Bonsai 2 27B is available today under Apache 2.0." Follow-ups show it running
  an agentic coding workflow with Cline and a computer-use workflow locally on an NVIDIA box.

## To harvest (each gets a file in this folder with the URL, the date read, and what it says)
- https://docs.prismml.com/  (the model card, the serving paths, KV4, the Hermes integration)
- https://github.com/PrismML-Eng/Bonsai-demo  (the official serving scripts; llama.cpp fork, branch prism)
- https://github.com/PrismML-Eng/image-studio  (notes concurrent requests serialize behind a lock)
- https://github.com/herlangga72/ternary-bonsai-inference  (single-stream Rust engine: "one generation at a time behind a mutex")
- https://blog.kubesimplify.com/bonsai-27b-rtx-pro-6000-dgx-spark  (Bonsai 27B on the DGX Spark: 28-43 tok/s single user; DGX OS, driver 580.159.03, CUDA 13.0, sm_121)
- https://alphasignal.ai/news/prismml-squeezes-qwen3-8-27b-into-5-9-gb-with-98-performance-retained
- https://blog.kubesimplify.com/day-2-anatomy-of-an-llm-inference-request-from-prompt-to-answer-step-by-step  (vLLM concurrency sweeps on a Spark, 2026-05-21)
- https://forums.developer.nvidia.com/t/new-inference-server-for-dgx-spark-cluster-running-mid-large-models-with-c4-55-90-tok-s-unquantized-and-no-speculative-decoding/377787
