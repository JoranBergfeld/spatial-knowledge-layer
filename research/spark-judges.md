# What the DGX Spark can host as the bench's judges

Research for #47, under map #45. Facts only, no pick. Gathered 2026-10-31.

Numbers marked (estimate) are my arithmetic, not a measurement. Everything else has a
source at the bottom. Nobody has run a judge model on our Spark yet, so every speed here
is someone else's box, someone else's container and someone else's prompts.

## Short version

- It fits. 128 GB holds a 70B dense judge at NVFP4 (42.7 GB of weights), gpt-oss-120b
  (65 GB), or Nemotron 3 Super NVFP4 (80 GB), next to the reranker's 10 GB cap. Not two
  big ones at once.
- Speed is about bandwidth (273 GB/s), so dense 70B is slow per stream (about 3 to 5
  tok/s) and only gets usable with lots of requests in flight. Sparse MoE models
  (gpt-oss-120b, 5.1B active) are much quicker per stream.
- For one bench run's worth of judging (a few hundred claims, about 1,500 checks), my
  estimate is 10 to 30 minutes with concurrency, and a couple of hours without it.
- The pinned container is the catch. The box runs NGC vLLM 26.02 (vLLM 0.15.1) because
  the driver is R580, and newer NGC tags want 595.58+. gpt-oss is supported there.
  Nemotron 3 Super is not (its card wants vLLM 0.20+, and v0.27.1 on Spark). NVFP4 speed
  on 0.15.1 is unknown.
- The NLI model has no reason to move for throughput: a run scores 15 to 32 passage
  pairs. vLLM can't serve DeBERTa anyway.
- The laptop can reach the Spark the way the reranker eval already does, over an SSH
  tunnel. UFW lets nothing but LXC 211 reach a model port today.

## What a judge would actually be asked to do

From the four runs in `runs/` (2026-10-17 x3, 2026-10-24):

|  | per run |
| -- | -- |
| turns asked | 230 to 248 |
| answers produced | 84 to 101 (the rest abstain, clarify or report a gap) |
| composition calls | 94 to 106 |
| composition output | 9,220 to 10,438 tokens, so about 100 tokens an answer |
| segments the NLI model scored | 12 to 23 |
| segment and passage pairs scored | 15 to 32 |
| passage length | about 840 to 1,020 characters on average, 1,457 max |
| run spend on Azure | $3.17 to $4.99 |

So today's NLI check is tiny. Most answers are tool facts (ids, counts), not passages,
and only the segments that cite a passage get scored.

A RAGChecker-style pass is bigger. The paper's checks are claim level: extract claims
from the response (and from a ground-truth answer), then check each claim against the
ground truth and against each retrieved chunk. Only faithfulness needs no ground-truth
answer. Precision, recall, claim recall, context utilization, noise sensitivity,
hallucination and self-knowledge all do. The bench's gold is ids and counts, not answer
text, so most of those metrics would need gold answers written first.

Rough size of one run (estimate): about 100 responses to extract from, 2 to 4 claims
each, so about 300 claims. Retrieval returns 5 passages (`TOP = 5` in
`packages/service/src/spatial_service/retrieval.py`), so checking each claim against
each passage is about 1,500 checker calls, and roughly double that with ground-truth
claims added. A checker call is a passage (about 250 tokens) plus a claim plus the
prompt, so call it 600 tokens in and a few tokens out. An extractor call is maybe 1,000
in and 150 out. That's about 1M prompt tokens and 30k generated tokens a run, before any
reasoning tokens a reasoning model adds.

## The box

- GB10, 128 GB LPDDR5x unified memory shared by CPU and GPU, 273 GB/s. 20 Arm cores.
- Already on it: the reranker, `nvidia/llama-nemotron-rerank-1b-v2` on
  `nvcr.io/nvidia/vllm:26.02-py3`, capped at `--gpu-memory-utilization 0.08` (about 10
  GB, using about 3.9 GiB). It serves the live RAG demo with a 15 s timeout. Also a
  `w2g-weather` training container that isn't ours. How much memory that one takes, I
  don't know.
- Driver 580.173.02 (CUDA 13.0). DGX OS 7.6.0, the current release, still ships that
  same driver for Spark. NGC vLLM tags from 26.04 on want driver 595.58 or newer. That's
  why the reranker sits on 26.02.
- NGC vLLM 26.02 is vLLM 0.15.1, CUDA 13.1.1 (it runs in forward-compat mode on R580).
  Its release notes list gpt-oss-20b/120b support as a key feature. Known issues, quoted
  more or less: vLLM grabs about all memory by default, which OOMs on unified memory, so
  set `--gpu-memory-utilization`. FP8 models on Spark may fail with CUDA stream capture
  errors in FlashInfer kernels.
- The reranker needed `--enforce-eager` because torch.compile warmup crashed on GB10 in
  0.15.1. A judge might hit the same thing. Eager costs a reranker nothing, but for a
  generating model CUDA graphs do help decode speed. Not measured.
- NVIDIA-first rule (homelab CLAUDE.md, quoted in `spark.md`): models and containers on
  this box come from NVIDIA unless there's a written reason otherwise.
- There's another route off the 26.02 pin: upstream `vllm/vllm-openai` images built for
  CUDA 13.0 (`cu130`). vLLM's docs say CUDA 13 needs R580 or newer, so these should run
  on the current driver without compat mode. NVIDIA's own Nemotron 3 Super card and its
  Spark playbook for Gemma 4 point at those upstream images. It's not NGC though, so the
  rule would need its written reason.

## Candidate judges

Weights are the summed safetensors sizes on Hugging Face. "In 26.02" means the
architecture is in vLLM 0.15.1's model registry and nothing I found says it needs newer.
Speeds are single stream unless they say otherwise.

| Model | Weights | In 26.02 | Decode, measured on a Spark | Notes |
| -- | -- | -- | -- | -- |
| `meta-llama/Meta-Llama-3-70B-Instruct` BF16 (what the paper used) | 141 GB | yes | n/a | Doesn't fit in BF16. |
| `nvidia/Llama-3.3-70B-Instruct-NVFP4` | 42.7 GB | arch yes, NVFP4 speed unknown | about 5 tok/s (TRT-LLM, forum). Bandwidth ceiling about 6.4 (estimate: 273 / 42.7) | In NVIDIA's vLLM-on-Spark matrix. Llama 3.3 licence. |
| `nvidia/Llama-3.3-70B-Instruct-FP8` | 72.7 GB | FP8 known issue on Spark | 2.7 tok/s for 3.1 70B FP8, prefill 803 tok/s (LMSYS, SGLang) |  |
| `openai/gpt-oss-120b` MXFP4 | 65.2 GB | yes, listed key feature | 55 to 61 tok/s (NVIDIA llama.cpp 55.4, ggml-org 58.7, forum vLLM 60.7); 33.5 in one vLLM run (Dendro Logic) | 117B total, 5.1B active. Reasoning model. Apache 2.0. In NVIDIA's matrix. |
| `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` | 80.3 GB | no, card wants vLLM 0.20.0+, Spark recipe uses `vllm/vllm-openai:v0.27.1` | 22.7 to 23.7 tok/s, prefill 1,636 to 1,884 (vLLM blog, used as an eval judge) | 12B active. Reasoning on/off. Nemotron open licence. In NVIDIA's matrix. |
| `nvidia/Qwen3-32B-NVFP4` | 20.7 GB | arch yes | no Spark number found for NVFP4 | In NVIDIA's matrix. Apache 2.0. |
| `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8` | 32.7 GB | arch yes, FP8 known issue | no first-party number found | 3B active. In NVIDIA's matrix (BF16 63.2 GB too). |
| `bespokelabs/Bespoke-MiniCheck-7B` (checker only) | 15.5 GB BF16 | arch yes (InternLM2) | n/a, it outputs one token | Document plus sentence in, supported or not out. Card says SOTA on LLM-AggreFact for its size, and more than 500 docs/min on one A6000 with vLLM. CC BY-NC 4.0. Not NVIDIA. |
| `dongyru/Mistral-7B-Claim-Extractor` (extractor only) | 14.5 GB BF16 | arch yes | n/a | RefChecker's own fine-tuned extractor. Apache 2.0. Not NVIDIA. |

Batched throughput, which is what a judge run would live on:

- gpt-oss-120b, vLLM, short outputs (Dendro Logic): 33.5 tok/s at 1 stream, 123 at 8,
  252 at 32, 565 at 128, 863 at 256 (aggregate generated tokens).
- gpt-oss-120b, llama.cpp batched bench (ggml-org): prompt processing about 2,400 tok/s
  flat; generation 57 at 1, 159 at 8, 353 at 32 (512-token prompts).
- Nemotron Super 49B v1.5 NVFP4, dense, vLLM (Dendro Logic): 5.79 tok/s at 1, 45 at 8,
  162 at 32, 695 at 256. That's the nearest published stand-in for a dense 70B sweep. I
  found none for Llama 3.3 70B itself.
- One forum post saw 116 to 153 aggregate at 8 streams for gpt-oss-120b on vLLM and says
  the concurrent numbers moved up to 30% between identical runs.

## Rough run times (estimate)

Using the workload above (1M prompt tokens, 30k generated, about 1,600 calls) and the
numbers above, with the runner keeping about 32 requests in flight. RAGChecker's own
examples use extractor and checker batch sizes of 32 to 64.

- gpt-oss-120b: prefill 1M / about 2,300 tok/s is about 7 min. Generation 30k / about
  250 tok/s is 2 min, but it's a reasoning model. At, say, 100 reasoning tokens a call
  that's another 160k tokens, about 11 min. So 10 to 20 minutes.
- Llama 3.3 70B NVFP4: prefill 1M / about 800 tok/s is about 21 min (the 803 figure is
  70B FP8 on SGLang, NVFP4 on vLLM may differ). Generation 30k at maybe 110 tok/s (49B
  sweep scaled by size) is about 5 min. So 25 to 30 minutes.
- Same model, one request at a time: about 0.75 s prefill plus 2 s of decode per check,
  and 30 s per extraction, so roughly 2 hours.
- Nemotron 3 Super: prefill alone about 9 min at 1,800 tok/s. No concurrency sweep
  found, and NVIDIA's Spark recipe sets `--max-num-seqs 4`.
- MiniCheck-7B as the checker: no instruction prompt to speak of, about 300 tokens a
  check, 450k tokens at a 7B's roughly 2,250 tok/s prefill (Qwen2 7B Q8 in llama.cpp on
  a Spark) is 3 to 4 minutes.

vLLM's prefix caching would shave the shared instruction part of every checker prompt.
How much, depends on the prompt. Not measured.

The reranker shares the GPU. A judge run at 32 streams would compete with live reranks
on the demo. No measurement, and I found no GPU partitioning on GB10.

## The NLI model

What's pinned: `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`, fp32 ONNX
(1.74 GB), HF commit `b3546ea6`, sha256 `9f304954...`, in
`personal-website/backend/scripts/nli-model-source.env`. Rule: drop a segment under
P(entailment) 0.5. 435M parameters.

Where it runs now:

- In production, the backend on LXC 211 loads it at boot (about 3.7 GB resident, CPU,
  ONNX Runtime) and checks every answer before it goes out.
- In the bench, `bench run` loads the same file locally through `OnnxNli`
  (`packages/bench/src/spatial_bench/nli.py`, CPU ONNX Runtime) and records its sha256
  so the run says it's the same file. The four runs in `runs/` all name
  `...wanli@sha256:9f304954c0c3ed1e`, so it has loaded for real. The README and the
  `nli.py` docstring still say it never has. That's stale.

Could it run on the Spark:

- Not through vLLM. `DebertaV2ForSequenceClassification` is not in vLLM's model
  registry, in 0.15.1 or on main today. There are PRs to add it (#20215, #42094), but
  neither made it into the registry.
- Through ONNX Runtime with CUDA, probably. PyPI `onnxruntime-gpu` 1.31.1 has a
  `manylinux_2_34_aarch64` wheel, and ORT says its GPU packages are built with CUDA 13.0
  from 1.27 on. Whether that wheel carries GB10 (sm_121) kernels, I didn't verify.
  Community builds for sm_121 exist, and one forum thread reports cuDNN crashes under
  batch load with ORT CUDA EP on GB10 that went away with driver 580.173.
- The CUDA execution provider is a different code path from the backend's CPU one.
  Scores near 0.5 could move. The bench's own docstring says a run has to score with the
  same model and the same rule as the backend, or the figure is about some other check.
- Throughput isn't a reason to move it. 15 to 32 pairs a run is nothing on a CPU.
- Moving the backend's copy would put the Spark in the visitor's request path. The Spark
  has no monitoring yet, and UFW opens model ports to 211 only.

RefChecker has a slot for this kind of model: besides LLM checkers it ships an
`NLIChecker` (RoBERTa-large NLI) and an `AlignScoreChecker`.

## Serving and reaching it

How the homelab serves a model today: NGC vLLM in Docker compose under a systemd unit,
host networking so UFW applies, `--api-key` from the vault, HF cache on disk, model
pinned to a commit. vLLM serves the OpenAI-compatible API (`/v1/chat/completions`).

- RefChecker and RAGChecker call models through litellm. For a local vLLM, RefChecker's
  README says to prefix the model with `openai/` and pass `api_base`
  (`http://host:port/v1`). So an OpenAI-compatible endpoint is what they want.
- The bench calls Azure through `agent_framework`'s `OpenAIChatCompletionClient` over an
  `AsyncAzureOpenAI` client (`azure_llm` in
  `packages/service/src/spatial_service/app.py`). A judge would be a new seam in
  `Wiring` either way. Today's `Wiring` has `llm` and `nli` and no judge.
- `bench run` refuses to start when a deployment it'll call has no rate in `rates.json`,
  and gives up on the first unpriced call. A local judge would need a rate (zero) or a
  rule change.
- vLLM's `--api-key` only guards `/v1/*`. The homelab found `/rerank` answering with no
  key. Anything that reaches the port gets the unguarded routes.
- Who can reach it: UFW on the Spark denies inbound except SSH from the LAN
  (192.168.0.0/16) and VPN (10.10.0.0/24), mDNS from the LAN, and port 8100 from LXC 211
  only. `bench run` runs on the laptop (it reads the service's env: the bench database,
  Azure, `SPATIAL_NLI_MODEL_DIR`). The reranker's laptop evals already go through
  `ssh -N -L 8100:127.0.0.1:8100 joran@192.168.5.6`. The same pattern works for any
  port, from the LAN or over WireGuard, with no new firewall rule.

## Open questions

- Whether NVFP4 runs at full speed in vLLM 0.15.1 on GB10. A December 2025 forum PSA
  said NVFP4 wasn't properly used on Spark in vLLM builds of that time. NVIDIA's later
  recipes force Marlin kernels (`VLLM_NVFP4_GEMM_BACKEND=marlin`). Only a test on the
  box answers this.
- Whether a judge needs `--enforce-eager` on 26.02 like the reranker did, and what that
  costs.
- How much memory `w2g-weather` takes, and when it trains.
- Whether a judge run hurts live rerank latency on the demo.
- How well any of these agrees with the paper's Llama3-70B, or with Joran, on this
  bench. The 45 labelled segments in `runs/labels-*.jsonl` are a start for that.
- RefChecker's real prompt lengths, and whether it batches several claims per checker
  call. I assumed one claim per call.
- Gold answer text: most RAGChecker metrics need it and the bench doesn't have it.
- One forum post reports thermal shutdowns on a single Spark at 32k+ context with
  Nemotron 3.5 Super. One report, not checked.

## Sources

This repo and the homelab:

- `runs/run-*.json`, `runs/segments-*.jsonl`, `runs/nli-precision.json` (main checkout,
  not in git)
- README, "Evaluation runs and the eval snapshot"
- `packages/bench/src/spatial_bench/nli.py`, `cli.py` (`wiring_from_env`), `evalrun.py`
  (`Wiring`)
- `packages/service/src/spatial_service/retrieval.py` (`TOP = 5`), `app.py`
  (`azure_llm`)
- `personal-website/backend/scripts/nli-model-source.env`
- homelab `docs/services/spark.md`, `docs/services/personal-website.md`,
  `docs/services/spatial-layer.md`, `roles/spark_reranker/`,
  `inventories/proxmox/host_vars/spark.yml`, `group_vars/all/network.yml` and `main.yml`

NVIDIA:

- DGX Spark hardware: https://docs.nvidia.com/dgx/dgx-spark/hardware.html
- Porting guide, system overview:
  https://docs.nvidia.com/dgx/dgx-spark-porting-guide/latest/overview.html
- DGX OS 7 release notes (Spark driver per release):
  https://docs.nvidia.com/dgx/dgx-os-7-user-guide/release_notes.html
- NGC vLLM 26.02 release notes:
  https://docs.nvidia.com/deeplearning/frameworks/vllm-release-notes/rel-26-02.html
- vLLM on Spark playbook and model matrix (last updated 2026-06-12):
  https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/vllm/README.md
- NVIDIA blog, Spark inference table (2025-10-24, updated 2026-02-11):
  https://developer.nvidia.com/blog/how-nvidia-dgx-sparks-performance-enables-intensive-ai-tasks/
- Nemotron 3 Super NVFP4 card:
  https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4
- Nemotron 3 Super Spark guide:
  https://github.com/NVIDIA-NeMo/Nemotron/blob/main/usage-cookbook/Nemotron-3-Super/SparkDeploymentGuide/README.md

vLLM, ONNX Runtime, model cards:

- vLLM blog, vLLM on the DGX Spark (2026-06-01):
  https://vllm.ai/blog/2026-06-01-vllm-dgx-spark
- vLLM GPU install (CUDA 13 needs R580+):
  https://docs.vllm.ai/en/stable/getting_started/installation/gpu/
- vLLM 0.15.1 model registry:
  https://github.com/vllm-project/vllm/blob/v0.15.1/vllm/model_executor/models/registry.py
- DeBERTa PRs: https://github.com/vllm-project/vllm/issues/20215,
  https://github.com/vllm-project/vllm/issues/42094
- ONNX Runtime CUDA EP:
  https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html, and
  https://pypi.org/project/onnxruntime-gpu/ (1.31.1 wheels)
- Hugging Face API metadata (sizes, licences) for each model in the table, e.g.
  https://huggingface.co/api/models/nvidia/Llama-3.3-70B-Instruct-NVFP4
- MiniCheck card: https://huggingface.co/bespokelabs/Bespoke-MiniCheck-7B

RAGChecker and RefChecker:

- RAGChecker paper (NeurIPS 2024 D&B):
  https://proceedings.neurips.cc/paper_files/paper/2024/file/27245589131d17368cccdfa990cbf16e-Paper-Datasets_and_Benchmarks_Track.pdf
- RAGChecker repo: https://github.com/amazon-science/RAGChecker
- RefChecker repo, "Choose Models for the Extractor and Checker":
  https://github.com/amazon-science/RefChecker

Benchmarks by others (secondary, read with care):

- LMSYS Spark review (2025-10-13):
  https://www.lmsys.org/blog/2025-10-13-nvidia-dgx-spark/
- ggml-org llama.cpp Spark benches (2026-02-05, driver 580.95.05):
  https://github.com/ggml-org/llama.cpp/blob/master/benches/dgx-spark/dgx-spark.md
- Dendro Logic concurrency sweep (2026-04-21):
  https://dendro-logic.com/engineering/nvidia-dgx-spark-concurrency-benchmark/
- NVIDIA forum, Llama 3.3 70B NVFP4 at 5 tok/s on TRT-LLM:
  https://forums.developer.nvidia.com/t/tensorrt-llm-nvidia-llama-3-3-70b-instruct-nvfp4-5-tok-s/357791
- NVIDIA forum, same harness across Ollama, llama.cpp and vLLM:
  https://forums.developer.nvidia.com/t/measured-inference-benchmarks-on-a-single-dgx-spark-same-harness-across-ollama-llama-cpp-and-vllm-notes-data-published/379766
- NVIDIA forum, NVFP4 in vLLM on Spark PSA (2025-12-01):
  https://forums.developer.nvidia.com/t/psa-state-of-fp4-nvfp4-support-for-dgx-spark-in-vllm/353069
- NVIDIA forum, ORT CUDA EP crashes fixed by driver 580.173:
  https://forums.developer.nvidia.com/t/gb10-dgx-spark-cudnn-fe-failure-11-and-cublas-status-internal-error-under-batch-load-fixed-by-driver-580-173/380948
