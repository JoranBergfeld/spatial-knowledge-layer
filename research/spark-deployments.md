# Could the service's deployments run on the DGX Spark

Research for #48, under map #45. Facts and estimates only. Which deployments the eval
measures is decided in its own ticket. Gathered 2026-10-31. Nothing here was run on the
Spark: every throughput number is an estimate from published benchmarks, and it says so.

## Short version

- Yes, mechanically. Agent Framework's `OpenAIChatCompletionClient` takes a `base_url`
  and vLLM is a listed target. But today's code hardwires `AsyncAzureOpenAI`, so it's a
  small code change, not an env var.
- The service doesn't use native tool calling at all. Both calls are one-shot structured
  output (`response_format` json_schema, strict). So what matters on the Spark is vLLM's
  grammar-constrained decoding, not tool-call parsers. BFCL is only a proxy.
- The big constraint is the Spark's driver. R580 pins NVIDIA's vLLM image to 26.02,
  which is vLLM 0.15.1. That rules out the newer models (Nemotron 3 Super, Qwen3.6)
  without a driver bump. What's left: gpt-oss-20b/120b, Nemotron 3 Nano, the Qwen3 2507
  line.
- gpt-5.4 (planning plus recovery) is 96 to 97.5% of a run's Azure spend, $3.2 to $5.0 a
  pass. Composition is 2.5 to 4%. Embeddings round to zero.
- Estimated model time per pass of 248 turns: about 23 min with a ~3B-active
  non-reasoning MoE on both stages, 40 to 70 min with gpt-oss-120b planning at low
  reasoning. K=5 is roughly 2 h or 3.5 to 6 h. The bench asks turns one at a time.
- The live service is on the same /24 as the Spark (LXC 217, `.47`, and Spark `.6`), so
  it could reach it with one more UFW rule. The bench from the laptop would go through
  an SSH tunnel like the reranker evals do.

## What the service sends today

From `packages/service/src/spatial_service/llm.py`, `app.py`, `turn.py`:

- Two "deployments", planning and composition (recovery uses planning). Each is a
  `Deployment(name, client)` with its own client, so they can point at different places.
- Each call is
  `Agent(client, instructions).run(json_payload, options={"response_format": Model})`.
  No `tools=`. The plan's "tools" (resolve_place, route, ...) are typed steps inside one
  JSON object, checked in code afterwards.
- `azure_llm()` builds
  `OpenAIChatCompletionClient(model=name, async_client=AsyncAzureOpenAI(...))`, timeout
  `SPATIAL_LLM_TIMEOUT_SECONDS` (30 s default), `max_retries=1`. The bench reuses this
  exact function (`spatial_bench/cli.py` imports `azure_llm`), so one change covers both
  the bench and the live service.
- The response format goes through `type_to_response_format_param`, which produces
  `{"type": "json_schema", "json_schema": {..., "strict": true}}`. Installed versions:
  agent-framework-core 1.20.0, agent-framework-openai 1.15.0, openai 3.24.0.

The plan schema (measured from `Plan`, 8.4k chars) uses `$defs`/`$ref` (30 refs),
`anyOf` (10), `pattern` (12, things like `^s[0-9]+$`), one `format: date`, `const`,
`enum`, `maxItems`, `maxLength`, `minimum`. The draft schema is tiny (645 chars).

## Tokens per turn, from `runs/`

Four real runs, all gpt-5.4 planning and gpt-5.4-mini composing (`spend.entries` in each
run file):

| Run | Turns | Planning in/out per call | Recovery calls | Composition calls, in/out per call | Spend |
| -- | -- | -- | -- | -- | -- |
| 2026-10-17-3ddaadac | 230 | 4,031 / 109 | 6 | 94, 1,049 / 98 | $3.17 |
| 2026-10-17-6f60db21 | 248 | 4,018 / 107 | 7 | 104, 1,062 / 100 | $3.41 |
| 2026-10-17-5d907262 | 248 | 6,123 / 106 | 6 | 106, 1,084 / 98 | $4.87 |
| 2026-10-24-b28d61d2 | 248 | 6,123 / 108 | 12 | 101, 1,088 / 92 | $4.99 |

Per pass (b28d61d2): about 1.60M input and 28.7k output tokens for planning plus
recovery, 110k in and 9.3k out for composition, 65 embedding tokens. Only about 40% of
turns compose at all. gpt-5.4's output is around 107 tokens a plan, so it's not
reasoning much (or the reasoning isn't counted; I didn't check which).

Spend split, from each run's `spend.entries`: planning alone is 92.5 to 94.7%, planning
plus recovery (both on gpt-5.4) 96.2 to 97.5%, composition 2.5 to 3.8%, embedding
$0.000001.

Prefix caching would help less than you'd think. The payload is
`json.dumps(..., sort_keys=True)`, so `question` comes before `venue.tree` and the only
shared prefix across turns is the system instructions, about 2k tokens of the 4k to 6k
(chars/4 estimate).

The runs don't record wall time, so there is no measured Azure duration to compare with.

## Could Agent Framework point at the Spark

Yes. Microsoft's docs:
[OpenAI-compatible endpoints](https://learn.microsoft.com/agent-framework/hosting/self-hosting/openai-endpoints#connecting-to-openai-compatible-endpoints-python)
say `OpenAIChatCompletionClient` takes `base_url` for "any OpenAI-compatible endpoint",
with vLLM at `http://localhost:8000/v1/` in their table, or `OPENAI_BASE_URL` from the
env. The installed source agrees: `__init__(..., api_key, base_url, async_client)`.

What it would take, as far as I can tell from the code:

- `azure_llm()` would need a sibling that builds
  `OpenAIChatCompletionClient(model=<served name>, base_url="http://<spark>:<port>/v1", api_key=<vLLM key>)`,
  or passes an `AsyncOpenAI(base_url=..., timeout=..., max_retries=1)` as
  `async_client`. `AsyncAzureOpenAI` builds
  `/openai/deployments/{name}/chat/completions?api-version=...` URLs, which vLLM doesn't
  serve.
- `Settings.from_env` requires `AZURE_OPENAI_ENDPOINT` and `AZURE_OPENAI_API_KEY`
  unconditionally, so it needs a way to name a different endpoint per deployment.
- Planning and composition must have different names (`ConfigError` otherwise). The
  check compares names only, and vLLM's `--served-model-name` can take more than one
  alias, so one process could answer as both. Two processes on two ports also works,
  since each `Deployment` holds its own client.
- Turning reasoning off is a per-request field for the Qwen and Nemotron models
  (`chat_template_kwargs: {"enable_thinking": false}` in the request body). `llm.py`
  only passes `response_format` today. Agent Framework's `_prepare_options` copies
  unknown option keys through to the request, so `extra_body` probably survives, but I
  didn't try it. gpt-oss can't turn reasoning off, only down to low.
- `rates.json` needs a row for every served name, `0.0` is fine. A deployment with no
  rate stops the run (`spend.py`, `SpendError`). Side effect worth knowing: the
  model-pair adoption rule's "this run was cheaper" half is always true at $0.
- Usage: vLLM returns `usage` on chat completions, and the client maps it into
  `usage_details`, so per-call token counts should still land in the trace. Not tried.

Embeddings are a separate story. `AzureEmbedder` (`search.py`) hardcodes
`/openai/deployments/{d}/embeddings` and an `api-key` header, so pointing it at vLLM's
`/v1/embeddings` is another code change. And the snapshot's index is tied to the
deployment that embedded it (manifest's `embedding_deployment`, a fixed `vector(n)`
column), so a different embedder means re-embedding the bench index. At $0.000001 a pass
there's no eval money in it.

## Structured output on vLLM

- vLLM does json_schema via
  [structured outputs](https://docs.vllm.ai/en/v0.15.1/features/structured_outputs/)
  with xgrammar or guidance, `auto` by default.
- In vLLM 0.15.1, xgrammar refuses a schema only for `multipleOf`, `uniqueItems`,
  `contains`, `minContains`, `maxContains`, `patternProperties`, `propertyNames`, or a
  string `format` outside its list, which includes `date`
  ([backend_xgrammar.py at v0.15.1](https://github.com/vllm-project/vllm/blob/v0.15.1/vllm/v1/structured_output/backend_xgrammar.py)).
  The plan schema uses none of those, so it should compile under xgrammar. Not tried.
- Grammar enforcement guarantees the plan parses. It doesn't make the plan right. The
  plan check in code still catches the rest (bad references, step cap) and buys a
  recovery call, same as today.
- Reasoning plus json_schema is where the bugs are. With a reasoning parser on, vLLM
  holds the grammar back until reasoning ends.
  [vLLM #44012](https://github.com/vllm-project/vllm/issues/44012) reports the Responses
  API path escaping the grammar, and the Chat Completions path (what the service uses)
  staying valid with thinking off.
- gpt-oss has had a string of these:
  [#23120](https://github.com/vllm-project/vllm/issues/23120) "Structured output is not
  correctly enforced when using GPT-OSS" was closed as stale in June 2026 with only a
  partial fix credited, and [#60664](https://github.com/vllm-project/vllm/issues/60664)
  reports Harmony json_schema constraints applying from token 0. Whether vLLM 0.15.1
  gets this right for gpt-oss on Chat Completions is unknown until someone tries it.

## What the Spark runs now, and the image it's stuck on

From the homelab repo (`docs/services/spark.md`, `roles/spark_reranker/`,
`inventories/proxmox/host_vars/spark.yml`):

- GB10, arm64, 128 GB unified memory, `192.168.5.6`, DGX OS 7.5, driver R580
  (580.173.02).
- Runs `nvidia/llama-nemotron-rerank-1b-v2` on `nvcr.io/nvidia/vllm:26.02-py3` (vLLM
  0.15.1), port 8100, `--gpu-memory-utilization 0.08` (about 10 GB cap, about 3.9 GiB
  actually used).
- Also on the box, not ours: a `w2g-weather` training container. Its memory use isn't
  written down anywhere I found.
- The pin: "Tags from 26.04 on want driver >= 595.58 and reportedly can't use compat
  mode on R580." So without a driver bump, any new model has to run on vLLM 0.15.1.
- vLLM's default takes the whole unified pool, so every extra process needs its own
  `--gpu-memory-utilization` cap, like the reranker has. The fractions are of the whole
  128 GB, which the OS and CPU processes share.
- Homelab rule: NVIDIA-first for models and containers on this box, community options
  only with a written reason.
- Not monitored. No node_exporter, no alert.

## Candidate stand-ins

Weights are the safetensors in the Hugging Face repo, summed from the HF API.

| Model | Total / active | Weights | Needs | On NGC 26.02 (vLLM 0.15.1)? | Reasoning off? | Tool-use numbers |
| -- | -- | -- | -- | -- | -- | -- |
| gpt-oss-120b (MXFP4) | 117B / 5.1B | 65.2 GB (60.8 GiB) |  | Yes, on NVIDIA's Spark playbook list | No, low/medium/high only | tau-bench retail 49.4 / 62.0 / 67.8 (low/med/high) |
| gpt-oss-20b (MXFP4) | 21B / 3.6B | 13.8 GB |  | Yes, playbook list | No | tau-bench retail 35.0 / 47.3 / 54.8 |
| Nemotron 3 Nano 30B-A3B | 30B / 3.5B | BF16 63.2, FP8 32.7, NVFP4 19.3 GB | vLLM >= 0.12 | Yes (card points Spark users at NGC 25.12.post1) | Yes, `enable_thinking: false` | BFCL v4 53.8 (BF16), 53.2 (FP8); TauBench V2 avg 49.0; IFBench 71.5 |
| Nemotron 3 Super 120B-A12B | 120B / 12B | NVFP4 80.3 GB | vLLM 0.20; card's Spark path is `vllm/vllm-openai:v0.27.1` | No, not without a driver bump (and that image isn't NGC) | Yes | TauBench V2 avg 60.5 (NVFP4); IFBench 73.3 |
| Qwen3-30B-A3B-Instruct-2507 | 30.5B / 3.3B | BF16 61.1, FP8 31.2 GB |  | Yes, Qwen3 MoE is old news for vLLM | It's the non-thinking line | BFCL v4 overall 41.4 (FC); single-turn AST non-live 85.8, live 77.9 |
| Qwen3.6-35B-A3B | 35B / 3B | BF16 71.9, FP8 37.5 GB | vLLM >= 0.19 | No | Yes | TAU3-Bench 67.2 |

Sources: [gpt-oss model card](https://arxiv.org/html/2508.10925),
[Nemotron 3 Nano FP8 card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8),
[Nemotron 3 Nano report](https://arxiv.org/pdf/2512.20848) (Table 3),
[Nemotron 3 Super NVFP4 card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4),
[Qwen3.6-35B-A3B card](https://huggingface.co/Qwen/Qwen3.6-35B-A3B),
[BFCL leaderboard CSV](https://gorilla.cs.berkeley.edu/data_overall.csv),
[NVIDIA's Spark vLLM playbook](https://github.com/NVIDIA/dgx-spark-playbooks/tree/main/nvidia/vllm).

Things to know when reading that table:

- The numbers aren't comparable across rows. Each card picks its own benchmarks. Only
  Nemotron 3 Nano reports BFCL v4 itself; NVIDIA measured Qwen3-30B-A3B-Thinking-2507 at
  46.4 on the same thing, and left gpt-oss-20b blank.
- BFCL (CSV last modified 2026-10-17) has no gpt-5.4, gpt-5.4-mini, gpt-oss or Nemotron
  3 rows. Nearest Azure-side proxies: GPT-5.2 (FC) overall 55.9, single-turn AST
  non-live 81.9 / live 70.4; GPT-5-mini (FC) 55.5, AST 69.9 / 58.6. On the single-turn
  AST columns the small Qwen3 models score higher than the GPT-5 rows (Qwen3-32B FC 88.8
  / 82.0). BFCL's overall is mostly multi-turn, web search and memory, which this
  service doesn't do. Single-turn AST (fill a call's arguments from a schema) is the
  closest thing to writing a plan, and it's still not the same job.
- Dense ~30B models are slow on this box. The Spark has 273 GB/s of memory bandwidth, so
  decode tracks active parameters. Ollama measured qwen3 32B (q4_K_M) at 9.4 tok/s
  decode ([Ollama](https://ollama.com/blog/nvidia-spark-performance)), against about 60
  for the 3B-active MoEs below. That's why the table is all MoE.
- NVFP4 on GB10 under vLLM 0.15.1: NVIDIA's playbook says not all architectures support
  it. I didn't confirm which ones work on 26.02.

### Do they fit next to what's there

Rough budget, weights only, plus the reranker's 10 GB cap. KV cache on top is small: the
bench asks one turn at a time and no call is over about 8k tokens, so a 16k
`--max-model-len` covers it.

| Combination | Weights | With reranker cap |
| -- | -- | -- |
| gpt-oss-120b plans, gpt-oss-20b composes | ~79 GB | ~89 GB |
| gpt-oss-120b for both (two aliases) | ~65 GB | ~75 GB |
| Nemotron 3 Nano FP8 for both | ~33 GB | ~43 GB |
| Qwen3-30B-A3B-2507 FP8 for both | ~31 GB | ~41 GB |
| gpt-oss-120b plans, Nemotron 3 Nano FP8 composes | ~98 GB | ~108 GB |

All of those leave something of 128 GB on paper. What's unknown is the `w2g-weather`
container's footprint and the OS's, so the gpt-oss-120b combinations could be tight
while it trains.

## Throughput: one pass and K passes

Published numbers on the Spark, single stream:

| Model | Prefill tok/s at 4k / 8k context | Decode tok/s at 4k / 8k | Source |
| -- | -- | -- | -- |
| gpt-oss-120b MXFP4 | 2,310 / 2,217 | 55.7 / 52.9 | llama.cpp, build 7941, Feb 2026 |
| gpt-oss-20b MXFP4 | 4,158 / 3,994 | 79.2 / 75.2 | same |
| Qwen3-Coder-30B-A3B Q8_0 | 2,633 / 2,354 | 54.8 / 48.0 | same |
| gpt-oss-120b MXFP4 | 1,725 | 55 | NVIDIA blog, Oct 2025, llama.cpp |

From the
[llama.cpp DGX Spark results](https://github.com/ggml-org/llama.cpp/blob/master/benches/dgx-spark/dgx-spark.md)
and [discussion #16578](https://github.com/ggml-org/llama.cpp/discussions/16578), and
[NVIDIA's blog](https://developer.nvidia.com/blog/?p=107650) for the last row. These are
llama.cpp, not vLLM. I found no clean first-party vLLM prefill figure for the Spark, and
vLLM numbers in write-ups vary a lot with container and kernels. Nemotron 3 Nano isn't
in the table; I'm treating it like the other ~3B-active MoE, which is a guess.

Estimate, model time only, using b28d61d2's tokens (1.60M in / 28.7k out planning +
recovery over 260 calls, 110k in / 9.3k out composition over 101 calls), no prefix
caching:

A. A ~3B-active non-reasoning MoE on both (Qwen3-30B-A3B-2507 or Nemotron 3 Nano with
thinking off), at ~2,500 prefill and ~51 decode:

- planning: 1.60M / 2,500 = 640 s prefill, 28.7k / 51 = 560 s decode
- composition: about 200 s
- about 1,400 s, so roughly 23 min a pass

B. gpt-oss-120b plans at low reasoning, gpt-oss-20b composes. The unknown is how many
reasoning tokens it writes per call. I assumed 150 to 450 (one bug report saw 430 on a
json_schema request; nobody has measured it on these prompts):

- planning: 1.60M / 2,200 = 730 s prefill, (28.7k + 260 x 150..450) / 53 = 1,280 to
  2,750 s decode
- composition: 30 s prefill, (9.3k + 101 x 150..450) / 77 = 320 to 710 s decode
- about 2,350 to 4,200 s, so roughly 40 to 70 min a pass

K repeats, model time:

| K | A (~3B MoE, no reasoning) | B (gpt-oss pair, low reasoning) |
| -- | -- | -- |
| 1 | ~23 min | ~40 to 70 min |
| 3 | ~70 min | ~2 to 3.5 h |
| 5 | ~2 h | ~3.3 to 5.9 h |
| 10 | ~4 h | ~6.5 to 11.7 h |

Plus whatever a turn spends outside the model (store, executor, NLI on the laptop),
which is the same as today and wasn't measured. Azure spend at the same K for
comparison: $3.2 to $5.0 a pass, so about $16 to $25 at K=5.

Notes on the estimate:

- The bench is serial (`Bench.ask_all` is a list comprehension, `evalrun.execute`
  loops), so vLLM's batching buys nothing as the code stands. Batched throughput on the
  Spark is much higher than single-stream, but a parallel bench would be a change to the
  bench.
- Per-call worst case in B is about 8k prefill (~4 s) plus up to ~1k output (~20 s),
  which is under the 30 s timeout but not by a lot. A reasoning run-on would hit it and
  come back `unavailable`.
- Everything above is single-request numbers. With two vLLM processes on the box, only
  one is busy at a time in a serial bench, but I have no measurement of what the idle
  one or the reranker costs the busy one.

## Could the live service reach it

From `docs/services/spatial-layer.md`, `roles/spatial_layer/`, `docs/network.md` in
homelab:

- The live service is LXC 217 at `192.168.5.47`, same `192.168.5.0/24` as the Spark at
  `.6`. Network-wise it's one hop. Its env comes from
  `roles/spatial_layer/templates/env.j2`, today pointing at `spatial-prod-foundry` with
  `gpt-5.4` and `gpt-5.4-mini`.
- The Spark's UFW denies all inbound except SSH (LAN + VPN), mDNS, and 8100 from LXC
  211\. A new port would need a rule from 217. Same pattern as the reranker's 211-only
  rule.
- The bench runs from the laptop. The reranker precedent is an SSH tunnel
  (`ssh -N -L 8100:127.0.0.1:8100 joran@192.168.5.6`) rather than a firewall rule, and
  SSH from the VPN (`10.10.0.0/24`) is already allowed.
- vLLM's `--api-key` only guards `/v1/*`. Chat completions is under `/v1`, so the key
  works for this, but anything that reaches the port gets the unguarded routes too. The
  spark doc already warns about this for the reranker.
- Things that change for the live path specifically: the Spark has no monitoring, and
  the service's `/health/ready` doesn't touch the model, so a down Spark shows up as
  `unavailable` turns (503). Eval runs and live turns would share the box, and so would
  the RAG demo's reranker. Live traffic is the site's demo, so presumably light, but
  nothing measures it on the Spark side.
- If the bench measures Spark models and the live service keeps Azure, `GET /eval`
  publishes figures for models the live service doesn't run. The eval snapshot names its
  deployments, so it would at least say so.

## Open questions nobody can answer without trying

- Does vLLM 0.15.1's xgrammar actually compile the plan schema and keep gpt-oss's or
  Nemotron Nano's output on it, through Chat Completions? Should, not tried.
- How many reasoning tokens gpt-oss-120b writes per plan at low. That one number moves
  pass time by a factor of two.
- Real vLLM prefill on GB10 at 4k to 8k context, for Nemotron 3 Nano in particular.
- How much memory `w2g-weather` holds while it trains.
- Whether `extra_body` / `chat_template_kwargs` pass through Agent Framework's options
  untouched.
- Plan quality of any of these against gpt-5.4 on this schema. No benchmark answers it;
  the bench itself would.
