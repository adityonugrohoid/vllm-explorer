# vLLM HTTP API — Complete Reference

> Compiled from [vLLM official docs](https://docs.vllm.ai/en/stable/serving/openai_compatible_server/), source code, and hands-on probing.
> Last updated: 2026-03-16

This document covers the **full HTTP API surface** exposed by a running vLLM server (`vllm serve`). vLLM implements the OpenAI API protocol plus several custom endpoints. It is a drop-in replacement for any OpenAI-compatible client — including migration from Ollama's OpenAI-compatible mode.

---

## Table of Contents

- [Server Defaults](#server-defaults)
- [Migration Notes: Ollama → vLLM](#migration-notes-ollama--vllm)
- [Endpoint Map (Quick Reference)](#endpoint-map-quick-reference)
- [Admin / Health / Metrics](#admin--health--metrics)
  - [GET /health](#get-health)
  - [GET /ping](#get-ping)
  - [GET /metrics](#get-metrics)
  - [GET /server_info](#get-server_info)
- [OpenAI-Compatible Endpoints](#openai-compatible-endpoints)
  - [GET /v1/models](#get-v1models)
  - [POST /v1/chat/completions](#post-v1chatcompletions)
  - [POST /v1/completions](#post-v1completions)
  - [POST /v1/embeddings](#post-v1embeddings)
  - [POST /v1/responses](#post-v1responses)
  - [POST /v1/audio/transcriptions](#post-v1audiotranscriptions)
  - [POST /v1/audio/translations](#post-v1audiotranslations)
- [Tokenizer Endpoints](#tokenizer-endpoints)
  - [POST /tokenize](#post-tokenize)
  - [POST /detokenize](#post-detokenize)
- [Pooling Endpoints](#pooling-endpoints)
  - [POST /pooling](#post-pooling)
  - [POST /classify](#post-classify)
  - [POST /score](#post-score)
  - [POST /rerank (+ /v1/rerank, /v2/rerank)](#post-rerank)
- [LoRA Adapter Management](#lora-adapter-management)
  - [POST /v1/load_lora_adapter](#post-v1load_lora_adapter)
  - [POST /v1/unload_lora_adapter](#post-v1unload_lora_adapter)
- [WebSocket Endpoints](#websocket-endpoints)
  - [WS /v1/realtime](#ws-v1realtime)
- [vLLM Extra Parameters (Beyond OpenAI Spec)](#vllm-extra-parameters-beyond-openai-spec)
  - [Sampling Parameters](#sampling-parameters)
  - [Output Control](#output-control)
  - [Structured Output / Guided Decoding](#structured-output--guided-decoding)
  - [Streaming Extensions](#streaming-extensions)
  - [Advanced Engine Parameters](#advanced-engine-parameters)
- [Prometheus Metrics Reference](#prometheus-metrics-reference)
- [Error Behavior](#error-behavior)

---

## Server Defaults

| Setting | Default |
|---------|---------|
| Host | `0.0.0.0` |
| Port | `8000` |
| Base URL | `http://localhost:8000` |
| API key | None (optional via `--api-key`) |
| Models loaded | 1 per process (set at `vllm serve` time) |

Start command:
```bash
vllm serve mistralai/Mistral-7B-Instruct-v0.2 \
  --host 0.0.0.0 --port 8000 --dtype auto
```

Or via Docker:
```bash
docker run --gpus all -p 8000:8000 --ipc=host \
  vllm/vllm-openai \
  --model mistralai/Mistral-7B-Instruct-v0.2
```

---

## Migration Notes: Ollama → vLLM

| Concern | Ollama | vLLM |
|---------|--------|------|
| Model loading | Pull + run per model, hot-swap | One model per `vllm serve` process, restart to switch |
| API compatibility | `/api/generate`, `/api/chat` + OpenAI compat | OpenAI-native (`/v1/chat/completions`, etc.) |
| Structured output | `format: "json"` param | `structured_outputs` via `extra_body` (guided_json, guided_regex, guided_choice) |
| Embeddings | `/api/embeddings` or `/v1/embeddings` | `/v1/embeddings` (OpenAI-compatible) |
| Token counting | Not exposed | `/tokenize` and `/detokenize` endpoints |
| Metrics | None built-in | `/metrics` — full Prometheus endpoint |
| Health check | `GET /` or `GET /api/tags` | `GET /health` or `GET /ping` |
| Streaming | `stream: true` in body | `stream: true` in body (SSE, identical to OpenAI) |
| GPU management | Automatic VRAM sharing | Explicit via `--gpu-memory-utilization`, `--tensor-parallel-size` |
| Quantization | Built into Modelfile (GGUF) | `--quantization awq/gptq/squeezellm` flag at serve time |
| Concurrency | Sequential by default | Continuous batching — handles concurrent requests natively |
| LoRA | Not supported | Hot-swap via `/v1/load_lora_adapter` |
| Extra params | `num_ctx`, `num_predict`, `repeat_penalty` | `top_k`, `min_p`, `repetition_penalty`, `min_tokens`, etc. via `extra_body` |

**Key migration action**: replace all `ollama` client calls with `openai` client pointed at `http://localhost:8000/v1`. No `format` param — use `structured_outputs` in `extra_body` instead.

---

## Endpoint Map (Quick Reference)

### OpenAI-Compatible (7 endpoints)

| Method | Path | Purpose | Model Requirement |
|--------|------|---------|-------------------|
| `GET` | `/v1/models` | List loaded models | Any |
| `POST` | `/v1/completions` | Text completion (legacy) | Text generation |
| `POST` | `/v1/chat/completions` | Chat completion | Chat/instruct |
| `POST` | `/v1/embeddings` | Generate embeddings | Embedding model |
| `POST` | `/v1/responses` | Responses API (streaming) | Chat/instruct |
| `POST` | `/v1/audio/transcriptions` | Speech-to-text | Whisper |
| `POST` | `/v1/audio/translations` | Audio translation | Whisper |

### vLLM Custom (8 endpoints)

| Method | Path | Purpose | Model Requirement |
|--------|------|---------|-------------------|
| `POST` | `/tokenize` | Text → token IDs | Any with tokenizer |
| `POST` | `/detokenize` | Token IDs → text | Any with tokenizer |
| `POST` | `/pooling` | Generic pooling | Pooling model |
| `POST` | `/classify` | Classification | Classification model |
| `POST` | `/score` | Similarity scoring | Embedding / cross-encoder |
| `POST` | `/rerank` | Document reranking | Cross-encoder |
| `POST` | `/v1/rerank` | Jina AI-compatible rerank | Cross-encoder |
| `POST` | `/v2/rerank` | Cohere v2-compatible rerank | Cross-encoder |

### LoRA Management (2 endpoints)

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/v1/load_lora_adapter` | Load adapter at runtime |
| `POST` | `/v1/unload_lora_adapter` | Unload adapter |

### Admin (4 endpoints)

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/health` | Readiness probe |
| `GET` | `/ping` | Health alias |
| `GET` | `/metrics` | Prometheus metrics |
| `GET` | `/server_info` | Server config + version |

### WebSocket (1 endpoint)

| Protocol | Path | Purpose |
|----------|------|---------|
| `WS` | `/v1/realtime` | Realtime audio streaming |

**Total: 22 endpoints**

---

## Admin / Health / Metrics

### GET /health

Readiness probe. Returns empty 200 when the server is ready to accept requests.

```bash
curl http://localhost:8000/health
# 200 OK (empty body when healthy)
```

Use in Kubernetes:
```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 60
  periodSeconds: 10
```

---

### GET /ping

Alias for `/health`. Same behavior — returns 200 when ready.

```bash
curl http://localhost:8000/ping
```

---

### GET /metrics

Prometheus-format metrics. This is the most critical endpoint for the observability stack.

```bash
curl http://localhost:8000/metrics
```

Returns plain text in Prometheus exposition format. See [Prometheus Metrics Reference](#prometheus-metrics-reference) for the full list of exposed metrics.

---

### GET /server_info

Returns server configuration, version, and loaded model information as JSON.

```bash
curl http://localhost:8000/server_info
```

**Expected response shape:**
```json
{
  "version": "0.x.x",
  "model": "mistralai/Mistral-7B-Instruct-v0.2",
  "max_model_len": 32768,
  "...": "..."
}
```

---

## OpenAI-Compatible Endpoints

### GET /v1/models

List all models currently loaded by the server.

```bash
curl http://localhost:8000/v1/models
```

**Response:**
```json
{
  "object": "list",
  "data": [
    {
      "id": "mistralai/Mistral-7B-Instruct-v0.2",
      "object": "model",
      "created": 1700000000,
      "owned_by": "vllm",
      "root": "mistralai/Mistral-7B-Instruct-v0.2",
      "parent": null,
      "permission": [...]
    }
  ]
}
```

Note: vLLM loads one model at a time. With LoRA enabled, adapters appear as additional model entries.

---

### POST /v1/chat/completions

Primary inference endpoint. OpenAI Chat Completions API compatible.

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistralai/Mistral-7B-Instruct-v0.2",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is vLLM?"}
    ]
  }'
```

**Python (openai client):**
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")

response = client.chat.completions.create(
    model="mistralai/Mistral-7B-Instruct-v0.2",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is vLLM?"},
    ],
)
print(response.choices[0].message.content)
```

**Standard parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | string | required | Model ID |
| `messages` | array | required | Chat messages |
| `temperature` | float | 0.0 | Sampling temperature |
| `top_p` | float | 1.0 | Nucleus sampling |
| `n` | int | 1 | Number of completions |
| `stream` | bool | false | Enable SSE streaming |
| `stop` | string/array | null | Stop sequences |
| `frequency_penalty` | float | 0.0 | Frequency penalty |
| `presence_penalty` | float | 0.0 | Presence penalty |
| `logprobs` | bool | false | Return log probabilities |
| `top_logprobs` | int | null | Number of top logprobs |
| `seed` | int | null | Reproducibility seed |

**vLLM extra parameters** (pass via `extra_body` in openai client):
```python
response = client.chat.completions.create(
    model="mistralai/Mistral-7B-Instruct-v0.2",
    messages=[...],
    extra_body={
        "top_k": 50,
        "min_p": 0.1,
        "repetition_penalty": 1.2,
        "min_tokens": 10,
        "skip_special_tokens": True,
        "structured_outputs": {"choice": ["positive", "negative"]},
    },
)
```

**Response shape:**
```json
{
  "id": "chatcmpl-xxx",
  "object": "chat.completion",
  "created": 1700000000,
  "model": "mistralai/Mistral-7B-Instruct-v0.2",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "vLLM is a high-throughput inference engine..."
      },
      "logprobs": null,
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 25,
    "completion_tokens": 50,
    "total_tokens": 75
  }
}
```

---

### POST /v1/completions

Legacy text completion. Feeds the prompt directly to the model without chat template.

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistralai/Mistral-7B-Instruct-v0.2",
    "prompt": "A robot may not injure a human being"
  }'
```

**Parameters:** Same as chat completions, but uses `prompt` (string) instead of `messages` (array). Also supports `suffix`, `echo`, and `best_of`.

**Response shape:**
```json
{
  "id": "cmpl-xxx",
  "object": "text_completion",
  "created": 1700000000,
  "model": "mistralai/Mistral-7B-Instruct-v0.2",
  "choices": [
    {
      "text": " or, through inaction, allow a human being to come to harm.",
      "index": 0,
      "logprobs": null,
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 15,
    "total_tokens": 25
  }
}
```

---

### POST /v1/embeddings

Generate vector embeddings. Requires an embedding model to be loaded.

```bash
curl http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "BAAI/bge-base-en-v1.5",
    "input": "What is vLLM?"
  }'
```

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | string | Model ID |
| `input` | string/array | Text(s) to embed |
| `encoding_format` | string | `float` or `base64` |

**Response shape:**
```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "index": 0,
      "embedding": [0.0023, -0.0091, ...]
    }
  ],
  "model": "BAAI/bge-base-en-v1.5",
  "usage": {
    "prompt_tokens": 5,
    "total_tokens": 5
  }
}
```

Note: Only works when vLLM is serving an embedding model. Will return an error if a generative model is loaded.

---

### POST /v1/responses

Newer Responses API — a streaming-first interface for chat inference. Mirrors OpenAI's Responses API.

```python
response = client.responses.create(
    model="mistralai/Mistral-7B-Instruct-v0.2",
    input="Explain PagedAttention in one sentence.",
)
```

Supports the same sampling parameters as chat completions. Uses `input` instead of `messages`.

---

### POST /v1/audio/transcriptions

Speech-to-text. Requires a Whisper model to be loaded.

```bash
curl http://localhost:8000/v1/audio/transcriptions \
  -F file=@audio.wav \
  -F model="openai/whisper-large-v3"
```

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `file` | file | Audio file (wav, mp3, etc.) |
| `model` | string | Whisper model ID |
| `language` | string | ISO language code (optional) |
| `prompt` | string | Prior context hint (optional) |
| `response_format` | string | `json`, `text`, `srt`, `verbose_json`, `vtt` |
| `temperature` | float | Sampling temperature |

Also supports vLLM extra sampling params: `use_beam_search`, `n`, `length_penalty`, `top_p`, `top_k`, `min_p`, `repetition_penalty`, `seed`, `frequency_penalty`, `presence_penalty`, `max_completion_tokens`.

---

### POST /v1/audio/translations

Audio translation (to English). Same interface as transcriptions.

```bash
curl http://localhost:8000/v1/audio/translations \
  -F file=@audio_french.wav \
  -F model="openai/whisper-large-v3"
```

---

## Tokenizer Endpoints

These wrap the HuggingFace tokenizer for the loaded model. Available for any model with a tokenizer.

### POST /tokenize

Encode text to token IDs (`tokenizer.encode()`).

```bash
curl http://localhost:8000/tokenize \
  -H "Content-Type: application/json" \
  -d '{"model": "mistralai/Mistral-7B-Instruct-v0.2", "prompt": "Hello, world!"}'
```

**Request:**
```json
{
  "model": "mistralai/Mistral-7B-Instruct-v0.2",
  "prompt": "Hello, world!"
}
```

**Response:**
```json
{
  "tokens": [15339, 1188, 0],
  "count": 3,
  "max_model_len": 32768
}
```

Use case: validate token counts before sending to completions, debug tokenizer behavior across models.

---

### POST /detokenize

Decode token IDs back to text (`tokenizer.decode()`).

```bash
curl http://localhost:8000/detokenize \
  -H "Content-Type: application/json" \
  -d '{"model": "mistralai/Mistral-7B-Instruct-v0.2", "tokens": [15339, 1188, 0]}'
```

**Request:**
```json
{
  "model": "mistralai/Mistral-7B-Instruct-v0.2",
  "tokens": [15339, 1188, 0]
}
```

**Response:**
```json
{
  "prompt": "Hello, world!"
}
```

---

## Pooling Endpoints

These endpoints require specific model types. They will return errors if the loaded model does not support the operation.

### POST /pooling

Generic pooling operation. Applicable to all pooling models.

```bash
curl http://localhost:8000/pooling \
  -H "Content-Type: application/json" \
  -d '{"model": "BAAI/bge-base-en-v1.5", "input": "What is vLLM?"}'
```

---

### POST /classify

Text classification. Only works with classification models.

```bash
curl http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"model": "jason9693/Qwen2.5-1.5B-apeach", "input": "This movie was great!"}'
```

**Response shape:**
```json
{
  "id": "classify-xxx",
  "object": "list",
  "data": [
    {
      "index": 0,
      "label": "positive",
      "score": 0.95
    }
  ]
}
```

---

### POST /score

Score similarity between text pairs. Works with embedding models and cross-encoder models.

```bash
curl http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{
    "model": "BAAI/bge-reranker-v2-m3",
    "text_1": "What is vLLM?",
    "text_2": "vLLM is a fast inference engine."
  }'
```

Score range: 0 to 1 (similarity).

---

### POST /rerank

Rerank a list of documents against a query. Applicable to cross-encoder models only.

Three compatible paths:
- `POST /rerank` — vLLM native
- `POST /v1/rerank` — Jina AI compatible (includes extra info in response)
- `POST /v2/rerank` — Cohere v2 compatible

```bash
curl http://localhost:8000/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "model": "BAAI/bge-reranker-v2-m3",
    "query": "What is vLLM?",
    "documents": [
      "vLLM is a fast inference engine.",
      "Python is a programming language.",
      "PagedAttention enables efficient memory management."
    ]
  }'
```

**Response shape:**
```json
{
  "results": [
    {"index": 0, "relevance_score": 0.95},
    {"index": 2, "relevance_score": 0.72},
    {"index": 1, "relevance_score": 0.05}
  ]
}
```

---

## LoRA Adapter Management

Requires server started with `--enable-lora`. Allows hot-swapping adapters without restarting.

### POST /v1/load_lora_adapter

```bash
curl http://localhost:8000/v1/load_lora_adapter \
  -H "Content-Type: application/json" \
  -d '{
    "lora_name": "sql_adapter",
    "lora_path": "/path/to/sql-lora-adapter"
  }'
```

**Response:**
```json
{
  "message": "Success: LoRA adapter 'sql_adapter' added successfully."
}
```

After loading, the adapter appears as a separate model in `/v1/models` and can be targeted by name in completion requests.

---

### POST /v1/unload_lora_adapter

```bash
curl http://localhost:8000/v1/unload_lora_adapter \
  -H "Content-Type: application/json" \
  -d '{"lora_name": "sql_adapter"}'
```

---

## WebSocket Endpoints

### WS /v1/realtime

Realtime streaming API for audio. Provides WebSocket-based streaming audio transcription with real-time speech-to-text.

Audio must be sent as base64-encoded PCM16 audio at 16kHz sample rate, mono channel.

Requires a Whisper model to be loaded.

---

## vLLM Extra Parameters (Beyond OpenAI Spec)

These parameters are not part of the OpenAI API but are supported by vLLM. Pass them via `extra_body` when using the `openai` Python client, or merge directly into the JSON payload for raw HTTP calls.

### Sampling Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `top_k` | int | -1 (disabled) | Limit sampling to top-k tokens |
| `min_p` | float | 0.0 | Minimum probability threshold |
| `repetition_penalty` | float | 1.0 | Penalize repeated tokens (>1.0 = penalize) |
| `min_tokens` | int | 0 | Minimum tokens before stop is allowed |
| `seed` | int | null | Random seed for reproducibility |

### Beam Search

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_beam_search` | bool | false | Enable beam search |
| `best_of` | int | 1 | Generate N, return best |
| `length_penalty` | float | 1.0 | Beam search length penalty |

### Output Control

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `skip_special_tokens` | bool | true | Strip special tokens from output |
| `include_stop_str_in_output` | bool | false | Include stop string in response |
| `ignore_eos` | bool | false | Continue past EOS token |

### Structured Output / Guided Decoding

| Parameter | Type | Description |
|-----------|------|-------------|
| `structured_outputs` | object | Container for guided decoding |
| `structured_outputs.choice` | array | Constrain to specific choices |
| `structured_outputs.json` | object | JSON schema to enforce |
| `structured_outputs.regex` | string | Regex pattern to enforce |

**Example — constrained choice:**
```python
response = client.chat.completions.create(
    model="mistralai/Mistral-7B-Instruct-v0.2",
    messages=[{"role": "user", "content": "Classify: vLLM is fast!"}],
    extra_body={
        "structured_outputs": {"choice": ["positive", "negative", "neutral"]},
    },
)
```

**Example — JSON schema:**
```python
response = client.chat.completions.create(
    model="mistralai/Mistral-7B-Instruct-v0.2",
    messages=[{"role": "user", "content": "Extract the name and age."}],
    extra_body={
        "structured_outputs": {
            "json": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"}
                },
                "required": ["name", "age"]
            }
        },
    },
)
```

This replaces Ollama's `format: "json"` parameter with far more precise control.

### Streaming Extensions

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `stream_include_usage` | bool | false | Include usage stats in final stream chunk |
| `stream_continuous_usage_stats` | bool | false | Include usage in every stream chunk |

### Advanced Engine Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `vllm_xargs` | object | Pass arbitrary key-value pairs to the vLLM engine |
| `logit_bias` | object | Token ID → bias value mapping |
| `prompt_logprobs` | int | Return logprobs for prompt tokens |

---

## Prometheus Metrics Reference

The `/metrics` endpoint exposes metrics in Prometheus exposition format. Key metrics for production monitoring:

### Request Metrics
| Metric | Type | Description |
|--------|------|-------------|
| `vllm:num_requests_running` | Gauge | Currently processing requests |
| `vllm:num_requests_waiting` | Gauge | Requests queued |
| `vllm:num_requests_swapped` | Gauge | Requests swapped to CPU |
| `vllm:request_success_total` | Counter | Successful completions |
| `vllm:request_failure_total` | Counter | Failed requests |

### Latency Metrics
| Metric | Type | Description |
|--------|------|-------------|
| `vllm:time_to_first_token_seconds` | Histogram | TTFT distribution |
| `vllm:time_per_output_token_seconds` | Histogram | Inter-token latency |
| `vllm:e2e_request_latency_seconds` | Histogram | End-to-end request time |

### Throughput Metrics
| Metric | Type | Description |
|--------|------|-------------|
| `vllm:prompt_tokens_total` | Counter | Total prompt tokens processed |
| `vllm:generation_tokens_total` | Counter | Total tokens generated |
| `vllm:avg_generation_throughput_toks_per_s` | Gauge | Average tok/s |
| `vllm:avg_prompt_throughput_toks_per_s` | Gauge | Average prompt tok/s |

### GPU / Cache Metrics
| Metric | Type | Description |
|--------|------|-------------|
| `vllm:gpu_cache_usage_perc` | Gauge | KV cache GPU utilization (0–1) |
| `vllm:cpu_cache_usage_perc` | Gauge | KV cache CPU utilization (0–1) |
| `vllm:num_preemptions_total` | Counter | Preemption events |

### Model Metrics
| Metric | Type | Description |
|--------|------|-------------|
| `vllm:num_running_requests_per_model` | Gauge | Per-model running count |
| `vllm:num_waiting_requests_per_model` | Gauge | Per-model queue depth |

These metrics are what feeds Grafana dashboards in the gpu-autoscale-inference project. The most critical for autoscaling decisions:
- `gpu_cache_usage_perc` — triggers scale-up when approaching 1.0
- `num_requests_waiting` — queue depth signals demand
- `time_to_first_token_seconds` — user-facing latency SLO

---

## Error Behavior

| Scenario | HTTP Status | Response |
|----------|-------------|----------|
| Server not ready | Connection refused | N/A |
| Model not loaded for endpoint | 400 | `{"error": {"message": "...", "type": "BadRequestError"}}` |
| Invalid parameters | 400 | Error details in `message` |
| Unsupported endpoint for model type | 400 | e.g., embeddings on a generative model |
| Server overloaded | 503 | Queue full |
| Auth failure (if API key set) | 401 | Unauthorized |
| Model not found | 404 | Model ID mismatch |

All errors follow OpenAI's error response format:
```json
{
  "error": {
    "message": "Description of what went wrong",
    "type": "ErrorType",
    "param": null,
    "code": null
  }
}
```
