# vLLM Endpoint Reference — Live Probe Results

> Generated from `probe_endpoints.py` against vLLM v0.17.1
> Model: `Qwen/Qwen2.5-1.5B-Instruct` on RTX 4060 8GB (WSL2)
> Date: 2026-03-16

## Summary

| Category | Available | Not Found | Skipped | Total |
|----------|-----------|-----------|---------|-------|
| Admin | 3 | 1 | 0 | 4 |
| OpenAI | 4 | 1 | 2 | 7 |
| Tokenizer | 2 | 0 | 0 | 2 |
| Pooling | 0 | 6 | 0 | 6 |
| LoRA | 0 | 2 | 0 | 2 |
| **Total** | **9** | **10** | **2** | **21** |

404s are expected — pooling/classify/score/rerank require specialized model types; LoRA requires `--enable-lora`; embeddings requires an embedding model. `/server_info` does not exist in vLLM v0.17.1.

---

## Admin Endpoints

### GET /health — 200 (1ms)

Readiness probe. Returns empty 200 body when server is ready.

```
Status: 200
Body: (empty)
Latency: 1ms
```

Used by Kubernetes `readinessProbe` and worker `wait_for_vllm()` retry loop.

---

### GET /ping — 200 (2ms)

Health alias. Identical behavior to `/health`.

```
Status: 200
Body: (empty)
Latency: 2ms
```

---

### GET /metrics — 200 (27ms)

Prometheus exposition format. Returns ~54KB of metrics text.

```
Status: 200
Content-Type: text/plain
Body: 53,872 chars
Latency: 27ms
```

Key metrics confirmed present:
- `vllm:num_requests_running` — currently processing
- `vllm:num_requests_waiting` — queued
- `vllm:kv_cache_usage_perc` — GPU KV cache utilization
- `vllm:prompt_tokens_total` — total prompt tokens processed
- `vllm:generation_tokens_total` — total tokens generated
- `vllm:request_success_total` — by finish reason (stop/length/abort/error)
- `vllm:num_preemptions_total` — KV cache preemption events
- `vllm:prefix_cache_hits_total` — prefix cache effectiveness

---

### GET /server_info — 404

Not available in vLLM v0.17.1 (V1 engine). May exist in older versions or future releases.

---

## OpenAI-Compatible Endpoints

### GET /v1/models — 200 (2ms)

Lists loaded models.

```json
{
  "object": "list",
  "data": [
    {
      "id": "Qwen/Qwen2.5-1.5B-Instruct",
      "object": "model",
      "created": 1773642306,
      "owned_by": "vllm",
      "root": "Qwen/Qwen2.5-1.5B-Instruct",
      "parent": null,
      "max_model_len": 4096,
      "permission": [...]
    }
  ]
}
```

Response shape: `{object, data[{id, object, created, owned_by, root, parent, max_model_len, permission}]}`

---

### POST /v1/chat/completions — 200 (869ms)

Primary inference endpoint. OpenAI Chat Completions API compatible.

**Request:**
```json
{
  "model": "Qwen/Qwen2.5-1.5B-Instruct",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Say hello in one sentence."}
  ],
  "temperature": 0.0
}
```

**Response shape:**
```
{
  id:                 "chatcmpl-<hex>"
  object:             "chat.completion"
  created:            <unix_timestamp>
  model:              "Qwen/Qwen2.5-1.5B-Instruct"
  choices: [{
    index:            int
    message:          {role, content}
    logprobs:         null
    finish_reason:    "stop" | "length"
    stop_reason:      null
    token_ids:        null
  }]
  service_tier:       null
  system_fingerprint: null
  usage: {
    prompt_tokens:         int
    completion_tokens:     int
    total_tokens:          int
    prompt_tokens_details: null
  }
  prompt_logprobs:    null
  prompt_token_ids:   null
  kv_transfer_params: null
}
```

**vLLM-specific fields** (not in OpenAI spec): `stop_reason`, `token_ids`, `prompt_logprobs`, `prompt_token_ids`, `kv_transfer_params`

---

### POST /v1/completions — 200 (322ms)

Legacy text completion. No chat template applied.

**Request:**
```json
{
  "model": "Qwen/Qwen2.5-1.5B-Instruct",
  "prompt": "A robot may not injure a human being",
  "temperature": 0.0
}
```

**Response shape:**
```
{
  id:                 "cmpl-<hex>"
  object:             "text_completion"
  created:            <unix_timestamp>
  model:              "Qwen/Qwen2.5-1.5B-Instruct"
  choices: [{
    index:            int
    text:             string
    logprobs:         null
    finish_reason:    "stop" | "length"
    stop_reason:      null
    token_ids:        null
    prompt_logprobs:  null
    prompt_token_ids: null
  }]
  service_tier:       null
  system_fingerprint: null
  usage: {
    prompt_tokens:         int
    completion_tokens:     int
    total_tokens:          int
    prompt_tokens_details: null
  }
  kv_transfer_params: null
}
```

---

### POST /v1/embeddings — 404

Requires an embedding model (e.g., `BAAI/bge-base-en-v1.5`). Not available with generative models.

---

### POST /v1/responses — 200 (296ms)

Newer Responses API. Uses `input` instead of `messages`.

**Request:**
```json
{
  "model": "Qwen/Qwen2.5-1.5B-Instruct",
  "input": "Say hello in one sentence."
}
```

**Response shape:**
```
{
  id:                  "resp_<hex>"
  created_at:          <unix_timestamp>
  object:              "response"
  model:               "Qwen/Qwen2.5-1.5B-Instruct"
  status:              "completed"
  output: [{
    id:                string
    content:           [{type, text}]
    role:              "assistant"
    status:            "completed"
    type:              "message"
  }]
  usage: {
    input_tokens:      int
    output_tokens:     int
    total_tokens:      int
    input_tokens_details:  {cached_tokens, ...}
    output_tokens_details: {reasoning_tokens, ...}
  }
  temperature:         0.7
  top_p:               0.8
  max_output_tokens:   4061
  parallel_tool_calls: true
  tool_choice:         "auto"
  tools:               []
  truncation:          "disabled"
  ...
}
```

Richer response than `/v1/chat/completions` — includes token detail breakdowns and tool-use metadata.

---

### POST /v1/audio/transcriptions — SKIPPED

Requires Whisper model + audio file upload. Not tested (generative model loaded).

---

### POST /v1/audio/translations — SKIPPED

Requires Whisper model + audio file upload. Not tested.

---

## Tokenizer Endpoints

### POST /tokenize — 200 (6ms)

Encode text to token IDs.

**Request:**
```json
{
  "model": "Qwen/Qwen2.5-1.5B-Instruct",
  "prompt": "Hello, world!"
}
```

**Response:**
```json
{
  "count": 4,
  "max_model_len": 4096,
  "tokens": [9707, 11, 1879, 0],
  "token_strs": null
}
```

Response shape: `{count, max_model_len, tokens[], token_strs}`

---

### POST /detokenize — 200 (6ms)

Decode token IDs back to text.

**Request:**
```json
{
  "model": "Qwen/Qwen2.5-1.5B-Instruct",
  "tokens": [9707, 11, 1879, 0]
}
```

**Response:**
```json
{
  "prompt": "Hello, world!"
}
```

Response shape: `{prompt}`

---

## Pooling Endpoints — All 404

These endpoints require specialized model types. With a generative model loaded, all return `{"detail": "Not Found"}`.

| Endpoint | Required Model Type |
|----------|-------------------|
| `POST /pooling` | Pooling model (e.g., `BAAI/bge-base-en-v1.5`) |
| `POST /classify` | Classification model (e.g., `jason9693/Qwen2.5-1.5B-apeach`) |
| `POST /score` | Embedding or cross-encoder model |
| `POST /rerank` | Cross-encoder model (e.g., `BAAI/bge-reranker-v2-m3`) |
| `POST /v1/rerank` | Cross-encoder (Jina AI compatible) |
| `POST /v2/rerank` | Cross-encoder (Cohere v2 compatible) |

These endpoints are documented in `docs/vllm-api-reference.md` with expected request/response shapes.

---

## LoRA Endpoints — 404

Requires vLLM started with `--enable-lora`. Without it, endpoints are not registered.

| Endpoint | Purpose |
|----------|---------|
| `POST /v1/load_lora_adapter` | Load adapter at runtime |
| `POST /v1/unload_lora_adapter` | Unload adapter |

---

## Performance Baseline

From `benchmark.py` (5 iterations per prompt length):

| Prompt | TTFT p50 | TTFT p95 | E2E p50 |
|--------|----------|----------|---------|
| Short (~10 tokens) | 48ms | 51ms | 427ms |
| Medium (~60 tokens) | 49ms | 54ms | 12,620ms |
| Long (~200 tokens) | 52ms | 60ms | 27,030ms |

TTFT is consistent across prompt lengths — prefill is fast on 1.5B parameters. E2E scales with output length (model generates longer responses for complex prompts).

---

## Parameter Support Matrix

From `test_model.py` (32/32 passed):

| Sweep | Values Tested | Result |
|-------|--------------|--------|
| Temperature | 0.0, 0.3, 0.7, 1.0, 1.5 | All pass |
| Top-K | 1, 10, 50, disabled | All pass |
| Top-P | 0.1, 0.5, 0.9, 1.0 | All pass |
| Min-P | 0.0, 0.05, 0.1, 0.2 | All pass |
| Repetition Penalty | 1.0, 1.1, 1.3, 1.5 | All pass |
| Structured Output | choice, json_schema, regex | All pass |
| Streaming | true, false | All pass |
| Stop Sequences | none, ".", "\n" | All pass |
| Seed Reproducibility | same seed × 2, different seed | All pass (deterministic) |

---

## Test Environment

| Setting | Value |
|---------|-------|
| vLLM version | 0.17.1 (V1 engine) |
| Model | Qwen/Qwen2.5-1.5B-Instruct |
| GPU | NVIDIA GeForce RTX 4060 Laptop (8GB VRAM) |
| Platform | WSL2 on Windows |
| vLLM flags | `--max-model-len 4096 --gpu-memory-utilization 0.8 --enforce-eager` |
| Quantization | None (bfloat16) |
| Max sequence length | 4096 |
