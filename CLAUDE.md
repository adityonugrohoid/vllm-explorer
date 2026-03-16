# vLLM Explorer

## Overview

Probes the full vLLM server API surface and builds a model capability catalog. Primary purpose is to inform model selection and parameter decisions for the `gpu-autoscale-inference` portfolio project.

## Purpose

- Document every vLLM HTTP endpoint: parameters, response shapes, error behavior
- Test selected model (`Qwen/Qwen2.5-1.5B-Instruct`) and alternatives for capability and performance comparison
- Benchmark TTFT and tokens/sec per model to support cold-start demo planning
- Understand the `/metrics` Prometheus endpoint (used by the main project's observability stack)

## Tech Stack

| Layer | Tool |
|---|---|
| HTTP client | httpx |
| OpenAI-compatible client | openai |
| CLI output | rich |
| Config | python-dotenv |
| Runtime | Python 3.12+ |

## Critical: Do NOT Install the vLLM Python Package

vLLM runs as a server exposing an OpenAI-compatible HTTP API. This project interacts with it **over HTTP only** — no in-process vLLM. Do not install the `vllm` pip package; it is a heavy CUDA dependency for running inference directly in Python, which is not what we do here.

Use `openai` (pip) or raw `httpx` calls against the running vLLM server.

## Architecture

```
vLLM Docker container (host, uses local GPU)
  └── HTTP API on localhost:8000

Scripts (host, .venv)
  └── probe/test/benchmark scripts → call localhost:8000 → write to data/
```

vLLM is external to these scripts. Scripts are pure HTTP clients.

## Prerequisites: Running vLLM

Start vLLM on the host before running any script:

```bash
docker run --gpus all \
  -p 8000:8000 \
  --ipc=host \
  vllm/vllm-openai \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.8 \
  --enforce-eager
```

Wait until the log shows `Application startup complete` before running scripts (~5–10s for Qwen2.5-1.5B).

**Required flags for 8GB VRAM (WSL2):** `--max-model-len 4096 --gpu-memory-utilization 0.8 --enforce-eager`

To test with a minimal model for endpoint validation only:
```bash
docker run --gpus all -p 8000:8000 --ipc=host vllm/vllm-openai --model facebook/opt-125m
```

Check readiness:
```bash
curl http://localhost:8000/health
```

## Commands

```bash
# Setup
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

# Probe all endpoints — logs response shapes to stdout + data/
python scripts/probe_endpoints.py

# Test a specific model with parameter sweeps
python scripts/test_model.py --model Qwen/Qwen2.5-1.5B-Instruct

# Build full catalog across all tested models
python scripts/build_catalog.py

# Run TTFT + tokens/sec benchmark
python scripts/benchmark.py --model Qwen/Qwen2.5-1.5B-Instruct
```

## vLLM API Endpoints to Probe

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Readiness check (used by worker in main project) |
| `/v1/models` | GET | List loaded models + metadata |
| `/v1/completions` | POST | Raw text completion |
| `/v1/chat/completions` | POST | Chat format with system prompts |
| `/v1/embeddings` | POST | Embedding generation (model-dependent) |
| `/metrics` | GET | Prometheus metrics (feeds Grafana in main project) |

## Key Patterns

- No token limits — let models run to natural stop (global convention)
- All results written to `data/` as JSON (gitignored)
- Use `rich` for progress bars and tables in CLI output
- `VLLM_BASE_URL` from `.env` — default `http://localhost:8000`

## Key Notes

- `/metrics` is the most important endpoint for the main project — verify what Prometheus metrics vLLM exposes natively (gpu_cache_usage_perc, num_requests_running, etc.)
- Model loading is one-time per `docker run` — no hot-swap here (that's a main project concern)
- `Qwen/Qwen2.5-1.5B-Instruct` is the selected model for both phases of the main project — benchmark this one thoroughly
- Platform is model-agnostic via `MODEL_ID` env var — Qwen2.5-1.5B selected for small footprint (~3GB), fast cold start (~5-10s), ungated, strong benchmark scores

## Project Structure

```
vllm-explorer/
├── scripts/
│   ├── probe_endpoints.py    # hit every endpoint, log shapes
│   ├── test_model.py         # single model parameter sweep
│   ├── build_catalog.py      # all probes → data/catalog.json
│   └── benchmark.py          # TTFT + tokens/sec per model
├── docs/
│   └── endpoint-reference.md # human-readable API reference (generated)
├── data/                     # runtime output — gitignored
├── .env.example
├── requirements.txt
├── ROADMAP.md
└── README.md
```

## Related Project

`~/projects/gpu-autoscale-inference` — the main portfolio project this explorer supports. Model selection and parameter decisions from this repo feed directly into the vLLM deployment config in that project. Model decision: `Qwen/Qwen2.5-1.5B-Instruct` for both phases.
