# vllm-explorer Roadmap

## v0.1 — Endpoint Catalog (active)

**Goal:** document the full vLLM HTTP API surface and test model behavior.

- [ ] `scripts/probe_endpoints.py` — hit every endpoint, log response shapes to stdout + `data/`
- [ ] `scripts/test_model.py` — single model parameter sweep (temperature, max_tokens behavior, stop tokens)
- [ ] `scripts/benchmark.py` — TTFT and tokens/sec per model
- [ ] `scripts/build_catalog.py` — run all probes across models, write `data/catalog.json`
- [ ] `docs/endpoint-reference.md` — human-readable API reference from probe results

**Models to test in v0.1:**
- `mistralai/Mistral-7B-Instruct-v0.2` (primary — used in gpu-autoscale-inference)
- `microsoft/Phi-3-mini-4k-instruct` (lightweight reference)
- `meta-llama/Meta-Llama-3-8B-Instruct` (fallback option for main project)

---

## v0.2 — Multi-Model Comparison

**Goal:** structured comparison across models to support definitive model selection.

- [ ] Side-by-side TTFT benchmark table (all models)
- [ ] VRAM usage per model (from `/metrics`)
- [ ] Output quality spot-check (same prompt, compare responses)
- [ ] `docs/model-comparison.md` — final recommendation with data

---

## v0.3 — HTML Browser (optional)

**Goal:** visual output consistent with nim-explorer and ollama-catalog pattern.

- [ ] `docs/index.html` — model browser with endpoint reference and benchmark results
- [ ] Vanilla HTML/CSS/JS, no framework
