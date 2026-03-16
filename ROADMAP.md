# vllm-explorer Roadmap

## v0.1 — Endpoint Catalog (active)

**Goal:** document the full vLLM HTTP API surface and test model behavior.

- [ ] `scripts/probe_endpoints.py` — hit every endpoint, log response shapes to stdout + `data/`
- [ ] `scripts/test_model.py` — single model parameter sweep (temperature, max_tokens behavior, stop tokens)
- [ ] `scripts/benchmark.py` — TTFT and tokens/sec per model
- [ ] `scripts/build_catalog.py` — run all probes across models, write `data/catalog.json`
- [ ] `docs/endpoint-reference.md` — human-readable API reference from probe results

**Primary model:**
- `Qwen/Qwen2.5-1.5B-Instruct` (selected for both phases of gpu-autoscale-inference — ~3GB disk, ~3.5GB VRAM, fast cold start, ungated)

**Optional comparison models:**
- `facebook/opt-125m` (endpoint validation only — loads instantly)
- `google/gemma-2-2b-it` (quality comparison)

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
