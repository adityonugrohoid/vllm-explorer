"""
probe_endpoints.py — Hit every known vLLM HTTP endpoint, log response shapes to stdout + data/

Probes all 22 known vLLM endpoints across 5 categories:
  - Admin (health, ping, metrics, server_info)
  - OpenAI-compatible (models, chat/completions, completions, embeddings, responses, audio)
  - Tokenizer (tokenize, detokenize)
  - Pooling (pooling, classify, score, rerank variants)
  - LoRA management (load/unload adapter)

Outputs:
  - Rich table to stdout with status, latency, response shape
  - data/probe_results.json with full response payloads
"""

import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

load_dotenv()

BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000")
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

console = Console()


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------

@dataclass
class ProbeResult:
    endpoint: str
    method: str
    category: str
    status_code: int | None = None
    latency_ms: float | None = None
    response_shape: dict[str, Any] | None = None
    response_body: Any = None
    error: str | None = None
    notes: str = ""


# ---------------------------------------------------------------------------
# Endpoint definitions
# ---------------------------------------------------------------------------

@dataclass
class EndpointDef:
    method: str
    path: str
    category: str
    body: dict[str, Any] | None = None
    notes: str = ""
    requires_model: bool = False
    headers: dict[str, str] = field(default_factory=lambda: {"Content-Type": "application/json"})


def build_endpoint_defs(model_id: str | None) -> list[EndpointDef]:
    """Build the full list of endpoints to probe. model_id is discovered at runtime."""

    m = model_id or "unknown"

    return [
        # --- Admin ---
        EndpointDef("GET", "/health", "admin", notes="Readiness probe"),
        EndpointDef("GET", "/ping", "admin", notes="Health alias"),
        EndpointDef("GET", "/metrics", "admin", notes="Prometheus metrics"),
        EndpointDef("GET", "/server_info", "admin", notes="Server config + version"),

        # --- OpenAI-compatible ---
        EndpointDef("GET", "/v1/models", "openai", notes="List loaded models"),
        EndpointDef(
            "POST", "/v1/chat/completions", "openai",
            body={
                "model": m,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Say hello in one sentence."},
                ],
                "temperature": 0.0,
            },
            requires_model=True,
            notes="Chat completion (primary inference path)",
        ),
        EndpointDef(
            "POST", "/v1/completions", "openai",
            body={
                "model": m,
                "prompt": "A robot may not injure a human being",
                "temperature": 0.0,
            },
            requires_model=True,
            notes="Text completion (legacy)",
        ),
        EndpointDef(
            "POST", "/v1/embeddings", "openai",
            body={"model": m, "input": "What is vLLM?"},
            requires_model=True,
            notes="Embeddings (requires embedding model)",
        ),
        EndpointDef(
            "POST", "/v1/responses", "openai",
            body={
                "model": m,
                "input": "Say hello in one sentence.",
            },
            requires_model=True,
            notes="Responses API (newer streaming interface)",
        ),
        EndpointDef(
            "POST", "/v1/audio/transcriptions", "openai",
            body=None,
            notes="Speech-to-text (requires Whisper model + audio file — skipped)",
        ),
        EndpointDef(
            "POST", "/v1/audio/translations", "openai",
            body=None,
            notes="Audio translation (requires Whisper model + audio file — skipped)",
        ),

        # --- Tokenizer ---
        EndpointDef(
            "POST", "/tokenize", "tokenizer",
            body={"model": m, "prompt": "Hello, world!"},
            requires_model=True,
            notes="Text → token IDs",
        ),
        EndpointDef(
            "POST", "/detokenize", "tokenizer",
            body={"model": m, "tokens": [1, 22557, 28725, 1526, 28808]},
            requires_model=True,
            notes="Token IDs → text",
        ),

        # --- Pooling ---
        EndpointDef(
            "POST", "/pooling", "pooling",
            body={"model": m, "input": "What is vLLM?"},
            requires_model=True,
            notes="Generic pooling (requires pooling model)",
        ),
        EndpointDef(
            "POST", "/classify", "pooling",
            body={"model": m, "input": "This movie was great!"},
            requires_model=True,
            notes="Classification (requires classification model)",
        ),
        EndpointDef(
            "POST", "/score", "pooling",
            body={
                "model": m,
                "text_1": "What is vLLM?",
                "text_2": "vLLM is a fast inference engine.",
            },
            requires_model=True,
            notes="Similarity scoring (requires embedding/cross-encoder model)",
        ),
        EndpointDef(
            "POST", "/rerank", "pooling",
            body={
                "model": m,
                "query": "What is vLLM?",
                "documents": [
                    "vLLM is a fast inference engine.",
                    "Python is a programming language.",
                ],
            },
            requires_model=True,
            notes="Rerank — vLLM native (requires cross-encoder model)",
        ),
        EndpointDef(
            "POST", "/v1/rerank", "pooling",
            body={
                "model": m,
                "query": "What is vLLM?",
                "documents": [
                    "vLLM is a fast inference engine.",
                    "Python is a programming language.",
                ],
            },
            requires_model=True,
            notes="Rerank — Jina AI compatible (requires cross-encoder model)",
        ),
        EndpointDef(
            "POST", "/v2/rerank", "pooling",
            body={
                "model": m,
                "query": "What is vLLM?",
                "documents": [
                    "vLLM is a fast inference engine.",
                    "Python is a programming language.",
                ],
            },
            requires_model=True,
            notes="Rerank — Cohere v2 compatible (requires cross-encoder model)",
        ),

        # --- LoRA management ---
        EndpointDef(
            "POST", "/v1/load_lora_adapter", "lora",
            body={"lora_name": "probe_test", "lora_path": "/nonexistent/path"},
            notes="Load LoRA adapter (expect error — probing availability only)",
        ),
        EndpointDef(
            "POST", "/v1/unload_lora_adapter", "lora",
            body={"lora_name": "probe_test"},
            notes="Unload LoRA adapter (expect error — probing availability only)",
        ),
    ]


# ---------------------------------------------------------------------------
# Shape extraction — summarize response structure without full payloads
# ---------------------------------------------------------------------------

def extract_shape(obj: Any, depth: int = 0, max_depth: int = 3) -> Any:
    """Recursively extract the shape/structure of a JSON response."""
    if depth >= max_depth:
        return type(obj).__name__

    if isinstance(obj, dict):
        return {k: extract_shape(v, depth + 1, max_depth) for k, v in obj.items()}
    elif isinstance(obj, list):
        if len(obj) == 0:
            return "list[empty]"
        return [extract_shape(obj[0], depth + 1, max_depth), f"...({len(obj)} items)"]
    elif isinstance(obj, str):
        if len(obj) > 100:
            return f"str({len(obj)} chars)"
        return obj
    elif isinstance(obj, (int, float, bool)):
        return obj
    elif obj is None:
        return None
    else:
        return type(obj).__name__


# ---------------------------------------------------------------------------
# Probing logic
# ---------------------------------------------------------------------------

async def probe_single(client: httpx.AsyncClient, ep: EndpointDef) -> ProbeResult:
    """Probe a single endpoint and return the result."""
    url = f"{BASE_URL}{ep.path}"
    result = ProbeResult(
        endpoint=ep.path,
        method=ep.method,
        category=ep.category,
        notes=ep.notes,
    )

    # Skip audio endpoints (require file upload + Whisper model)
    if "/audio/" in ep.path:
        result.status_code = None
        result.error = "skipped"
        result.notes = ep.notes
        return result

    try:
        start = time.perf_counter()
        if ep.method == "GET":
            resp = await client.get(url)
        else:
            resp = await client.post(url, json=ep.body, headers=ep.headers)
        elapsed = (time.perf_counter() - start) * 1000

        result.status_code = resp.status_code
        result.latency_ms = round(elapsed, 1)

        # Parse response
        content_type = resp.headers.get("content-type", "")
        if "application/json" in content_type:
            body = resp.json()
            result.response_body = body
            result.response_shape = extract_shape(body)
        elif "text/plain" in content_type or "text/html" in content_type:
            text = resp.text
            if len(text) > 2000:
                result.response_body = f"<text, {len(text)} chars>"
                result.response_shape = {"type": "text", "length": len(text)}
            else:
                result.response_body = text
                result.response_shape = {"type": "text", "length": len(text)}
        else:
            # Empty body or binary
            result.response_body = resp.text[:500] if resp.text else None
            result.response_shape = {"type": content_type or "empty", "length": len(resp.text)}

    except httpx.ConnectError:
        result.error = "connection_refused"
    except httpx.TimeoutException:
        result.error = "timeout"
    except Exception as e:
        result.error = f"{type(e).__name__}: {e}"

    return result


async def discover_model(client: httpx.AsyncClient) -> str | None:
    """Hit /v1/models to discover the loaded model ID."""
    try:
        resp = await client.get(f"{BASE_URL}/v1/models")
        if resp.status_code == 200:
            data = resp.json()
            models = data.get("data", [])
            if models:
                return models[0].get("id")
    except Exception:
        pass
    return None


async def run_probes() -> list[ProbeResult]:
    """Run all endpoint probes."""
    timeout = httpx.Timeout(30.0, connect=5.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        # Step 1: Check server reachability
        console.print(f"\n[bold]Target:[/bold] {BASE_URL}")
        try:
            health = await client.get(f"{BASE_URL}/health")
            console.print(f"[green]Server reachable[/green] — /health returned {health.status_code}\n")
        except httpx.ConnectError:
            console.print("[red bold]Server unreachable[/red bold] — cannot connect to vLLM")
            console.print(f"Start vLLM first: docker run --gpus all -p 8000:8000 --ipc=host vllm/vllm-openai --model <model>")
            console.print("\nProbing anyway to document endpoint availability...\n")

        # Step 2: Discover loaded model
        model_id = await discover_model(client)
        if model_id:
            console.print(f"[bold]Model:[/bold] {model_id}\n")
        else:
            console.print("[yellow]Could not discover model — using placeholder in requests[/yellow]\n")

        # Step 3: Build endpoint list with discovered model
        endpoints = build_endpoint_defs(model_id)

        # Step 4: Probe all endpoints sequentially (to avoid overwhelming server)
        results: list[ProbeResult] = []
        for ep in endpoints:
            result = await probe_single(client, ep)
            results.append(result)

            # Live feedback
            status = result.status_code
            if result.error == "skipped":
                icon = "[dim]SKIP[/dim]"
            elif result.error:
                icon = f"[red]ERR [/red]"
            elif status and 200 <= status < 300:
                icon = f"[green]{status} [/green]"
            elif status and 400 <= status < 500:
                icon = f"[yellow]{status} [/yellow]"
            else:
                icon = f"[red]{status} [/red]"

            latency = f"{result.latency_ms:.0f}ms" if result.latency_ms else "—"
            console.print(f"  {icon} {ep.method:4s} {ep.path:<35s} {latency:>8s}  {ep.notes}")

        return results


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_summary_table(results: list[ProbeResult]) -> None:
    """Print a rich summary table."""
    table = Table(title="\nvLLM Endpoint Probe Results", show_lines=True)
    table.add_column("Category", style="bold")
    table.add_column("Method", style="cyan")
    table.add_column("Endpoint")
    table.add_column("Status", justify="center")
    table.add_column("Latency", justify="right")
    table.add_column("Response Shape")
    table.add_column("Notes")

    for r in results:
        if r.error == "skipped":
            status_str = "[dim]SKIP[/dim]"
        elif r.error:
            status_str = f"[red]{r.error}[/red]"
        elif r.status_code and 200 <= r.status_code < 300:
            status_str = f"[green]{r.status_code}[/green]"
        elif r.status_code and 400 <= r.status_code < 500:
            status_str = f"[yellow]{r.status_code}[/yellow]"
        else:
            status_str = f"[red]{r.status_code}[/red]"

        latency_str = f"{r.latency_ms:.0f}ms" if r.latency_ms else "—"

        shape_str = ""
        if r.response_shape:
            if isinstance(r.response_shape, dict):
                keys = list(r.response_shape.keys())
                shape_str = "{" + ", ".join(keys[:6])
                if len(keys) > 6:
                    shape_str += ", ..."
                shape_str += "}"
            else:
                shape_str = str(r.response_shape)[:60]

        table.add_row(
            r.category,
            r.method,
            r.endpoint,
            status_str,
            latency_str,
            shape_str,
            r.notes,
        )

    console.print(table)


def print_category_stats(results: list[ProbeResult]) -> None:
    """Print per-category summary."""
    categories: dict[str, dict[str, int]] = {}
    for r in results:
        cat = categories.setdefault(r.category, {"total": 0, "ok": 0, "error": 0, "skip": 0})
        cat["total"] += 1
        if r.error == "skipped":
            cat["skip"] += 1
        elif r.error or (r.status_code and r.status_code >= 400):
            cat["error"] += 1
        else:
            cat["ok"] += 1

    console.print("\n[bold]Category Summary:[/bold]")
    for cat, stats in categories.items():
        ok = stats["ok"]
        total = stats["total"]
        skip = stats["skip"]
        err = stats["error"]
        console.print(f"  {cat:<12s}  {ok}/{total} OK   {err} errors   {skip} skipped")


def save_results(results: list[ProbeResult]) -> Path:
    """Save full probe results to data/probe_results.json."""
    output = {
        "base_url": BASE_URL,
        "probe_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_endpoints": len(results),
        "results": [],
    }

    for r in results:
        entry = asdict(r)
        # Truncate large response bodies for the JSON dump
        if entry.get("response_body") and isinstance(entry["response_body"], str) and len(entry["response_body"]) > 5000:
            entry["response_body"] = entry["response_body"][:5000] + "...<truncated>"
        output["results"].append(entry)

    out_path = DATA_DIR / "probe_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    console.print(Panel.fit(
        "[bold]vLLM Endpoint Probe[/bold]\n"
        "Probing all 22 known vLLM HTTP endpoints",
        border_style="blue",
    ))

    results = await run_probes()
    print_summary_table(results)
    print_category_stats(results)

    out_path = save_results(results)
    console.print(f"\n[bold]Results saved to:[/bold] {out_path}\n")


if __name__ == "__main__":
    asyncio.run(main())
