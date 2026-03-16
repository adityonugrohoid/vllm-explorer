"""
test_model.py — Single model parameter sweep against a running vLLM server.

Tests a loaded model across parameter combinations to map its behavior:
  - Temperature sweep (0.0, 0.3, 0.7, 1.0, 1.5)
  - Top-k sweep (1, 10, 50, -1/disabled)
  - Top-p sweep (0.1, 0.5, 0.9, 1.0)
  - Min-p sweep (0.0, 0.05, 0.1, 0.2)
  - Repetition penalty sweep (1.0, 1.1, 1.3, 1.5)
  - Structured output (choice constraint, JSON schema, regex)
  - Streaming vs non-streaming
  - Stop sequence behavior
  - Seed reproducibility check

Outputs:
  - Rich tables to stdout per sweep category
  - data/test_model_{model_slug}.json with full results

Usage:
  python scripts/test_model.py
  python scripts/test_model.py --model google/gemma-3-1b-it
"""

import argparse
import asyncio
import json
import os
import time
from dataclasses import dataclass, asdict
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

CHAT_ENDPOINT = f"{BASE_URL}/v1/chat/completions"
TOKENIZE_ENDPOINT = f"{BASE_URL}/tokenize"

# Standard test prompt — short enough for fast sweeps, complex enough to show differences
TEST_PROMPT = "Explain what PagedAttention is and why it matters for LLM inference, in exactly two sentences."
SYSTEM_PROMPT = "You are a concise technical assistant."


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------

@dataclass
class SweepResult:
    sweep_name: str
    param_name: str
    param_value: Any
    status_code: int | None = None
    latency_ms: float | None = None
    output_text: str = ""
    output_tokens: int | None = None
    prompt_tokens: int | None = None
    finish_reason: str = ""
    error: str | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def chat_request(
    client: httpx.AsyncClient,
    model: str,
    extra_body: dict[str, Any] | None = None,
    messages: list[dict] | None = None,
    stream: bool = False,
) -> tuple[int | None, dict | None, float, str | None]:
    """Send a chat completion request. Returns (status, body, latency_ms, error)."""
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages or [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": TEST_PROMPT},
        ],
        "temperature": 0.0,
    }
    if extra_body:
        payload.update(extra_body)
    payload["stream"] = stream

    try:
        start = time.perf_counter()
        if not stream:
            resp = await client.post(CHAT_ENDPOINT, json=payload)
            elapsed = (time.perf_counter() - start) * 1000
            if resp.status_code == 200:
                return resp.status_code, resp.json(), elapsed, None
            return resp.status_code, None, elapsed, resp.text[:500]
        else:
            # Streaming: collect chunks
            collected_text = ""
            chunk_count = 0
            usage = {}
            finish_reason = ""
            async with client.stream("POST", CHAT_ENDPOINT, json=payload) as resp:
                if resp.status_code != 200:
                    elapsed = (time.perf_counter() - start) * 1000
                    body = await resp.aread()
                    return resp.status_code, None, elapsed, body.decode()[:500]

                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        chunk_count += 1
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        if "content" in delta and delta["content"]:
                            collected_text += delta["content"]
                        fr = chunk.get("choices", [{}])[0].get("finish_reason")
                        if fr:
                            finish_reason = fr
                        if "usage" in chunk and chunk["usage"]:
                            usage = chunk["usage"]
                    except json.JSONDecodeError:
                        continue

            elapsed = (time.perf_counter() - start) * 1000
            body = {
                "choices": [{"message": {"content": collected_text}, "finish_reason": finish_reason}],
                "usage": usage,
                "_stream_chunks": chunk_count,
            }
            return 200, body, elapsed, None

    except httpx.ConnectError:
        return None, None, 0, "connection_refused"
    except httpx.TimeoutException:
        return None, None, 0, "timeout"
    except Exception as e:
        return None, None, 0, f"{type(e).__name__}: {e}"


def extract_result(
    sweep_name: str,
    param_name: str,
    param_value: Any,
    status: int | None,
    body: dict | None,
    latency: float,
    error: str | None,
) -> SweepResult:
    """Extract a SweepResult from a chat response."""
    r = SweepResult(
        sweep_name=sweep_name,
        param_name=param_name,
        param_value=param_value,
        status_code=status,
        latency_ms=round(latency, 1),
        error=error,
    )
    if body and body.get("choices"):
        choice = body["choices"][0]
        msg = choice.get("message", {})
        r.output_text = msg.get("content", "")
        r.finish_reason = choice.get("finish_reason", "")
    if body and body.get("usage"):
        r.prompt_tokens = body["usage"].get("prompt_tokens")
        r.output_tokens = body["usage"].get("completion_tokens")
    return r


async def discover_model(client: httpx.AsyncClient) -> str | None:
    """Discover the loaded model from /v1/models."""
    try:
        resp = await client.get(f"{BASE_URL}/v1/models")
        if resp.status_code == 200:
            models = resp.json().get("data", [])
            if models:
                return models[0].get("id")
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Sweep runners
# ---------------------------------------------------------------------------

async def sweep_temperature(client: httpx.AsyncClient, model: str) -> list[SweepResult]:
    """Sweep temperature values."""
    values = [0.0, 0.3, 0.7, 1.0, 1.5]
    results = []
    for temp in values:
        status, body, latency, error = await chat_request(
            client, model, extra_body={"temperature": temp}
        )
        results.append(extract_result("temperature", "temperature", temp, status, body, latency, error))
    return results


async def sweep_top_k(client: httpx.AsyncClient, model: str) -> list[SweepResult]:
    """Sweep top_k values."""
    values = [1, 10, 50, -1]
    results = []
    for k in values:
        status, body, latency, error = await chat_request(
            client, model, extra_body={"temperature": 0.7, "top_k": k}
        )
        label = k if k > 0 else "disabled"
        results.append(extract_result("top_k", "top_k", label, status, body, latency, error))
    return results


async def sweep_top_p(client: httpx.AsyncClient, model: str) -> list[SweepResult]:
    """Sweep top_p values."""
    values = [0.1, 0.5, 0.9, 1.0]
    results = []
    for p in values:
        status, body, latency, error = await chat_request(
            client, model, extra_body={"temperature": 0.7, "top_p": p}
        )
        results.append(extract_result("top_p", "top_p", p, status, body, latency, error))
    return results


async def sweep_min_p(client: httpx.AsyncClient, model: str) -> list[SweepResult]:
    """Sweep min_p values."""
    values = [0.0, 0.05, 0.1, 0.2]
    results = []
    for mp in values:
        status, body, latency, error = await chat_request(
            client, model, extra_body={"temperature": 0.7, "min_p": mp}
        )
        results.append(extract_result("min_p", "min_p", mp, status, body, latency, error))
    return results


async def sweep_repetition_penalty(client: httpx.AsyncClient, model: str) -> list[SweepResult]:
    """Sweep repetition_penalty values."""
    values = [1.0, 1.1, 1.3, 1.5]
    results = []
    for rp in values:
        status, body, latency, error = await chat_request(
            client, model, extra_body={"temperature": 0.7, "repetition_penalty": rp}
        )
        results.append(extract_result("repetition_penalty", "repetition_penalty", rp, status, body, latency, error))
    return results


async def sweep_structured_output(client: httpx.AsyncClient, model: str) -> list[SweepResult]:
    """Test structured output modes."""
    results = []

    # Choice constraint
    status, body, latency, error = await chat_request(
        client, model,
        messages=[
            {"role": "user", "content": "Is vLLM fast? Answer with one word."},
        ],
        extra_body={"structured_outputs": {"choice": ["yes", "no", "maybe"]}},
    )
    results.append(extract_result("structured_output", "mode", "choice", status, body, latency, error))

    # JSON schema
    status, body, latency, error = await chat_request(
        client, model,
        messages=[
            {"role": "user", "content": "Give me a name and age for a fictional character."},
        ],
        extra_body={
            "structured_outputs": {
                "json": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"},
                    },
                    "required": ["name", "age"],
                },
            },
        },
    )
    results.append(extract_result("structured_output", "mode", "json_schema", status, body, latency, error))

    # Regex
    status, body, latency, error = await chat_request(
        client, model,
        messages=[
            {"role": "user", "content": "Give me a US phone number."},
        ],
        extra_body={"structured_outputs": {"regex": r"\d{3}-\d{3}-\d{4}"}},
    )
    results.append(extract_result("structured_output", "mode", "regex", status, body, latency, error))

    return results


async def sweep_streaming(client: httpx.AsyncClient, model: str) -> list[SweepResult]:
    """Compare streaming vs non-streaming."""
    results = []

    # Non-streaming
    status, body, latency, error = await chat_request(client, model, stream=False)
    results.append(extract_result("streaming", "stream", False, status, body, latency, error))

    # Streaming
    status, body, latency, error = await chat_request(
        client, model,
        extra_body={"stream_include_usage": True},
        stream=True,
    )
    r = extract_result("streaming", "stream", True, status, body, latency, error)
    if body and "_stream_chunks" in body:
        r.error = f"{body['_stream_chunks']} chunks"
    results.append(r)

    return results


async def sweep_stop_sequences(client: httpx.AsyncClient, model: str) -> list[SweepResult]:
    """Test stop sequence behavior."""
    results = []

    # No stop
    status, body, latency, error = await chat_request(client, model)
    results.append(extract_result("stop_sequences", "stop", "none", status, body, latency, error))

    # Stop on period
    status, body, latency, error = await chat_request(
        client, model, extra_body={"stop": ["."]}
    )
    results.append(extract_result("stop_sequences", "stop", "[\".\"]", status, body, latency, error))

    # Stop on newline
    status, body, latency, error = await chat_request(
        client, model, extra_body={"stop": ["\n"]}
    )
    results.append(extract_result("stop_sequences", "stop", "[\"\\n\"]", status, body, latency, error))

    return results


async def sweep_seed_reproducibility(client: httpx.AsyncClient, model: str) -> list[SweepResult]:
    """Test that same seed produces same output."""
    results = []

    # Two requests with same seed
    status1, body1, lat1, err1 = await chat_request(
        client, model, extra_body={"temperature": 0.7, "seed": 42}
    )
    r1 = extract_result("seed", "seed", "42 (run 1)", status1, body1, lat1, err1)
    results.append(r1)

    status2, body2, lat2, err2 = await chat_request(
        client, model, extra_body={"temperature": 0.7, "seed": 42}
    )
    r2 = extract_result("seed", "seed", "42 (run 2)", status2, body2, lat2, err2)

    # Check if outputs match
    if r1.output_text and r2.output_text:
        match = r1.output_text == r2.output_text
        r2.error = f"match={match}"
    results.append(r2)

    # Different seed
    status3, body3, lat3, err3 = await chat_request(
        client, model, extra_body={"temperature": 0.7, "seed": 99}
    )
    r3 = extract_result("seed", "seed", "99", status3, body3, lat3, err3)
    if r1.output_text and r3.output_text:
        r3.error = f"differs={r1.output_text != r3.output_text}"
    results.append(r3)

    return results


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_sweep_table(sweep_name: str, results: list[SweepResult]) -> None:
    """Print a rich table for one sweep category."""
    table = Table(title=f"\n{sweep_name}", show_lines=True)
    table.add_column("Param", style="cyan")
    table.add_column("Value", style="bold")
    table.add_column("Status", justify="center")
    table.add_column("Latency", justify="right")
    table.add_column("Tokens", justify="right")
    table.add_column("Finish", style="dim")
    table.add_column("Output (truncated)")
    table.add_column("Notes")

    for r in results:
        if r.error and r.status_code is None:
            status_str = f"[red]{r.error}[/red]"
        elif r.status_code and 200 <= r.status_code < 300:
            status_str = f"[green]{r.status_code}[/green]"
        elif r.status_code and 400 <= r.status_code < 500:
            status_str = f"[yellow]{r.status_code}[/yellow]"
        else:
            status_str = f"[red]{r.status_code}[/red]"

        latency_str = f"{r.latency_ms:.0f}ms" if r.latency_ms else "—"
        tokens_str = str(r.output_tokens) if r.output_tokens else "—"
        output_preview = r.output_text[:80] + "..." if len(r.output_text) > 80 else r.output_text
        notes = r.error if r.error and r.status_code and r.status_code < 400 else ""

        table.add_row(
            r.param_name,
            str(r.param_value),
            status_str,
            latency_str,
            tokens_str,
            r.finish_reason,
            output_preview,
            notes,
        )

    console.print(table)


def save_results(model: str, all_results: dict[str, list[SweepResult]]) -> Path:
    """Save all sweep results to JSON."""
    slug = model.replace("/", "_").replace("-", "_").lower()
    output = {
        "model": model,
        "base_url": BASE_URL,
        "test_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "prompt": TEST_PROMPT,
        "system_prompt": SYSTEM_PROMPT,
        "sweeps": {},
    }

    for sweep_name, results in all_results.items():
        output["sweeps"][sweep_name] = [asdict(r) for r in results]

    out_path = DATA_DIR / f"test_model_{slug}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    parser = argparse.ArgumentParser(description="Test a vLLM model with parameter sweeps")
    parser.add_argument("--model", type=str, default=None, help="Model ID (auto-discovered if omitted)")
    args = parser.parse_args()

    console.print(Panel.fit(
        "[bold]vLLM Model Parameter Sweep[/bold]\n"
        "Testing model behavior across parameter combinations",
        border_style="blue",
    ))

    timeout = httpx.Timeout(120.0, connect=5.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        # Check server
        console.print(f"\n[bold]Target:[/bold] {BASE_URL}")
        try:
            await client.get(f"{BASE_URL}/health")
            console.print("[green]Server reachable[/green]\n")
        except httpx.ConnectError:
            console.print("[red bold]Server unreachable[/red bold] — start vLLM first")
            return

        # Discover model
        model = args.model or await discover_model(client)
        if not model:
            console.print("[red]Could not discover model. Pass --model explicitly.[/red]")
            return
        console.print(f"[bold]Model:[/bold] {model}\n")

        # Run all sweeps
        sweeps = [
            ("Temperature", sweep_temperature),
            ("Top-K", sweep_top_k),
            ("Top-P", sweep_top_p),
            ("Min-P", sweep_min_p),
            ("Repetition Penalty", sweep_repetition_penalty),
            ("Structured Output", sweep_structured_output),
            ("Streaming", sweep_streaming),
            ("Stop Sequences", sweep_stop_sequences),
            ("Seed Reproducibility", sweep_seed_reproducibility),
        ]

        all_results: dict[str, list[SweepResult]] = {}
        total_requests = 0

        for sweep_name, sweep_fn in sweeps:
            console.print(f"[bold cyan]Running:[/bold cyan] {sweep_name} sweep...")
            results = await sweep_fn(client, model)
            all_results[sweep_name] = results
            total_requests += len(results)
            print_sweep_table(sweep_name, results)

        # Save
        out_path = save_results(model, all_results)

        # Summary
        ok = sum(1 for rs in all_results.values() for r in rs if r.status_code and 200 <= r.status_code < 300)
        err = total_requests - ok
        console.print(f"\n[bold]Summary:[/bold] {total_requests} requests — {ok} OK, {err} errors/skipped")
        console.print(f"[bold]Results saved to:[/bold] {out_path}\n")


if __name__ == "__main__":
    asyncio.run(main())
