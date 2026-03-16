"""
benchmark.py — TTFT and tokens/sec benchmark for a vLLM model.

Measures:
  - Time to First Token (TTFT) via streaming
  - End-to-end latency
  - Output tokens/sec (generation throughput)
  - Prompt tokens/sec (prefill throughput)

Runs multiple iterations per prompt length to get stable measurements.
Tests short, medium, and long prompts to show how TTFT scales with input.

Outputs:
  - Rich tables to stdout with percentile breakdowns
  - data/benchmark_{model_slug}.json with raw timings

Usage:
  python scripts/benchmark.py
  python scripts/benchmark.py --model google/gemma-3-1b-it
  python scripts/benchmark.py --iterations 10
"""

import argparse
import asyncio
import json
import os
import statistics
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

# Prompts of varying length to test TTFT scaling with input size
BENCHMARK_PROMPTS = {
    "short": {
        "messages": [
            {"role": "user", "content": "What is vLLM?"},
        ],
        "description": "Short prompt (~10 tokens)",
    },
    "medium": {
        "messages": [
            {"role": "system", "content": "You are a technical writer who explains complex systems clearly and concisely."},
            {"role": "user", "content": (
                "Explain the difference between PagedAttention and standard attention mechanisms "
                "in transformer-based language models. Cover memory efficiency, throughput implications, "
                "and why this matters for production inference serving."
            )},
        ],
        "description": "Medium prompt (~60 tokens)",
    },
    "long": {
        "messages": [
            {"role": "system", "content": (
                "You are a senior systems architect reviewing infrastructure decisions. "
                "You provide detailed, actionable analysis with concrete recommendations."
            )},
            {"role": "user", "content": (
                "We are building a multi-tenant LLM inference platform on Kubernetes. The platform needs to "
                "serve multiple models (7B-70B parameters) with automatic scaling based on request queue depth "
                "and GPU memory utilization. We're evaluating vLLM as the inference engine, with Prometheus "
                "for metrics collection and a custom HPA controller for autoscaling decisions.\n\n"
                "Key requirements:\n"
                "1. Cold start time under 60 seconds for 7B models\n"
                "2. P99 TTFT under 500ms at 50 concurrent requests\n"
                "3. GPU utilization above 80% during peak hours\n"
                "4. Graceful scale-down without dropping in-flight requests\n"
                "5. Support for LoRA adapter hot-swapping per tenant\n\n"
                "What are the critical architecture decisions we need to make, and what are the main risks "
                "with this approach? Provide specific vLLM configuration recommendations."
            )},
        ],
        "description": "Long prompt (~200 tokens)",
    },
}


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkRun:
    prompt_category: str
    iteration: int
    ttft_ms: float | None = None
    e2e_latency_ms: float | None = None
    output_tokens: int | None = None
    prompt_tokens: int | None = None
    generation_toks_per_sec: float | None = None
    prefill_toks_per_sec: float | None = None
    finish_reason: str = ""
    error: str | None = None


@dataclass
class BenchmarkSummary:
    prompt_category: str
    description: str
    iterations: int
    ttft_p50_ms: float | None = None
    ttft_p95_ms: float | None = None
    ttft_p99_ms: float | None = None
    ttft_mean_ms: float | None = None
    e2e_p50_ms: float | None = None
    e2e_p95_ms: float | None = None
    e2e_mean_ms: float | None = None
    gen_tps_mean: float | None = None
    gen_tps_p50: float | None = None
    prefill_tps_mean: float | None = None
    avg_output_tokens: float | None = None
    errors: int = 0


# ---------------------------------------------------------------------------
# Benchmark logic
# ---------------------------------------------------------------------------

async def benchmark_single(
    client: httpx.AsyncClient,
    model: str,
    messages: list[dict],
    prompt_category: str,
    iteration: int,
) -> BenchmarkRun:
    """Run a single benchmark iteration using streaming to measure TTFT."""
    run = BenchmarkRun(prompt_category=prompt_category, iteration=iteration)

    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "stream": True,
        "stream_include_usage": True,
    }

    try:
        output_text = ""
        first_token_time = None
        usage = {}
        finish_reason = ""

        start = time.perf_counter()

        async with client.stream("POST", CHAT_ENDPOINT, json=payload) as resp:
            if resp.status_code != 200:
                body = await resp.aread()
                run.error = f"HTTP {resp.status_code}: {body.decode()[:200]}"
                return run

            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str.strip() == "[DONE]":
                    break

                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                delta = chunk.get("choices", [{}])[0].get("delta", {})
                content = delta.get("content", "")
                if content and first_token_time is None:
                    first_token_time = time.perf_counter()

                if content:
                    output_text += content

                fr = chunk.get("choices", [{}])[0].get("finish_reason")
                if fr:
                    finish_reason = fr

                if "usage" in chunk and chunk["usage"]:
                    usage = chunk["usage"]

        end = time.perf_counter()

        # Calculate metrics
        run.e2e_latency_ms = round((end - start) * 1000, 1)
        run.finish_reason = finish_reason

        if first_token_time:
            run.ttft_ms = round((first_token_time - start) * 1000, 1)

        run.prompt_tokens = usage.get("prompt_tokens")
        run.output_tokens = usage.get("completion_tokens")

        # Tokens/sec calculations
        if run.output_tokens and run.ttft_ms and run.e2e_latency_ms:
            generation_time_s = (run.e2e_latency_ms - run.ttft_ms) / 1000
            if generation_time_s > 0:
                run.generation_toks_per_sec = round(run.output_tokens / generation_time_s, 1)

        if run.prompt_tokens and run.ttft_ms:
            prefill_time_s = run.ttft_ms / 1000
            if prefill_time_s > 0:
                run.prefill_toks_per_sec = round(run.prompt_tokens / prefill_time_s, 1)

    except httpx.ConnectError:
        run.error = "connection_refused"
    except httpx.TimeoutException:
        run.error = "timeout"
    except Exception as e:
        run.error = f"{type(e).__name__}: {e}"

    return run


def percentile(data: list[float], p: float) -> float:
    """Calculate percentile from a sorted list."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * (p / 100)
    f = int(k)
    c = f + 1
    if c >= len(sorted_data):
        return sorted_data[f]
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


def summarize_runs(
    prompt_category: str,
    description: str,
    runs: list[BenchmarkRun],
) -> BenchmarkSummary:
    """Compute summary statistics from a list of runs."""
    summary = BenchmarkSummary(
        prompt_category=prompt_category,
        description=description,
        iterations=len(runs),
    )

    ok_runs = [r for r in runs if r.error is None]
    summary.errors = len(runs) - len(ok_runs)

    ttfts = [r.ttft_ms for r in ok_runs if r.ttft_ms is not None]
    e2es = [r.e2e_latency_ms for r in ok_runs if r.e2e_latency_ms is not None]
    gen_tps = [r.generation_toks_per_sec for r in ok_runs if r.generation_toks_per_sec is not None]
    prefill_tps = [r.prefill_toks_per_sec for r in ok_runs if r.prefill_toks_per_sec is not None]
    out_tokens = [r.output_tokens for r in ok_runs if r.output_tokens is not None]

    if ttfts:
        summary.ttft_mean_ms = round(statistics.mean(ttfts), 1)
        summary.ttft_p50_ms = round(percentile(ttfts, 50), 1)
        summary.ttft_p95_ms = round(percentile(ttfts, 95), 1)
        summary.ttft_p99_ms = round(percentile(ttfts, 99), 1)

    if e2es:
        summary.e2e_mean_ms = round(statistics.mean(e2es), 1)
        summary.e2e_p50_ms = round(percentile(e2es, 50), 1)
        summary.e2e_p95_ms = round(percentile(e2es, 95), 1)

    if gen_tps:
        summary.gen_tps_mean = round(statistics.mean(gen_tps), 1)
        summary.gen_tps_p50 = round(percentile(gen_tps, 50), 1)

    if prefill_tps:
        summary.prefill_tps_mean = round(statistics.mean(prefill_tps), 1)

    if out_tokens:
        summary.avg_output_tokens = round(statistics.mean(out_tokens), 1)

    return summary


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
# Output
# ---------------------------------------------------------------------------

def print_run_table(prompt_category: str, runs: list[BenchmarkRun]) -> None:
    """Print detailed per-iteration results."""
    table = Table(title=f"\n{prompt_category} — Per-iteration Results", show_lines=False)
    table.add_column("#", justify="right", style="dim")
    table.add_column("TTFT (ms)", justify="right", style="cyan")
    table.add_column("E2E (ms)", justify="right")
    table.add_column("Out Tokens", justify="right")
    table.add_column("Gen tok/s", justify="right", style="green")
    table.add_column("Prefill tok/s", justify="right", style="blue")
    table.add_column("Finish")
    table.add_column("Error", style="red")

    for r in runs:
        table.add_row(
            str(r.iteration),
            f"{r.ttft_ms:.0f}" if r.ttft_ms else "—",
            f"{r.e2e_latency_ms:.0f}" if r.e2e_latency_ms else "—",
            str(r.output_tokens) if r.output_tokens else "—",
            f"{r.generation_toks_per_sec:.1f}" if r.generation_toks_per_sec else "—",
            f"{r.prefill_toks_per_sec:.1f}" if r.prefill_toks_per_sec else "—",
            r.finish_reason,
            r.error or "",
        )

    console.print(table)


def print_summary_table(summaries: list[BenchmarkSummary]) -> None:
    """Print the final summary comparison table."""
    table = Table(title="\nBenchmark Summary", show_lines=True)
    table.add_column("Prompt", style="bold")
    table.add_column("Iters", justify="right")
    table.add_column("TTFT p50", justify="right", style="cyan")
    table.add_column("TTFT p95", justify="right", style="cyan")
    table.add_column("TTFT p99", justify="right", style="cyan")
    table.add_column("E2E p50", justify="right")
    table.add_column("E2E p95", justify="right")
    table.add_column("Gen tok/s", justify="right", style="green")
    table.add_column("Prefill tok/s", justify="right", style="blue")
    table.add_column("Avg Out Tok", justify="right")
    table.add_column("Errors", justify="right")

    for s in summaries:
        table.add_row(
            f"{s.prompt_category}\n[dim]{s.description}[/dim]",
            str(s.iterations),
            f"{s.ttft_p50_ms:.0f}ms" if s.ttft_p50_ms else "—",
            f"{s.ttft_p95_ms:.0f}ms" if s.ttft_p95_ms else "—",
            f"{s.ttft_p99_ms:.0f}ms" if s.ttft_p99_ms else "—",
            f"{s.e2e_p50_ms:.0f}ms" if s.e2e_p50_ms else "—",
            f"{s.e2e_p95_ms:.0f}ms" if s.e2e_p95_ms else "—",
            f"{s.gen_tps_mean:.1f}" if s.gen_tps_mean else "—",
            f"{s.prefill_tps_mean:.1f}" if s.prefill_tps_mean else "—",
            f"{s.avg_output_tokens:.0f}" if s.avg_output_tokens else "—",
            str(s.errors),
        )

    console.print(table)


def save_results(
    model: str,
    all_runs: dict[str, list[BenchmarkRun]],
    summaries: list[BenchmarkSummary],
) -> Path:
    """Save full benchmark data to JSON."""
    slug = model.replace("/", "_").replace("-", "_").lower()
    output = {
        "model": model,
        "base_url": BASE_URL,
        "benchmark_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runs": {k: [asdict(r) for r in v] for k, v in all_runs.items()},
        "summaries": [asdict(s) for s in summaries],
    }

    out_path = DATA_DIR / f"benchmark_{slug}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark TTFT and tokens/sec for a vLLM model")
    parser.add_argument("--model", type=str, default=None, help="Model ID (auto-discovered if omitted)")
    parser.add_argument("--iterations", type=int, default=5, help="Iterations per prompt category (default: 5)")
    args = parser.parse_args()

    console.print(Panel.fit(
        "[bold]vLLM Benchmark[/bold]\n"
        "TTFT + tokens/sec across prompt lengths",
        border_style="blue",
    ))

    timeout = httpx.Timeout(300.0, connect=5.0)
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
        console.print(f"[bold]Model:[/bold] {model}")
        console.print(f"[bold]Iterations:[/bold] {args.iterations} per prompt\n")

        # Warmup
        console.print("[dim]Warmup request...[/dim]")
        await benchmark_single(client, model, BENCHMARK_PROMPTS["short"]["messages"], "warmup", 0)

        # Run benchmarks
        all_runs: dict[str, list[BenchmarkRun]] = {}
        summaries: list[BenchmarkSummary] = []

        for category, prompt_def in BENCHMARK_PROMPTS.items():
            console.print(f"\n[bold cyan]Benchmarking:[/bold cyan] {category} — {prompt_def['description']}")
            runs: list[BenchmarkRun] = []

            for i in range(args.iterations):
                run = await benchmark_single(
                    client, model, prompt_def["messages"], category, i + 1
                )
                runs.append(run)

                # Live progress
                ttft = f"{run.ttft_ms:.0f}ms" if run.ttft_ms else "err"
                tps = f"{run.generation_toks_per_sec:.1f} tok/s" if run.generation_toks_per_sec else "—"
                console.print(f"  [{i+1}/{args.iterations}] TTFT={ttft}  Gen={tps}")

            all_runs[category] = runs
            print_run_table(category, runs)

            summary = summarize_runs(category, prompt_def["description"], runs)
            summaries.append(summary)

        # Final summary
        print_summary_table(summaries)

        # Save
        out_path = save_results(model, all_runs, summaries)
        console.print(f"\n[bold]Results saved to:[/bold] {out_path}\n")


if __name__ == "__main__":
    asyncio.run(main())
