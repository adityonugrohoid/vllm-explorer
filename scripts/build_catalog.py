"""
build_catalog.py — Aggregate probe, test, and benchmark results into data/catalog.json.

Reads all JSON outputs from data/ and builds a unified model capability catalog:
  - Server info and endpoint availability (from probe_results.json)
  - Parameter support matrix (from test_model_*.json)
  - Performance benchmarks (from benchmark_*.json)

This catalog is the primary deliverable — it feeds into the gpu-autoscale-inference
project for model selection and vLLM configuration decisions.

Usage:
  python scripts/build_catalog.py
"""

import json
import time
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

console = Console()


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_json(path: Path) -> dict | None:
    """Load a JSON file, return None if missing/invalid."""
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        console.print(f"[yellow]Warning: Could not load {path.name}: {e}[/yellow]")
        return None


def find_files(pattern: str) -> list[Path]:
    """Find files matching a glob pattern in data/."""
    return sorted(DATA_DIR.glob(pattern))


# ---------------------------------------------------------------------------
# Catalog builders
# ---------------------------------------------------------------------------

def build_endpoint_catalog(probe_data: dict) -> dict:
    """Build endpoint availability section from probe results."""
    endpoints = {}
    for result in probe_data.get("results", []):
        path = result["endpoint"]
        status = result.get("status_code")
        error = result.get("error")

        if error == "skipped":
            availability = "skipped"
        elif error == "connection_refused":
            availability = "server_down"
        elif error:
            availability = "error"
        elif status and 200 <= status < 300:
            availability = "available"
        elif status and status == 400:
            availability = "rejected"  # endpoint exists but model doesn't support it
        elif status and status == 404:
            availability = "not_found"
        elif status and status == 405:
            availability = "method_not_allowed"
        else:
            availability = "unknown"

        endpoints[path] = {
            "method": result["method"],
            "category": result["category"],
            "status_code": status,
            "availability": availability,
            "latency_ms": result.get("latency_ms"),
            "response_shape": result.get("response_shape"),
            "notes": result.get("notes", ""),
        }

    return {
        "base_url": probe_data.get("base_url"),
        "probe_timestamp": probe_data.get("probe_timestamp"),
        "total_endpoints": len(endpoints),
        "available": sum(1 for e in endpoints.values() if e["availability"] == "available"),
        "rejected": sum(1 for e in endpoints.values() if e["availability"] == "rejected"),
        "endpoints": endpoints,
    }


def build_model_test_catalog(test_data: dict) -> dict:
    """Build parameter support matrix from test results."""
    model = test_data.get("model", "unknown")
    sweeps = test_data.get("sweeps", {})

    param_matrix: dict[str, dict] = {}

    for sweep_name, results in sweeps.items():
        sweep_summary = {
            "total_tests": len(results),
            "passed": 0,
            "failed": 0,
            "values_tested": [],
        }

        for r in results:
            status = r.get("status_code")
            value = r.get("param_value")
            sweep_summary["values_tested"].append(value)

            if status and 200 <= status < 300:
                sweep_summary["passed"] += 1
            else:
                sweep_summary["failed"] += 1

        param_matrix[sweep_name] = sweep_summary

    return {
        "model": model,
        "test_timestamp": test_data.get("test_timestamp"),
        "prompt": test_data.get("prompt"),
        "parameter_matrix": param_matrix,
    }


def build_benchmark_catalog(bench_data: dict) -> dict:
    """Build performance section from benchmark results."""
    return {
        "model": bench_data.get("model", "unknown"),
        "benchmark_timestamp": bench_data.get("benchmark_timestamp"),
        "summaries": bench_data.get("summaries", []),
    }


# ---------------------------------------------------------------------------
# Catalog assembly
# ---------------------------------------------------------------------------

def build_full_catalog() -> dict:
    """Assemble the complete catalog from all available data."""
    catalog: dict = {
        "catalog_version": "0.1",
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "endpoints": None,
        "models": {},
    }

    # Load probe results
    probe_data = load_json(DATA_DIR / "probe_results.json")
    if probe_data:
        catalog["endpoints"] = build_endpoint_catalog(probe_data)
        console.print("[green]Loaded[/green] probe_results.json")
    else:
        console.print("[yellow]Missing[/yellow] probe_results.json — run probe_endpoints.py first")

    # Load test results (one per model)
    test_files = find_files("test_model_*.json")
    for tf in test_files:
        test_data = load_json(tf)
        if test_data:
            model = test_data.get("model", "unknown")
            model_entry = catalog["models"].setdefault(model, {})
            model_entry["parameters"] = build_model_test_catalog(test_data)
            console.print(f"[green]Loaded[/green] {tf.name}")

    # Load benchmark results (one per model)
    bench_files = find_files("benchmark_*.json")
    for bf in bench_files:
        bench_data = load_json(bf)
        if bench_data:
            model = bench_data.get("model", "unknown")
            model_entry = catalog["models"].setdefault(model, {})
            model_entry["benchmarks"] = build_benchmark_catalog(bench_data)
            console.print(f"[green]Loaded[/green] {bf.name}")

    return catalog


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_catalog_overview(catalog: dict) -> None:
    """Print a rich overview of the catalog."""
    tree = Tree("[bold]vLLM Catalog[/bold]")

    # Endpoints
    ep = catalog.get("endpoints")
    if ep:
        ep_branch = tree.add(f"[cyan]Endpoints[/cyan] — {ep['total_endpoints']} total, {ep['available']} available")
        by_cat: dict[str, list] = {}
        for path, info in ep["endpoints"].items():
            by_cat.setdefault(info["category"], []).append((path, info))
        for cat, items in by_cat.items():
            avail = sum(1 for _, i in items if i["availability"] == "available")
            cat_branch = ep_branch.add(f"{cat} ({avail}/{len(items)})")
            for path, info in items:
                icon = "[green]OK[/green]" if info["availability"] == "available" else f"[dim]{info['availability']}[/dim]"
                cat_branch.add(f"{info['method']} {path} — {icon}")
    else:
        tree.add("[yellow]Endpoints — not probed yet[/yellow]")

    # Models
    models = catalog.get("models", {})
    if models:
        models_branch = tree.add(f"[cyan]Models[/cyan] — {len(models)} tested")
        for model_id, data in models.items():
            model_branch = models_branch.add(f"[bold]{model_id}[/bold]")

            if "parameters" in data:
                pm = data["parameters"]["parameter_matrix"]
                total_tests = sum(s["total_tests"] for s in pm.values())
                total_passed = sum(s["passed"] for s in pm.values())
                param_branch = model_branch.add(f"Parameters — {total_passed}/{total_tests} passed")
                for sweep_name, sweep in pm.items():
                    icon = "[green]OK[/green]" if sweep["failed"] == 0 else f"[yellow]{sweep['failed']} failed[/yellow]"
                    param_branch.add(f"{sweep_name}: {icon}")

            if "benchmarks" in data:
                summaries = data["benchmarks"].get("summaries", [])
                bench_branch = model_branch.add("Benchmarks")
                for s in summaries:
                    ttft = f"TTFT p50={s.get('ttft_p50_ms', '—')}ms"
                    tps = f"Gen={s.get('gen_tps_mean', '—')} tok/s"
                    bench_branch.add(f"{s['prompt_category']}: {ttft}, {tps}")
    else:
        tree.add("[yellow]Models — no test/benchmark data yet[/yellow]")

    console.print(tree)


def print_endpoint_table(catalog: dict) -> None:
    """Print endpoint availability table."""
    ep = catalog.get("endpoints")
    if not ep:
        return

    table = Table(title="\nEndpoint Availability", show_lines=True)
    table.add_column("Category", style="bold")
    table.add_column("Method", style="cyan")
    table.add_column("Path")
    table.add_column("Status", justify="center")
    table.add_column("Latency", justify="right")
    table.add_column("Notes")

    for path, info in ep["endpoints"].items():
        avail = info["availability"]
        if avail == "available":
            status_str = "[green]available[/green]"
        elif avail == "rejected":
            status_str = "[yellow]rejected[/yellow]"
        elif avail == "skipped":
            status_str = "[dim]skipped[/dim]"
        else:
            status_str = f"[red]{avail}[/red]"

        latency = f"{info['latency_ms']:.0f}ms" if info.get("latency_ms") else "—"

        table.add_row(info["category"], info["method"], path, status_str, latency, info.get("notes", ""))

    console.print(table)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    console.print(Panel.fit(
        "[bold]vLLM Catalog Builder[/bold]\n"
        "Aggregating probe, test, and benchmark results",
        border_style="blue",
    ))

    console.print(f"\n[bold]Data directory:[/bold] {DATA_DIR}\n")

    # Check what's available
    available_files = list(DATA_DIR.glob("*.json"))
    if not available_files:
        console.print("[red]No data files found.[/red] Run the following scripts first:")
        console.print("  python scripts/probe_endpoints.py")
        console.print("  python scripts/test_model.py")
        console.print("  python scripts/benchmark.py")
        return

    console.print(f"Found {len(available_files)} data file(s):\n")

    # Build catalog
    catalog = build_full_catalog()

    # Display
    console.print()
    print_catalog_overview(catalog)
    print_endpoint_table(catalog)

    # Save
    out_path = DATA_DIR / "catalog.json"
    with open(out_path, "w") as f:
        json.dump(catalog, f, indent=2, default=str)

    console.print(f"\n[bold]Catalog saved to:[/bold] {out_path}")

    # Stats
    models = catalog.get("models", {})
    ep = catalog.get("endpoints")
    ep_count = ep["total_endpoints"] if ep else 0
    console.print(f"[bold]Contents:[/bold] {ep_count} endpoints, {len(models)} model(s)\n")


if __name__ == "__main__":
    main()
