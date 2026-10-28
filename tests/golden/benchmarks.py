"""Compare benchmark results with a baseline and report changes above a threshold.

    uv run python tests/golden/benchmarks.py benchmark.json [--baseline FILE] [--threshold 5]

``benchmark.json`` comes from ``pytest -m benchmark --benchmark-json=benchmark.json``.
The baseline is another such file (an earlier run on the same machine), or by default
``runtime_baseline.json``: the runtime of the generator at qgis_dhydamo 9a324bc,
recorded with the golden masters and only comparable on that machine. The best
(minimum) time per scenario is compared, as the legacy baseline records the best of
its runs.

Timings vary a lot between moments on a laptop (up to a factor 2 was seen), so for the
5% rule measure before and after a change back to back on the same machine.

The report is a markdown table on stdout, also appended to ``$GITHUB_STEP_SUMMARY``
when that is set; in GitHub Actions every scenario slower than the threshold gets a
warning annotation. The exit code is 0 unless ``--fail`` is given and a scenario is
slower than the threshold.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

RUNTIME_BASELINE = Path(__file__).parent / "runtime_baseline.json"


def best_times(path: Path) -> tuple[dict[str, float], str]:
    """Best time per scenario and a description of the machine, from either format."""
    data = json.loads(path.read_text())
    if "benchmarks" in data:  # pytest-benchmark
        times = {
            b["extra_info"].get("scenario", b["name"]): float(b["stats"]["min"])
            for b in data["benchmarks"]
        }
        machine = data.get("machine_info", {})
        cpu = machine.get("cpu", {}).get("brand_raw", machine.get("processor", "?"))
        return (
            times,
            f"{machine.get('system', '?')}, {cpu}, Python {machine.get('python_version', '?')}",
        )
    times = {name: float(s["seconds"]) for name, s in data["scenarios"].items()}
    machine = data.get("machine", {})
    return times, (
        f"{data.get('generator', 'legacy generator')} on {machine.get('platform', '?')}, "
        f"{machine.get('processor', '?')}, Python {machine.get('python', '?')}"
    )


def compare(current: dict[str, float], baseline: dict[str, float], threshold: float):
    rows = []
    for name in sorted(current):
        now = current[name]
        before = baseline.get(name)
        change = None if before is None else (now - before) / before * 100.0
        slower = change is not None and change > threshold
        rows.append((name, before, now, change, slower))
    return rows


def report(rows, threshold: float, current_machine: str, baseline_machine: str) -> str:
    lines = [
        f"Benchmarks (best time; threshold +{threshold:g}%)",
        "",
        f"- now: {current_machine}",
        f"- baseline: {baseline_machine}",
        "",
        "| scenario | baseline (s) | now (s) | change | |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for name, before, now, change, slower in rows:
        before_text = "-" if before is None else f"{before:.3f}"
        change_text = "-" if change is None else f"{change:+.1f}%"
        flag = "slower" if slower else ""
        lines.append(f"| {name} | {before_text} | {now:.3f} | {change_text} | {flag} |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results", type=Path, help="pytest-benchmark JSON of this run")
    parser.add_argument("--baseline", type=Path, default=RUNTIME_BASELINE)
    parser.add_argument("--threshold", type=float, default=5.0, help="percent (default 5)")
    parser.add_argument("--fail", action="store_true", help="exit 1 when a scenario is slower")
    args = parser.parse_args(argv)

    current, current_machine = best_times(args.results)
    baseline, baseline_machine = best_times(args.baseline)
    rows = compare(current, baseline, args.threshold)
    text = report(rows, args.threshold, current_machine, baseline_machine)
    print(text)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write(text)
    slower = [r for r in rows if r[4]]
    if os.environ.get("GITHUB_ACTIONS"):
        for name, before, now, change, _ in slower:
            print(
                f"::warning title=Benchmark {name}::{change:+.1f}% "
                f"({before:.3f} s -> {now:.3f} s), above +{args.threshold:g}%"
            )
    return 1 if (args.fail and slower) else 0


if __name__ == "__main__":
    sys.exit(main())
