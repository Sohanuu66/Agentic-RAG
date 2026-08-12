"""
backend/evaluation/check_regression.py
----------------------------------------
Regression gating script for the CI pipeline.

Compares current RAGAS evaluation results against a stored baseline.
Exits with code 1 (and prints a diff table) if any metric has degraded
beyond the allowed threshold.

Usage
-----
    # From the backend/ directory:
    python -m evaluation.check_regression \\
        --results  evaluation/results/eval_20240101T120000Z.json \\
        --baseline evaluation/baselines/baseline.json \\
        --threshold 0.02

    # Shorter form — reads paths from .env defaults:
    python -m evaluation.check_regression --results path/to/results.json

Exit codes
----------
    0  All metrics are within threshold (or improved).
    1  At least one metric regressed beyond the threshold.
    2  Input error (missing file, bad JSON, missing metric).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

METRICS = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
]

DEFAULT_THRESHOLD = 0.02  # 2 percentage-point regression allowed


# ---------------------------------------------------------------------------
# Core comparison logic
# ---------------------------------------------------------------------------


def load_scores(path: Path) -> Dict[str, float]:
    """
    Load aggregate_scores from a result or baseline JSON file.

    Accepts files produced by evaluate.py (which have a top-level
    ``aggregate_scores`` key) as well as bare ``{metric: value}`` dicts.
    """
    if not path.exists():
        print(f"[ERROR] File not found: {path}", file=sys.stderr)
        sys.exit(2)

    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)

    # Support both formats
    if "aggregate_scores" in data:
        return data["aggregate_scores"]
    # Bare dict (e.g., a hand-crafted baseline or ablation output)
    return data


def compare_metrics(
    current: Dict[str, float],
    baseline: Dict[str, float],
    threshold: float,
    metrics: List[str] = METRICS,
) -> Tuple[List[dict], bool]:
    """
    Compare each metric in *current* against *baseline*.

    Returns
    -------
    rows : list of dicts with keys metric, baseline, current, delta, status
    regression : True if at least one metric regressed beyond threshold
    """
    rows = []
    regression = False

    for metric in metrics:
        base_val = baseline.get(metric)
        curr_val = current.get(metric)

        # Skip metrics that are missing from either file
        if base_val is None or curr_val is None:
            rows.append(
                {
                    "metric": metric,
                    "baseline": "N/A",
                    "current": "N/A",
                    "delta": "N/A",
                    "status": "SKIP (missing)",
                }
            )
            continue

        if isinstance(base_val, str) or isinstance(curr_val, str):
            rows.append(
                {
                    "metric": metric,
                    "baseline": str(base_val),
                    "current": str(curr_val),
                    "delta": "N/A",
                    "status": "SKIP (non-numeric)",
                }
            )
            continue

        # Treat -1.0 as stub values — skip
        if base_val < 0 or curr_val < 0:
            rows.append(
                {
                    "metric": metric,
                    "baseline": f"{base_val:.4f}",
                    "current": f"{curr_val:.4f}",
                    "delta": "N/A",
                    "status": "SKIP (stub -1)",
                }
            )
            continue

        delta = curr_val - base_val  # positive = improvement

        if delta < -threshold:
            status = f"REGRESSION  (dropped {abs(delta):.4f} > {threshold})"
            regression = True
        elif delta < 0:
            status = f"WITHIN THRESHOLD  (dropped {abs(delta):.4f} <= {threshold})"
        elif delta == 0:
            status = "NO CHANGE"
        else:
            status = f"IMPROVED  (+{delta:.4f})"

        rows.append(
            {
                "metric": metric,
                "baseline": f"{base_val:.4f}",
                "current": f"{curr_val:.4f}",
                "delta": f"{delta:+.4f}",
                "status": status,
            }
        )

    return rows, regression


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def format_table(rows: List[dict], threshold: float) -> str:
    """Render a nicely aligned plain-text table (also valid GitHub Markdown)."""
    col_widths = {
        "metric":   max(len("Metric"),   max(len(r["metric"]) for r in rows)),
        "baseline": max(len("Baseline"), max(len(str(r["baseline"])) for r in rows)),
        "current":  max(len("Current"),  max(len(str(r["current"])) for r in rows)),
        "delta":    max(len("Delta"),    max(len(str(r["delta"])) for r in rows)),
        "status":   max(len("Status"),   max(len(r["status"]) for r in rows)),
    }

    def row_line(metric, baseline, current, delta, status):
        return (
            f"| {metric:<{col_widths['metric']}} "
            f"| {baseline:>{col_widths['baseline']}} "
            f"| {current:>{col_widths['current']}} "
            f"| {delta:>{col_widths['delta']}} "
            f"| {status:<{col_widths['status']}} |"
        )

    sep_line = (
        f"|{'-' * (col_widths['metric'] + 2)}"
        f"|{'-' * (col_widths['baseline'] + 2)}"
        f"|{'-' * (col_widths['current'] + 2)}"
        f"|{'-' * (col_widths['delta'] + 2)}"
        f"|{'-' * (col_widths['status'] + 2)}|"
    )

    lines = [
        f"\nRAGAS Regression Check  (threshold = {threshold})",
        "=" * (sum(col_widths.values()) + 16),
        row_line("Metric", "Baseline", "Current", "Delta", "Status"),
        sep_line,
    ]
    for r in rows:
        lines.append(row_line(r["metric"], r["baseline"], r["current"], r["delta"], r["status"]))

    return "\n".join(lines)


def format_github_summary(rows: List[dict], threshold: float, regression: bool) -> str:
    """Render a GitHub Actions step summary / PR comment (Markdown)."""
    status_emoji = "REGRESSION DETECTED" if regression else "ALL METRICS PASSING"
    lines = [
        f"## RAGAS Regression Check - {status_emoji}",
        f"",
        f"**Threshold:** `{threshold}` (metrics may not drop by more than this amount)",
        f"",
        "| Metric | Baseline | Current | Delta | Status |",
        "|--------|----------|---------|-------|--------|",
    ]
    for r in rows:
        lines.append(
            f"| {r['metric']} | {r['baseline']} | {r['current']} | {r['delta']} | {r['status']} |"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m evaluation.check_regression",
        description=(
            "Compare RAGAS evaluation results against a baseline. "
            "Exits with code 1 if any metric regressed beyond the threshold."
        ),
    )
    parser.add_argument(
        "--results",
        required=True,
        help="Path to the current evaluation result JSON (output of evaluate.py)",
    )
    parser.add_argument(
        "--baseline",
        default=None,
        help=(
            "Path to the baseline JSON (default: reads baseline_path from .env). "
            "Accepts a full evaluate.py result or a bare {metric: value} dict."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            f"Maximum allowed metric drop before flagging a regression "
            f"(default: {DEFAULT_THRESHOLD}, or EVAL_REGRESSION_THRESHOLD from .env)"
        ),
    )
    parser.add_argument(
        "--output-markdown",
        default=None,
        metavar="PATH",
        help="Write a GitHub-Markdown summary to this file (for Actions step summary).",
    )
    parser.add_argument(
        "--github-summary",
        action="store_true",
        help="Print GitHub Markdown table to stdout (useful for Actions step summary).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)

    # ------------------------------------------------------------------
    # Resolve defaults from .env / settings
    # ------------------------------------------------------------------
    try:
        _backend_dir = Path(__file__).parent.parent
        if str(_backend_dir) not in sys.path:
            sys.path.insert(0, str(_backend_dir))
        from app.config import settings  # type: ignore[import]
        default_baseline = settings.baseline_path
        default_threshold = settings.eval_regression_threshold
    except Exception:
        default_baseline = "./evaluation/baselines/baseline.json"
        default_threshold = DEFAULT_THRESHOLD

    baseline_path = Path(args.baseline or default_baseline)
    threshold = args.threshold if args.threshold is not None else default_threshold
    results_path = Path(args.results)

    # ------------------------------------------------------------------
    # Load scores
    # ------------------------------------------------------------------
    print(f"Loading results  : {results_path}")
    print(f"Loading baseline : {baseline_path}")
    print(f"Threshold        : {threshold}")
    print()

    current_scores = load_scores(results_path)
    baseline_scores = load_scores(baseline_path)

    # ------------------------------------------------------------------
    # Compare
    # ------------------------------------------------------------------
    rows, regression = compare_metrics(current_scores, baseline_scores, threshold)

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    table = format_table(rows, threshold)
    print(table)
    print()

    if args.github_summary:
        print(format_github_summary(rows, threshold, regression))

    if args.output_markdown:
        md_path = Path(args.output_markdown)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        with open(md_path, "w", encoding="utf-8") as fh:
            fh.write(format_github_summary(rows, threshold, regression))
        print(f"Markdown summary written to: {md_path}")

    if regression:
        print("REGRESSION DETECTED -- exiting with code 1.", file=sys.stderr)
        sys.exit(1)
    else:
        print("All metrics within threshold -- no regression.")
        sys.exit(0)


if __name__ == "__main__":
    main()
