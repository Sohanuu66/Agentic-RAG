"""
backend.evaluation — RAGAS-based evaluation and ablation experiments

Modules:
    evaluate         → run full pipeline on golden_qa.json, compute RAGAS metrics
    ablation         → sweep chunking/retrieval configs, compare metric tables
    check_regression → compare current results against baseline.json, exit 1 if regression
"""
