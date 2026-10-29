"""Command-line benchmark runner for Instruction Memory Playground.

Runs an evaluation grid across:
  (models x strategies x configs x variants)
Features:
- Incremental JSONL streaming to results/evaluations.jsonl
- Aggregate CSV summary generation to results/summary.csv
- Resume support (--resume) to continue interrupted runs
- Quick run limiter (--limit N)
- Diagnostic alerts if pass rates saturate near 0% or 100%
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

from checker import score
from generator import SessionConfig, SessionSample, generate_session
from models import get_model
from strategies import get_strategy
from tasks import TASKS, get_all_tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Instruction Memory Playground benchmark grid.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["mock"],
        help="Model specs to evaluate (e.g. mock, openai/gpt-4o-mini, anthropic/claude-3-5-haiku-20241022, ollama/llama3.2).",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["full_context", "restate", "notes", "summarize"],
        help="Inference strategies to evaluate.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["primary", "zero_gap", "inline"],
        help="Variants to evaluate: primary, zero_gap, inline.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["all"],
        help="Task IDs to evaluate or 'all'.",
    )
    parser.add_argument(
        "--initial-rules",
        nargs="+",
        type=int,
        default=[2, 3],
        help="List of initial rule counts (1..5).",
    )
    parser.add_argument(
        "--updates",
        nargs="+",
        type=int,
        default=[0, 1],
        help="List of rule update counts (0..3).",
    )
    parser.add_argument(
        "--gaps",
        nargs="+",
        type=int,
        default=[0, 3, 10],
        help="List of distractor gap lengths (0..40).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic generation.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum total sample evaluations to execute.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save evaluations.jsonl and summary.csv.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume previous run by skipping completed (model, strategy, sample_id) records.",
    )
    return parser.parse_args()


def load_completed_keys(jsonl_path: Path) -> Set[Tuple[str, str, str]]:
    """Read existing (model, strategy, sample_id) keys from JSONL file."""
    completed: Set[Tuple[str, str, str]] = set()
    if not jsonl_path.exists():
        return completed
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                completed.add((rec["model"], rec["strategy"], rec["sample_id"]))
            except Exception:
                continue
    return completed


def compute_and_save_summary(jsonl_path: Path, csv_path: Path) -> pd.DataFrame:
    """Aggregate evaluations into a summary dataframe and write to CSV."""
    records: List[Dict[str, Any]] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass

    if not records:
        empty_df = pd.DataFrame()
        empty_df.to_csv(csv_path, index=False)
        return empty_df

    df = pd.DataFrame(records)

    # Flatten config fields if present
    if "config" in df.columns:
        df["gap_length"] = df["config"].apply(lambda c: c.get("gap_length", 0) if isinstance(c, dict) else 0)
        df["num_initial_rules"] = df["config"].apply(lambda c: c.get("num_initial_rules", 0) if isinstance(c, dict) else 0)
        df["num_updates"] = df["config"].apply(lambda c: c.get("num_updates", 0) if isinstance(c, dict) else 0)

    df["has_superseded_violation"] = df["superseded_violations"].apply(lambda v: 1 if v and len(v) > 0 else 0)
    df["has_forgot"] = df["forgot"].apply(lambda v: 1 if v and len(v) > 0 else 0)

    group_cols = ["model", "strategy", "variant"]
    agg_dict = {
        "correctness": ["count", "mean"],
        "adherence": ["mean"],
        "total": ["mean"],
        "has_superseded_violation": ["mean"],
        "has_forgot": ["mean"],
        "latency": ["mean"],
    }

    summary = df.groupby(group_cols).agg(agg_dict).reset_index()
    # Flatten multi-level columns
    summary.columns = [
        f"{c[0]}_{c[1]}" if c[1] else c[0]
        for c in summary.columns
    ]
    summary = summary.rename(
        columns={
            "correctness_count": "count",
            "correctness_mean": "mean_correctness",
            "adherence_mean": "mean_adherence",
            "total_mean": "mean_total",
            "has_superseded_violation_mean": "superseded_rate",
            "has_forgot_mean": "forgot_rate",
            "latency_mean": "avg_latency_s",
        }
    )

    summary.to_csv(csv_path, index=False)
    return summary


def run_benchmark(args: argparse.Namespace):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "evaluations.jsonl"
    csv_path = output_dir / "summary.csv"

    completed_keys = load_completed_keys(jsonl_path) if args.resume else set()
    if completed_keys:
        print(f"[INFO] Resuming run. Found {len(completed_keys)} previously completed evaluations.")

    # Determine tasks
    all_tasks = get_all_tasks()
    if "all" in args.tasks:
        selected_tasks = all_tasks
    else:
        selected_tasks = [t for t in all_tasks if t.id in args.tasks]

    # Instantiate model backends
    models: Dict[str, Any] = {m_spec: get_model(m_spec) for m_spec in args.models}

    # Generate benchmark grid configs
    grid_configs: List[SessionConfig] = []
    seed_counter = args.seed
    for task in selected_tasks:
        for n_rules in args.initial_rules:
            for n_updates in args.updates:
                for gap in args.gaps:
                    cfg = SessionConfig(
                        num_initial_rules=n_rules,
                        num_updates=n_updates,
                        gap_length=gap,
                        task_id=task.id,
                        seed=seed_counter,
                    )
                    grid_configs.append(cfg)
                    seed_counter += 1

    print(f"[INFO] Generated {len(grid_configs)} session base configs.")
    print(f"[INFO] Models: {args.models} | Strategies: {args.strategies} | Variants: {args.variants}")

    eval_count = 0
    new_evals = 0

    with open(jsonl_path, "a" if args.resume else "w", encoding="utf-8") as f_out:
        for cfg in grid_configs:
            primary, zero_gap, inline = generate_session(cfg)
            samples_by_variant = {
                "primary": primary,
                "zero_gap": zero_gap,
                "inline": inline,
            }

            for v_name in args.variants:
                if v_name not in samples_by_variant:
                    continue
                sample = samples_by_variant[v_name]

                for strat_name in args.strategies:
                    strat = get_strategy(strat_name)

                    for m_spec in args.models:
                        key = (m_spec, strat_name, sample.sample_id)
                        if key in completed_keys:
                            continue

                        model_backend = models[m_spec]

                        # 1. Prepare messages according to strategy
                        try:
                            prompt_messages = strat.prepare_messages(sample, model=model_backend)
                        except Exception as e:
                            print(f"[WARN] Strategy {strat_name} failed on {sample.sample_id}: {e}")
                            continue

                        # 2. Call model
                        try:
                            resp = model_backend.generate(prompt_messages, temperature=0.0)
                        except Exception as e:
                            print(f"[ERROR] Model call failed for {m_spec}: {e}")
                            continue

                        # 3. Score completion
                        score_res = score(sample, resp.text)

                        eval_record = {
                            "model": m_spec,
                            "strategy": strat_name,
                            "variant": v_name,
                            "sample_id": sample.sample_id,
                            "task_id": sample.task_id,
                            "config": sample.config,
                            "active_rules": sample.active_rules,
                            "superseded_rules": sample.superseded_rules,
                            "correctness": score_res["correctness"],
                            "adherence": score_res["adherence"],
                            "superseded_violations": score_res["superseded_violations"],
                            "forgot": score_res["forgot"],
                            "total": score_res["total"],
                            "tests_passed": score_res["tests_passed"],
                            "tests_total": score_res["tests_total"],
                            "latency": resp.latency,
                            "cached": resp.cached,
                            "completion": resp.text,
                            "error": score_res["error"],
                        }

                        f_out.write(json.dumps(eval_record, ensure_ascii=False) + "\n")
                        f_out.flush()

                        new_evals += 1
                        eval_count += 1

                        print(
                            f"[{eval_count}] {m_spec} | {strat_name:12s} | {v_name:8s} | "
                            f"{sample.task_id[:16]:16s} | corr: {score_res['correctness']:.2f} | "
                            f"adh: {score_res['adherence']:.2f} | tot: {score_res['total']:.2f} | "
                            f"sup_viols: {len(score_res['superseded_violations'])}"
                        )

                        if args.limit and eval_count >= args.limit:
                            print(f"[INFO] Reached --limit {args.limit}. Stopping execution.")
                            break
                    if args.limit and eval_count >= args.limit:
                        break
                if args.limit and eval_count >= args.limit:
                    break
            if args.limit and eval_count >= args.limit:
                break

    # Compute and display summary
    summary_df = compute_and_save_summary(jsonl_path, csv_path)
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    if not summary_df.empty:
        print(summary_df.to_string(index=False))
        print("=" * 80)

        # Difficulty tuning alerts
        overall_adh = summary_df["mean_adherence"].mean()
        if overall_adh >= 0.98:
            print("[FLAG / WARNING] Overall adherence is near 100% (ceiling effect).")
            print("Consider increasing gap lengths (D > 20) or adding more concurrent rules to increase difficulty.")
        elif overall_adh <= 0.05:
            print("[FLAG / WARNING] Overall adherence is near 0% (floor effect).")
            print("Check model prompting or rule feasibility to prevent excessive difficulty.")
    else:
        print("No evaluations recorded.")


if __name__ == "__main__":
    cli_args = parse_args()
    run_benchmark(cli_args)
