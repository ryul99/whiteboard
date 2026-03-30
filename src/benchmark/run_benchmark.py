#!/usr/bin/env python3
"""
Confidence-Gated Interleaved Reasoning — Benchmark Runner

Compares baseline (single-call) vs interleaved (confidence-gated) performance
using lm-evaluation-harness.

Usage:
    python run_benchmark.py --config config.yaml
    python run_benchmark.py --config config.yaml --tasks gpqa_diamond_zeroshot --limit 20
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

import lm_eval
from lm_eval.tasks import TaskManager, get_task_dict

from confidence_model import BaselineChatLM, InterleavedChatLM, CallStats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("benchmark")


# ──────────────────────────────────────────────
# Evaluation helpers
# ──────────────────────────────────────────────


def run_eval(
    lm,
    tasks: list[str],
    num_fewshot: int = 0,
    limit: int | None = None,
    log_samples: bool = True,
) -> dict:
    """Run evaluation using lm_eval.evaluate()."""
    task_manager = TaskManager()
    task_dict = get_task_dict(tasks, task_manager)

    # Override num_fewshot on each task config if non-default
    if num_fewshot != 0:
        for task in task_dict.values():
            task.config.num_fewshot = num_fewshot

    results = lm_eval.evaluate(
        lm=lm,
        task_dict=task_dict,
        limit=limit,
        log_samples=log_samples,
    )

    return results


# ──────────────────────────────────────────────
# Results display
# ──────────────────────────────────────────────


def _pick_primary_metric(metrics: dict) -> str | None:
    # lm_eval uses "metric,filter" keys (e.g. "acc_norm,none")
    for prefix in ("acc_norm", "acc", "exact_match"):
        for key in metrics:
            if key.split(",")[0] == prefix:
                return key
    return next(iter(metrics)) if metrics else None


def print_results_table(
    baseline_eval: dict,
    interleaved_eval: dict,
    baseline_lm,
    interleaved_lm,
    tasks: list[str],
):
    try:
        from tabulate import tabulate
    except ImportError:
        tabulate = None

    print("\n" + "=" * 80)
    print("  BENCHMARK RESULTS: Baseline vs Confidence-Gated Interleaved Reasoning")
    print("=" * 80)

    # ── Summary table ──
    rows = []
    for task_name in tasks:
        b_metrics = baseline_eval["results"].get(task_name, {})
        i_metrics = interleaved_eval["results"].get(task_name, {})

        metric_key = _pick_primary_metric(b_metrics)
        if metric_key is None:
            rows.append([task_name, "NO METRICS", "", "", "", ""])
            continue

        b_val = float(b_metrics.get(metric_key, 0))
        i_val = float(i_metrics.get(metric_key, 0))
        delta = i_val - b_val

        n_samples_raw = baseline_eval.get("n-samples", {}).get(task_name, "?")
        n_samples = (
            n_samples_raw.get("effective", n_samples_raw)
            if isinstance(n_samples_raw, dict)
            else n_samples_raw
        )

        rows.append(
            [
                task_name,
                n_samples,
                f"{b_val:.1%}",
                f"{i_val:.1%}",
                f"{delta * 100:+.1f}%",
                metric_key,
            ]
        )

    headers = ["Task", "N", "Baseline", "Interleaved", "Delta", "Metric"]

    if tabulate:
        print(tabulate(rows, headers=headers, tablefmt="rounded_grid"))
    else:
        header_str = " | ".join(f"{h:<20}" for h in headers)
        print(header_str)
        print("-" * len(header_str))
        for row in rows:
            print(" | ".join(f"{str(v):<20}" for v in row))

    # ── Statistics summary ──
    for task_name in tasks:
        print(f"\n--- {task_name} Detailed Statistics ---")
        print(
            f"  Interleaved avg confidence:  {interleaved_lm.stats.avg_confidence:.3f}"
        )
        print(f"  Interleaved retry rate:      {interleaved_lm.stats.retry_rate:.1%}")
        print(f"  Interleaved total API calls: {interleaved_lm.stats.total_calls}")
        print(f"  Baseline total API calls:    {baseline_lm.stats.total_calls}")
        print(
            f"  Token overhead:              "
            f"{interleaved_lm.stats.total_input_tokens + interleaved_lm.stats.total_output_tokens} "
            f"vs {baseline_lm.stats.total_input_tokens + baseline_lm.stats.total_output_tokens}"
        )

    # ── Per-sample improvement / regression analysis ──
    for task_name in tasks:
        b_samples = baseline_eval.get("samples", {}).get(task_name, [])
        i_samples = interleaved_eval.get("samples", {}).get(task_name, [])

        if not b_samples or not i_samples:
            continue

        # Match by doc_id
        b_by_id = {s["doc_id"]: s for s in b_samples}
        i_by_id = {s["doc_id"]: s for s in i_samples}

        metric_key = _pick_primary_metric(baseline_eval["results"].get(task_name, {}))
        if metric_key is None:
            continue

        # For multiple-choice tasks: check which choice was selected
        # For generate_until tasks: check filtered_resps against target
        fixed = []
        broken = []

        for doc_id in b_by_id:
            if doc_id not in i_by_id:
                continue
            bs = b_by_id[doc_id]
            is_ = i_by_id[doc_id]

            b_correct = _is_correct(bs, metric_key)
            i_correct = _is_correct(is_, metric_key)

            if i_correct and not b_correct:
                fixed.append((doc_id, bs, is_))
            elif not i_correct and b_correct:
                broken.append((doc_id, bs, is_))

        print(f"\n--- {task_name} Improvement / Regression Analysis ---")
        print(f"  Additional samples Interleaved got correct: {len(fixed)}")
        print(f"  Additional samples Interleaved got wrong:   {len(broken)}")
        print(f"  Net improvement: {len(fixed) - len(broken)}")

        if fixed:
            print("\n  [Improved samples (up to 3)]")
            for doc_id, bs, is_ in fixed[:3]:
                b_resp = _format_resp(bs)
                i_resp = _format_resp(is_)
                print(
                    f"    #{doc_id}: target={bs.get('target', '?')}, "
                    f"baseline='{b_resp[:30]}', "
                    f"interleaved='{i_resp[:30]}'"
                )

        if broken:
            print("\n  [Regressed samples (up to 3)]")
            for doc_id, bs, is_ in broken[:3]:
                b_resp = _format_resp(bs)
                i_resp = _format_resp(is_)
                print(
                    f"    #{doc_id}: target={bs.get('target', '?')}, "
                    f"baseline='{b_resp[:30]}', "
                    f"interleaved='{i_resp[:30]}'"
                )

    print()


def _is_correct(sample: dict, metric_key: str) -> bool:
    """Check if a sample was answered correctly."""
    # For multiple-choice / loglikelihood tasks, filtered_resps contains results
    filtered = sample.get("filtered_resps", [])
    if isinstance(filtered, list) and filtered:
        # For mc tasks, filtered_resps[0] is the selected choice index
        return bool(filtered[0])
    # Fallback: check metrics
    metrics = sample.get("metrics", {})
    return metrics.get(metric_key, 0) == 1.0


def _format_resp(sample: dict) -> str:
    resps = sample.get("resps", [])
    if resps and isinstance(resps[0], list):
        return str(resps[0][0]) if resps[0] else ""
    return str(resps[0]) if resps else ""


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Confidence-Gated Interleaved Reasoning Benchmark"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="YAML config file path",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=None,
        help="Override tasks from config",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Override sample limit",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override confidence threshold",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=None,
        help="Override max retries",
    )
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config not found: {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    model_cfg = config["model"]
    interleaved_cfg = config["interleaved"]
    eval_cfg = config["eval"]

    # CLI overrides
    tasks = args.tasks or config["tasks"]
    limit = args.limit if args.limit is not None else eval_cfg.get("limit")
    threshold = args.threshold or interleaved_cfg["confidence_threshold"]
    max_retries = (
        args.max_retries
        if args.max_retries is not None
        else interleaved_cfg["max_retries"]
    )

    # Verify API key
    if not os.environ.get("OPENAI_API_KEY") and not model_cfg.get("api_base"):
        logger.error(
            "Set the OPENAI_API_KEY environment variable or specify api_base in config."
        )
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("Confidence-Gated Interleaved Reasoning Benchmark")
    logger.info("=" * 60)
    logger.info(f"Model:     {model_cfg['model_name']}")
    logger.info(f"Tasks:     {tasks}")
    logger.info(f"Threshold: {threshold}")
    logger.info(f"Retries:   {max_retries}")
    logger.info(f"Limit:     {limit or 'all'}")

    # Initialize models
    common_kwargs = dict(
        model_name=model_cfg["model_name"],
        api_base=model_cfg.get("api_base"),
        temperature=model_cfg.get("temperature", 0.0),
        max_tokens=model_cfg.get("max_tokens", 1024),
    )

    answer_system_prompt = (
        "You are an expert at answering questions. "
        "Read the question carefully and provide ONLY the answer label "
        "(e.g., A, B, C, or D for multiple choice). "
        "Think step by step, then give your final answer."
    )

    baseline_lm = BaselineChatLM(
        **common_kwargs,
        system_prompt=answer_system_prompt,
    )

    interleaved_lm = InterleavedChatLM(
        **common_kwargs,
        system_prompt=answer_system_prompt,
        confidence_threshold=threshold,
        max_retries=max_retries,
        retry_temperature=interleaved_cfg.get("retry_temperature", 0.3),
        confidence_guide=interleaved_cfg.get("confidence_guide", ""),
    )

    # ── Run evaluations ──
    start_time = time.time()

    logger.info("Running baseline evaluation...")
    baseline_eval = run_eval(
        lm=baseline_lm,
        tasks=tasks,
        num_fewshot=eval_cfg.get("num_fewshot", 0),
        limit=limit,
        log_samples=eval_cfg.get("log_samples", True),
    )

    interleaved_lm.stats = CallStats()

    logger.info("Running interleaved evaluation...")
    interleaved_eval = run_eval(
        lm=interleaved_lm,
        tasks=tasks,
        num_fewshot=eval_cfg.get("num_fewshot", 0),
        limit=limit,
        log_samples=eval_cfg.get("log_samples", True),
    )

    elapsed = time.time() - start_time

    # Print results
    print_results_table(
        baseline_eval, interleaved_eval, baseline_lm, interleaved_lm, tasks
    )
    logger.info(f"Total time: {elapsed:.1f}s")

    # Save as JSON
    output_path = args.output or os.path.join(
        eval_cfg.get("output_dir", "./results"),
        f"benchmark_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json",
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    output_data = {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": model_cfg["model_name"],
            "confidence_threshold": threshold,
            "max_retries": max_retries,
            "elapsed_seconds": round(elapsed, 1),
        },
        "baseline_results": baseline_eval["results"],
        "interleaved_results": interleaved_eval["results"],
        "baseline_stats": baseline_lm.stats.summary(),
        "interleaved_stats": interleaved_lm.stats.summary(),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2, default=str)

    logger.info(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
