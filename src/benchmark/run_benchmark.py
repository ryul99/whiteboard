#!/usr/bin/env python3
"""
Confidence-Gated Interleaved Reasoning — Benchmark Runner

Compares baseline (single-call) vs interleaved (confidence-gated) performance
using lm-evaluation-harness task data and metrics.

Usage:
    python run_benchmark.py --config config.yaml
    python run_benchmark.py --config config.yaml --tasks gpqa_diamond_zeroshot --limit 20
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from lm_eval.tasks import TaskManager, get_task_dict

from confidence_model import BaselineChatLM, InterleavedChatLM

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("benchmark")


# ──────────────────────────────────────────────
# Task data extraction helpers
# ──────────────────────────────────────────────


def load_task_docs(
    task_name: str, limit: int | None = None, num_fewshot: int = 0
) -> list[dict]:
    """
    Extract evaluation documents (docs) and prompts from an lm_eval task.
    Returns: [{"doc": original_doc, "prompt": str, "target": str, "until": list[str]}, ...]
    """
    task_manager = TaskManager()
    task_dict = get_task_dict([task_name], task_manager)

    task_obj = task_dict[task_name]

    # Access config if ConfigurableTask
    if hasattr(task_obj, "config"):
        cfg = task_obj.config
    else:
        cfg = None

    # Configure fewshot
    if hasattr(task_obj, "set_fewshot_seed"):
        task_obj.set_fewshot_seed(seed=1234)

    # Build dataset
    if hasattr(task_obj, "build_all_requests"):
        pass  # Required in some versions

    # Get test set
    if hasattr(task_obj, "test_docs"):
        docs = list(task_obj.test_docs())
    elif hasattr(task_obj, "eval_docs"):
        docs = list(task_obj.eval_docs())
    elif hasattr(task_obj, "validation_docs"):
        docs = list(task_obj.validation_docs())
    else:
        raise ValueError(f"No evaluation documents found for task {task_name}.")

    if limit:
        docs = docs[:limit]

    results = []
    for doc in docs:
        # Generate prompt
        prompt = task_obj.doc_to_text(doc)
        if isinstance(prompt, list):
            # Chat format
            prompt = "\n".join(
                f"{m.get('role', 'user')}: {m.get('content', '')}" for m in prompt
            )

        # Extract target answer
        target = task_obj.doc_to_target(doc)
        if isinstance(target, list):
            target = target[0] if target else ""
        target = str(target).strip()

        # Build fewshot prompt
        if num_fewshot > 0 and hasattr(task_obj, "fewshot_context"):
            try:
                ctx = task_obj.fewshot_context(doc, num_fewshot)
                if ctx:
                    prompt = ctx
            except Exception:
                pass  # Fall back to zero-shot if fewshot fails

        # Extract stop tokens from generation_kwargs
        until = []
        if cfg and hasattr(cfg, "generation_kwargs") and cfg.generation_kwargs:
            until = cfg.generation_kwargs.get("until", [])
        if not until:
            until = ["\n", "</s>", "<|im_end|>"]

        results.append(
            {
                "doc": doc,
                "prompt": str(prompt),
                "target": target,
                "until": until,
            }
        )

    return results


def extract_answer_label(text: str) -> str:
    """Extract answer label (A, B, C, D, etc.) from response."""
    text = text.strip()

    # Exact label matching: (A), (B), ... or A, B, ...
    patterns = [
        r"\(([A-D])\)",  # (A), (B), (C), (D)
        r"^([A-D])[\s\.\)]",  # A. or A) at start
        r"^([A-D])$",  # just A
        r"answer is\s*\(?([A-D])\)?",  # "answer is (A)" or "answer is A"
        r"([A-D])(?:\s|$)",  # any standalone A-D
    ]
    for pattern in patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return m.group(1).upper()

    return text[:50]  # Fallback: first 50 characters


def check_match(prediction: str, target: str) -> bool:
    """Check if prediction matches target (flexible matching)."""
    pred = prediction.strip().upper()
    tgt = target.strip().upper()

    # Exact match
    if pred == tgt:
        return True

    # Compare after label extraction
    pred_label = extract_answer_label(pred)
    tgt_label = extract_answer_label(tgt)

    if pred_label and tgt_label and pred_label == tgt_label:
        return True

    # Compare after removing parentheses
    pred_clean = pred.replace("(", "").replace(")", "").strip()
    tgt_clean = tgt.replace("(", "").replace(")", "").strip()
    if pred_clean == tgt_clean:
        return True

    return False


# ──────────────────────────────────────────────
# Benchmark runner
# ──────────────────────────────────────────────


def run_single_task(
    task_name: str,
    baseline_lm: BaselineChatLM,
    interleaved_lm: InterleavedChatLM,
    limit: int | None = None,
    num_fewshot: int = 0,
    log_samples: bool = True,
) -> dict:
    """Run baseline vs interleaved comparison for a single task."""
    logger.info(f"{'=' * 60}")
    logger.info(f"Task: {task_name}")
    logger.info(f"{'=' * 60}")

    # Load task data
    logger.info("Loading task data...")
    try:
        task_docs = load_task_docs(task_name, limit=limit, num_fewshot=num_fewshot)
    except Exception as e:
        logger.error(f"Failed to load task {task_name}: {e}")
        return {"task": task_name, "error": str(e)}

    n_samples = len(task_docs)
    logger.info(f"Loaded {n_samples} samples")

    # Store results
    baseline_correct = 0
    interleaved_correct = 0
    sample_logs = []

    for i, item in enumerate(task_docs):
        prompt = item["prompt"]
        target = item["target"]
        until = item["until"]

        if (i + 1) % 10 == 0 or i == 0:
            logger.info(f"  [{i + 1}/{n_samples}] processing...")

        # ---- Baseline ----
        try:
            baseline_answer = baseline_lm.generate(prompt, until=until)
        except Exception as e:
            logger.warning(f"  Baseline error on sample {i}: {e}")
            baseline_answer = ""

        baseline_match = check_match(baseline_answer, target)
        if baseline_match:
            baseline_correct += 1

        # ---- Interleaved ----
        try:
            interleaved_answer = interleaved_lm.generate(prompt, until=until)
        except Exception as e:
            logger.warning(f"  Interleaved error on sample {i}: {e}")
            interleaved_answer = ""

        interleaved_match = check_match(interleaved_answer, target)
        if interleaved_match:
            interleaved_correct += 1

        # Per-sample logging
        if log_samples:
            sample_logs.append(
                {
                    "index": i,
                    "target": target,
                    "baseline_answer": baseline_answer,
                    "baseline_correct": baseline_match,
                    "interleaved_answer": interleaved_answer,
                    "interleaved_correct": interleaved_match,
                    "interleaved_confidence": (
                        interleaved_lm.stats.confidence_values[-1]
                        if interleaved_lm.stats.confidence_values
                        else None
                    ),
                    "interleaved_retries": (
                        interleaved_lm.stats.retry_counts[-1]
                        if interleaved_lm.stats.retry_counts
                        else None
                    ),
                }
            )

    # Compute metrics
    baseline_acc = baseline_correct / n_samples if n_samples else 0
    interleaved_acc = interleaved_correct / n_samples if n_samples else 0
    delta = interleaved_acc - baseline_acc

    result = {
        "task": task_name,
        "n_samples": n_samples,
        "metrics": {
            "baseline": {
                "exact_match": round(baseline_acc, 4),
                "correct": baseline_correct,
                **baseline_lm.stats.summary(),
            },
            "interleaved": {
                "exact_match": round(interleaved_acc, 4),
                "correct": interleaved_correct,
                **interleaved_lm.stats.summary(),
            },
            "delta": {
                "exact_match": round(delta, 4),
                "exact_match_pct": f"{delta * 100:+.1f}%",
            },
        },
    }

    if log_samples:
        result["samples"] = sample_logs

    return result


def print_results_table(results: list[dict]):
    """Print results as a formatted table."""
    try:
        from tabulate import tabulate
    except ImportError:
        tabulate = None

    print("\n" + "=" * 80)
    print("  BENCHMARK RESULTS: Baseline vs Confidence-Gated Interleaved Reasoning")
    print("=" * 80)

    rows = []
    for r in results:
        if "error" in r:
            rows.append([r["task"], "ERROR", "", "", "", ""])
            continue

        m = r["metrics"]
        rows.append(
            [
                r["task"],
                r["n_samples"],
                f"{m['baseline']['exact_match']:.1%}",
                f"{m['interleaved']['exact_match']:.1%}",
                m["delta"]["exact_match_pct"],
                f"{m['interleaved']['avg_calls_per_sample']:.1f}",
            ]
        )

    headers = ["Task", "N", "Baseline", "Interleaved", "Delta", "Avg API calls"]

    if tabulate:
        print(tabulate(rows, headers=headers, tablefmt="rounded_grid"))
    else:
        # Fallback: simple table
        header_str = " | ".join(f"{h:<20}" for h in headers)
        print(header_str)
        print("-" * len(header_str))
        for row in rows:
            print(" | ".join(f"{str(v):<20}" for v in row))

    # Statistics summary
    for r in results:
        if "error" in r:
            continue
        m = r["metrics"]
        print(f"\n--- {r['task']} Detailed Statistics ---")
        print(
            f"  Interleaved avg confidence:  {m['interleaved']['avg_confidence']:.3f}"
        )
        print(f"  Interleaved retry rate:      {m['interleaved']['retry_rate']:.1%}")
        print(f"  Interleaved total API calls: {m['interleaved']['total_api_calls']}")
        print(f"  Baseline total API calls:    {m['baseline']['total_api_calls']}")
        print(
            f"  Token overhead:              "
            f"{m['interleaved']['total_input_tokens'] + m['interleaved']['total_output_tokens']} "
            f"vs {m['baseline']['total_input_tokens'] + m['baseline']['total_output_tokens']}"
        )

    # Correct/incorrect case analysis
    for r in results:
        if "error" in r or "samples" not in r:
            continue
        samples = r["samples"]

        # Cases where Interleaved was correct but Baseline was wrong
        fixed = [
            s for s in samples if s["interleaved_correct"] and not s["baseline_correct"]
        ]
        # Cases where Baseline was correct but Interleaved was wrong
        broken = [
            s for s in samples if not s["interleaved_correct"] and s["baseline_correct"]
        ]

        print(f"\n--- {r['task']} Improvement / Regression Analysis ---")
        print(f"  Additional samples Interleaved got correct: {len(fixed)}")
        print(f"  Additional samples Interleaved got wrong:   {len(broken)}")
        print(f"  Net improvement: {len(fixed) - len(broken)}")

        if fixed:
            print("\n  [Improved samples (up to 3)]")
            for s in fixed[:3]:
                print(
                    f"    #{s['index']}: target={s['target']}, "
                    f"baseline='{s['baseline_answer'][:30]}', "
                    f"interleaved='{s['interleaved_answer'][:30]}' "
                    f"(conf={s['interleaved_confidence']:.2f}, retries={s['interleaved_retries']})"
                )

        if broken:
            print("\n  [Regressed samples (up to 3)]")
            for s in broken[:3]:
                print(
                    f"    #{s['index']}: target={s['target']}, "
                    f"baseline='{s['baseline_answer'][:30]}', "
                    f"interleaved='{s['interleaved_answer'][:30]}' "
                    f"(conf={s['interleaved_confidence']:.2f}, retries={s['interleaved_retries']})"
                )

    print()


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

    # ── Use structured output + system prompt for Baseline as well
    # (Use the same system prompt for a fair comparison)
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

    # Run per task
    all_results = []
    start_time = time.time()

    for task_name in tasks:
        # Reset stats for each task
        baseline_lm.stats = __import__("confidence_model").CallStats()
        interleaved_lm.stats = __import__("confidence_model").CallStats()

        result = run_single_task(
            task_name=task_name,
            baseline_lm=baseline_lm,
            interleaved_lm=interleaved_lm,
            limit=limit,
            num_fewshot=eval_cfg.get("num_fewshot", 0),
            log_samples=eval_cfg.get("log_samples", True),
        )
        all_results.append(result)

    elapsed = time.time() - start_time

    # Print results
    print_results_table(all_results)
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
        "results": all_results,
    }

    # Remove doc from samples (to avoid serialization issues)
    for r in output_data["results"]:
        if "samples" in r:
            for s in r["samples"]:
                s.pop("doc", None)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    logger.info(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
