#!/usr/bin/env python3
"""
Confidence-Gated Interleaved Reasoning — Benchmark Runner

lm-evaluation-harness의 태스크 데이터와 메트릭을 활용하여
baseline(단일 호출) vs interleaved(confidence-gated) 성능을 비교합니다.

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
    lm_eval 태스크에서 평가 문서(doc)와 프롬프트를 추출합니다.
    Returns: [{"doc": original_doc, "prompt": str, "target": str, "until": list[str]}, ...]
    """
    task_manager = TaskManager()
    task_dict = get_task_dict([task_name], task_manager)

    task_obj = task_dict[task_name]

    # ConfigurableTask인 경우 config 접근
    if hasattr(task_obj, "config"):
        cfg = task_obj.config
    else:
        cfg = None

    # fewshot 설정
    if hasattr(task_obj, "set_fewshot_seed"):
        task_obj.set_fewshot_seed(seed=1234)

    # 데이터셋 빌드
    if hasattr(task_obj, "build_all_requests"):
        pass  # 일부 버전에서는 이게 필요

    # test set 가져오기
    if hasattr(task_obj, "test_docs"):
        docs = list(task_obj.test_docs())
    elif hasattr(task_obj, "eval_docs"):
        docs = list(task_obj.eval_docs())
    elif hasattr(task_obj, "validation_docs"):
        docs = list(task_obj.validation_docs())
    else:
        raise ValueError(f"Task {task_name}에서 평가용 문서를 찾을 수 없습니다.")

    if limit:
        docs = docs[:limit]

    results = []
    for doc in docs:
        # 프롬프트 생성
        prompt = task_obj.doc_to_text(doc)
        if isinstance(prompt, list):
            # chat format
            prompt = "\n".join(
                f"{m.get('role', 'user')}: {m.get('content', '')}" for m in prompt
            )

        # 정답 추출
        target = task_obj.doc_to_target(doc)
        if isinstance(target, list):
            target = target[0] if target else ""
        target = str(target).strip()

        # fewshot 프롬프트 구성
        if num_fewshot > 0 and hasattr(task_obj, "fewshot_context"):
            try:
                ctx = task_obj.fewshot_context(doc, num_fewshot)
                if ctx:
                    prompt = ctx
            except Exception:
                pass  # fewshot 실패 시 zero-shot으로

        # generation_kwargs에서 until 토큰 추출
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
    """응답에서 답변 레이블 (A, B, C, D 등) 추출"""
    text = text.strip()

    # 정확한 레이블 매칭: (A), (B), ... 또는 A, B, ...
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

    return text[:50]  # fallback: 첫 50자


def check_match(prediction: str, target: str) -> bool:
    """예측과 정답의 매칭 여부 확인 (유연한 매칭)"""
    pred = prediction.strip().upper()
    tgt = target.strip().upper()

    # 정확 일치
    if pred == tgt:
        return True

    # 레이블 추출 후 비교
    pred_label = extract_answer_label(pred)
    tgt_label = extract_answer_label(tgt)

    if pred_label and tgt_label and pred_label == tgt_label:
        return True

    # 괄호 제거 후 비교
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
    """단일 태스크에 대해 baseline vs interleaved 비교 실행"""
    logger.info(f"{'=' * 60}")
    logger.info(f"Task: {task_name}")
    logger.info(f"{'=' * 60}")

    # 태스크 데이터 로드
    logger.info("Loading task data...")
    try:
        task_docs = load_task_docs(task_name, limit=limit, num_fewshot=num_fewshot)
    except Exception as e:
        logger.error(f"Task {task_name} 로드 실패: {e}")
        return {"task": task_name, "error": str(e)}

    n_samples = len(task_docs)
    logger.info(f"Loaded {n_samples} samples")

    # 결과 저장
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

        # 샘플별 로그
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

    # 메트릭 계산
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
    """결과를 보기 좋은 테이블로 출력"""
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
        # fallback: 간단한 테이블
        header_str = " | ".join(f"{h:<20}" for h in headers)
        print(header_str)
        print("-" * len(header_str))
        for row in rows:
            print(" | ".join(f"{str(v):<20}" for v in row))

    # 통계 요약
    for r in results:
        if "error" in r:
            continue
        m = r["metrics"]
        print(f"\n--- {r['task']} 상세 통계 ---")
        print(
            f"  Interleaved 평균 confidence: {m['interleaved']['avg_confidence']:.3f}"
        )
        print(f"  Interleaved retry 비율:      {m['interleaved']['retry_rate']:.1%}")
        print(f"  Interleaved 총 API 호출:     {m['interleaved']['total_api_calls']}")
        print(f"  Baseline 총 API 호출:        {m['baseline']['total_api_calls']}")
        print(
            f"  Token overhead:              "
            f"{m['interleaved']['total_input_tokens'] + m['interleaved']['total_output_tokens']} "
            f"vs {m['baseline']['total_input_tokens'] + m['baseline']['total_output_tokens']}"
        )

    # 잘/못 된 케이스 분석
    for r in results:
        if "error" in r or "samples" not in r:
            continue
        samples = r["samples"]

        # Interleaved가 맞추고 Baseline이 틀린 케이스
        fixed = [
            s for s in samples if s["interleaved_correct"] and not s["baseline_correct"]
        ]
        # Baseline이 맞추고 Interleaved가 틀린 케이스
        broken = [
            s for s in samples if not s["interleaved_correct"] and s["baseline_correct"]
        ]

        print(f"\n--- {r['task']} 개선/퇴보 분석 ---")
        print(f"  Interleaved가 추가로 맞춘 샘플: {len(fixed)}")
        print(f"  Interleaved가 추가로 틀린 샘플: {len(broken)}")
        print(f"  순 개선: {len(fixed) - len(broken)}")

        if fixed:
            print("\n  [개선된 샘플 예시 (최대 3개)]")
            for s in fixed[:3]:
                print(
                    f"    #{s['index']}: target={s['target']}, "
                    f"baseline='{s['baseline_answer'][:30]}', "
                    f"interleaved='{s['interleaved_answer'][:30]}' "
                    f"(conf={s['interleaved_confidence']:.2f}, retries={s['interleaved_retries']})"
                )

        if broken:
            print("\n  [퇴보한 샘플 예시 (최대 3개)]")
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

    # Config 로드
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

    # API key 확인
    if not os.environ.get("OPENAI_API_KEY") and not model_cfg.get("api_base"):
        logger.error(
            "OPENAI_API_KEY 환경변수를 설정하거나 config에 api_base를 지정하세요."
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

    # 모델 초기화
    common_kwargs = dict(
        model_name=model_cfg["model_name"],
        api_base=model_cfg.get("api_base"),
        temperature=model_cfg.get("temperature", 0.0),
        max_tokens=model_cfg.get("max_tokens", 1024),
    )

    # ── Baseline에도 structured output + system prompt 사용
    # (공정한 비교를 위해 동일한 시스템 프롬프트 사용)
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

    # 태스크별 실행
    all_results = []
    start_time = time.time()

    for task_name in tasks:
        # 태스크마다 stats 리셋
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

    # 결과 출력
    print_results_table(all_results)
    logger.info(f"Total time: {elapsed:.1f}s")

    # JSON으로 저장
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

    # samples에서 doc 제거 (직렬화 문제 방지)
    for r in output_data["results"]:
        if "samples" in r:
            for s in r["samples"]:
                s.pop("doc", None)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    logger.info(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
