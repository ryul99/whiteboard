from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import lm_eval
from lm_eval.utils import handle_non_serializable

from benchmark.whiteboard_model import WhiteboardOpenAI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

TASK_CHOICES = [
    "gpqa_diamond_cot_zeroshot",
    "gpqa_diamond_cot_n_shot",
    "gpqa_diamond_generative_n_shot",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GPQA Diamond Benchmark with Whiteboard Tools",
    )
    parser.add_argument(
        "--model", required=True, help="OpenAI model name (e.g. gpt-4o)"
    )
    parser.add_argument("--base-url", default=None, help="OpenAI API base URL")
    parser.add_argument("--api-key", default=None, help="OpenAI API key")
    parser.add_argument(
        "--task",
        default=TASK_CHOICES[0],
        choices=TASK_CHOICES,
        help="GPQA Diamond task variant",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=10,
        help="Max tool-calling turns per question",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit number of examples"
    )
    parser.add_argument("--output", default="results.json", help="Output results file")
    parser.add_argument(
        "--log-samples", action="store_true", help="Log individual samples"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    lm = WhiteboardOpenAI(
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        max_turns=args.max_turns,
    )

    logger.info("Running %s with model=%s", args.task, args.model)

    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=[args.task],
        limit=args.limit,
        log_samples=args.log_samples,
    )

    with open(args.output, "w") as f:
        json.dump(results, f, default=handle_non_serializable, indent=2)
    logger.info("Results saved to %s", args.output)

    for task_name, task_results in results["results"].items():
        logger.info("=== %s ===", task_name)
        for metric, value in task_results.items():
            if not metric.endswith(",stderr"):
                logger.info("  %s: %.4f", metric, value)


if __name__ == "__main__":
    main()
