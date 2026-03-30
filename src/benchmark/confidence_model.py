"""
Confidence-Gated Interleaved Reasoning - Custom LM for lm-evaluation-harness

Custom LM class with two modes:
  1. BaselineChatLM       — standard single call
  2. InterleavedChatLM    — confidence-gated iterative reasoning
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field

from openai import OpenAI
from lm_eval.api.model import LM

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Structured output schema
# ------------------------------------------------------------------
INTERLEAVED_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "interleaved_response",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "thought": {
                    "type": "string",
                    "description": "Internal reasoning (hidden from user)",
                },
                "answer": {
                    "type": "string",
                    "description": "The answer to present (exact answer label like A, B, C, D or the answer text)",
                },
                "confidence": {
                    "type": "number",
                    "description": "Self-assessed confidence 0.0-1.0",
                },
                "confidence_reason": {
                    "type": "string",
                    "description": "Why this confidence level (e.g. 'direct calculation' vs 'guessing')",
                },
            },
            "required": ["thought", "answer", "confidence", "confidence_reason"],
            "additionalProperties": False,
        },
    },
}


@dataclass
class CallStats:
    """Track API call statistics."""

    total_calls: int = 0
    total_retries: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    confidence_values: list[float] = field(default_factory=list)
    retry_counts: list[int] = field(default_factory=list)

    @property
    def avg_calls_per_sample(self) -> float:
        n = len(self.retry_counts)
        return self.total_calls / n if n else 0

    @property
    def avg_confidence(self) -> float:
        return (
            sum(self.confidence_values) / len(self.confidence_values)
            if self.confidence_values
            else 0
        )

    @property
    def retry_rate(self) -> float:
        return (
            sum(1 for r in self.retry_counts if r > 0) / len(self.retry_counts)
            if self.retry_counts
            else 0
        )

    def summary(self) -> dict:
        return {
            "total_samples": len(self.retry_counts),
            "total_api_calls": self.total_calls,
            "avg_calls_per_sample": round(self.avg_calls_per_sample, 2),
            "avg_confidence": round(self.avg_confidence, 3),
            "retry_rate": round(self.retry_rate, 3),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
        }


class ChatLMBase(LM):
    """Base LM class using OpenAI-compatible API."""

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        api_base: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        system_prompt: str | None = None,
    ):
        super().__init__()
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = (
            system_prompt
            or "You are a helpful assistant that answers questions accurately and concisely."
        )

        kwargs = {}
        if api_base:
            kwargs["base_url"] = api_base
        if api_key:
            kwargs["api_key"] = api_key

        self.client = OpenAI(**kwargs)
        self.stats = CallStats()

    def _call_api(self, messages: list[dict], **kwargs) -> dict:
        """Single API call with rate limit retry."""
        for attempt in range(5):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    **kwargs,
                )
                self.stats.total_calls += 1

                # Track token usage
                if response.usage:
                    self.stats.total_input_tokens += response.usage.prompt_tokens
                    self.stats.total_output_tokens += response.usage.completion_tokens

                return response
            except Exception as e:
                if "rate_limit" in str(e).lower() or "429" in str(e):
                    wait = 2**attempt
                    logger.warning(f"Rate limited. Waiting {wait}s...")
                    time.sleep(wait)
                else:
                    raise
        raise RuntimeError("Max API retries exceeded")

    # ── lm_eval.api.model.LM interface ──

    @property
    def batch_size(self):
        return 1

    @property
    def max_length(self):
        return 128000

    @property
    def max_gen_toks(self):
        return self.max_tokens

    def generate_until(self, requests):
        results = []
        for request in requests:
            context, gen_kwargs = request.args
            until = gen_kwargs.get("until", ["\n"])
            answer = self.generate(context, until=until)
            results.append(answer)
        return results

    def loglikelihood(self, requests):
        results = []
        for request in requests:
            context, continuation = request.args
            ll, ig = self._compute_loglikelihood(context, continuation)
            results.append((ll, ig))
        return results

    def loglikelihood_rolling(self, requests):
        results = []
        for request in requests:
            (text,) = request.args
            ll, ig = self._compute_loglikelihood("", text)
            results.append((ll, ig))
        return results

    def _compute_loglikelihood(
        self, context: str, continuation: str
    ) -> tuple[float, bool]:
        if not continuation:
            return (0.0, False)

        full_text = context + continuation
        context_char_len = len(context)

        try:
            response = self.client.completions.create(
                model=self.model_name,
                prompt=full_text,
                max_tokens=1,
                echo=True,
                logprobs=1,
                temperature=0,
            )
            self.stats.total_calls += 1

            lp = response.choices[0].logprobs

            # Find where continuation starts via character offsets
            cont_start = None
            for i, offset in enumerate(lp.text_offset):
                if offset is not None and offset >= context_char_len:
                    cont_start = i
                    break

            if cont_start is None:
                return (0.0, False)

            # Sum log-probs for continuation tokens (exclude generated token)
            n_prompt_tokens = len(lp.tokens) - 1
            cont_lps = [
                p
                for p in lp.token_logprobs[cont_start:n_prompt_tokens]
                if p is not None
            ]

            log_likelihood = sum(cont_lps) if cont_lps else 0.0
            return (log_likelihood, True)

        except Exception as e:
            logger.warning(f"loglikelihood computation failed: {e}")
            return (0.0, False)


class BaselineChatLM(ChatLMBase):
    """Standard single-call baseline."""

    def generate(self, prompt: str, until: list[str] | None = None) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]

        response = self._call_api(
            messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        text = response.choices[0].message.content or ""
        self.stats.confidence_values.append(1.0)  # Baseline has no confidence tracking
        self.stats.retry_counts.append(0)

        # Truncate at stop tokens
        if until:
            for stop in until:
                if stop in text:
                    text = text[: text.index(stop)]

        return text.strip()


class InterleavedChatLM(ChatLMBase):
    """Confidence-gated interleaved reasoning"""

    def __init__(
        self,
        confidence_threshold: float = 0.7,
        max_retries: int = 2,
        retry_temperature: float = 0.3,
        confidence_guide: str = "",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.confidence_threshold = confidence_threshold
        self.max_retries = max_retries
        self.retry_temperature = retry_temperature

        # Append confidence guide to system prompt
        self.system_prompt = (
            self.system_prompt
            + "\n\n"
            + confidence_guide
            + "\n\nYou must respond in the exact JSON format requested."
        )

    def generate(self, prompt: str, until: list[str] | None = None) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]

        previous_thoughts: list[dict] = []
        retries = 0
        best_answer = ""
        best_confidence = 0.0

        while True:
            # On retry, include previous thoughts in context
            call_messages = list(messages)
            if previous_thoughts:
                retry_context = "Previous reasoning attempts (improve upon these):\n"
                for i, t in enumerate(previous_thoughts, 1):
                    retry_context += (
                        f"\n--- Attempt {i} (confidence: {t['confidence']}) ---\n"
                        f"Thought: {t['thought']}\n"
                        f"Answer: {t['answer']}\n"
                        f"Issue: {t['confidence_reason']}\n"
                    )
                retry_context += (
                    "\nYour previous confidence was low. "
                    "Try a different approach, verify your reasoning, "
                    "or check for errors. Do NOT repeat the same reasoning."
                )
                call_messages.append({"role": "user", "content": retry_context})

            try:
                response = self._call_api(
                    call_messages,
                    temperature=self.retry_temperature
                    if retries > 0
                    else self.temperature,
                    max_tokens=self.max_tokens,
                    response_format=INTERLEAVED_SCHEMA,
                )

                raw = response.choices[0].message.content or "{}"
                parsed = json.loads(raw)

                thought = parsed.get("thought", "")
                answer = parsed.get("answer", "")
                confidence = float(parsed.get("confidence", 0.0))
                reason = parsed.get("confidence_reason", "")

            except (json.JSONDecodeError, KeyError, TypeError) as e:
                logger.warning(f"Failed to parse structured output: {e}")
                # Fallback to plain call if structured output fails
                fallback_response = self._call_api(
                    messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                answer = fallback_response.choices[0].message.content or ""
                confidence = 0.5
                thought = ""
                reason = "fallback"

            # Track best-confidence answer
            if confidence > best_confidence:
                best_confidence = confidence
                best_answer = answer

            # Confidence sufficient -> exit
            if confidence >= self.confidence_threshold:
                break

            # Max retries exceeded -> return best effort
            if retries >= self.max_retries:
                logger.info(
                    f"Max retries ({self.max_retries}) reached. "
                    f"Returning best answer (confidence={best_confidence:.2f})"
                )
                answer = best_answer
                break

            # Retry
            previous_thoughts.append(
                {
                    "thought": thought,
                    "answer": answer,
                    "confidence": confidence,
                    "confidence_reason": reason,
                }
            )
            retries += 1
            self.stats.total_retries += 1

        # Record statistics
        self.stats.confidence_values.append(best_confidence)
        self.stats.retry_counts.append(retries)

        # Truncate at stop tokens
        if until:
            for stop in until:
                if stop in answer:
                    answer = answer[: answer.index(stop)]

        return answer.strip()
