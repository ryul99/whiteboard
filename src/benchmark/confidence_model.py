"""
Confidence-Gated Interleaved Reasoning - Custom LM for lm-evaluation-harness

두 가지 모드를 제공하는 커스텀 LM 클래스:
  1. BaselineChatLM       — 표준 단일 호출
  2. InterleavedChatLM    — confidence-gated 반복 추론
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field

from openai import OpenAI

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
    """API 호출 통계 추적"""

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


class ChatLMBase:
    """OpenAI-compatible API를 사용하는 기본 LM 클래스"""

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        api_base: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        system_prompt: str | None = None,
    ):
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
        """단일 API 호출 (rate limit 재시도 포함)"""
        for attempt in range(5):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    **kwargs,
                )
                self.stats.total_calls += 1

                # 토큰 사용량 추적
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


class BaselineChatLM(ChatLMBase):
    """표준 단일 호출 baseline"""

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
        self.stats.confidence_values.append(1.0)  # baseline은 confidence 추적 없음
        self.stats.retry_counts.append(0)

        # until 토큰으로 자르기
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

        # System prompt에 confidence 가이드 추가
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
            # 재시도 시 이전 사고를 context에 추가
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
                # Structured output 실패 시 일반 호출로 fallback
                fallback_response = self._call_api(
                    messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                answer = fallback_response.choices[0].message.content or ""
                confidence = 0.5
                thought = ""
                reason = "fallback"

            # 최고 confidence 답변 추적
            if confidence > best_confidence:
                best_confidence = confidence
                best_answer = answer

            # Confidence 충분 → 종료
            if confidence >= self.confidence_threshold:
                break

            # 재시도 한도 초과 → best effort 반환
            if retries >= self.max_retries:
                logger.info(
                    f"Max retries ({self.max_retries}) reached. "
                    f"Returning best answer (confidence={best_confidence:.2f})"
                )
                answer = best_answer
                break

            # 재시도
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

        # 통계 기록
        self.stats.confidence_values.append(best_confidence)
        self.stats.retry_counts.append(retries)

        # until 토큰으로 자르기
        if until:
            for stop in until:
                if stop in answer:
                    answer = answer[: answer.index(stop)]

        return answer.strip()
