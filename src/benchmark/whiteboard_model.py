from __future__ import annotations

import json
import logging
import os

from openai import OpenAI

from lm_eval.api.model import LM
from lm_eval.api.registry import register_model

logger = logging.getLogger(__name__)

WHITEBOARD_TOOLS = [
    {
        "type": "function",
        "name": "write",
        "description": "Write the thoughts at note and name the note as `key`",
        "parameters": {
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": "Name of the note",
                },
                "text": {
                    "type": "string",
                    "description": "Content to write",
                },
            },
            "required": ["key", "text"],
        },
    },
    {
        "type": "function",
        "name": "read",
        "description": "Recall the note named as `key`",
        "parameters": {
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": "Name of the note to recall",
                },
            },
            "required": ["key"],
        },
    },
]


@register_model("whiteboard-openai")
class WhiteboardOpenAI(LM):
    """OpenAI model with whiteboard tools (read/write) for reasoning."""

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        batch_size: int = 1,
        max_gen_toks: int = 1024,
        max_turns: int = 10,
        **kwargs,
    ):
        super().__init__()
        self.model_name = model
        self._batch_size = batch_size
        self._max_gen_toks = max_gen_toks
        self._max_turns = max_turns

        self.client = OpenAI(
            base_url=base_url or os.environ.get("OPENAI_BASE_URL"),
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
        )

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def generate_until(self, requests) -> list[str]:
        results = []
        for request in requests:
            context, gen_kwargs = request.args
            stop = gen_kwargs.get("until", [])
            max_tokens = gen_kwargs.get("max_gen_toks", self._max_gen_toks)

            result = self._run_tool_loop(context, stop, max_tokens)
            results.append(result)
        return results

    def _run_tool_loop(
        self, prompt: str, stop_sequences: list[str], max_tokens: int
    ) -> str:
        whiteboard: dict[str, str] = {}

        response = self.client.responses.create(
            model=self.model_name,
            input=prompt,
            tools=WHITEBOARD_TOOLS,
            max_output_tokens=max_tokens,
            temperature=0.0,
        )

        for _ in range(self._max_turns):
            function_calls = [
                item for item in response.output if item.type == "function_call"
            ]

            if not function_calls:
                text = response.output_text or ""
                for stop in stop_sequences:
                    if stop in text:
                        text = text[: text.index(stop)]
                return text

            tool_outputs = []
            for fc in function_calls:
                result = self._execute_tool(fc, whiteboard)
                logger.debug(
                    "Tool call: %s(%s) -> %s",
                    fc.name,
                    fc.arguments,
                    result,
                )
                tool_outputs.append(
                    {
                        "type": "function_call_output",
                        "call_id": fc.call_id,
                        "output": result,
                    }
                )

            response = self.client.responses.create(
                model=self.model_name,
                previous_response_id=response.id,
                input=tool_outputs,
                tools=WHITEBOARD_TOOLS,
                max_output_tokens=max_tokens,
                temperature=0.0,
            )

        logger.warning("Max turns (%d) reached without final answer", self._max_turns)
        return ""

    @staticmethod
    def _execute_tool(function_call, whiteboard: dict[str, str]) -> str:
        args = json.loads(function_call.arguments)
        if function_call.name == "write":
            whiteboard[args["key"]] = args["text"]
            return f"Saved to '{args['key']}'."
        if function_call.name == "read":
            key = args["key"]
            if key not in whiteboard:
                return f"No data found for '{key}'."
            return whiteboard[key]
        return "Unknown tool."

    def loglikelihood(self, requests):
        raise NotImplementedError("WhiteboardOpenAI only supports generate_until tasks")

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError("WhiteboardOpenAI only supports generate_until tasks")
