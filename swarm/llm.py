"""Backend shim: OpenRouter for smoke tests, vLLM (OpenAI-compatible) for the real run.

Switching backends changes only the base URL, the model names, and how thinking is enabled.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
import uuid
from dataclasses import dataclass, field

import openai
from openai import AsyncOpenAI

from .config import Config

THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def salvage_tool_calls(content: str) -> list[dict]:
    """Recover calls the hermes parser drops: several JSON objects in one <tool_call> block, or no closing tag."""
    if "<tool_call>" not in content:
        return []
    text = content.replace("<tool_call>", "\n").replace("</tool_call>", "\n")
    dec, i, calls = json.JSONDecoder(), 0, []
    while (i := text.find("{", i)) != -1:
        try:
            obj, i = dec.raw_decode(text, i)
        except ValueError:
            i += 1
            continue
        if isinstance(obj, dict) and isinstance(obj.get("name"), str):
            args = obj.get("arguments", {})
            calls.append({"id": f"salvaged-{uuid.uuid4().hex[:12]}", "name": obj["name"],
                          "arguments": args if isinstance(args, str) else json.dumps(args)})
    return calls

DEFAULT_URLS = {
    "openrouter": "https://openrouter.ai/api/v1",
    "vllm": "http://localhost:8000/v1",
}
KEY_ENV = {"openrouter": "OPENROUTER_API_KEY", "vllm": "VLLM_API_KEY"}


@dataclass
class Completion:
    content: str
    reasoning: str
    tool_calls: list[dict]          # [{"id", "name", "arguments"}]; arguments is the raw JSON string
    finish_reason: str | None
    prompt_tokens: int
    completion_tokens: int
    latency: float
    provider: str | None = None
    think_unclosed: bool = False    # stopped inside <think>: the reply (and any tool call) landed in reasoning
    salvaged: bool = False          # tool calls recovered from malformed <tool_call> text (salvage_tool_calls)
    raw_extra: dict = field(default_factory=dict)


class LLM:
    def __init__(self, cfg: Config, log=None):
        self.cfg = cfg
        self.log = log
        self.client = AsyncOpenAI(
            base_url=cfg.base_url or DEFAULT_URLS[cfg.backend],
            api_key=os.environ.get(KEY_ENV[cfg.backend], "EMPTY"),
            timeout=600,
        )
        self.sem = asyncio.Semaphore(cfg.concurrency)

    async def server_max_len(self) -> int | None:
        """vLLM reports max_model_len per model on GET /models; the smallest one among our models wins."""
        models = (await self.client.models.list()).data
        ours = {self.cfg.base_model, self.cfg.mo_model, self.cfg.summarizer_model}
        lens = [(m.model_extra or {}).get("max_model_len") for m in models if m.id in ours] or \
               [(m.model_extra or {}).get("max_model_len") for m in models]
        lens = [n for n in lens if n]
        return min(lens) if lens else None

    def _extra_body(self, thinking: bool, tools: bool) -> dict:
        if self.cfg.backend == "openrouter":
            body = {"reasoning": {"enabled": thinking}}
            if tools:
                # Route only to providers that support every request parameter (some lack tool calling).
                body["provider"] = {"require_parameters": True}
            return body
        # vLLM: needs the stock Qwen3 chat template + --reasoning-parser qwen3 for thinking-on.
        return {"chat_template_kwargs": {"enable_thinking": thinking}}

    async def chat(self, model: str, messages: list[dict], tools: list[dict] | None = None,
                   max_tokens: int | None = None, thinking: bool | None = None) -> Completion:
        thinking = self.cfg.thinking if thinking is None else thinking
        kwargs = dict(
            model=model, messages=messages, temperature=self.cfg.temperature,
            max_tokens=max_tokens or self.cfg.max_tokens,
            extra_body=self._extra_body(thinking, bool(tools)),
        )
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        delay = 2.0
        last_err: Exception | None = None
        for attempt in range(7):
            async with self.sem:
                t0 = time.time()
                try:
                    r = await self.client.chat.completions.create(**kwargs)
                except openai.APIStatusError as e:
                    if 400 <= e.status_code < 500 and e.status_code not in (408, 409, 429):
                        raise  # bad request (e.g. context overflow): retrying cannot help
                    last_err, r = e, None
                except Exception as e:  # network / timeout
                    last_err, r = e, None
            if r is not None and r.choices:
                return self._parse(r, time.time() - t0)
            if r is not None:
                last_err = RuntimeError(f"empty choices: {getattr(r, 'model_extra', None)}")
            if self.log:
                self.log.write("llm_retry", model=model, attempt=attempt + 1, error=repr(last_err)[:500])
            await asyncio.sleep(delay)
            delay = min(delay * 2, 60)
        raise RuntimeError(f"LLM call failed after retries: {last_err!r}")

    @staticmethod
    def _parse(r, latency: float) -> Completion:
        ch = r.choices[0]
        msg = ch.message
        extra = msg.model_extra or {}
        reasoning = extra.get("reasoning") or extra.get("reasoning_content") or ""
        content = msg.content or ""
        # Fallback: thinking emitted inline in content (no reasoning parser).
        inline = THINK_RE.findall(content)
        if inline:
            reasoning = (reasoning + "\n" + "\n".join(inline)).strip()
            content = THINK_RE.sub("", content)
        if "</think>" in content and "<think>" not in content:  # template pre-filled the opening tag
            head, _, tail = content.partition("</think>")
            reasoning = (reasoning + "\n" + head).strip()
            content = tail
        if "<think>" in content and "</think>" not in content:  # truncated thinking
            head, _, tail = content.partition("<think>")
            reasoning = (reasoning + "\n" + tail).strip()
            content = head
        tool_calls = [
            {"id": tc.id, "name": tc.function.name, "arguments": tc.function.arguments or ""}
            for tc in (msg.tool_calls or [])
        ]
        salvaged = False
        if not tool_calls and (found := salvage_tool_calls(content)):
            tool_calls, content, salvaged = found, "", True
        u = r.usage
        unclosed = bool(reasoning) and not content.strip() and not tool_calls and ch.finish_reason == "stop"
        return Completion(
            content=content.strip(), reasoning=reasoning, tool_calls=tool_calls,
            finish_reason=ch.finish_reason,
            prompt_tokens=getattr(u, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(u, "completion_tokens", 0) or 0,
            latency=latency, provider=(r.model_extra or {}).get("provider"), think_unclosed=unclosed, salvaged=salvaged,
        )
