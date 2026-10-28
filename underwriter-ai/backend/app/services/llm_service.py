"""Centralised LLM access through OpenRouter's OpenAI-compatible API.

All model calls go through `LLMService`, which owns retries with exponential backoff,
timeouts, optional JSON output and secret-safe logging. The API key is read from settings
(environment) only and never returned, logged or sent to the frontend.
"""

import json
import logging
import random
import time
from dataclasses import dataclass
from typing import Any

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI, RateLimitError

from app.core.config import Settings, get_settings
from app.core.logging import scrub

log = logging.getLogger(__name__)


class LLMNotConfiguredError(RuntimeError):
    """Raised when OPENROUTER_API_KEY is missing."""


class LLMError(RuntimeError):
    """Raised when the LLM call fails after all retries."""


@dataclass
class LLMResponse:
    text: str
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


_RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


class LLMService:
    """Thin, testable wrapper around the OpenRouter chat-completions endpoint."""

    def __init__(self, settings: Settings | None = None, client: Any | None = None) -> None:
        self.settings = settings or get_settings()
        self._client = client

    @property
    def configured(self) -> bool:
        return self._client is not None or self.settings.llm_configured

    def _get_client(self) -> Any:
        if self._client is None:
            if not self.settings.llm_configured:
                raise LLMNotConfiguredError("OPENROUTER_API_KEY is not set; the LLM is not configured.")
            self._client = OpenAI(
                api_key=self.settings.openrouter_api_key.get_secret_value(),
                base_url=self.settings.openrouter_base_url,
                timeout=self.settings.llm_timeout_seconds,
                max_retries=0,  # retries are handled here, with logging and backoff
                default_headers={"X-Title": "UnderwriterAI"},
            )
        return self._client

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        model: str | None = None,
        json_output: bool = False,
        temperature: float | None = None,
        max_tokens: int = 2000,
    ) -> LLMResponse:
        """Run one chat completion with retries. Raises LLMNotConfiguredError / LLMError."""
        client = self._get_client()
        model = model or self.settings.openrouter_model
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            "temperature": self.settings.llm_temperature if temperature is None else temperature,
            "max_tokens": max_tokens,
        }
        if json_output:
            kwargs["response_format"] = {"type": "json_object"}

        attempts = self.settings.llm_max_retries + 1
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                resp = client.chat.completions.create(**kwargs)
                choice = resp.choices[0].message.content or ""
                usage = getattr(resp, "usage", None)
                return LLMResponse(
                    text=choice,
                    model=getattr(resp, "model", model) or model,
                    prompt_tokens=getattr(usage, "prompt_tokens", None),
                    completion_tokens=getattr(usage, "completion_tokens", None),
                )
            except (RateLimitError, APITimeoutError, APIConnectionError) as e:
                last_error = e
            except APIStatusError as e:
                last_error = e
                # Some models reject response_format; retry once without it.
                if json_output and e.status_code == 400 and "response_format" in kwargs:
                    log.warning("Model %s rejected JSON mode; retrying without response_format", model)
                    kwargs.pop("response_format")
                    continue
                if e.status_code not in _RETRYABLE_STATUS:
                    break
            if attempt < attempts:
                delay = min(30.0, (2 ** (attempt - 1)) * 1.0) + random.uniform(0, 0.5)
                log.warning("LLM call failed (attempt %d/%d): %s; retrying in %.1fs",
                            attempt, attempts, scrub(str(last_error))[:300], delay)
                time.sleep(delay)
        log.error("LLM call failed after %d attempts: %s", attempts, scrub(str(last_error))[:500])
        raise LLMError(f"LLM request failed: {type(last_error).__name__}") from last_error

    def chat_json(self, system_prompt: str, user_prompt: str, **kwargs: Any) -> dict[str, Any]:
        """Chat expecting a JSON object back; tolerant of fenced or prefixed JSON."""
        resp = self.chat(system_prompt, user_prompt, json_output=True, **kwargs)
        return parse_json_object(resp.text)


def parse_json_object(text: str) -> dict[str, Any]:
    """Extract the first JSON object from model text (handles ```json fences)."""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t
        t = t.rsplit("```", 1)[0]
    start, end = t.find("{"), t.rfind("}")
    if start == -1 or end == -1:
        raise LLMError("LLM did not return a JSON object")
    try:
        return json.loads(t[start : end + 1])
    except json.JSONDecodeError as e:
        raise LLMError(f"LLM returned invalid JSON: {e}") from e
