from __future__ import annotations

import time
from typing import Any, Dict, List

import litellm

from symprompt.config import DEFAULT_LLM_CONFIG
from symprompt.llm.litellm_client import _strip_markdown_code_blocks


class LiteLLMSyncClient:
    """
    Synchronous LiteLLM-based client exposing a .complete(prompt) API.

    This is used by the CLI and evaluation harness so that all LLM calls
    go through LiteLLM with a consistent configuration.
    """

    def __init__(
        self,
        model_name: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        timeout_seconds: int,
        retries: int,
        retry_delay_seconds: float,
    ) -> None:
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds
        self.retries = retries
        self.retry_delay_seconds = retry_delay_seconds

    def complete(self, prompt: str) -> str:
        messages: List[Dict[str, str]] = [{"role": "user", "content": prompt}]
        params: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }

        for attempt in range(self.retries + 1):
            try:
                response = litellm.completion(timeout=self.timeout_seconds, **params)
                choice = response.choices[0]
                content = getattr(choice.message, "content", "") or ""
                return _strip_markdown_code_blocks(content)
            except Exception:
                if attempt >= self.retries:
                    raise
                time.sleep(self.retry_delay_seconds)


def build_default_sync_client() -> LiteLLMSyncClient:
    cfg = DEFAULT_LLM_CONFIG
    return LiteLLMSyncClient(
        model_name=cfg.model_name,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        max_tokens=cfg.max_tokens,
        timeout_seconds=cfg.timeout_seconds,
        retries=cfg.retries,
        retry_delay_seconds=cfg.retry_delay_seconds,
    )

