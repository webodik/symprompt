from __future__ import annotations

import logging
import time
from typing import Any, Dict, List

import litellm

from symprompt.config import DEFAULT_LLM_CONFIG
from symprompt.llm.litellm_client import _strip_markdown_code_blocks

logger = logging.getLogger(__name__)


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

        last_error = None
        for attempt in range(self.retries + 1):
            try:
                response = litellm.completion(timeout=self.timeout_seconds, **params)
                choice = response.choices[0]
                content = getattr(choice.message, "content", "") or ""
                return _strip_markdown_code_blocks(content)
            except Exception as exc:
                last_error = exc
                if attempt < self.retries:
                    logger.warning(
                        "LiteLLM sync error on attempt %s/%s: %s. Retrying...",
                        attempt + 1,
                        self.retries + 1,
                        exc,
                    )
                    time.sleep(self.retry_delay_seconds)

        # Return empty string instead of crashing worker processes
        logger.error(
            "LiteLLM sync error after %s attempts: %s (returning empty response)",
            self.retries + 1,
            last_error,
        )
        return ""


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

