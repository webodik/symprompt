from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List

import litellm
from openevolve.config import LLMModelConfig
from openevolve.llm.base import LLMInterface

logger = logging.getLogger(__name__)


def _strip_markdown_code_blocks(text: str) -> str:
    """
    Remove markdown code fences from LLM output.

    LLMs sometimes wrap code in ```python ... ``` blocks.
    This strips those markers to get raw Python code.
    """
    if not text:
        return text

    lines = text.split("\n")
    result_lines = []
    in_code_block = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            if not in_code_block:
                in_code_block = True
                continue
            else:
                in_code_block = False
                continue
        result_lines.append(line)

    return "\n".join(result_lines)


class LiteLLMLLM(LLMInterface):
    """
    LLMInterface implementation that uses the LiteLLM Python library directly.

    This allows OpenEvolve to call various providers via LiteLLM
    without running a separate HTTP proxy process.
    """

    def __init__(self, model_cfg: LLMModelConfig):
        self.model = model_cfg.name

        self.temperature = model_cfg.temperature
        self.top_p = model_cfg.top_p
        self.max_tokens = model_cfg.max_tokens
        self.timeout = model_cfg.timeout
        self.retries = model_cfg.retries or 0
        self.retry_delay = model_cfg.retry_delay or 0
        self.random_seed = getattr(model_cfg, "random_seed", None)
        self.reasoning_effort = getattr(model_cfg, "reasoning_effort", None)

        if not hasattr(logger, "_initialized_models"):
            logger._initialized_models = set()
        if self.model not in logger._initialized_models:
            logger.info("Initialized LiteLLM client with model: %s", self.model)
            logger._initialized_models.add(self.model)

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        return await self.generate_with_context(
            system_message=kwargs.pop("system_message", ""),
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )

    async def generate_with_context(
        self,
        system_message: str,
        messages: List[Dict[str, str]],
        **kwargs: Any,
    ) -> str:
        chat_messages: List[Dict[str, str]] = []
        if system_message:
            chat_messages.append({"role": "system", "content": system_message})
        chat_messages.extend(messages)

        params: Dict[str, Any] = {
            "model": self.model,
            "messages": chat_messages,
        }

        temperature = kwargs.get("temperature", self.temperature)
        if temperature is not None:
            params["temperature"] = temperature

        top_p = kwargs.get("top_p", self.top_p)
        if top_p is not None:
            params["top_p"] = top_p

        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        if max_tokens is not None:
            params["max_tokens"] = max_tokens

        reasoning_effort = kwargs.get("reasoning_effort", self.reasoning_effort)
        if reasoning_effort is not None:
            params["reasoning_effort"] = reasoning_effort

        seed = kwargs.get("seed", self.random_seed)
        if seed is not None:
            params["seed"] = seed

        timeout = kwargs.get("timeout", self.timeout)
        retries = kwargs.get("retries", self.retries)
        retry_delay = kwargs.get("retry_delay", self.retry_delay)

        for attempt in range(retries + 1):
            try:
                response = await litellm.acompletion(timeout=timeout, **params)
                choice = response.choices[0]
                content = choice.message.content or ""
                content = _strip_markdown_code_blocks(content)
                logger.debug("LiteLLM response content: %s", content)
                return content
            except asyncio.TimeoutError:
                if attempt < retries:
                    logger.warning(
                        "LiteLLM timeout on attempt %s/%s. Retrying...",
                        attempt + 1,
                        retries + 1,
                    )
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(
                        "LiteLLM timeout after %s attempts", retries + 1
                    )
                    raise
            except Exception as exc:
                if attempt < retries:
                    logger.warning(
                        "LiteLLM error on attempt %s/%s: %s. Retrying...",
                        attempt + 1,
                        retries + 1,
                        exc,
                    )
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(
                        "LiteLLM error after %s attempts: %s", retries + 1, exc
                    )
                    raise
