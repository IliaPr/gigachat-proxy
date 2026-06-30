from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

import httpx

from gigachat_openai_proxy.config import Settings


logger = logging.getLogger("gigachat_openai_proxy.gigachat")

MODEL_ALIASES = {
    "GigaChat-2-Lite": "GigaChat",
    "gpt-3.5-turbo": "GigaChat",
    "gpt-4": "GigaChat",
    "gpt-4-turbo": "GigaChat",
    "gpt-4.1": "GigaChat",
    "gpt-4.1-mini": "GigaChat",
    "gpt-4o": "GigaChat",
    "gpt-4o-mini": "GigaChat",
    "openai/gpt-4o": "GigaChat",
}


def normalize_model(model: str) -> str:
    if model.startswith("gpt-") or model.startswith("openai/"):
        return MODEL_ALIASES.get(model, "GigaChat")
    return MODEL_ALIASES.get(model, model)


class GigaChatError(Exception):
    def __init__(self, status_code: int, message: str, details: Any | None = None):
        self.status_code = status_code
        self.message = message
        self.details = details
        super().__init__(message)


class GigaChatClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._access_token: str | None = None
        self._expires_at = 0.0
        self._token_lock = asyncio.Lock()
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(settings.request_timeout),
            verify=settings.httpx_verify,
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def chat_completions(self, payload: dict[str, Any]) -> httpx.Response:
        request_payload = self._normalize_payload(payload)
        response = await self._client.post(
            f"{self.settings.gigachat_api_base_url}/chat/completions",
            headers=await self._headers(request_payload),
            json=request_payload,
        )
        logger.info(
            "gigachat chat completion status=%s model=%s stream=%s",
            response.status_code,
            request_payload.get("model"),
            request_payload.get("stream", False),
        )
        if response.status_code >= 400:
            logger.warning(
                "gigachat chat completion error status=%s body=%s",
                response.status_code,
                response.text[:1000],
            )
        return response

    async def stream_chat_completions(
        self, payload: dict[str, Any]
    ) -> AsyncIterator[bytes]:
        request_payload = self._normalize_payload(payload)
        async with self._client.stream(
            "POST",
            f"{self.settings.gigachat_api_base_url}/chat/completions",
            headers=await self._headers(request_payload),
            json=request_payload,
        ) as response:
            if response.status_code >= 400:
                details = await self._read_error_details(response)
                raise GigaChatError(
                    response.status_code,
                    "GigaChat API request failed",
                    details,
                )

            async for chunk in response.aiter_bytes():
                yield chunk

    async def _headers(self, payload: dict[str, Any]) -> dict[str, str]:
        accept = "text/event-stream" if payload.get("stream") is True else "application/json"
        return {
            "Accept": accept,
            "Authorization": f"Bearer {await self._access_token_value()}",
            "Content-Type": "application/json",
        }

    async def _access_token_value(self) -> str:
        if self._access_token and time.time() < self._expires_at - 30:
            return self._access_token

        async with self._token_lock:
            if self._access_token and time.time() < self._expires_at - 30:
                return self._access_token

            response = await self._client.post(
                self.settings.gigachat_token_url,
                headers={
                    "Accept": "application/json",
                    "Authorization": f"Basic {self.settings.gigachat_auth_key}",
                    "Content-Type": "application/x-www-form-urlencoded",
                    "RqUID": str(uuid.uuid4()),
                },
                data={"scope": self.settings.gigachat_scope},
            )

            if response.status_code >= 400:
                logger.warning(
                    "gigachat oauth error status=%s body=%s",
                    response.status_code,
                    response.text[:1000],
                )
                raise GigaChatError(
                    response.status_code,
                    "GigaChat OAuth request failed",
                    self._json_or_text(response),
                )

            data = response.json()
            token = data.get("access_token")
            if not token:
                raise GigaChatError(
                    502,
                    "GigaChat OAuth response does not contain access_token",
                    data,
                )

            self._access_token = token
            self._expires_at = self._parse_expires_at(data)
            logger.info("gigachat oauth token refreshed expires_at=%s", int(self._expires_at))
            return token

    def _normalize_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(payload)
        normalized.setdefault("model", self.settings.gigachat_model)
        normalized["model"] = normalize_model(normalized["model"])
        normalized["messages"] = normalize_messages(normalized.get("messages", []))

        if "max_completion_tokens" in normalized and "max_tokens" not in normalized:
            normalized["max_tokens"] = normalized.pop("max_completion_tokens")

        # Common OpenAI fields that GigaChat may reject.
        for key in (
            "function_call",
            "functions",
            "frequency_penalty",
            "logit_bias",
            "logprobs",
            "n",
            "parallel_tool_calls",
            "presence_penalty",
            "response_format",
            "seed",
            "service_tier",
            "store",
            "stream_options",
            "top_logprobs",
            "tool_choice",
            "tools",
            "user",
        ):
            normalized.pop(key, None)

        return normalized

    @staticmethod
    def _parse_expires_at(data: dict[str, Any]) -> float:
        expires_at = data.get("expires_at")
        if isinstance(expires_at, int | float):
            return float(expires_at / 1000 if expires_at > 10_000_000_000 else expires_at)

        return time.time() + int(data.get("expires_in", 1800))

    @staticmethod
    def _json_or_text(response: httpx.Response) -> Any:
        try:
            return response.json()
        except ValueError:
            return response.text

    @classmethod
    async def _read_error_details(cls, response: httpx.Response) -> Any:
        raw = await response.aread()
        if not raw:
            return None
        try:
            return response.json()
        except ValueError:
            return raw.decode("utf-8", errors="replace")


def normalize_messages(messages: Any) -> list[dict[str, Any]]:
    if not isinstance(messages, list):
        return []

    normalized_messages: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue

        normalized = dict(message)
        normalized.pop("tool_calls", None)
        normalized.pop("function_call", None)

        if normalized.get("role") == "tool":
            tool_name = normalized.get("name") or normalized.get("tool_call_id") or "tool"
            normalized["role"] = "user"
            normalized["content"] = f"Tool result from {tool_name}:\n{normalized.get('content', '')}"
            normalized.pop("name", None)
            normalized.pop("tool_call_id", None)

        content = normalized.get("content")
        if isinstance(content, list):
            normalized["content"] = content_parts_to_text(content)
        if normalized.get("role") == "assistant" and not normalized.get("content"):
            continue
        normalized_messages.append(normalized)

    return normalized_messages


def content_parts_to_text(parts: list[Any]) -> str:
    text_parts: list[str] = []
    for part in parts:
        if isinstance(part, str):
            text_parts.append(part)
            continue

        if not isinstance(part, dict):
            continue

        if part.get("type") == "text":
            text = part.get("text", "")
            if isinstance(text, str):
                text_parts.append(text)
        elif part.get("type") in {"image_url", "input_image"}:
            text_parts.append("[Image attachment omitted: vision is not supported by this proxy yet.]")

    return "\n".join(text for text in text_parts if text)
