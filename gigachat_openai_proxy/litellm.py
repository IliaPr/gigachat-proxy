from __future__ import annotations

import logging
import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from gigachat_openai_proxy.config import Settings


logger = logging.getLogger("gigachat_openai_proxy.litellm")


class LiteLLMClient:
    def __init__(self, settings: Settings):
        if not settings.litellm_base_url:
            raise RuntimeError("LITELLM_BASE_URL is required")

        self.settings = settings
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(settings.request_timeout),
            verify=settings.httpx_verify,
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def chat_completions(self, payload: dict[str, Any]) -> httpx.Response:
        response = await self._client.post(
            f"{self.settings.litellm_base_url}/chat/completions",
            headers=self._headers(stream=False),
            json=payload,
        )
        logger.info(
            "litellm chat completion status=%s model=%s stream=%s",
            response.status_code,
            payload.get("model"),
            payload.get("stream", False),
        )
        if response.status_code >= 400:
            logger.warning(
                "litellm chat completion error status=%s body=%s",
                response.status_code,
                response.text[:1000],
            )
        return response

    async def stream_chat_completions(
        self, payload: dict[str, Any]
    ) -> AsyncIterator[bytes]:
        async with self._client.stream(
            "POST",
            f"{self.settings.litellm_base_url}/chat/completions",
            headers=self._headers(stream=True),
            json=payload,
        ) as response:
            if response.status_code >= 400:
                body = await response.aread()
                logger.warning(
                    "litellm streaming error status=%s body=%s",
                    response.status_code,
                    body[:1000].decode("utf-8", errors="replace"),
                )
                yield litellm_stream_error_chunk(response.status_code, body)
                return

            async for chunk in response.aiter_bytes():
                yield chunk

    def _headers(self, *, stream: bool) -> dict[str, str]:
        headers = {
            "Accept": "text/event-stream" if stream else "application/json",
            "Content-Type": "application/json",
        }
        if self.settings.litellm_api_key:
            headers["Authorization"] = f"Bearer {self.settings.litellm_api_key}"
        return headers


def litellm_stream_error_chunk(status_code: int, body: bytes) -> bytes:
    text = body.decode("utf-8", errors="replace")
    if len(text) > 500:
        text = text[:500]
    payload = {
        "choices": [
            {
                "index": 0,
                "delta": {
                    "content": f"LiteLLM API request failed with HTTP {status_code}: {text}"
                },
                "finish_reason": "stop",
            }
        ]
    }
    return f"data: {json.dumps(payload, ensure_ascii=True)}\n\ndata: [DONE]\n\n".encode(
        "utf-8"
    )
