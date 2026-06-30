from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.requests import ClientDisconnect

from gigachat_openai_proxy.config import Settings
from gigachat_openai_proxy.gigachat import (
    MODEL_ALIASES,
    GigaChatClient,
    GigaChatError,
    normalize_model,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("gigachat_openai_proxy")
FILE_ID_RE = re.compile(r"\b[a-z0-9]{26}\b")
FILE_CONTAINER_KEYS = {
    "attachment",
    "attachments",
    "file",
    "files",
    "file_info",
    "file_infos",
    "image",
    "images",
    "input_file",
}
FILE_ID_KEYS = {
    "file_id",
    "file_ids",
    "fileId",
    "fileIds",
    "fileID",
    "fileIDs",
}

settings = Settings.from_env()
served_model = normalize_model(settings.gigachat_model)
gigachat = GigaChatClient(settings)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    try:
        yield
    finally:
        await gigachat.close()


app = FastAPI(
    title="GigaChat OpenAI Proxy",
    version="0.1.0",
    lifespan=lifespan,
)


class RequestLoggingMiddleware:
    def __init__(self, inner_app: Any):
        self.inner_app = inner_app

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.inner_app(scope, receive, send)
            return

        started_at = time.perf_counter()
        headers = {
            key.decode("latin-1").lower(): value.decode("latin-1")
            for key, value in scope.get("headers", [])
        }
        status_code: int | None = None

        logger.info(
            "incoming request method=%s path=%s content_length=%s transfer_encoding=%s expect=%s content_type=%s user_agent=%s",
            scope.get("method"),
            scope.get("path"),
            headers.get("content-length"),
            headers.get("transfer-encoding"),
            headers.get("expect"),
            headers.get("content-type"),
            headers.get("user-agent"),
        )

        async def send_wrapper(message: dict[str, Any]) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
            await send(message)

        try:
            await self.inner_app(scope, receive, send_wrapper)
        finally:
            duration_ms = (time.perf_counter() - started_at) * 1000
            logger.info(
                "request complete method=%s path=%s status=%s duration_ms=%.1f",
                scope.get("method"),
                scope.get("path"),
                status_code,
                duration_ms,
            )


app.add_middleware(RequestLoggingMiddleware)


async def require_proxy_auth(authorization: str | None = Header(default=None)) -> None:
    if not settings.proxy_api_key:
        return

    token = ""
    if authorization and authorization.startswith("Bearer "):
        token = authorization.removeprefix("Bearer ").strip()

    if token != settings.proxy_api_key:
        raise HTTPException(status_code=401, detail="unauthorized")


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
    message = exc.detail if isinstance(exc.detail, str) else "request failed"
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"message": message}},
        headers=exc.headers,
    )


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/v1")
async def v1_info() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "gigachat-openai-proxy",
        "model": served_model,
        "endpoints": [
            "GET /v1/models",
            "POST /v1/chat/completions",
        ],
    }


@app.get("/v1/models", dependencies=[Depends(require_proxy_auth)])
async def models() -> dict[str, Any]:
    model_ids = [served_model]
    for alias in MODEL_ALIASES:
        if alias not in model_ids:
            model_ids.append(alias)

    return {
        "object": "list",
        "data": [
            {
                "id": model_id,
                "object": "model",
                "created": 0,
                "owned_by": "gigachat",
            }
            for model_id in model_ids
        ],
    }


@app.post("/v1/chat/completions", dependencies=[Depends(require_proxy_auth)])
async def chat_completions(request: Request) -> Response:
    try:
        payload = await read_json_payload(request)
    except RequestBodyReadTimeout as exc:
        logger.warning(
            "timed out reading request body after %.1fs idle timeout; bytes=%s expected=%s elapsed_ms=%.1f",
            exc.idle_timeout,
            exc.bytes_read,
            exc.expected_bytes,
            exc.elapsed_ms,
        )
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": (
                        "incomplete request body "
                        f"after receiving {exc.bytes_read} of {exc.expected_bytes or 'unknown'} bytes"
                    ),
                    "type": "incomplete_request_body",
                }
            },
            headers={"Connection": "close"},
        )
    except TimeoutError:
        logger.warning(
            "timed out reading request body after %.1fs content_length=%s",
            settings.request_body_timeout,
            request.headers.get("content-length"),
        )
        return JSONResponse(
            status_code=408,
            content={"error": {"message": "timed out reading request body"}},
        )
    except ClientDisconnect:
        logger.warning("client disconnected before request body was read")
        return Response(status_code=499)
    except ValueError:
        raise HTTPException(
            status_code=400, detail="invalid JSON request body")

    if not isinstance(payload, dict):
        raise HTTPException(
            status_code=400, detail="request body must be a JSON object")

    try:
        logger.info(
            "chat completion request model=%s stream=%s messages=%s tools=%s tool_names=%s",
            payload.get("model"),
            payload.get("stream", False),
            len(payload.get("messages", [])) if isinstance(
                payload.get("messages"), list) else 0,
            len(payload.get("tools", [])) if isinstance(
                payload.get("tools"), list) else 0,
            tool_names(payload.get("tools")),
        )
        logger.info("message diagnostics=%s",
                    message_diagnostics(payload.get("messages")))
        if settings.enable_mattermost_tools:
            read_file_completion = maybe_read_file_tool_call_completion(
                payload)
            if read_file_completion is not None:
                logger.info(
                    "returning synthetic read_file tool calls count=%s",
                    len(read_file_completion["choices"]
                        [0]["message"]["tool_calls"]),
                )
                if payload.get("stream") is True:
                    return StreamingResponse(
                        openai_sse_from_tool_call_completion(
                            read_file_completion),
                        media_type="text/event-stream",
                        headers=stream_headers(close_connection=True),
                    )
                return JSONResponse(status_code=200, content=read_file_completion)

            generic_tool_completion = await maybe_generic_tool_completion(payload)
            if generic_tool_completion is not None:
                choice = generic_tool_completion["choices"][0]
                has_tool_calls = bool(choice["message"].get("tool_calls"))
                logger.info(
                    "returning generic MCP completion has_tool_calls=%s", has_tool_calls)
                if payload.get("stream") is True:
                    stream = (
                        openai_sse_from_tool_call_completion(
                            generic_tool_completion)
                        if has_tool_calls
                        else openai_sse_from_completion(generic_tool_completion)
                    )
                    return StreamingResponse(
                        stream,
                        media_type="text/event-stream",
                        headers=stream_headers(close_connection=True),
                    )
                return JSONResponse(status_code=200, content=generic_tool_completion)
        else:
            logger.info(
                "mattermost tool handling disabled; forwarding chat to GigaChat")

        if payload.get("stream") is True:
            return StreamingResponse(
                openai_sse_from_gigachat_completion(payload),
                media_type="text/event-stream",
                headers=stream_headers(close_connection=True),
            )

        response = await gigachat.chat_completions(payload)
        if response.status_code >= 400:
            return openai_error_response(response)

        return JSONResponse(
            status_code=200,
            content=normalize_chat_completion_response(
                response.json(), payload),
        )
    except GigaChatError as exc:
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "message": exc.message,
                    "details": exc.details,
                }
            },
        )


class RequestBodyReadTimeout(TimeoutError):
    def __init__(
        self,
        *,
        bytes_read: int,
        expected_bytes: int | None,
        elapsed_ms: float,
        idle_timeout: float,
    ):
        self.bytes_read = bytes_read
        self.expected_bytes = expected_bytes
        self.elapsed_ms = elapsed_ms
        self.idle_timeout = idle_timeout
        super().__init__("timed out reading request body")


async def read_json_payload(request: Request) -> Any:
    started_at = time.perf_counter()
    chunks: list[bytes] = []
    total_bytes = 0
    expected_bytes = parse_content_length(
        request.headers.get("content-length"))
    first_chunk_logged = False
    deadline_at = started_at + settings.request_body_timeout
    receive = request.receive

    while True:
        remaining_timeout = deadline_at - time.perf_counter()
        if remaining_timeout <= 0:
            raise RequestBodyReadTimeout(
                bytes_read=total_bytes,
                expected_bytes=expected_bytes,
                elapsed_ms=(time.perf_counter() - started_at) * 1000,
                idle_timeout=settings.request_body_timeout,
            )

        idle_timeout = min(
            settings.request_body_idle_timeout, remaining_timeout)
        try:
            message = await asyncio.wait_for(receive(), timeout=idle_timeout)
        except asyncio.TimeoutError as exc:
            raise RequestBodyReadTimeout(
                bytes_read=total_bytes,
                expected_bytes=expected_bytes,
                elapsed_ms=(time.perf_counter() - started_at) * 1000,
                idle_timeout=idle_timeout,
            ) from exc

        if message["type"] == "http.disconnect":
            raise ClientDisconnect()

        if message["type"] == "http.request":
            chunk = message.get("body", b"")
            if not first_chunk_logged:
                logger.info(
                    "request body first chunk bytes=%s expected=%s more_body=%s after_ms=%.1f",
                    len(chunk),
                    expected_bytes,
                    message.get("more_body", False),
                    (time.perf_counter() - started_at) * 1000,
                )
                first_chunk_logged = True
            if chunk:
                chunks.append(chunk)
                total_bytes += len(chunk)
                logger.info(
                    "request body chunk bytes=%s total=%s expected=%s more_body=%s",
                    len(chunk),
                    total_bytes,
                    expected_bytes,
                    message.get("more_body", False),
                )

            if not message.get("more_body", False):
                break

    logger.info(
        "request body read complete bytes=%s expected=%s duration_ms=%.1f",
        total_bytes,
        expected_bytes,
        (time.perf_counter() - started_at) * 1000,
    )
    if expected_bytes is not None and total_bytes != expected_bytes:
        logger.warning(
            "request body length mismatch bytes=%s expected=%s",
            total_bytes,
            expected_bytes,
        )
    body = b"".join(chunks)
    return json.loads(body)


def parse_content_length(value: str | None) -> int | None:
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def stream_headers(close_connection: bool) -> dict[str, str]:
    headers = {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    }
    if close_connection:
        headers["Connection"] = "close"
    return headers


def maybe_read_file_tool_call_completion(payload: dict[str, Any]) -> dict[str, Any] | None:
    read_file_tool_name = find_read_file_tool_name(payload.get("tools"))
    if read_file_tool_name is None:
        logger.info("read_file tool not present in request")
        return None

    messages = payload.get("messages")
    if not isinstance(messages, list):
        logger.info("read_file tool present but messages payload is not a list")
        return None

    if has_tool_results_after_latest_user(messages):
        logger.info(
            "read_file tool result present; forwarding tool result to GigaChat")
        return None

    file_ids = find_file_ids_in_latest_user_message(messages)
    if not file_ids:
        logger.warning(
            "read_file tool present but no Mattermost file IDs found in message text; message_snippets=%s",
            message_snippets(messages),
        )
        return None

    logger.info("read_file tool present; found file_ids=%s",
                ",".join(file_ids))

    tool_calls = [
        {
            "id": f"call_{uuid.uuid4().hex[:24]}",
            "type": "function",
            "function": {
                "name": read_file_tool_name,
                "arguments": json.dumps(
                    {"file_id": file_id, "offset": 0, "limit": 20000},
                    ensure_ascii=False,
                ),
            },
        }
        for file_id in file_ids
    ]

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": normalize_model(str(payload.get("model") or settings.gigachat_model)),
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": tool_calls,
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


async def maybe_generic_tool_completion(payload: dict[str, Any]) -> dict[str, Any] | None:
    tools = payload.get("tools")
    messages = payload.get("messages")
    if not isinstance(tools, list) or not tools:
        return None
    if not isinstance(messages, list) or has_tool_results_after_latest_user(messages):
        return None

    decision_payload = build_tool_decision_payload(payload, tools, messages)
    response = await gigachat.chat_completions(decision_payload)
    if response.status_code >= 400:
        logger.warning(
            "generic MCP tool decision failed status=%s", response.status_code)
        return None

    completion = normalize_chat_completion_response(
        response.json(), decision_payload)
    content = completion["choices"][0]["message"].get("content") or ""
    decision = parse_tool_decision(content)
    if decision is None:
        logger.warning(
            "generic MCP tool decision was not parseable: %s", content[:500])
        return None

    tool_calls = decision.get("tool_calls")
    if isinstance(tool_calls, list) and tool_calls:
        valid_tool_calls = build_openai_tool_calls(tool_calls, tools)
        if valid_tool_calls:
            return tool_call_completion(
                model=normalize_model(
                    str(payload.get("model") or settings.gigachat_model)),
                tool_calls=valid_tool_calls,
            )

    final_answer = decision.get("final_answer")
    if isinstance(final_answer, str) and final_answer.strip():
        return text_completion(
            model=normalize_model(
                str(payload.get("model") or settings.gigachat_model)),
            content=final_answer.strip(),
        )

    return None


def build_tool_decision_payload(
    payload: dict[str, Any],
    tools: list[Any],
    messages: list[Any],
) -> dict[str, Any]:
    tool_specs = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function")
        if isinstance(function, dict):
            tool_specs.append(
                {
                    "name": function.get("name"),
                    "description": function.get("description", ""),
                    "parameters": function.get("parameters") or {},
                }
            )
        elif isinstance(tool.get("name"), str):
            tool_specs.append(
                {
                    "name": tool.get("name"),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema") or tool.get("parameters") or {},
                }
            )

    conversation = []
    for message in messages[-8:]:
        if not isinstance(message, dict):
            continue
        conversation.append(
            {
                "role": message.get("role"),
                "content": message_text(message),
            }
        )

    system_prompt = (
        "You are an OpenAI-compatible tool router for Mattermost MCP tools. "
        "Return ONLY valid JSON, without markdown. "
        "If the user request requires Mattermost data or an action, choose the best tool. "
        "If no tool is needed, answer directly. "
        "Use exactly one of these JSON shapes: "
        '{"tool_calls":[{"name":"tool_name","arguments":{}}]} '
        'or {"final_answer":"text"}. '
        "Do not invent tool names. Arguments must match the tool schema as well as possible. "
        "Never put human-readable channel names, user names, or display names into arguments "
        "whose names end with _id. If the user gave a name but the target tool needs an id, "
        "first choose a lookup or search tool that can resolve that name."
    )

    user_prompt = json.dumps(
        {
            "available_tools": tool_specs,
            "conversation": conversation,
        },
        ensure_ascii=False,
    )

    return {
        "model": payload.get("model") or settings.gigachat_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0,
        "stream": False,
    }


def parse_tool_decision(content: str) -> dict[str, Any] | None:
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            data = json.loads(text[start: end + 1])
        except json.JSONDecodeError:
            return None

    return data if isinstance(data, dict) else None


def build_openai_tool_calls(tool_calls: list[Any], tools: list[Any]) -> list[dict[str, Any]]:
    allowed_names = set(tool_names(tools))
    openai_tool_calls: list[dict[str, Any]] = []
    for tool_call in tool_calls[:3]:
        if not isinstance(tool_call, dict):
            continue
        name = tool_call.get("name")
        if not isinstance(name, str) or name not in allowed_names:
            continue
        arguments = tool_call.get("arguments") or {}
        if isinstance(arguments, str):
            arguments_json = arguments
        else:
            arguments_json = json.dumps(arguments, ensure_ascii=False)
        openai_tool_calls.append(
            {
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": arguments_json,
                },
            }
        )
    return openai_tool_calls


def tool_call_completion(model: str, tool_calls: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": tool_calls,
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def text_completion(model: str, content: str) -> dict[str, Any]:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def find_read_file_tool_name(tools: Any) -> str | None:
    if not isinstance(tools, list):
        return None

    fallback_name: str | None = None
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function")
        if isinstance(function, dict) and function.get("name") == "read_file":
            return "read_file"
        if tool.get("name") == "read_file":
            return "read_file"

        name = None
        if isinstance(function, dict) and isinstance(function.get("name"), str):
            name = function["name"]
        elif isinstance(tool.get("name"), str):
            name = tool["name"]

        if name and name.lower().endswith("read_file"):
            fallback_name = name

    return fallback_name


def tool_names(tools: Any) -> list[str]:
    names: list[str] = []
    if not isinstance(tools, list):
        return names
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function")
        if isinstance(function, dict) and isinstance(function.get("name"), str):
            names.append(function["name"])
        elif isinstance(tool.get("name"), str):
            names.append(tool["name"])
    return names


def has_tool_results_after_latest_user(messages: list[Any]) -> bool:
    latest_user_index = -1
    for index, message in enumerate(messages):
        if isinstance(message, dict) and message.get("role") == "user":
            latest_user_index = index

    for message in messages[latest_user_index + 1:]:
        if isinstance(message, dict) and message.get("role") == "tool":
            return True
    return False


def find_file_ids(messages: list[Any]) -> list[str]:
    seen: set[str] = set()
    file_ids: list[str] = []
    for message in messages:
        for match in extract_file_ids_from_message(message):
            if match not in seen:
                seen.add(match)
                file_ids.append(match)
    return file_ids


def find_file_ids_in_latest_user_message(messages: list[Any]) -> list[str]:
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "user":
            return find_file_ids([message])
    return find_file_ids(messages)


def extract_file_ids_from_message(message: Any) -> list[str]:
    if not isinstance(message, dict):
        return []

    file_ids = FILE_ID_RE.findall(message_text(message))
    content = message.get("content")
    if isinstance(content, list):
        for part in content:
            file_ids.extend(extract_file_ids_from_value(
                part, in_file_context=True))

    for key in FILE_CONTAINER_KEYS | FILE_ID_KEYS:
        if key in message:
            file_ids.extend(extract_file_ids_from_value(
                message[key], in_file_context=True))

    return file_ids


def extract_file_ids_from_value(value: Any, *, in_file_context: bool) -> list[str]:
    if isinstance(value, str):
        return FILE_ID_RE.findall(value) if in_file_context else []

    if isinstance(value, list):
        file_ids: list[str] = []
        for item in value:
            file_ids.extend(extract_file_ids_from_value(
                item, in_file_context=in_file_context))
        return file_ids

    if not isinstance(value, dict):
        return []

    value_type = value.get("type")
    nested_file_context = in_file_context or (
        isinstance(value_type, str)
        and any(marker in value_type.lower() for marker in ("file", "image", "attachment"))
    )

    file_ids: list[str] = []
    for key, nested_value in value.items():
        key_file_context = (
            nested_file_context
            or key in FILE_ID_KEYS
            or key in FILE_CONTAINER_KEYS
            or key.lower().endswith("file_id")
            or key.lower().endswith("file_ids")
        )
        file_ids.extend(
            extract_file_ids_from_value(
                nested_value, in_file_context=key_file_context)
        )
    return file_ids


def message_text(message: Any) -> str:
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
        return "\n".join(parts)
    return ""


def message_snippets(messages: list[Any]) -> list[str]:
    snippets: list[str] = []
    for message in messages[-5:]:
        text = message_text(message).replace("\n", " ").strip()
        if len(text) > 240:
            text = text[:240] + "..."
        if text:
            snippets.append(text)
    return snippets


def message_diagnostics(messages: Any) -> list[dict[str, Any]]:
    if not isinstance(messages, list):
        return []

    diagnostics: list[dict[str, Any]] = []
    for message in messages[-6:]:
        if not isinstance(message, dict):
            continue
        text = message_text(message).replace("\n", " ").strip()
        if len(text) > 220:
            text = text[:220] + "..."
        diagnostics.append(
            {
                "role": message.get("role"),
                "has_tool_calls": bool(message.get("tool_calls")),
                "tool_call_id": message.get("tool_call_id"),
                "file_ids": extract_file_ids_from_message(message),
                "snippet": text,
            }
        )
    return diagnostics


def normalize_chat_completion_response(
    payload: dict[str, Any],
    request_payload: dict[str, Any],
) -> dict[str, Any]:
    now = int(time.time())
    model = payload.get("model") or normalize_model(
        str(request_payload.get("model") or settings.gigachat_model)
    )
    choices = payload.get("choices")
    if not isinstance(choices, list):
        choices = []

    normalized_choices: list[dict[str, Any]] = []
    for index, choice in enumerate(choices):
        if not isinstance(choice, dict):
            continue
        message = choice.get("message")
        if not isinstance(message, dict):
            message = {
                "role": "assistant",
                "content": choice.get("text", ""),
            }
        message.setdefault("role", "assistant")
        message.setdefault("content", "")
        normalized_choices.append(
            {
                "index": choice.get("index", index),
                "message": message,
                "finish_reason": normalize_finish_reason(choice.get("finish_reason")),
            }
        )

    if not normalized_choices:
        normalized_choices.append(
            {
                "index": 0,
                "message": {"role": "assistant", "content": ""},
                "finish_reason": "stop",
            }
        )

    return {
        "id": payload.get("id") or f"chatcmpl-{uuid.uuid4().hex}",
        "object": payload.get("object") or "chat.completion",
        "created": payload.get("created") or now,
        "model": model,
        "choices": normalized_choices,
        "usage": payload.get("usage")
        or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def openai_error_response(response: Any) -> JSONResponse:
    details: Any
    try:
        details = response.json()
    except ValueError:
        details = response.text

    message = "GigaChat API request failed"
    if isinstance(details, dict):
        message = str(details.get("message")
                      or details.get("error") or message)
    elif isinstance(details, str) and details:
        message = details[:500]

    return JSONResponse(
        status_code=response.status_code,
        content={
            "error": {
                "message": message,
                "type": "gigachat_error",
                "code": response.status_code,
                "details": details,
            }
        },
    )


async def openai_sse_from_completion(completion: dict[str, Any]) -> AsyncIterator[bytes]:
    choice = completion["choices"][0]
    message = choice.get("message", {})
    content = message.get("content") or ""
    model = completion["model"]
    chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    start_chunk = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
    }
    yield sse_data(start_chunk)

    stop_finish_reason = normalize_finish_reason(choice.get("finish_reason"))
    final_chunk = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {"content": content},
                "finish_reason": stop_finish_reason,
            }
        ],
    }
    yield sse_data(final_chunk)
    logger.info("openai SSE complete finish_reason=%s", stop_finish_reason)
    yield b"data: [DONE]\n\n"


async def openai_sse_from_gigachat_stream(payload: dict[str, Any]) -> AsyncIterator[bytes]:
    try:
        async with asyncio.timeout(settings.request_timeout):
            async for chunk in gigachat.stream_chat_completions(payload):
                yield chunk
    except GigaChatError as exc:
        logger.warning(
            "gigachat streaming request failed status=%s message=%s details=%s",
            exc.status_code,
            exc.message,
            exc.details,
        )
        yield openai_sse_error_text(
            f"GigaChat API request failed: {exc.message}",
            payload,
        )
    except TimeoutError:
        logger.warning(
            "gigachat streaming request timed out after %.1fs",
            settings.request_timeout,
        )
        yield openai_sse_error_text(
            f"GigaChat API request timed out after {settings.request_timeout:.0f}s",
            payload,
        )
    except Exception as exc:
        logger.exception("gigachat streaming request crashed")
        yield openai_sse_error_text(
            f"GigaChat API request failed: {type(exc).__name__}: {exc}",
            payload,
        )


def openai_sse_error_text(message: str, payload: dict[str, Any]) -> bytes:
    now = int(time.time())
    model = normalize_model(
        str(payload.get("model") or settings.gigachat_model))
    chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
    error_payload = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": now,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": message},
                "finish_reason": None,
            }
        ],
    }
    stop_payload = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": now,
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    return sse_data(error_payload) + sse_data(stop_payload) + b"data: [DONE]\n\n"


async def openai_sse_from_gigachat_completion(
    payload: dict[str, Any],
) -> AsyncIterator[bytes]:
    non_stream_payload = dict(payload)
    non_stream_payload["stream"] = False
    model = normalize_model(
        str(payload.get("model") or settings.gigachat_model))
    chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    yield sse_data(
        {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        }
    )

    try:
        async with asyncio.timeout(settings.request_timeout):
            response = await gigachat.chat_completions(non_stream_payload)
        if response.status_code >= 400:
            content = gigachat_error_text(response)
            finish_reason = "stop"
        else:
            completion = normalize_chat_completion_response(
                response.json(), payload)
            choice = completion["choices"][0]
            content = choice.get("message", {}).get("content") or ""
            finish_reason = normalize_finish_reason(
                choice.get("finish_reason"))
            log_completion_summary(
                "gigachat completion normalized for SSE",
                content=content,
                finish_reason=finish_reason,
            )
    except GigaChatError as exc:
        logger.warning("gigachat completion request failed: %s", exc.message)
        content = f"GigaChat API request failed: {exc.message}"
        finish_reason = "stop"
    except TimeoutError:
        logger.warning(
            "gigachat completion request timed out after %.1fs",
            settings.request_timeout,
        )
        content = f"GigaChat API request timed out after {settings.request_timeout:.0f}s"
        finish_reason = "stop"
    except Exception as exc:
        logger.exception("gigachat completion request crashed")
        content = f"GigaChat API request failed: {type(exc).__name__}: {exc}"
        finish_reason = "stop"

    yield sse_data(
        {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": content},
                    "finish_reason": finish_reason,
                }
            ],
        }
    )
    logger.info("openai SSE complete finish_reason=%s", finish_reason)
    yield b"data: [DONE]\n\n"


def gigachat_error_text(response: Any) -> str:
    details: Any
    try:
        details = response.json()
    except ValueError:
        details = response.text

    if isinstance(details, dict):
        message = details.get("message") or details.get("error")
        if message:
            return f"GigaChat API request failed: {message}"

    if isinstance(details, str) and details:
        return f"GigaChat API request failed: {details[:500]}"

    return f"GigaChat API request failed with HTTP {response.status_code}"


def normalize_finish_reason(value: Any) -> str:
    if isinstance(value, str) and value:
        return value
    return "stop"


def log_completion_summary(prefix: str, *, content: Any, finish_reason: str) -> None:
    text = content if isinstance(content, str) else str(content)
    logger.info(
        "%s content_chars=%s finish_reason=%s content_preview=%r",
        prefix,
        len(text),
        finish_reason,
        text[:160],
    )


async def openai_sse_from_tool_call_completion(completion: dict[str, Any]) -> AsyncIterator[bytes]:
    choice = completion["choices"][0]
    message = choice["message"]
    model = completion["model"]
    chunk_id = completion["id"]
    created = int(time.time())

    yield sse_data(
        {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        }
    )

    for index, tool_call in enumerate(message["tool_calls"]):
        yield sse_data(
            {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": index,
                                    "id": tool_call["id"],
                                    "type": "function",
                                    "function": tool_call["function"],
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
            }
        )

    yield sse_data(
        {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
        }
    )
    yield b"data: [DONE]\n\n"


def sse_data(payload: dict[str, Any]) -> bytes:
    return f"data: {json.dumps(payload, ensure_ascii=True)}\n\n".encode("utf-8")


def run() -> None:
    uvicorn.run(
        "gigachat_openai_proxy.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
        timeout_keep_alive=0,
    )


if __name__ == "__main__":
    run()
