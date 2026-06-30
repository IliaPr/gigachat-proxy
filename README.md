# GigaChat OpenAI-compatible proxy

FastAPI proxy for using GigaChat from clients that only support an OpenAI-compatible provider, including Mattermost agents.

The proxy exposes:

- `GET /healthz`
- `GET /v1`
- `GET /v1/models`
- `POST /v1/chat/completions`

It handles GigaChat OAuth locally, caches the access token in memory, refreshes it before expiry, and forwards chat completion requests to GigaChat.

## Run

Install dependencies with Poetry:

```bash
poetry env use python3.13
poetry install
```

```bash
cp .env.example .env
```

Edit `.env`, then export it into the shell:

```bash
set -a
source .env
set +a
poetry run uvicorn gigachat_openai_proxy.main:app --host 127.0.0.1 --port 8080
```

The service will listen on `http://127.0.0.1:8080` by default.

You can also run the console script:

```bash
poetry run gigachat-openai-proxy
```

## LiteLLM proxy

The project can also run through LiteLLM Proxy instead of the local FastAPI proxy.
This is the preferred compatibility test for Mattermost Agents because LiteLLM
already implements an OpenAI-compatible proxy and a GigaChat provider.

```bash
poetry env use python3.13
poetry install
set -a
source .env
set +a
poetry run litellm --config litellm_config.yaml --host 127.0.0.1 --port 4000
```

Configure Mattermost Agents for LiteLLM with:

- Base URL: `http://127.0.0.1:4000/v1`
- API key: value of `PROXY_API_KEY`
- Model: `GigaChat`
- Use Responses API: disabled

## Mattermost configuration

Configure the Mattermost agents OpenAI-compatible provider with:

- Base URL: `http://127.0.0.1:8080/v1`
- API key: value of `PROXY_API_KEY`, or any non-empty value if `PROXY_API_KEY` is unset
- Model: value of `GIGACHAT_MODEL`, for example `GigaChat`
- Use Responses API: disabled

If Mattermost runs in Docker, `127.0.0.1` points to the container itself. Use the host address reachable from that container instead, for example `http://host.docker.internal:8080/v1` on Docker Desktop.

By default, Mattermost tool handling is disabled and the proxy forwards plain chat messages to GigaChat after removing OpenAI-only fields that GigaChat rejects. This is the most stable mode for normal dialog.

Set `ENABLE_MATTERMOST_TOOLS=true` to enable experimental Mattermost MCP tool routing and attachment reading.

Mattermost Agents can expose readable attachments through its `read_file` tool. When tool handling is enabled, this proxy returns synthetic OpenAI-compatible `read_file` tool calls when it sees Mattermost file IDs, then converts the following tool result into plain text context for GigaChat.

Office/PDF/TXT attachments are supported through that `read_file` loop when bot tools are enabled.

If an attachment is not read, check the proxy logs for:

- `tool_names=[...]`: must contain `read_file` or a namespaced name ending with `read_file`
- `message diagnostics=... file_ids=[...]`: must show the attached Mattermost file ID on the latest user message
- `returning synthetic read_file tool calls count=...`: confirms the proxy asked Mattermost Agents to read the file

When `ENABLE_MATTERMOST_TOOLS=true`, Mattermost MCP/tools are exposed through an OpenAI-compatible tool loop. The proxy asks GigaChat to choose a tool, returns `tool_calls` to Mattermost, then converts the following `role=tool` result into plain text context for GigaChat.

Image attachments are different: Mattermost Agents treats supported images as multimodal files, while GigaChat expects images to be uploaded to `/files` first and then referenced from `messages[].attachments`. This proxy does not yet translate OpenAI-style image payloads into GigaChat file attachments.

For GigaChat Lite, use model id `GigaChat`. The B2B `/models` endpoint may not expose `GigaChat-2-Lite`; if Mattermost sends that name, GigaChat returns `No such model`.

## Smoke test

```bash
curl http://127.0.0.1:8080/healthz
```

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Authorization: Bearer change-me" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "GigaChat",
    "messages": [{"role": "user", "content": "Привет"}],
    "stream": false
  }'
```

## TLS notes

If GigaChat TLS verification fails because the required CA is missing locally, set `GIGACHAT_CA_BUNDLE` to a PEM bundle. `GIGACHAT_VERIFY_SSL=false` is configured in `.env.example` for the requested environment, but a CA bundle is preferable in production.
