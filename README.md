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
poetry env use python3.12
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
poetry env use python3.12
poetry install
set -a
source .env
set +a
poetry run litellm --config litellm_config.yaml --host 127.0.0.1 --port 4000
```

Configure Mattermost Agents for LiteLLM with:

- Base URL: `http://127.0.0.1:4000/v1`
- API key: value of `PROXY_API_KEY`
- Model: `GigaChat` or `gpt-4o`
- Use Responses API: disabled

This direct route is simplest, but it bypasses the FastAPI adapter and therefore
cannot use the adapter's Mattermost attachment/tool handling.

`litellm_config.yaml` also registers common OpenAI model aliases and a wildcard
route to GigaChat, because some Mattermost Agents requests can still send
`gpt-4o` even when the configured provider model is different. The LiteLLM
config disables SSL verification for GigaChat requests in the same environment
where `GIGACHAT_VERIFY_SSL=false` is needed by the local FastAPI proxy.

## Adapter in front of LiteLLM

For Mattermost attachment handling while still using LiteLLM as the model
upstream, run both services:

```bash
poetry run litellm --config litellm_config.yaml --host 127.0.0.1 --port 4000
```

In another shell, set:

```bash
LITELLM_BASE_URL=http://127.0.0.1:4000/v1
LITELLM_API_KEY=$PROXY_API_KEY
ENABLE_MATTERMOST_TOOLS=true
```

Then run the FastAPI adapter:

```bash
poetry run uvicorn gigachat_openai_proxy.main:app --host 127.0.0.1 --port 8080
```

Configure Mattermost Agents with the adapter URL:

- Base URL: `http://127.0.0.1:8080/v1`
- API key: value of `PROXY_API_KEY`
- Model: `GigaChat` or `gpt-4o`
- Use Responses API: disabled

With `LITELLM_BASE_URL` set, `/v1/chat/completions` forwards normal model calls
to LiteLLM. The adapter still intercepts Mattermost tool loops, including the
synthetic `read_file` calls used for attachments.

## File processor tools

For generated files, the model should call a tool rather than trying to return
binary data in a chat response. The adapter exposes a tool-friendly Excel
processor endpoint:

```bash
curl http://127.0.0.1:8080/file-processing/excel/edit \
  -H "Authorization: Bearer $PROXY_API_KEY" \
  -F file=@/path/to/input.xlsx \
  -F output_filename=edited.xlsx \
  -F 'operations={
    "operations": [
      {"op": "set_cell", "sheet": "Sheet1", "cell": "A1", "value": "Updated"},
      {"op": "append_row", "sheet": "Sheet1", "values": ["Total", 123]},
      {"op": "set_column_width", "sheet": "Sheet1", "column": "A", "width": 24}
    ]
  }' \
  --output edited.xlsx
```

Supported Excel operations:

- `create_sheet`: `{"op":"create_sheet","title":"Report"}`
- `rename_sheet`: `{"op":"rename_sheet","sheet":"Sheet1","title":"Report"}`
- `delete_sheet`: `{"op":"delete_sheet","sheet":"Old"}`
- `set_cell`: `{"op":"set_cell","sheet":"Report","cell":"B2","value":42}`
- `append_row`: `{"op":"append_row","sheet":"Report","values":["A","B"]}`
- `set_column_width`: `{"op":"set_column_width","sheet":"Report","column":"A","width":20}`
- `style_cell`: `{"op":"style_cell","sheet":"Report","cell":"A1","bold":true,"fill_color":"#FFE599"}`

The intended Agents integration is:

1. Agents/tool gets the original file bytes from Mattermost.
2. The tool asks the model for explicit JSON operations.
3. The tool calls `/file-processing/excel/edit`.
4. The tool uploads the returned `.xlsx` back to the same thread.

If Agents does not expose an upload/create-file tool and this external adapter
must upload directly through the Mattermost REST API, also set:

```bash
MATTERMOST_SITE_URL=https://mattermost.example.com
MATTERMOST_ACCESS_TOKEN=mattermost-personal-or-bot-access-token
```

This is not required when Mattermost Agents exposes its own tool for uploading
or creating files. In that case the adapter should route a tool call back to
Agents instead of using Mattermost REST credentials.

The upload/post helper is exposed as an authenticated multipart endpoint for
trusted server-side tools:

```bash
curl http://127.0.0.1:8080/mattermost/file-posts \
  -H "Authorization: Bearer $PROXY_API_KEY" \
  -F channel_id=CHANNEL_ID \
  -F message="Edited file" \
  -F file=@/path/to/edited.xlsx
```

## Mattermost configuration

Configure the Mattermost agents OpenAI-compatible provider with:

- Base URL: `http://127.0.0.1:8080/v1`
- API key: value of `PROXY_API_KEY`, or any non-empty value if `PROXY_API_KEY` is unset
- Model: value of `GIGACHAT_MODEL`, for example `GigaChat`
- Use Responses API: disabled

If Mattermost runs in Docker, `127.0.0.1` points to the container itself. Use the host address reachable from that container instead, for example `http://host.docker.internal:8080/v1` on Docker Desktop.

By default, Mattermost tool handling is disabled and the proxy forwards plain
chat messages to the configured upstream. Without `LITELLM_BASE_URL`, the
upstream is GigaChat directly. With `LITELLM_BASE_URL`, the upstream is LiteLLM.

Set `ENABLE_MATTERMOST_TOOLS=true` to enable experimental Mattermost MCP tool routing and attachment reading.

Mattermost Agents can expose readable attachments through its `read_file` tool.
When tool handling is enabled, this proxy returns synthetic OpenAI-compatible
`read_file` tool calls when it sees Mattermost file IDs, then converts the
following tool result into plain text context for the configured upstream.

Office/PDF/TXT attachments are supported through that `read_file` loop when bot tools are enabled.

If an attachment is not read, check the proxy logs for:

- `tool_names=[...]`: must contain `read_file` or a namespaced name ending with `read_file`
- `message diagnostics=... file_ids=[...]`: must show the attached Mattermost file ID on the latest user message
- `returning synthetic read_file tool calls count=...`: confirms the proxy asked Mattermost Agents to read the file

When `ENABLE_MATTERMOST_TOOLS=true`, Mattermost MCP/tools are exposed through an
OpenAI-compatible tool loop. The proxy asks the configured upstream to choose a
tool, returns `tool_calls` to Mattermost, then converts the following
`role=tool` result into plain text context for the upstream.

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
