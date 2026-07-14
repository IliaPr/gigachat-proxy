from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Any

import httpx

from gigachat_openai_proxy.config import Settings


class MattermostClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(settings.request_timeout),
            verify=settings.httpx_verify,
        )

    @property
    def is_configured(self) -> bool:
        return bool(self.settings.mattermost_site_url and self.settings.mattermost_access_token)

    async def close(self) -> None:
        await self._client.aclose()

    async def download_file(self, file_id: str, destination: Path) -> Path:
        self._require_configured()
        response = await self._client.get(
            f"{self.settings.mattermost_site_url}/api/v4/files/{file_id}",
            headers=self._headers(),
        )
        response.raise_for_status()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(response.content)
        return destination

    async def upload_file(self, channel_id: str, path: Path) -> list[str]:
        self._require_configured()
        mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        with path.open("rb") as file_handle:
            response = await self._client.post(
                f"{self.settings.mattermost_site_url}/api/v4/files",
                headers=self._headers(),
                data={"channel_id": channel_id},
                files={"files": (path.name, file_handle, mime_type)},
            )
        response.raise_for_status()
        payload = response.json()
        file_infos = payload.get("file_infos")
        if not isinstance(file_infos, list):
            return []
        return [
            file_info["id"]
            for file_info in file_infos
            if isinstance(file_info, dict) and isinstance(file_info.get("id"), str)
        ]

    async def upload_file_bytes(
        self,
        *,
        channel_id: str,
        filename: str,
        content: bytes,
        content_type: str | None = None,
    ) -> list[str]:
        self._require_configured()
        mime_type = content_type or mimetypes.guess_type(filename)[
            0] or "application/octet-stream"
        response = await self._client.post(
            f"{self.settings.mattermost_site_url}/api/v4/files",
            headers=self._headers(),
            data={"channel_id": channel_id},
            files={"files": (filename, content, mime_type)},
        )
        response.raise_for_status()
        payload = response.json()
        file_infos = payload.get("file_infos")
        if not isinstance(file_infos, list):
            return []
        return [
            file_info["id"]
            for file_info in file_infos
            if isinstance(file_info, dict) and isinstance(file_info.get("id"), str)
        ]

    async def create_post(
        self,
        *,
        channel_id: str,
        message: str,
        file_ids: list[str] | None = None,
        root_id: str | None = None,
    ) -> dict[str, Any]:
        self._require_configured()
        payload: dict[str, Any] = {
            "channel_id": channel_id,
            "message": message,
        }
        if file_ids:
            payload["file_ids"] = file_ids
        if root_id:
            payload["root_id"] = root_id

        response = await self._client.post(
            f"{self.settings.mattermost_site_url}/api/v4/posts",
            headers={**self._headers(), "Content-Type": "application/json"},
            json=payload,
        )
        response.raise_for_status()
        data = response.json()
        return data if isinstance(data, dict) else {"response": data}

    async def upload_file_post(
        self,
        *,
        channel_id: str,
        path: Path,
        message: str,
        root_id: str | None = None,
    ) -> dict[str, Any]:
        file_ids = await self.upload_file(channel_id, path)
        return await self.create_post(
            channel_id=channel_id,
            message=message,
            file_ids=file_ids,
            root_id=root_id,
        )

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.settings.mattermost_access_token}"}

    def _require_configured(self) -> None:
        if not self.is_configured:
            raise RuntimeError(
                "MATTERMOST_SITE_URL and MATTERMOST_ACCESS_TOKEN are required"
            )
