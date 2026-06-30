from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


DEFAULT_TOKEN_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
DEFAULT_API_BASE_URL = "https://gigachat.devices.sberbank.ru/api/v1"
DEFAULT_SCOPE = "GIGACHAT_API_B2B"
DEFAULT_MODEL = "GigaChat"


@dataclass(frozen=True)
class Settings:
    host: str
    port: int
    proxy_api_key: str | None
    gigachat_auth_key: str
    gigachat_scope: str
    gigachat_token_url: str
    gigachat_api_base_url: str
    gigachat_model: str
    enable_mattermost_tools: bool
    request_timeout: float
    request_body_timeout: float
    request_body_idle_timeout: float
    verify_ssl: bool
    ca_bundle: str | None

    @classmethod
    def from_env(cls) -> "Settings":
        auth_key = (
            os.getenv("GIGACHAT_CREDENTIALS")
            or os.getenv("GIGACHAT_AUTH_KEY")
            or ""
        ).strip()
        if not auth_key:
            raise RuntimeError("GIGACHAT_CREDENTIALS is required")

        return cls(
            host=os.getenv("HOST", "127.0.0.1"),
            port=int(os.getenv("PORT", "8080")),
            proxy_api_key=os.getenv("PROXY_API_KEY") or None,
            gigachat_auth_key=auth_key,
            gigachat_scope=os.getenv("GIGACHAT_SCOPE", DEFAULT_SCOPE),
            gigachat_token_url=os.getenv(
                "GIGACHAT_TOKEN_URL", DEFAULT_TOKEN_URL),
            gigachat_api_base_url=os.getenv(
                "GIGACHAT_API_BASE_URL", DEFAULT_API_BASE_URL
            ).rstrip("/"),
            gigachat_model=os.getenv("GIGACHAT_MODEL", DEFAULT_MODEL),
            enable_mattermost_tools=os.getenv(
                "ENABLE_MATTERMOST_TOOLS", "false"
            ).lower()
            in {"1", "true", "yes"},
            request_timeout=float(os.getenv("REQUEST_TIMEOUT", "120")),
            request_body_timeout=float(
                os.getenv("REQUEST_BODY_TIMEOUT", "60")),
            request_body_idle_timeout=float(
                os.getenv("REQUEST_BODY_IDLE_TIMEOUT", "10")),
            verify_ssl=os.getenv("GIGACHAT_VERIFY_SSL", "true").lower()
            not in {"0", "false", "no"},
            ca_bundle=os.getenv("GIGACHAT_CA_BUNDLE") or None,
        )

    @property
    def httpx_verify(self) -> bool | str:
        if not self.verify_ssl:
            return False
        return self.ca_bundle or True
