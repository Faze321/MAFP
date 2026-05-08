from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class OpenRouterConfig:
    api_key: str | None
    base_url: str
    model: str
    http_referer: str | None = None
    title: str | None = None
    timeout_seconds: float = 90.0

    @classmethod
    def from_env(cls, model: str | None = None) -> "OpenRouterConfig":
        return cls(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            model=model or os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct"),
            http_referer=os.getenv("OPENROUTER_HTTP_REFERER"),
            title=os.getenv("OPENROUTER_TITLE", "MAPF UrbanEV"),
            timeout_seconds=float(os.getenv("OPENROUTER_TIMEOUT_SECONDS", "90")),
        )
