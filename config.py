from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import os


@dataclass(frozen=True)
class OpenRouterConfig:
    api_key: str | None
    base_url: str
    model: str
    http_referer: str | None = None
    title: str | None = None
    timeout_seconds: float = 90.0

    @classmethod
    def from_json(
        cls,
        path: Path,
        *,
        model: str | None = None,
        required: bool = True,
    ) -> "OpenRouterConfig":
        if not path.exists():
            if required:
                raise FileNotFoundError(
                    f"Config file not found: {path}. Copy config.example.json to config.json."
                )
            return cls.default(model=model)

        raw = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(raw, dict):
            raise ValueError(f"Config file must contain a JSON object: {path}")
        settings = raw.get("openrouter", raw)
        if not isinstance(settings, dict):
            raise ValueError('Config key "openrouter" must contain a JSON object')

        return cls(
            api_key=optional_str(settings.get("api_key", os.getenv("OPENROUTER_API_KEY"))),
            base_url=optional_str(settings.get("base_url")) or "https://openrouter.ai/api/v1",
            model=model or optional_str(settings.get("model")) or "meta-llama/llama-3.1-8b-instruct",
            http_referer=optional_str(settings.get("http_referer")),
            title=optional_str(settings.get("title")) or "MAPF UrbanEV",
            timeout_seconds=float(settings.get("timeout_seconds", 90)),
        )

    @classmethod
    def default(cls, *, model: str | None = None) -> "OpenRouterConfig":
        return cls(
            api_key=None,
            base_url="https://openrouter.ai/api/v1",
            model=model or "meta-llama/llama-3.1-8b-instruct",
            title="MAPF UrbanEV",
        )


def optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
