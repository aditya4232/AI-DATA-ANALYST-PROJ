from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class AppConfig:
    provider: str
    api_key: str
    base_url: str
    model: str
    temperature: float = 0.0
    max_rows_preview: int = 8

    @classmethod
    def from_env(cls) -> "AppConfig":
        provider = os.getenv("LLM_PROVIDER", "nvidia").strip().lower()
        api_key = _first_non_empty(
            os.getenv("LLM_API_KEY"),
            os.getenv("NVIDIA_API_KEY"),
            os.getenv("OPENAI_API_KEY"),
            "",
        )
        base_url = _first_non_empty(
            os.getenv("LLM_BASE_URL"),
            os.getenv("NVIDIA_BASE_URL"),
            "https://integrate.api.nvidia.com/v1",
        )
        model = _first_non_empty(
            os.getenv("LLM_MODEL"),
            os.getenv("NVIDIA_MODEL"),
            "meta/llama-3.1-70b-instruct",
        )
        temperature = float(os.getenv("LLM_TEMPERATURE", "0"))
        return cls(provider=provider, api_key=api_key, base_url=base_url, model=model, temperature=temperature)

    def resolved(self, *, api_key: str | None = None, model: str | None = None, base_url: str | None = None) -> "AppConfig":
        return AppConfig(
            provider=self.provider,
            api_key=(api_key or self.api_key).strip(),
            base_url=(base_url or self.base_url).strip(),
            model=(model or self.model).strip(),
            temperature=self.temperature,
            max_rows_preview=self.max_rows_preview,
        )

    @property
    def provider_label(self) -> str:
        return self.provider.replace("_", " ").title()


def _first_non_empty(*values: str) -> str:
    for value in values:
        if value and value.strip():
            return value.strip()
    return ""
