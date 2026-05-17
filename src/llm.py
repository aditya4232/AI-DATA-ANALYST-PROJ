from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - supports older OpenAI SDKs
    OpenAI = None

import openai as legacy_openai

from config import AppConfig
from prompts import build_system_prompt


@dataclass(frozen=True)
class AnalysisPlan:
    answer_kind: str
    summary: str
    code: str
    chart_title: str = ""
    key_insights: list[str] | None = None
    caveats: list[str] | None = None
    next_step: str = ""
    raw_text: str = ""

    @property
    def answer(self) -> str:
        return self.summary


class LLMError(RuntimeError):
    pass


def generate_analysis_plan(prompt: str, config: AppConfig) -> AnalysisPlan:
    if not config.api_key:
        raise LLMError(
            f"Missing API key for {config.provider_label}. Set NVIDIA_API_KEY, OPENAI_API_KEY, or LLM_API_KEY."
        )

    content = _create_chat_completion(prompt, config)
    return parse_analysis_plan(content)


def _create_chat_completion(prompt: str, config: AppConfig) -> str:
    messages = [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": prompt},
    ]

    if OpenAI is not None:
        client = OpenAI(api_key=config.api_key, base_url=config.base_url)
        response = client.chat.completions.create(
            model=config.model,
            temperature=config.temperature,
            messages=messages,
        )
        return response.choices[0].message.content or ""

    legacy_openai.api_key = config.api_key
    if config.base_url:
        legacy_openai.api_base = config.base_url
    response = legacy_openai.ChatCompletion.create(
        model=config.model,
        temperature=config.temperature,
        messages=messages,
    )
    return response["choices"][0]["message"]["content"] or ""


def parse_analysis_plan(raw_text: str) -> AnalysisPlan:
    content = _strip_code_fences(raw_text)
    content = _extract_json_object(content)
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise LLMError(f"The model did not return valid JSON: {exc}") from exc

    answer_kind = str(payload.get("answer_kind", "text")).strip().lower()
    summary = str(payload.get("summary", payload.get("answer", ""))).strip()
    code = str(payload.get("code", "")).strip()
    chart_title = str(payload.get("chart_title", "")).strip()
    key_insights = _coerce_text_list(payload.get("key_insights") or payload.get("insights"))
    caveats = _coerce_text_list(payload.get("caveats") or payload.get("notes"))
    next_step = str(payload.get("next_step", "")).strip()

    if answer_kind not in {"table", "chart", "text", "clarification"}:
        answer_kind = "text"

    return AnalysisPlan(
        answer_kind=answer_kind,
        summary=summary,
        code=code,
        chart_title=chart_title,
        key_insights=key_insights,
        caveats=caveats,
        next_step=next_step,
        raw_text=raw_text,
    )


def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json|javascript|python)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def _extract_json_object(text: str) -> str:
    if text.startswith("{") and text.endswith("}"):
        return text
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        return match.group(0)
    return text


def _coerce_text_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if value:
        item = str(value).strip()
        return [item] if item else []
    return []
