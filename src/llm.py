from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any

from openai import OpenAI

from .config import AppConfig
from .prompts import build_system_prompt


@dataclass(frozen=True)
class AnalysisPlan:
    answer_kind: str
    answer: str
    code: str
    chart_title: str = ""
    notes: list[str] | None = None
    raw_text: str = ""


class LLMError(RuntimeError):
    pass


def generate_analysis_plan(prompt: str, config: AppConfig) -> AnalysisPlan:
    if not config.api_key:
        raise LLMError(
            f"Missing API key for {config.provider_label}. Set NVIDIA_API_KEY, OPENAI_API_KEY, or LLM_API_KEY."
        )

    client = OpenAI(api_key=config.api_key, base_url=config.base_url)
    response = client.chat.completions.create(
        model=config.model,
        temperature=config.temperature,
        messages=[
            {"role": "system", "content": build_system_prompt()},
            {"role": "user", "content": prompt},
        ],
    )
    content = response.choices[0].message.content or ""
    return parse_analysis_plan(content)


def parse_analysis_plan(raw_text: str) -> AnalysisPlan:
    content = _strip_code_fences(raw_text)
    content = _extract_json_object(content)
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise LLMError(f"The model did not return valid JSON: {exc}") from exc

    answer_kind = str(payload.get("answer_kind", "text")).strip().lower()
    answer = str(payload.get("answer", "")).strip()
    code = str(payload.get("code", "")).strip()
    chart_title = str(payload.get("chart_title", "")).strip()
    notes = payload.get("notes")
    if isinstance(notes, list):
        notes_list = [str(item).strip() for item in notes if str(item).strip()]
    elif notes:
        notes_list = [str(notes).strip()]
    else:
        notes_list = []

    if answer_kind not in {"table", "chart", "text"}:
        answer_kind = "text"

    return AnalysisPlan(
        answer_kind=answer_kind,
        answer=answer,
        code=code,
        chart_title=chart_title,
        notes=notes_list,
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
