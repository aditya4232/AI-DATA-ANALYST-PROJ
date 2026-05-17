from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Callable

import pandas as pd


@dataclass(frozen=True)
class FallbackPlan:
    answer_kind: str
    answer: str
    code: str
    chart_title: str = ""
    notes: list[str] | None = None


def build_fallback_plan(question: str, df: pd.DataFrame) -> FallbackPlan:
    normalized = question.strip().lower()
    numeric_columns = df.select_dtypes(include="number").columns.tolist()
    text_columns = [column for column in df.columns if column not in numeric_columns]

    if not normalized:
        return FallbackPlan(
            answer_kind="text",
            answer="Ask a question about the uploaded dataset.",
            code="result = 'Ask a question about the uploaded dataset.'",
        )

    if _contains(normalized, ["missing", "null", "na", "nan"]):
        code = "result = df.isna().sum().sort_values(ascending=False).to_frame(name='missing_cells')"
        return FallbackPlan(
            answer_kind="table",
            answer="Here is the missing-value summary for each column.",
            code=code,
            notes=["Sorted from highest to lowest missing count."],
        )

    if _contains(normalized, ["highest total sales", "top sales by region", "sales by region"]):
        group_column = _pick_column(df, ["region", "state", "city", "location", "department"], text_columns)
        value_column = _pick_column(df, ["sales", "salary", "amount", "profit", "revenue"], numeric_columns)
        if group_column and value_column:
            chart = "plt.figure(figsize=(8, 4)); result.plot(kind='bar', color='#1f6feb'); plt.title('" + group_column.title() + " by Total " + value_column.title() + "'); plt.tight_layout()"
            code = (
                f"result = df.groupby('{group_column}')['{value_column}'].sum().sort_values(ascending=False)\n"
                f"{chart}"
            )
            return FallbackPlan(
                answer_kind="chart",
                answer=f"This shows total {value_column} by {group_column}.",
                code=code,
                chart_title=f"{group_column.title()} by Total {value_column.title()}",
            )

    if _contains(normalized, ["top 2 products", "top products", "highest products"]):
        category_column = _pick_column(df, ["product", "course", "stream", "branch", "specialization"], text_columns)
        value_column = _pick_column(df, ["sales", "salary", "amount", "profit", "revenue"], numeric_columns)
        if category_column and value_column:
            code = (
                f"result = df.groupby('{category_column}')['{value_column}'].sum().sort_values(ascending=False).head(2)\n"
                "result = result.to_frame(name='total_" + value_column + "')"
            )
            return FallbackPlan(
                answer_kind="table",
                answer=f"Here are the top 2 {category_column} values by total {value_column}.",
                code=code,
            )

    if _contains(normalized, ["average salary", "mean salary", "salary by"]):
        group_column = _pick_column(df, ["placed", "placement status", "status", "department", "branch", "stream", "specialization"], list(df.columns))
        value_column = _pick_column(df, ["salary", "ctc", "package", "income"], numeric_columns)
        if group_column and value_column:
            code = f"result = df.groupby('{group_column}')['{value_column}'].mean().sort_values(ascending=False)"
            return FallbackPlan(
                answer_kind="table",
                answer=f"This shows the average {value_column} by {group_column}.",
                code=code,
            )

    if _contains(normalized, ["placement rate", "placed status", "placement status"]):
        group_column = _pick_column(df, ["placed", "placement status", "status"], list(df.columns))
        if group_column:
            code = (
                f"result = df.groupby('{group_column}').size().sort_values(ascending=False).to_frame(name='student_count')"
            )
            return FallbackPlan(
                answer_kind="table",
                answer=f"This shows the student count by {group_column}.",
                code=code,
            )

    if _contains(normalized, ["plot", "chart", "bar", "visualize"]):
        group_column = _pick_column(df, ["region", "department", "stream", "status", "category", "product"], text_columns)
        value_column = _pick_column(df, ["sales", "salary", "amount", "profit", "revenue"], numeric_columns)
        if group_column and value_column:
            code = (
                f"result = df.groupby('{group_column}')['{value_column}'].sum().sort_values(ascending=False)\n"
                "plt.figure(figsize=(8, 4))\n"
                "result.plot(kind='bar', color='#1f6feb')\n"
                f"plt.title('{group_column.title()} by Total {value_column.title()}')\n"
                "plt.tight_layout()"
            )
            return FallbackPlan(
                answer_kind="chart",
                answer=f"This chart compares total {value_column} across {group_column}.",
                code=code,
                chart_title=f"{group_column.title()} by Total {value_column.title()}",
            )

    if numeric_columns:
        primary_column = numeric_columns[0]
        code = f"result = df['{primary_column}'].describe().to_frame(name='{primary_column}')"
        return FallbackPlan(
            answer_kind="table",
            answer=f"Here is a summary for {primary_column}.",
            code=code,
        )

    fallback_column = str(df.columns[0]) if len(df.columns) else "value"
    code = f"result = df['{fallback_column}'].value_counts().head(10)"
    return FallbackPlan(
        answer_kind="table",
        answer=f"Here are the most frequent values in {fallback_column}.",
        code=code,
    )


def _pick_column(df: pd.DataFrame, candidates: list[str], available_columns: list[str]) -> str | None:
    normalized_lookup = {column.lower(): column for column in available_columns}
    for candidate in candidates:
        for column in available_columns:
            if candidate in column.lower():
                return column
        if candidate in normalized_lookup:
            return normalized_lookup[candidate]
    return None


def _contains(text: str, phrases: list[str]) -> bool:
    return any(phrase in text for phrase in phrases)