from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Callable

import pandas as pd


@dataclass(frozen=True)
class FallbackPlan:
    answer_kind: str
    summary: str
    code: str
    chart_title: str = ""
    key_insights: list[str] | None = None
    caveats: list[str] | None = None
    next_step: str = ""

    @property
    def answer(self) -> str:
        return self.summary


def build_fallback_plan(question: str, df: pd.DataFrame) -> FallbackPlan:
    normalized = question.strip().lower()
    numeric_columns = df.select_dtypes(include="number").columns.tolist()
    text_columns = [column for column in df.columns if column not in numeric_columns]
    column_names = [column.lower() for column in df.columns]

    if _contains(normalized, ["confidence"]) and not any("confidence" in column for column in column_names):
        return FallbackPlan(
            answer_kind="clarification",
            summary="I can’t find a confidence column in this dataset.",
            code="result = 'I can\'t find a confidence column in this dataset. Please choose a column name from the dataset.'",
            key_insights=["Available columns are shown in the dataset profile and raw schema tabs."],
            caveats=["The request may refer to a column that does not exist in this CSV."],
            next_step="Ask again using an existing column name, for example `cgpa`, `python_skill`, `aptitude_score`, or `salary_lpa`.",
        )

    if not normalized:
        return FallbackPlan(
            answer_kind="text",
            summary="Ask a question about the uploaded dataset.",
            code="result = 'Ask a question about the uploaded dataset.'",
            next_step="Try a question like: Which branch has the highest average salary_lpa?",
        )

    if _contains(normalized, ["missing", "null", "na", "nan"]):
        code = "result = df.isna().sum().sort_values(ascending=False).to_frame(name='missing_cells')"
        return FallbackPlan(
            answer_kind="table",
            summary="Here is the missing-value summary for each column.",
            code=code,
            key_insights=["Columns with the highest missing counts appear first."],
            caveats=["Rows with missing values may affect downstream aggregations."],
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
                summary=f"This shows total {value_column} by {group_column}.",
                code=code,
                chart_title=f"{group_column.title()} by Total {value_column.title()}",
                key_insights=[f"Groups are ordered from highest to lowest total {value_column}."],
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
                summary=f"Here are the top 2 {category_column} values by total {value_column}.",
                code=code,
                key_insights=[f"The list is sorted by total {value_column} in descending order."],
            )

    if _contains(normalized, ["average salary", "mean salary", "salary by"]):
        group_column = _pick_column(df, ["placed", "placement status", "status", "department", "branch", "stream", "specialization"], list(df.columns))
        value_column = _pick_column(df, ["salary", "ctc", "package", "income"], numeric_columns)
        if group_column and value_column:
            code = f"result = df.groupby('{group_column}')['{value_column}'].mean().sort_values(ascending=False)"
            return FallbackPlan(
                answer_kind="table",
                summary=f"This shows the average {value_column} by {group_column}.",
                code=code,
                key_insights=[f"Higher rows indicate a larger mean {value_column}."],
            )

    if _contains(normalized, ["placement rate", "placed status", "placement status"]):
        group_column = _pick_column(df, ["placed", "placement status", "status"], list(df.columns))
        if group_column:
            code = (
                f"result = df.groupby('{group_column}').size().sort_values(ascending=False).to_frame(name='student_count')"
            )
            return FallbackPlan(
                answer_kind="table",
                summary=f"This shows the student count by {group_column}.",
                code=code,
                key_insights=[f"Counts are grouped by {group_column}."],
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
                summary=f"This chart compares total {value_column} across {group_column}.",
                code=code,
                chart_title=f"{group_column.title()} by Total {value_column.title()}",
                key_insights=[f"The tallest bar represents the highest total {value_column}."],
            )

    if numeric_columns:
        primary_column = numeric_columns[0]
        code = f"result = df['{primary_column}'].describe().to_frame(name='{primary_column}')"
        return FallbackPlan(
            answer_kind="table",
            summary=f"Here is a summary for {primary_column}.",
            code=code,
            key_insights=["This is a quick numeric summary when no stronger intent match is found."],
        )

    fallback_column = str(df.columns[0]) if len(df.columns) else "value"
    code = f"result = df['{fallback_column}'].value_counts().head(10)"
    return FallbackPlan(
        answer_kind="table",
        summary=f"Here are the most frequent values in {fallback_column}.",
        code=code,
        key_insights=[f"This helps identify the most common category in {fallback_column}."],
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