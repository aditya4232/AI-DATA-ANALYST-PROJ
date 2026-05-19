from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd


# ---------------------------------------------------------------------------
# Matplotlib chart templates (used in generated code strings)
# ---------------------------------------------------------------------------

def _bar_chart(title: str, xlabel: str = "", ylabel: str = "") -> str:
    return (
        "plt.figure(figsize=(9, 4.5))\n"
        "ax = result.plot(kind='bar', color='#1f6feb', edgecolor='white', width=0.7)\n"
        f"plt.title('{title}', fontsize=13, fontweight='bold')\n"
        f"plt.xlabel('{xlabel}')\n"
        f"plt.ylabel('{ylabel}')\n"
        "plt.xticks(rotation=35, ha='right')\n"
        "plt.grid(axis='y', alpha=0.3)\n"
        "plt.tight_layout()"
    )


def _horizontal_bar(title: str, xlabel: str = "", ylabel: str = "") -> str:
    return (
        "plt.figure(figsize=(9, 4.5))\n"
        "ax = result.plot(kind='barh', color='#1f6feb', edgecolor='white', height=0.7)\n"
        f"plt.title('{title}', fontsize=13, fontweight='bold')\n"
        f"plt.xlabel('{xlabel}')\n"
        f"plt.ylabel('{ylabel}')\n"
        "plt.grid(axis='x', alpha=0.3)\n"
        "plt.tight_layout()"
    )


def _histogram(column: str, bins: int = 30) -> str:
    return (
        "plt.figure(figsize=(9, 4.5))\n"
        f"df['{column}'].plot(kind='hist', bins={bins}, color='#1f6feb', edgecolor='white', alpha=0.8)\n"
        f"plt.title('Distribution of {column}', fontsize=13, fontweight='bold')\n"
        f"plt.xlabel('{column}')\n"
        "plt.ylabel('Frequency')\n"
        "plt.grid(axis='y', alpha=0.3)\n"
        "plt.tight_layout()"
    )


def _box_plot(value_col: str, group_col: str) -> str:
    return (
        "plt.figure(figsize=(9, 4.5))\n"
        f"df.boxplot(column='{value_col}', by='{group_col}', "
        "patch_artist=True, "
        "boxprops=dict(facecolor='#bfdbfe', color='#1f6feb'), "
        "medianprops=dict(color='#1f6feb', linewidth=2), "
        "whiskerprops=dict(color='#1f6feb'), "
        "capprops=dict(color='#1f6feb'))\n"
        f"plt.title('{value_col} by {group_col}', fontsize=13, fontweight='bold')\n"
        "plt.suptitle('')\n"
        "plt.grid(axis='y', alpha=0.3)\n"
        "plt.xticks(rotation=35, ha='right')\n"
        "plt.tight_layout()"
    )


def _pie_chart(column: str) -> str:
    return (
        "plt.figure(figsize=(7, 5))\n"
        f"counts = df['{column}'].value_counts()\n"
        "colors = ['#1f6feb', '#3b82f6', '#60a5fa', '#93c5fd', '#bfdbfe', '#dbeafe']\n"
        "counts.plot(kind='pie', autopct='%1.1f%%', colors=colors[:len(counts)],\n"
        "           startangle=90, wedgeprops=dict(edgecolor='white', linewidth=1.5))\n"
        f"plt.title('{column} Distribution', fontsize=13, fontweight='bold')\n"
        "plt.ylabel('')\n"
        "plt.tight_layout()"
    )


# ---------------------------------------------------------------------------
# FallbackPlan
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Deterministic intent matching
# ---------------------------------------------------------------------------


def build_fallback_plan(question: str, df: pd.DataFrame) -> FallbackPlan:
    normalized = question.strip().lower()
    numeric_columns = df.select_dtypes(include="number").columns.tolist()
    text_columns = [column for column in df.columns if column not in numeric_columns]
    column_names = [column.lower() for column in df.columns]

    # --- Confidence column check ---
    if _contains(normalized, ["confidence"]) and not any("confidence" in column for column in column_names):
        return FallbackPlan(
            answer_kind="clarification",
            summary="I can't find a confidence column in this dataset.",
            code="result = 'I can\\'t find a confidence column in this dataset. Please choose a column name from the dataset.'",
            key_insights=["Available columns are shown in the dataset profile and raw schema tabs."],
            caveats=["The request may refer to a column that does not exist in this CSV."],
            next_step="Ask again using an existing column name, for example `cgpa`, `python_skill`, `aptitude_score`, or `salary_lpa`.",
        )

    # --- Empty question ---
    if not normalized:
        return FallbackPlan(
            answer_kind="text",
            summary="Ask a question about the uploaded dataset.",
            code="result = 'Ask a question about the uploaded dataset.'",
            next_step="Try a question like: Which branch has the highest placement rate?",
        )

    # --- Missing values ---
    if _contains(normalized, ["missing", "null", "na", "nan"]):
        code = "result = df.isna().sum().sort_values(ascending=False).to_frame(name='missing_cells')"
        return FallbackPlan(
            answer_kind="table",
            summary="Here is the missing-value summary for each column.",
            code=code,
            key_insights=["Columns with the highest missing counts appear first."],
            caveats=["Rows with missing values may affect downstream aggregations."],
        )

    # --- Distribution / histogram ---
    if _contains(normalized, ["distribution of", "histogram", "density"]):
        target = _pick_column(df, ["cgpa", "salary", "coding_score", "aptitude_score", "resume_score"], numeric_columns)
        if target:
            code = (
                f"result = df['{target}'].describe()\n"
                + _histogram(target)
            )
            return FallbackPlan(
                answer_kind="chart",
                summary=f"Here is the distribution of {target}.",
                code=code,
                chart_title=f"Distribution of {target}",
                key_insights=[f"The histogram shows the frequency of {target} values across students."],
            )

    # --- Correlation / relationship ---
    if _contains(normalized, ["correlation", "correlate", "relationship", "heatmap"]):
        if len(numeric_columns) >= 3:
            code = (
                "corr = df.select_dtypes(include='number').corr()\n"
                "result = corr\n"
                "plt.figure(figsize=(10, 7))\n"
                "im = plt.imshow(corr, cmap='Blues', aspect='auto', vmin=-1, vmax=1)\n"
                "plt.colorbar(im, shrink=0.8)\n"
                "plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha='right', fontsize=8)\n"
                "plt.yticks(range(len(corr.columns)), corr.columns, fontsize=8)\n"
                "for i in range(len(corr.columns)):\n"
                "    for j in range(len(corr.columns)):\n"
                "        plt.text(j, i, f'{corr.iloc[i, j]:.2f}', ha='center', va='center', fontsize=7)\n"
                "plt.title('Correlation Heatmap', fontsize=14, fontweight='bold')\n"
                "plt.tight_layout()"
            )
            return FallbackPlan(
                answer_kind="chart",
                summary="Here is the correlation heatmap of numeric features.",
                code=code,
                chart_title="Correlation Heatmap",
                key_insights=["Dark blue indicates strong positive correlation.", "Light/white indicates weak or no correlation."],
            )

    # --- Salary analysis ---
    salary_col = _pick_column(df, ["salary_lpa", "salary", "ctc", "package", "income"], numeric_columns)
    if salary_col and _contains(normalized, ["average salary", "mean salary", "salary by", "avg salary"]):
        group_col = _pick_column(df, ["branch", "placed", "company_type", "college_tier", "job_role"], list(df.columns))
        if group_col:
            code = (
                f"result = df.groupby('{group_col}')['{salary_col}'].mean().sort_values(ascending=False)\n"
                + _bar_chart(f"Average {salary_col} by {group_col}", xlabel=group_col, ylabel=f"Average {salary_col}")
            )
            return FallbackPlan(
                answer_kind="chart",
                summary=f"This shows the average {salary_col} by {group_col}.",
                code=code,
                chart_title=f"Average {salary_col} by {group_col}",
                key_insights=[f"Higher bars indicate a larger mean {salary_col}."],
            )

    if salary_col and _contains(normalized, ["salary distribution", "salary range", "salary spread"]):
        code = (
            f"result = df['{salary_col}'].describe()\n"
            + _histogram(salary_col, bins=40)
        )
        return FallbackPlan(
            answer_kind="chart",
            summary=f"Here is the distribution of {salary_col}.",
            code=code,
            chart_title=f"{salary_col} Distribution",
            key_insights=[f"Mean: {df[salary_col].mean():.2f}, Median: {df[salary_col].median():.2f}, Std: {df[salary_col].std():.2f}"],
        )

    if salary_col and _contains(normalized, ["salary by branch", "branch salary"]):
        if "branch" in df.columns:
            code = (
                f"result = df.groupby('branch')['{salary_col}'].mean().sort_values(ascending=False)\n"
                + _horizontal_bar(f"Average {salary_col} by Branch", xlabel=f"Average {salary_col}", ylabel="Branch")
            )
            return FallbackPlan(
                answer_kind="chart",
                summary=f"Average {salary_col} by branch, from highest to lowest.",
                code=code,
                chart_title=f"Average {salary_col} by Branch",
                key_insights=["The branch with the highest average salary is shown at the top."],
            )

    # --- Placement rate analysis ---
    if _contains(normalized, ["placement rate", "placed status", "placement status", "placement by branch", "placement rate by"]):
        if "placed" in df.columns and "branch" in df.columns:
            code = (
                "placement_rate = df.groupby('branch')['placed'].mean().sort_values(ascending=False) * 100\n"
                "result = placement_rate.to_frame(name='placement_rate_%')\n"
                + _bar_chart("Placement Rate by Branch (%)", xlabel="Branch", ylabel="Placement Rate (%)")
            )
            return FallbackPlan(
                answer_kind="chart",
                summary="This shows the placement rate percentage by branch.",
                code=code,
                chart_title="Placement Rate by Branch",
                key_insights=["Branches with the highest placement rate appear first."],
            )
        if "placed" in df.columns:
            code = (
                "result = df['placed'].value_counts().sort_index()\n"
                + _pie_chart("placed")
            )
            return FallbackPlan(
                answer_kind="chart",
                summary="This shows the proportion of placed versus not placed students.",
                code=code,
                chart_title="Placement Breakdown",
                key_insights=[f"{df['placed'].mean() * 100:.1f}% of students are placed."],
            )

    # --- Skills analysis ---
    if _contains(normalized, ["skill", "proficiency", "python", "dsa", "ml", "web"]):
        skill_cols = [c for c in ["python_skill", "dsa_skill", "ml_skill", "web_dev_skill"] if c in df.columns]
        if skill_cols and "branch" in df.columns:
            code = (
                "skill_data = df.groupby('branch')[[" + ", ".join(f"'{c}'" for c in skill_cols) + "]].mean() * 100\n"
                "result = skill_data\n"
                "ax = skill_data.plot(kind='bar', figsize=(10, 5), color=['#1f6feb', '#3b82f6', '#60a5fa', '#93c5fd'], edgecolor='white', width=0.75)\n"
                "plt.title('Skills Proficiency by Branch (%)', fontsize=13, fontweight='bold')\n"
                "plt.xlabel('Branch')\n"
                "plt.ylabel('Proficiency (%)')\n"
                "plt.xticks(rotation=35, ha='right')\n"
                "plt.legend(title='Skill')\n"
                "plt.grid(axis='y', alpha=0.3)\n"
                "plt.tight_layout()"
            )
            return FallbackPlan(
                answer_kind="chart",
                summary="This shows the skill proficiency percentage by branch.",
                code=code,
                chart_title="Skills Proficiency by Branch",
                key_insights=["Higher values indicate a larger proportion of students with that skill."],
            )

    # --- CGPA analysis ---
    if _contains(normalized, ["cgpa distribution", "cgpa by", "average cgpa"]):
        if "cgpa" in df.columns:
            if _contains(normalized, ["by branch", "by placed", "by stream"]):
                group_col = _pick_column(df, ["branch", "placed", "company_type"], list(df.columns))
                if group_col:
                    code = (
                        f"result = df.groupby('{group_col}')['cgpa'].mean().sort_values(ascending=False)\n"
                        + _bar_chart(f"Average CGPA by {group_col}", xlabel=group_col, ylabel="Average CGPA")
                    )
                    return FallbackPlan(
                        answer_kind="chart",
                        summary=f"This shows the average CGPA by {group_col}.",
                        code=code,
                        chart_title=f"Average CGPA by {group_col}",
                        key_insights=[f"Students in different {group_col} categories show varying average CGPA."],
                    )
            else:
                code = (
                    "result = df['cgpa'].describe()\n"
                    + _histogram("cgpa")
                )
                return FallbackPlan(
                    answer_kind="chart",
                    summary="Here is the CGPA distribution.",
                    code=code,
                    chart_title="CGPA Distribution",
                    key_insights=[f"Mean CGPA: {df['cgpa'].mean():.2f}, Median: {df['cgpa'].median():.2f}"],
                )

    # --- Top N queries ---
    if _contains(normalized, ["top 5", "top 10", "top 3", "highest", "top "]):
        category_col = _pick_column(df, ["branch", "stream", "company_type", "job_role", "department", "specialization"], text_columns)
        value_col = _pick_column(df, ["salary_lpa", "salary", "cgpa", "coding_score", "aptitude_score", "placement"], numeric_columns)
        if category_col and value_col:
            top_n = 5
            for token in normalized.split():
                if token.startswith("top") and token[3:].isdigit():
                    top_n = int(token[3:])
                    break
            code = (
                f"result = df.groupby('{category_col}')['{value_col}'].mean().sort_values(ascending=False).head({top_n})\n"
                + _bar_chart(f"Top {top_n} {category_col} by Average {value_col}", xlabel=category_col, ylabel=f"Average {value_col}")
            )
            return FallbackPlan(
                answer_kind="chart",
                summary=f"Here are the top {top_n} {category_col} by average {value_col}.",
                code=code,
                chart_title=f"Top {top_n} {category_col} by Average {value_col}",
                key_insights=["The chart is sorted from highest to lowest."],
            )

    # --- Generic plot / chart / visualize ---
    if _contains(normalized, ["plot", "chart", "bar", "visualize", "graph"]):
        group_col = _pick_column(df, ["branch", "department", "stream", "status", "company_type", "job_role"], text_columns)
        value_col = _pick_column(df, ["salary_lpa", "salary", "cgpa", "coding_score", "aptitude_score", "resume_score"], numeric_columns)
        if group_col and value_col:
            code = (
                f"result = df.groupby('{group_col}')['{value_col}'].mean().sort_values(ascending=False)\n"
                + _bar_chart(f"Average {value_col} by {group_col}", xlabel=group_col, ylabel=f"Average {value_col}")
            )
            return FallbackPlan(
                answer_kind="chart",
                summary=f"This chart compares average {value_col} across {group_col}.",
                code=code,
                chart_title=f"Average {value_col} by {group_col}",
                key_insights=[f"The tallest bar represents the highest average {value_col}."],
            )

    # --- Numeric summary fallback ---
    if numeric_columns:
        primary_column = numeric_columns[0]
        code = (
            f"result = df['{primary_column}'].describe().to_frame(name='{primary_column}')\n"
            + _histogram(primary_column)
        )
        return FallbackPlan(
            answer_kind="chart",
            summary=f"Here is a summary for {primary_column}.",
            code=code,
            chart_title=f"{primary_column} Summary",
            key_insights=["This is a quick numeric summary when no stronger intent match is found."],
        )

    # --- Ultimate fallback ---
    fallback_column = str(df.columns[0]) if len(df.columns) else "value"
    code = f"result = df['{fallback_column}'].value_counts().head(10)"
    return FallbackPlan(
        answer_kind="table",
        summary=f"Here are the most frequent values in {fallback_column}.",
        code=code,
        key_insights=[f"This helps identify the most common category in {fallback_column}."],
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pick_column(df: pd.DataFrame, candidates: list[str], available_columns: list[str]) -> str | None:
    for candidate in candidates:
        for column in available_columns:
            if candidate in column.lower():
                return column
    normalized_lookup = {column.lower(): column for column in available_columns}
    for candidate in candidates:
        if candidate in normalized_lookup:
            return normalized_lookup[candidate]
    return None


def _contains(text: str, phrases: list[str]) -> bool:
    return any(phrase in text for phrase in phrases)
