from __future__ import annotations

from typing import Any

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ---------------------------------------------------------------------------
# Matplotlib helpers (kept for LLM-generated chart rendering)
# ---------------------------------------------------------------------------

def render_result(result: Any) -> tuple[list[pd.DataFrame], str]:
    if result is None:
        return [], "No result produced."
    if isinstance(result, pd.DataFrame):
        return [result], "DataFrame result"
    if isinstance(result, pd.Series):
        name = result.name or "value"
        return [result.to_frame(name=name)], "Series result"
    if isinstance(result, (list, tuple)):
        try:
            frame = pd.DataFrame(result)
            return [frame], "Collection result"
        except Exception:
            return [], str(result)
    return [], str(result)


def format_figure_title(fig: matplotlib.figure.Figure, fallback: str = "Chart") -> str:
    if fig._suptitle is not None and fig._suptitle.get_text():
        return fig._suptitle.get_text()
    axes = fig.get_axes()
    if axes and axes[0].get_title():
        return axes[0].get_title()
    return fallback


def clear_matplotlib() -> None:
    plt.close("all")


# ---------------------------------------------------------------------------
# Plotly colour palette matching the app theme
# ---------------------------------------------------------------------------
THEME_COLORS = px.colors.qualitative.Set2
PRIMARY = "#1F6FEB"
SECONDARY = "#0F172A"
SEQUENTIAL = ["#1F6FEB", "#3b82f6", "#60a5fa", "#93c5fd", "#bfdbfe"]


def _branded_layout(title: str = "") -> dict:
    return {
        "title": {"text": title, "font": {"color": SECONDARY, "size": 16}},
        "font": {"family": "sans-serif", "color": "#475569"},
        "plot_bgcolor": "rgba(0,0,0,0)",
        "paper_bgcolor": "rgba(0,0,0,0)",
        "margin": {"t": 50, "b": 40, "l": 50, "r": 20},
        "hovermode": "x unified",
    }


# ---------------------------------------------------------------------------
# 1. Distribution plots (histogram + KDE overlay)
# ---------------------------------------------------------------------------

def plot_distribution(df: pd.DataFrame, column: str, color: str = PRIMARY) -> go.Figure:
    """Histogram with KDE overlay for a numeric column."""
    fig = px.histogram(
        df,
        x=column,
        marginal="rug",
        nbins=40,
        color_discrete_sequence=[color],
        opacity=0.75,
    )
    fig.update_layout(**_branded_layout(f"Distribution of {column}"))
    fig.update_traces(hovertemplate=f"{column}=%{{x}}<br>Count=%{{y}}")
    return fig


# ---------------------------------------------------------------------------
# 2. Categorical bar chart
# ---------------------------------------------------------------------------

def plot_categorical_counts(
    df: pd.DataFrame,
    column: str,
    title: str | None = None,
    limit: int = 20,
    color: str = PRIMARY,
) -> go.Figure:
    """Value counts bar chart for a categorical column, sorted descending."""
    counts = df[column].value_counts().head(limit).reset_index()
    counts.columns = [column, "count"]
    fig = px.bar(
        counts,
        x=column,
        y="count",
        text="count",
        color_discrete_sequence=[color],
    )
    fig.update_traces(textposition="outside", textfont_size=11)
    fig.update_layout(
        **_branded_layout(title or f"Distribution of {column}"),
        xaxis={"title": column},
        yaxis={"title": "Count"},
    )
    return fig


# ---------------------------------------------------------------------------
# 3. Grouped bar (categorical x categorical)
# ---------------------------------------------------------------------------

def plot_grouped_bar(
    df: pd.DataFrame,
    group_col: str,
    split_col: str,
    title: str | None = None,
    agg: str = "size",
    value_col: str | None = None,
    colors: list[str] | None = None,
) -> go.Figure:
    """Grouped bar chart of group_col split by split_col."""
    if agg == "size":
        table = df.groupby([group_col, split_col]).size().reset_index(name="count")
        y_label, y_col = "Count", "count"
    else:
        table = df.groupby([group_col, split_col])[value_col].mean().reset_index(name="mean")
        y_label, y_col = f"Mean {value_col}", "mean"

    fig = px.bar(
        table,
        x=group_col,
        y=y_col,
        color=split_col,
        barmode="group",
        text=y_col,
        color_discrete_sequence=colors or THEME_COLORS,
    )
    fig.update_traces(textposition="outside", textfont_size=10)
    fig.update_layout(
        **_branded_layout(title or f"{y_label} by {group_col} and {split_col}"),
        xaxis={"title": group_col},
        yaxis={"title": y_label},
        legend={"title": split_col},
    )
    return fig


# ---------------------------------------------------------------------------
# 4. Correlation heatmap
# ---------------------------------------------------------------------------

def plot_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """Correlation heatmap of all numeric columns."""
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        fig = go.Figure()
        fig.update_layout(**_branded_layout("Not enough numeric columns for a correlation heatmap"))
        return fig

    corr = numeric_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    # Build annotated heatmap labels
    corr_text = np.where(mask, "", corr.round(2).astype(str))

    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            text=corr_text,
            texttemplate="%{text}",
            textfont={"size": 9},
            colorscale="Blues",
            zmin=-1,
            zmax=1,
            hovertemplate="%{x} vs %{y}: %{z:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        **_branded_layout("Correlation Heatmap"),
        width=None,
        height=max(400, 40 * len(corr.columns)),
        xaxis={"side": "bottom", "tickangle": -45},
        yaxis={"autorange": "reversed"},
    )
    return fig


# ---------------------------------------------------------------------------
# 5. Box plot
# ---------------------------------------------------------------------------

def plot_box(
    df: pd.DataFrame,
    value_col: str,
    group_col: str | None = None,
    title: str | None = None,
    colors: list[str] | None = None,
) -> go.Figure:
    """Box plot of value_col, optionally grouped by group_col."""
    fig = px.box(
        df,
        x=group_col,
        y=value_col,
        color=group_col if group_col else None,
        color_discrete_sequence=colors or THEME_COLORS,
        points="outliers",
        notched=False,
    )
    fig.update_layout(**_branded_layout(title or f"Distribution of {value_col}"))
    fig.update_traces(hovertemplate=f"{value_col}=%{{y}}<br>Group=%{{x}}")
    return fig


# ---------------------------------------------------------------------------
# 6. Pie / donut chart
# ---------------------------------------------------------------------------

def plot_pie(
    df: pd.DataFrame,
    column: str,
    title: str | None = None,
    colors: list[str] | None = None,
) -> go.Figure:
    """Donut chart for a categorical column."""
    counts = df[column].value_counts().reset_index()
    counts.columns = [column, "count"]
    fig = px.pie(
        counts,
        names=column,
        values="count",
        hole=0.4,
        color_discrete_sequence=colors or THEME_COLORS,
    )
    fig.update_traces(
        textinfo="label+percent",
        hovertemplate="%{label}<br>%{value} students (%{percent})",
    )
    fig.update_layout(**_branded_layout(title or f"Distribution of {column}"))
    return fig


# ---------------------------------------------------------------------------
# 7. Scatter plot
# ---------------------------------------------------------------------------

def plot_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: str | None = None,
    title: str | None = None,
) -> go.Figure:
    """Scatter plot of two numeric columns, optionally coloured."""
    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color=color_col,
        color_discrete_sequence=THEME_COLORS,
        opacity=0.6,
        trendline="lowess" if len(df) < 5000 else None,
        trendline_color_override=PRIMARY,
    )
    fig.update_layout(**_branded_layout(title or f"{y_col} vs {x_col}"))
    fig.update_traces(
        hovertemplate=f"{x_col}=%{{x}}<br>{y_col}=%{{y}}<br>%{{text}}",
    )
    # Sample for performance with large datasets
    return fig


# ---------------------------------------------------------------------------
# 8. Salary analysis dashboard (dataset-specific)
# ---------------------------------------------------------------------------

def plot_salary_analysis(df: pd.DataFrame) -> go.Figure:
    """Multi-panel salary analysis: distribution, by placement, by branch."""
    has_salary = "salary_lpa" in df.columns
    has_placed = "placed" in df.columns
    has_branch = "branch" in df.columns

    if not has_salary:
        fig = go.Figure()
        fig.update_layout(**_branded_layout("No salary column found"))
        return fig

    rows = 1 + int(has_placed) + int(has_branch)
    spec_types = [{"type": "xy"}] * rows
    fig = make_subplots(
        rows=rows,
        cols=1,
        subplot_titles=[
            "Salary Distribution",
            "Salary by Placement Status" if has_placed else None,
            "Salary by Branch" if has_branch else None,
        ][:rows],
        vertical_spacing=0.12 / rows,
    )

    # Row 1: histogram
    row = 1
    fig.add_trace(
        go.Histogram(x=df["salary_lpa"], nbinsx=50, marker_color=PRIMARY, name="salary_lpa"),
        row=row, col=1,
    )

    # Row 2: box by placed
    if has_placed:
        row += 1
        placed_map = {0: "Not Placed", 1: "Placed"}
        df_plot = df.copy()
        df_plot["placed_label"] = df_plot["placed"].map(placed_map).fillna("Unknown")
        for i, (val, label) in enumerate(placed_map.items()):
            subset = df_plot[df_plot["placed"] == val]
            fig.add_trace(
                go.Box(
                    y=subset["salary_lpa"],
                    name=label,
                    marker_color=THEME_COLORS[i % len(THEME_COLORS)],
                    boxmean=True,
                ),
                row=row, col=1,
            )

    # Row 3: box by branch
    if has_branch:
        row += 1
        branches = df["branch"].dropna().unique()
        for i, branch in enumerate(sorted(branches)):
            subset = df[df["branch"] == branch]
            fig.add_trace(
                go.Box(
                    y=subset["salary_lpa"],
                    name=str(branch),
                    marker_color=THEME_COLORS[i % len(THEME_COLORS)],
                    boxmean=True,
                ),
                row=row, col=1,
            )

    fig.update_layout(
        height=220 * rows,
        showlegend=True,
        **_branded_layout("Salary Analysis"),
    )
    fig.update_xaxes(title_text="Salary (LPA)", row=rows, col=1)
    return fig


# ---------------------------------------------------------------------------
# 9. Skills analysis (radar / stacked)
# ---------------------------------------------------------------------------

def plot_skills_analysis(df: pd.DataFrame) -> go.Figure:
    """Stacked bar chart of binary skill columns by branch."""
    skill_cols = [c for c in ["python_skill", "dsa_skill", "ml_skill", "web_dev_skill"] if c in df.columns]
    if not skill_cols:
        fig = go.Figure()
        fig.update_layout(**_branded_layout("No binary skill columns found"))
        return fig

    if "branch" not in df.columns:
        fig = go.Figure()
        fig.update_layout(**_branded_layout("No branch column for skills breakdown"))
        return fig

    # Count students with each skill per branch
    skill_data = df.groupby("branch")[skill_cols].sum().reset_index()
    skill_data_melted = skill_data.melt(
        id_vars="branch",
        var_name="skill",
        value_name="count",
    )
    # Rename skills for display
    skill_labels = {
        "python_skill": "Python",
        "dsa_skill": "DSA",
        "ml_skill": "ML",
        "web_dev_skill": "Web Dev",
    }
    skill_data_melted["skill"] = skill_data_melted["skill"].map(skill_labels).fillna(skill_data_melted["skill"])

    fig = px.bar(
        skill_data_melted,
        x="branch",
        y="count",
        color="skill",
        barmode="group",
        text="count",
        color_discrete_sequence=THEME_COLORS,
    )
    fig.update_traces(textposition="outside", textfont_size=10)
    fig.update_layout(
        **_branded_layout("Skills Proficiency by Branch"),
        xaxis={"title": "Branch"},
        yaxis={"title": "Students with Skill"},
        legend={"title": "Skill"},
    )
    return fig


# ---------------------------------------------------------------------------
# 10. Full EDA dashboard -- generates all charts for a dataset
# ---------------------------------------------------------------------------

class EDADashboard:
    """Container for all auto-generated EDA charts."""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.charts: dict[str, go.Figure] = {}
        self._build()

    def _build(self) -> None:
        df = self.df
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = [c for c in df.columns if c not in numeric_cols]
        # Also treat low-cardinality numeric as categorical
        for c in numeric_cols[:]:
            if df[c].nunique() <= 10:
                numeric_cols.remove(c)
                categorical_cols.append(c)

        # --- Overview ---
        # 1. Placement rate pie
        if "placed" in df.columns:
            self.charts["Placement Rate"] = plot_pie(df, "placed", "Placement Rate")

        # 2. Branch distribution
        if "branch" in df.columns:
            self.charts["Branch Distribution"] = plot_categorical_counts(df, "branch")

        # 3. Company type distribution
        if "company_type" in df.columns:
            self.charts["Company Type"] = plot_pie(df.dropna(subset=["company_type"]), "company_type")

        # 4. Job role distribution
        if "job_role" in df.columns:
            self.charts["Job Roles"] = plot_categorical_counts(df.dropna(subset=["job_role"]), "job_role")

        # --- Numeric distributions ---
        preferred_num = ["cgpa", "coding_score", "aptitude_score", "resume_score", "salary_lpa"]
        for col in preferred_num:
            if col in df.columns and col in numeric_cols:
                self.charts[f"{col.title()} Distribution"] = plot_distribution(df, col)

        # --- Correlation ---
        if len(numeric_cols) >= 2:
            self.charts["Correlation Heatmap"] = plot_correlation_heatmap(df)

        # --- Grouped analyses ---
        if "placed" in df.columns and "branch" in df.columns:
            self.charts["Placement by Branch"] = plot_grouped_bar(
                df, "branch", "placed",
                title="Placement Status by Branch",
            )

        if "salary_lpa" in df.columns and "placed" in df.columns:
            self.charts["Salary by Placement"] = plot_box(
                df, "salary_lpa", "placed",
                title="Salary Distribution by Placement Status",
            )

        if "salary_lpa" in df.columns and "branch" in df.columns:
            self.charts["Salary by Branch"] = plot_box(
                df, "salary_lpa", "branch",
                title="Salary Distribution by Branch",
            )

        # --- Skills ---
        skill_cols = [c for c in ["python_skill", "dsa_skill", "ml_skill", "web_dev_skill"] if c in df.columns]
        if skill_cols:
            self.charts["Skills Analysis"] = plot_skills_analysis(df)

        # --- Scatter (CGPA vs Salary) ---
        if "cgpa" in df.columns and "salary_lpa" in df.columns:
            self.charts["CGPA vs Salary"] = plot_scatter(
                df, "cgpa", "salary_lpa",
                color_col="placed" if "placed" in df.columns else None,
                title="CGPA vs Salary (coloured by Placement)",
            )

        # --- Misc additional ---
        if "college_tier" in df.columns and "salary_lpa" in df.columns:
            self.charts["Salary by College Tier"] = plot_box(
                df, "salary_lpa", "college_tier",
                title="Salary Distribution by College Tier",
            )

        if "internships" in df.columns and "placed" in df.columns:
            self.charts["Internships vs Placement"] = plot_grouped_bar(
                df, "internships", "placed",
                title="Placement Status by Number of Internships",
            )

    @property
    def chart_list(self) -> list[tuple[str, go.Figure]]:
        return list(self.charts.items())


# ---------------------------------------------------------------------------
# Convenience: build dashboard from dataframe
# ---------------------------------------------------------------------------

def build_eda_dashboard(df: pd.DataFrame) -> EDADashboard:
    return EDADashboard(df)
