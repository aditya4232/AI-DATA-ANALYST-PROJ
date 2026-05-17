from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class ColumnProfile:
    name: str
    dtype: str
    non_null: int
    missing: int
    unique: int
    sample_values: list[str]


@dataclass(frozen=True)
class DatasetProfile:
    row_count: int
    column_count: int
    columns: list[ColumnProfile]
    numeric_columns: list[str]
    categorical_columns: list[str]
    datetime_columns: list[str]
    missing_cells: int
    duplicate_rows: int


def build_profile(df: pd.DataFrame) -> DatasetProfile:
    columns: list[ColumnProfile] = []
    for column_name in df.columns:
        series = df[column_name]
        sample_values = [format_value(value) for value in series.dropna().head(5).tolist()]
        columns.append(
            ColumnProfile(
                name=str(column_name),
                dtype=str(series.dtype),
                non_null=int(series.notna().sum()),
                missing=int(series.isna().sum()),
                unique=int(series.nunique(dropna=True)),
                sample_values=sample_values,
            )
        )

    numeric_columns = [str(column) for column in df.select_dtypes(include="number").columns.tolist()]
    datetime_columns = [str(column) for column in df.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns.tolist()]
    categorical_columns = [
        str(column)
        for column in df.columns
        if column not in numeric_columns and column not in datetime_columns
    ]

    return DatasetProfile(
        row_count=int(len(df)),
        column_count=int(df.shape[1]),
        columns=columns,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        datetime_columns=datetime_columns,
        missing_cells=int(df.isna().sum().sum()),
        duplicate_rows=int(df.duplicated().sum()),
    )


def profile_to_text(profile: DatasetProfile) -> str:
    lines = [
        f"Rows: {profile.row_count}",
        f"Columns: {profile.column_count}",
        f"Missing cells: {profile.missing_cells}",
        f"Duplicate rows: {profile.duplicate_rows}",
        f"Numeric columns: {', '.join(profile.numeric_columns) if profile.numeric_columns else 'None'}",
        f"Categorical columns: {', '.join(profile.categorical_columns) if profile.categorical_columns else 'None'}",
        f"Datetime columns: {', '.join(profile.datetime_columns) if profile.datetime_columns else 'None'}",
        "Column details:",
    ]

    for column in profile.columns:
        sample_text = ", ".join(column.sample_values) if column.sample_values else "No non-null samples"
        lines.append(
            f"- {column.name} | dtype={column.dtype} | non_null={column.non_null} | missing={column.missing} | unique={column.unique} | samples={sample_text}"
        )

    return "\n".join(lines)


def dataframe_preview_text(df: pd.DataFrame, rows: int = 8) -> str:
    preview = df.head(rows).copy()
    if preview.empty:
        return "<empty dataframe>"
    return preview.to_string(index=False)


def format_value(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)
