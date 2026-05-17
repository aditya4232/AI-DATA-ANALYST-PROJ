from __future__ import annotations

from textwrap import dedent

from profiling import DatasetProfile, dataframe_preview_text, profile_to_text


def build_analysis_prompt(question: str, profile: DatasetProfile, df_preview: str) -> str:
    return dedent(
        f"""
        You are a senior data analyst working inside a Streamlit app.

                Your job is to answer the user's question using only the provided dataframe named df.

        Rules:
                - Return valid JSON only.
        - Do not wrap the JSON in markdown fences.
                - Use python code that relies on df, pd, np, and plt only.
        - Do not import modules.
        - Do not access files, the network, or the operating system.
                - If the question cannot be answered from the available columns, return a clarification request instead of guessing.
                - If a chart is useful, create it with matplotlib and leave the figure open.
                - Put the final computed value in a variable named result.
                - If the answer is tabular, return a pandas DataFrame or Series in result.
                - If the answer is textual, still compute result when possible.
                - Prefer explicit pandas operations, clear grouping, and deterministic ordering.
                - Keep the explanation concise, professional, and easy to scan.

        Required JSON schema:
        {{
                    "answer_kind": "table" | "chart" | "text" | "clarification",
                    "summary": "short markdown-friendly direct answer",
                    "key_insights": ["optional insight 1", "optional insight 2"],
          "code": "python code string",
          "chart_title": "optional chart title",
                    "caveats": ["optional short notes or limitations"],
                    "next_step": "optional follow-up suggestion"
        }}

        Dataset profile:
        {profile_to_text(profile)}

        Preview rows:
        {df_preview}

        User question:
        {question.strip()}
        """
    ).strip()


def build_system_prompt() -> str:
    return (
        "You produce trustworthy pandas analysis plans. "
        "Respond with JSON only, keep code concise, and prefer code that can run safely in a restricted namespace."
    )
