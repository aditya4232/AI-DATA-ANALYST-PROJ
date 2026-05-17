from __future__ import annotations

import io
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from src.charts import format_figure_title, render_result
from src.config import AppConfig
from src.execution import AnalysisCodeError, execute_analysis_code
from src.fallback import FallbackPlan, build_fallback_plan
from src.llm import LLMError, generate_analysis_plan
from src.profiling import build_profile, dataframe_preview_text, profile_to_text
from src.prompts import build_analysis_prompt


st.set_page_config(page_title="AI Data Analyst Pro", page_icon="📊", layout="wide")


APP_DIR = Path(__file__).resolve().parent
SAMPLE_DATASET_PATH = APP_DIR / "student_placement_salary_elite_v2.csv"


SAMPLE_QUESTIONS = [
    "Which stream has the highest placement rate?",
    "What is the average salary_lpa by placed status?",
    "Plot CGPA distribution by placed status.",
    "Which features are most associated with placement?",
    "Show the top 5 streams by average salary_lpa.",
]


def _first_non_empty(*values: str) -> str:
    for value in values:
        if value and value.strip():
            return value.strip()
    return ""


def _secret_value(name: str) -> str:
    try:
        value = st.secrets.get(name, "")
    except Exception:
        return ""
    return str(value).strip() if value else ""


def _resolve_runtime_config() -> AppConfig:
    env_config = AppConfig.from_env()
    provider = _first_non_empty(os.getenv("LLM_PROVIDER"), _secret_value("LLM_PROVIDER"), env_config.provider)
    api_key = _first_non_empty(
        os.getenv("LLM_API_KEY"),
        os.getenv("NVIDIA_API_KEY"),
        os.getenv("OPENAI_API_KEY"),
        _secret_value("LLM_API_KEY"),
        _secret_value("NVIDIA_API_KEY"),
        _secret_value("OPENAI_API_KEY"),
        env_config.api_key,
    )
    base_url = _first_non_empty(
        os.getenv("LLM_BASE_URL"),
        os.getenv("NVIDIA_BASE_URL"),
        _secret_value("LLM_BASE_URL"),
        _secret_value("NVIDIA_BASE_URL"),
        env_config.base_url,
    )
    model = _first_non_empty(
        os.getenv("LLM_MODEL"),
        os.getenv("NVIDIA_MODEL"),
        _secret_value("LLM_MODEL"),
        _secret_value("NVIDIA_MODEL"),
        env_config.model,
    )
    return AppConfig(
        provider=provider,
        api_key=api_key,
        base_url=base_url,
        model=model,
        temperature=env_config.temperature,
        max_rows_preview=env_config.max_rows_preview,
    )


def load_css() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(31,111,235,0.08), transparent 28%),
                radial-gradient(circle at top right, rgba(15,23,42,0.05), transparent 24%),
                linear-gradient(180deg, #f8fbff 0%, #eef4fb 100%);
        }
        .hero {
            padding: 1.1rem 1.3rem;
            border-radius: 1.25rem;
            background: linear-gradient(135deg, rgba(15,23,42,0.96), rgba(31,111,235,0.94));
            color: white;
            box-shadow: 0 18px 48px rgba(15,23,42,0.18);
            margin-bottom: 1rem;
        }
        .hero h1 {
            font-size: 2.15rem;
            margin: 0;
            line-height: 1.1;
        }
        .hero p {
            margin: 0.45rem 0 0;
            color: rgba(255,255,255,0.86);
            font-size: 1rem;
        }
        .metric-card {
            padding: 1rem 1.1rem;
            border-radius: 1rem;
            background: rgba(255,255,255,0.75);
            border: 1px solid rgba(148,163,184,0.22);
            box-shadow: 0 12px 24px rgba(15,23,42,0.06);
        }
        .section-title {
            font-size: 1.05rem;
            font-weight: 700;
            margin: 0.4rem 0 0.35rem;
        }
        .status-pill {
            display: inline-block;
            padding: 0.25rem 0.6rem;
            border-radius: 999px;
            font-size: 0.82rem;
            background: rgba(255,255,255,0.18);
            color: white;
            border: 1px solid rgba(255,255,255,0.25);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_csv(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes), low_memory=False)


@st.cache_data(show_spinner=False)
def cached_profile(file_bytes: bytes) -> tuple[pd.DataFrame, str, str]:
    df = load_csv(file_bytes)
    profile = build_profile(df)
    return df, profile_to_text(profile), dataframe_preview_text(df, rows=8)


def init_state() -> None:
    st.session_state.setdefault("question_input", "")
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("last_plan", None)
    st.session_state.setdefault("last_result", None)
    st.session_state.setdefault("last_error", None)
    st.session_state.setdefault("last_code", "")
    st.session_state.setdefault("fallback_mode", False)


def set_example_question(question: str) -> None:
    st.session_state["question_input"] = question


def rerun_app() -> None:
    rerun = getattr(st, "rerun", None)
    if callable(rerun):
        rerun()
        return
    experimental_rerun = getattr(st, "experimental_rerun", None)
    if callable(experimental_rerun):
        experimental_rerun()


def render_response_sections(plan: object) -> None:
    summary = getattr(plan, "summary", getattr(plan, "answer", ""))
    key_insights = list(getattr(plan, "key_insights", None) or [])
    caveats = list(getattr(plan, "caveats", None) or [])
    next_step = str(getattr(plan, "next_step", "") or "").strip()
    answer_kind = str(getattr(plan, "answer_kind", "text") or "text").lower()

    st.markdown("### Structured answer")
    if summary:
        st.write(summary)

    if key_insights:
        st.markdown("**Key insights**")
        for insight in key_insights:
            st.markdown(f"- {insight}")

    if caveats:
        st.markdown("**Caveats**")
        for caveat in caveats:
            st.markdown(f"- {caveat}")

    if next_step:
        st.info(next_step)

    st.caption(f"Response type: {answer_kind}")


def main() -> None:
    init_state()
    load_css()
    runtime_config = _resolve_runtime_config()

    header_col, badge_col = st.columns([4, 1])
    with header_col:
        st.markdown(
            """
            <div class="hero">
                <h1>AI Data Analyst Pro</h1>
                <p>Upload the Kaggle student placement and salary CSV, ask analytical questions in plain English, and get reproducible pandas results with charts.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with badge_col:
        st.markdown(
            """
            <div class="metric-card">
                <div class="section-title">Engine</div>
                <div>NVIDIA / OpenAI-compatible</div>
                <div style="margin-top:0.3rem;color:#475569;">Safe code execution</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.sidebar:
        st.header("Workspace")
        st.caption("Configure the model provider and choose a dataset source.")
        st.markdown("[Kaggle dataset source](https://www.kaggle.com/datasets/amaymishra11/student-placement-and-salary-dataset-skills-based)")
        data_source_options = ["Upload CSV"]
        if SAMPLE_DATASET_PATH.exists():
            data_source_options.append("Use bundled sample CSV")
        default_data_source = "Use bundled sample CSV" if "Use bundled sample CSV" in data_source_options else "Upload CSV"
        data_source = st.selectbox("Data source", data_source_options, index=data_source_options.index(default_data_source))
        provider = st.text_input("Provider", value=runtime_config.provider)
        model = st.text_input("Model", value=runtime_config.model)
        base_url = st.text_input("Base URL", value=runtime_config.base_url)
        with st.expander("Optional: bring your own API key", expanded=False):
            api_key_override = st.text_input("API key override", value="", type="password")
            st.caption("Leave this blank to use the secret configured in your deployment.")
        api_key = api_key_override.strip() or runtime_config.api_key
        key_state = "configured" if api_key else "not configured"
        st.markdown(f"<span class='status-pill'>API key {key_state}</span>", unsafe_allow_html=True)
        st.divider()
        st.subheader("Example questions")
        for question in SAMPLE_QUESTIONS:
            if st.button(question, use_container_width=True, key=f"sample_{question}"):
                set_example_question(question)
                rerun_app()
        st.divider()
        st.caption("Tip: for NVIDIA free/hosted access, set NVIDIA_API_KEY in your environment.")

    file_bytes: bytes | None = None
    if data_source == "Use bundled sample CSV":
        if not SAMPLE_DATASET_PATH.exists():
            st.error("Bundled sample CSV not found.")
            return
        st.info(f"Using bundled sample dataset: {SAMPLE_DATASET_PATH.name}")
        file_bytes = SAMPLE_DATASET_PATH.read_bytes()
    else:
        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
        if uploaded_file:
            file_bytes = uploaded_file.getvalue()

    if not file_bytes:
        st.info("Choose a dataset source or upload a CSV to begin. The app will profile the data before answering any questions.")
        return

    try:
        df, profile_text, preview_text = cached_profile(file_bytes)
    except Exception as exc:
        st.error(f"Could not read the CSV: {exc}")
        return

    profile = build_profile(df)
    resolved_config = runtime_config.resolved(api_key=api_key, model=model, base_url=base_url)
    resolved_config = AppConfig(
        provider=provider,
        api_key=resolved_config.api_key,
        base_url=resolved_config.base_url,
        model=resolved_config.model,
        temperature=runtime_config.temperature,
        max_rows_preview=runtime_config.max_rows_preview,
    )
    fallback_active = not bool(resolved_config.api_key)

    overview_col, profile_col = st.columns([1.15, 1])
    with overview_col:
        st.subheader("Data preview")
        st.dataframe(df.head(20), use_container_width=True)
    with profile_col:
        st.subheader("Dataset profile")
        st.code(profile_text, language="text")

    metrics = st.columns(4)
    with metrics[0]:
        st.metric("Rows", f"{len(df):,}")
    with metrics[1]:
        st.metric("Columns", f"{len(df.columns):,}")
    with metrics[2]:
        st.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
    with metrics[3]:
        if "placed" in df.columns:
            placed_rate = pd.to_numeric(df["placed"], errors="coerce").fillna(0).mean() * 100
            st.metric("Placed rate", f"{placed_rate:.1f}%")
        else:
            st.metric("Non-empty rows", f"{int(df.dropna(how='all').shape[0]):,}")

    tabs = st.tabs(["Ask", "History", "Raw schema"])

    with tabs[0]:
        st.subheader("Chat with your CSV")
        st.caption("Ask one question at a time. The model will generate a guarded pandas analysis plan.")
        if fallback_active:
            st.info("Fallback mode is active because no API key is configured. Common questions will still work locally.")

        with st.form("question_form", clear_on_submit=False):
            question = st.text_area(
                "Question",
                key="question_input",
                height=120,
                placeholder="Example: Which region has the highest total sales?",
            )
            submitted = st.form_submit_button("Analyze")

        if submitted:
            question = question.strip()
            if not question:
                st.warning("Type a question before running analysis.")
            else:
                prompt = build_analysis_prompt(question, profile, preview_text)
                try:
                    with st.spinner("Generating analysis plan..."):
                        plan = generate_analysis_plan(prompt, resolved_config)
                        st.session_state["fallback_mode"] = False
                except LLMError:
                    plan = build_fallback_plan(question, df)
                    st.session_state["fallback_mode"] = True
                except Exception:
                    plan = build_fallback_plan(question, df)
                    st.session_state["fallback_mode"] = True
                try:
                    st.session_state["last_plan"] = plan
                    st.session_state["last_code"] = plan.code
                    with st.expander("Generated plan", expanded=False):
                        st.write({
                            "answer_kind": plan.answer_kind,
                            "summary": getattr(plan, "summary", getattr(plan, "answer", "")),
                            "key_insights": getattr(plan, "key_insights", None),
                            "chart_title": plan.chart_title,
                            "caveats": getattr(plan, "caveats", None),
                            "next_step": getattr(plan, "next_step", ""),
                        })
                        st.code(plan.code, language="python")
                    render_response_sections(plan)

                    should_execute = bool(plan.code.strip()) and plan.answer_kind != "clarification"
                    if should_execute:
                        with st.spinner("Executing analysis..."):
                            outcome = execute_analysis_code(plan.code, df.copy())
                        st.session_state["last_result"] = outcome
                        st.session_state["last_error"] = None

                        tables, scalar_text = render_result(outcome.result)
                        for table in tables:
                            st.dataframe(table, use_container_width=True)
                        if not tables and scalar_text:
                            st.write(scalar_text)

                        if outcome.figures:
                            st.markdown("### Chart")
                            for index, fig in enumerate(outcome.figures, start=1):
                                title = format_figure_title(fig, fallback=f"Chart {index}")
                                st.caption(title)
                                st.pyplot(fig, clear_figure=False, use_container_width=True)
                    else:
                        st.session_state["last_result"] = None
                        st.session_state["last_error"] = None

                    st.session_state["history"].append(
                        {
                            "question": question,
                            "answer": getattr(plan, "summary", getattr(plan, "answer", "")),
                            "code": plan.code,
                            "result_type": "clarification" if plan.answer_kind == "clarification" else (outcome.result_type if should_execute else "None"),
                            "mode": "fallback" if st.session_state.get("fallback_mode") else "llm",
                        }
                    )
                except (LLMError, AnalysisCodeError, Exception) as exc:
                    st.session_state["last_error"] = str(exc)
                    st.error(str(exc))

        if st.session_state.get("last_error"):
            st.error(st.session_state["last_error"])

    with tabs[1]:
        st.subheader("Analysis history")
        if not st.session_state["history"]:
            st.info("No questions answered yet.")
        else:
            for index, item in enumerate(reversed(st.session_state["history"]), start=1):
                with st.expander(f"{index}. {item['question']}", expanded=index == 1):
                    st.write(item.get("answer") or "")
                    st.caption(f"Mode: {item.get('mode', 'llm')} | Result type: {item.get('result_type', 'None')}")
                    st.code(item.get("code", ""), language="python")

    with tabs[2]:
        st.subheader("Raw schema")
        st.write(df.dtypes.astype(str).to_frame(name="dtype"))
        st.write("Shape:", df.shape)
        st.write("Missing values:")
        st.write(df.isna().sum().to_frame(name="missing_cells"))
        st.write("Preview text used in the prompt")
        st.code(preview_text, language="text")

    st.divider()
    st.caption("Built for CSV analysis workflows. The generated code is validated before execution and runs in a restricted namespace.")


if __name__ == "__main__":
    main()
