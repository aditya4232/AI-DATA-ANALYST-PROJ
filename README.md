# AI Data Analyst Pro

A research-grade Streamlit app for chatting with CSV files. Upload a dataset, ask questions in natural language, and get tables or charts powered by a configurable LLM workflow.

## Dataset source

This project is tailored to the Kaggle dataset you shared:

https://www.kaggle.com/datasets/amaymishra11/student-placement-and-salary-dataset-skills-based

The app remains dataset-agnostic, but the examples and analysis defaults are aligned to the student placement and salary use case.

## Download the dataset

Install the extra dependency and run the helper script:

```bash
pip install -r requirements.txt
python scripts/download_kaggle_dataset.py
```

The script uses the KaggleHub call you provided and prints the local path to the downloaded files.

## Features

- CSV upload with automatic schema and quality profiling
- Natural-language questions over any tabular dataset
- Constrained LLM prompts that produce JSON analysis plans
- Guarded local execution of generated pandas code
- Matplotlib chart rendering
- Chat-style history for prior questions
- NVIDIA API / OpenAI-compatible provider support

## Project layout

- `app.py` - Streamlit entry point and UI
- `src/` - profiling, prompts, execution, LLM, fallback, and chart helpers
- `scripts/` - utility scripts such as the Kaggle dataset downloader
- `student_placement_salary_elite_v2.csv` - bundled sample dataset

## Quick start

1. Create and activate a Python virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set your API key and optional model settings:

```bash
set NVIDIA_API_KEY=your_key_here
set LLM_PROVIDER=nvidia
set LLM_MODEL=meta/llama-3.1-70b-instruct
```

4. Launch the app:

```bash
streamlit run app.py
```

Inside the app, choose **Use bundled sample CSV** in the sidebar to start from the included dataset immediately.

## Docker deployment

Build and run locally with Docker:

```bash
docker build -t ai-data-analyst-pro .
docker run -p 8501:8501 -e NVIDIA_API_KEY=your_key_here ai-data-analyst-pro
```

The container starts Streamlit on port `8501` and is suitable for most container platforms.

## Streamlit Cloud deployment

1. Push this repo to GitHub.
2. Deploy it on Streamlit Cloud and point the app entry to `app.py`.
3. Add secrets or environment variables in the Streamlit Cloud settings:

```text
NVIDIA_API_KEY=your_key_here
LLM_PROVIDER=nvidia
LLM_MODEL=meta/llama-3.1-70b-instruct
```

4. Confirm `student_placement_salary_elite_v2.csv` stays in the repository so the bundled sample mode works.
5. The app keeps the configured API key hidden; users can optionally paste their own key in the sidebar if they want to override the deployment secret.

## Secret setup locally

For local development, you can create `.streamlit/secrets.toml`:

```toml
NVIDIA_API_KEY = "your_key_here"
LLM_PROVIDER = "nvidia"
LLM_MODEL = "meta/llama-3.1-70b-instruct"
```

The app will use these values by default and will not show the key in the interface.

## Deployment checklist

- `app.py` starts without extra setup when the bundled CSV is present.
- `requirements.txt` includes every runtime dependency.
- `Dockerfile` builds the app into a Streamlit container.
- The app falls back to local deterministic analysis if no API key is set.
- The app prepends `src/` to `sys.path`, so Streamlit Cloud can import the helper modules reliably.
- Streamlit width warnings are removed by using the newer `width="stretch"` parameter.

## Environment variables

- `LLM_PROVIDER` - defaults to `nvidia`
- `LLM_API_KEY` - generic override for any provider
- `NVIDIA_API_KEY` - NVIDIA API key
- `NVIDIA_BASE_URL` - defaults to `https://integrate.api.nvidia.com/v1`
- `NVIDIA_MODEL` - defaults to `meta/llama-3.1-70b-instruct`
- `OPENAI_API_KEY` - fallback if you want to use an OpenAI-compatible provider
- `LLM_BASE_URL` - custom OpenAI-compatible endpoint
- `LLM_MODEL` - custom model name
- `LLM_TEMPERATURE` - defaults to `0`

## Suggested questions for the placement dataset

- Which stream has the highest placement rate?
- What is the average salary_lpa by placed status?
- Which features are most associated with placement?
- Plot CGPA distribution by placed status.
- Show the top 5 streams by average salary_lpa.

## Notes

The app validates generated code before execution and blocks imports, filesystem access, and other dangerous operations. It is designed for analytical workflows, not arbitrary code execution.
If no API key is configured, the app switches to a deterministic fallback planner for common CSV questions so the deployment still works offline for basic analysis.
