# AI Data Analyst Pro

A research-grade Streamlit app for chatting with CSV files. Upload a dataset, ask questions in natural language, and get tables or charts powered by a configurable LLM workflow.

## Features

- CSV upload with automatic schema and quality profiling
- Natural-language questions over any tabular dataset
- Constrained LLM prompts that produce JSON analysis plans
- Guarded local execution of generated pandas code
- Matplotlib chart rendering
- Chat-style history for prior questions
- NVIDIA API / OpenAI-compatible provider support

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
- What is the average salary by placement status?
- Which feature is most associated with placement?
- Plot CGPA distribution by placement status.
- Show top 5 streams by average salary.

## Notes

The app validates generated code before execution and blocks imports, filesystem access, and other dangerous operations. It is designed for analytical workflows, not arbitrary code execution.
