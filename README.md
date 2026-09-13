# Model Evaluation Bake-off, CS496

A benchmark tool to evaluate and compare the reasoning and accuracy of cloud and local LLMs against a verified ground-truth dataset.

## Overview

This project runs an automated evaluation across 50 mathematical questions of varying difficulty (ranging from basic arithmetic to algebraic expressions). Each item features human-made, verified answers.

We evaluate three model tiers:

1. **Top API Model:** `nvidia/nemotron-3-ultra-550b-a55b:free` (via OpenRouter)
2. **Cheap API Model:** `cohere/north-mini-code:free` (via OpenRouter)
3. **Local Open-Weights Model:** `qwen2.5:7b` (via Ollama)

The pipeline executes all 50 items against each model, compares output correctness against the ground-truth targets, logs latencies/errors, and exports the findings into a report.

---

## Prerequisites & Requirements

- **Python 3.10+**
- **OpenRouter API Key:** Required for cloud model inference.
- **Ollama:** Installed and running locally with the Qwen model pulled:
  ```bash
  ollama run qwen2.5:7b
  Quickstart & Execution
  ```

1. Set Up Environment
   Create a virtual environment and install dependencies:

# Create virtual environment

```
python -m venv .venv
```

# Activate environment

# On Linux/macOS:

```
source .venv/bin/activate
```

# On Windows:

```
.venv\Scripts\activate
```

# Install requirements

```
pip install -r requirements.txt
```

2. Configure Environment Variables

Create a `.env` file in the root directory and add your OpenRouter API key, and you local ollama url:

```
OPENROUTER_API_KEY=your_openrouter_api_key_here
OLLAMA_BASE_URL=http://localhost:11434/v1

```

3. Run the Bake-off Benchmark
   Execute the evaluation script to query all models and produce raw performance metrics:

```
python bake-off.py
```

Outputs: results.csv (contains responses, latency, and pass/fail status per item).

4. Generate the HTML Report
   Run the report generator to convert raw results into an interactive HTML dashboard:

```
python report_generator.py
```

Outputs: report.html (open in any browser to inspect metrics, charts, and failure analysis).
