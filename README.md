# Model Evaluation Bake-off, CS496

A benchmark tool to evaluate and compare the reasoning and accuracy of cloud and local LLMs against a verified ground-truth dataset.

## Overview

This project runs an automated evaluation across 50 math questions (arithmetic, algebra, word problems, basic finance), each with a single human-verified correct answer. Scoring is fully automatic (SymPy equivalence check, no human or LLM judge).

We evaluate three model tiers:

1. **Top API Model:** `openai/gpt-oss-120b` (via Groq)
2. **Cheap API Model:** `openai/gpt-oss-20b` (via Groq)
3. **Local Open-Weights Model:** `qwen2.5:7b` (via Ollama, run on our own hardware)

The pipeline runs all 50 items against each model, scores correctness against the ground-truth answers, logs latency and real token usage, and produces `results/per_item.csv`, `results/summary.csv`, and an HTML report.

---

## Prerequisites

- Python 3.10+
- A **Groq API key** (free at [console.groq.com](https://console.groq.com/keys)) — used for both API tiers
- **Ollama** installed and running locally with the model pulled:
  ```bash
  ollama pull qwen2.5:7b
  ```

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file in the repo root:

```
GROQ_API_KEY=your_groq_api_key_here
OLLAMA_BASE_URL=http://localhost:11434/v1
```

Run the benchmark, then generate the report:

```bash
python bake-off.py
python report_generator.py
```

Outputs:
- `results/per_item.csv` — one row per (model, item): prediction, correctness, latency, real token counts, real cost
- `results/summary.csv` — one row per model: accuracy, p50/p95 latency, cost per 1k requests (now and at 100x traffic)
- `results/report.html` — human-readable dashboard

**Before running `report_generator.py`**, set `HARDWARE_HOURLY_COST_USD` at the top of that file to the real hourly cost of the machine Ollama ran on (a comparable cloud instance's price, or your electricity rate) — see `results/hardware_note.md`.

## No API keys in this repo

`.env` is git-ignored. Never commit real keys.

## Contributions

<!-- TODO: fill in with real names before freezing the repo -->
- **Rahma Sbei** — initial repo setup, task design (50 math items), SymPy scorer, HTML report generator
- **[Name 2]** — [what they did]
- **[Name 3]** — [what they did]
- **[Name 4]** — [what they did]
