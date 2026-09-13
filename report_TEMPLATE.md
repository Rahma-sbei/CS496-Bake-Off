<!--
  Fill in every [BRACKET] using results/summary.csv and results/per_item.csv after
  you rerun bake-off.py + report_generator.py with real API keys. Then convert this
  file to a 2-page report.pdf (pandoc, or paste into Google Docs/Word and export).
  Keep it to 2 pages - trim prose, not numbers.
-->

# Project 0 — Bake-Off Report

**Team:** [Name 1, Name 2, Name 3, Name 4] · **Date:** [YYYY-MM-DD]

## 1. The task

We evaluate three models on 50 short math problems (arithmetic, algebra, word
problems, basic finance), one model call per item, scored automatically by
exact/symbolic equivalence (SymPy) against a human-verified answer.

**Example item:**
> Q: "A rectangle's length is 3m longer than width w. Perimeter is 26m. Calculate width w."
> Expected answer: `5`

## 2. The data

- **Source:** hand-written by the team, covering 3 difficulty bands (10 arithmetic,
  10 algebraic substitution, 30 word problems / finance).
- **Label check:** [describe how two people independently checked all 50 expected
  answers — e.g. "two team members solved each item independently and compared."]
- **Dev/test split:** [state it — e.g. "no separate dev set; all 50 items used as the
  single frozen test set, prompt was written and fixed before any model was run."]

## 3. Setup

| Tier | Model | Provider |
|---|---|---|
| Top API | `openai/gpt-oss-120b` | Groq |
| Cheap API | `openai/gpt-oss-20b` | Groq |
| Local (self-hosted) | `qwen2.5:7b` | Ollama, [your hardware] |

Temperature 0, max_tokens 256, identical system prompt for all three models
(forces `\boxed{answer}`-only output), same 50 items in the same order, same
parser. Run on [date].

## 4. Results

| Tier | Model | Score | Accuracy | p50 latency | p95 latency | $/1k now | $/1k @100x |
|---|---|---|---|---|---|---|---|
| Top API | openai/gpt-oss-120b | [n]/50 | [X]% | [X] ms | [X] ms | $[X] | $[X] |
| Cheap API | openai/gpt-oss-20b | [n]/50 | [X]% | [X] ms | [X] ms | $[X] | $[X] |
| Local | qwen2.5:7b | [n]/50 | [X]% | [X] ms | [X] ms | $[X] | $[X] |

**3 wrong answers per model** (from `results/per_item.csv`):

- Top: [item_id] — expected `[X]`, got `[X]`
- Cheap: [item_id] — expected `[X]`, got `[X]`
- Local: [item_id] — expected `[X]`, got `[X]`

## 5. Our choice

[One paragraph: which model you'd actually use for this task, and why — weigh
accuracy vs. the cost/latency numbers above. Then: what would make you switch —
e.g. "if traffic exceeds N requests/hour, self-hosting becomes cheaper" or "if
accuracy on harder items matters more than cost, use the top-tier model
regardless of price."]

## 6. Cost at scale

- At 100x today's traffic: [state each model's $/1k from the table above — API
  models are flat since there's no bulk discount on these list prices; the local
  model's $/1k falls because the fixed hourly hardware cost is spread over more
  requests, up to its real throughput ceiling.]
- **Break-even volume:** local hosting becomes as cheap as [API model] at
  ~[X] requests/hour (from `results/report.html`, section 2). Below that, the API
  is cheaper because you're paying for idle local hardware time.
