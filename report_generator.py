import os
import csv
import numpy as np


HARDWARE_HOURLY_COST_USD = 0

MODEL_TIERS = {
    "openai/gpt-oss-120b": "Top API Model",
    "openai/gpt-oss-20b": "Cheap API Model",
    "qwen2.5:7b": "Local Open-Weights Model",
}

LOCAL_MODEL_NAME = "qwen2.5:7b"

# --------------------------------------------------------------------------
# Report metadata. Purely descriptive - none of this feeds the math above.
# Edit these per run to keep the report's narrative sections in sync with
# whatever produced results/per_item.csv.
# --------------------------------------------------------------------------
REPORT_TITLE = "Project 0 — Model Bake-Off Report"
REPORT_SUBTITLE = "Three models, one frozen test set, judged on accuracy, latency and cost."
TEAM_MEMBERS = ["Nadia Ben Njima", "Ahmed Brini", "Rahma Sbei", "Mariem Lahmar"]
REPORT_DATE = "2026-09-13"

TASK_DESCRIPTION = "Evaluate three models on 50 short math problems against a human-verified answer."
EXAMPLE_ITEM = {
    "question": "A rectangle's length is 3m longer than width w. Perimeter is 26m. Calculate width w.",
    "expected": "5",
}
DATASET_INFO = {
    "Source": "Hand-written by the team",
    "Composition": "10 arithmetic, 10 algebraic substitution, 30 word problems / finance",
    "Label check": "Two team members independently solved each item; all ground-truth answers matched",
    "Dev/test split": "None — all 50 items served as the single frozen test set",
}

MODEL_PROVIDERS = {
    "openai/gpt-oss-120b": "Groq",
    "openai/gpt-oss-20b": "Groq",
    "qwen2.5:7b": "Ollama (self-hosted)",
}
RUN_PARAMETERS = {
    "Temperature": "0",
    "Max tokens": "256",
    "System prompt": "Identical across models, forces \\boxed{answer}-only output",
    "Item order": "Same 50 items, same order, same parser for all models",
    "Run date": REPORT_DATE,
}

HARDWARE_INFO = {
    "Machine": "NVIDIA GeForce RTX 3050, 24GB RAM",
    "Ollama version": "0.34.0",
}


def generate_report(csv_path: str = "results/per_item.csv",
                     output_html_path: str = "results/report.html",
                     summary_csv_path: str = "results/summary.csv"):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Source CSV file not found at '{csv_path}'. Run bake-off.py first.")

    if HARDWARE_HOURLY_COST_USD is None:
        raise ValueError(
            "Set HARDWARE_HOURLY_COST_USD at the top of this file before generating the report "
            "(the real $/hour your Ollama machine would cost to rent, or your electricity cost). "
            "This is required for the cost-at-100x and break-even numbers the brief asks for."
        )

    # Group evaluation items by model name
    models_data = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            model_name = row["model"]
            if model_name not in models_data:
                models_data[model_name] = {
                    "tier": row.get("tier") or MODEL_TIERS.get(model_name, "API Model"),
                    "items": [],
                }
            models_data[model_name]["items"].append(row)

    summary_results = []

    for model_name, data in models_data.items():
        items = data["items"]
        total_items = len(items)
        correct_count = sum(1 for r in items if r["is_correct"].strip().lower() == "true")
        wrong_count = total_items - correct_count
        latencies = sorted(float(r["latency_ms"]) for r in items)
        p50 = float(np.percentile(latencies, 50)) if latencies else 0.0
        p95 = float(np.percentile(latencies, 95)) if latencies else 0.0

        is_local = model_name == LOCAL_MODEL_NAME

        if is_local:
            # Brief's formula: hardware cost per hour / requests per hour.
            # Requests/hour here = current measured serial throughput (3600s / p50 latency).
            requests_per_hour_now = 3600.0 / (p50 / 1000.0) if p50 > 0 else 0.0
            cost_per_req_now = (HARDWARE_HOURLY_COST_USD / requests_per_hour_now) if requests_per_hour_now else 0.0
            cost_per_1k_now = cost_per_req_now * 1000
            # Same box, 100x more requests routed through it (only valid if the
            # hardware can actually sustain that throughput - call this out in the report).
            cost_per_1k_100x = cost_per_1k_now / 100.0
            comp_tokens = [int(r["completion_tokens"]) for r in items if r.get("completion_tokens")]
            lat_s = [float(r["latency_ms"]) / 1000.0 for r in items if r.get("completion_tokens")]
            tok_per_sec = (
                float(np.mean([c / l for c, l in zip(comp_tokens, lat_s) if l > 0]))
                if comp_tokens else 0.0
            )
        else:
            costs = [float(r["cost_usd"]) for r in items if r.get("cost_usd")]
            cost_per_req_now = float(np.mean(costs)) if costs else 0.0
            cost_per_1k_now = cost_per_req_now * 1000
            # No committed-use/bulk discount on these list prices -> flat per-request cost at any volume.
            cost_per_1k_100x = cost_per_1k_now
            tok_per_sec = None

        wrong_examples = []
        for r in items:
            if r["is_correct"].strip().lower() != "true" and len(wrong_examples) < 3:
                wrong_examples.append({
                    "id": r["item_id"], "question": r["question"],
                    "expected": r["expected"], "actual": r["prediction"],
                })

        summary_results.append({
            "tier": data["tier"],
            "model_name": model_name,
            "is_local": is_local,
            "correct": correct_count,
            "wrong": wrong_count,
            "total": total_items,
            "accuracy_pct": round(100.0 * correct_count / total_items, 2) if total_items else 0.0,
            "p50_latency": round(p50, 2),
            "p95_latency": round(p95, 2),
            "cost_per_1k_now": round(cost_per_1k_now, 4),
            "cost_per_1k_100x": round(cost_per_1k_100x, 4),
            "tokens_per_sec": round(tok_per_sec, 1) if tok_per_sec else None,
            "wrong_examples": wrong_examples,
        })

    # --- Break-even volume: local cost/1k (which falls as traffic rises, since the
    # hourly hardware rate is fixed) vs each API model's flat cost/1k. Solve for the
    # requests/hour at which local's amortized cost matches the API's flat cost.
    local_row = next((s for s in summary_results if s["is_local"]), None)
    breakeven_notes = []
    if local_row:
        p50_s = local_row["p50_latency"] / 1000.0
        for s in summary_results:
            if s["is_local"] or s["cost_per_1k_now"] <= 0:
                continue
            api_cost_per_req = s["cost_per_1k_now"] / 1000.0
            breakeven_req_per_hour = HARDWARE_HOURLY_COST_USD / api_cost_per_req if api_cost_per_req else float("inf")
            breakeven_notes.append({
                "vs": s["model_name"],
                "requests_per_hour": round(breakeven_req_per_hour, 1),
            })

    # --- Write results/summary.csv ---
    os.makedirs(os.path.dirname(summary_csv_path) or ".", exist_ok=True)
    with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["tier", "model_name", "correct", "wrong", "total", "accuracy_pct",
                      "p50_latency_ms", "p95_latency_ms", "cost_per_1k_usd_now",
                      "cost_per_1k_usd_at_100x", "tokens_per_sec"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in summary_results:
            writer.writerow({
                "tier": s["tier"], "model_name": s["model_name"],
                "correct": s["correct"], "wrong": s["wrong"], "total": s["total"],
                "accuracy_pct": s["accuracy_pct"],
                "p50_latency_ms": s["p50_latency"], "p95_latency_ms": s["p95_latency"],
                "cost_per_1k_usd_now": s["cost_per_1k_now"],
                "cost_per_1k_usd_at_100x": s["cost_per_1k_100x"],
                "tokens_per_sec": s["tokens_per_sec"] or "",
            })

    html_content = _build_html(summary_results, breakeven_notes, local_row)

    os.makedirs(os.path.dirname(output_html_path), exist_ok=True)
    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"Wrote {summary_csv_path} and {output_html_path}.")


# ==========================================================================
# Everything below builds the HTML. No evaluation math lives here - it only
# formats summary_results / breakeven_notes, which are computed above.
# ==========================================================================

def _esc(value) -> str:
    """Minimal HTML escaping for values pulled from the CSV."""
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _pick_recommendation(summary_results):
    """Pick the model with the best accuracy, breaking ties on lowest cost/1k now.
    Purely descriptive - used only to write the conclusion paragraph."""
    if not summary_results:
        return None
    best_acc = max(s["accuracy_pct"] for s in summary_results)
    top = [s for s in summary_results if s["accuracy_pct"] == best_acc]
    return min(top, key=lambda s: s["cost_per_1k_now"])


def _build_conclusion(summary_results, breakeven_notes, local_row):
    """Generate a short, data-driven recommendation paragraph from summary_results.
    This narrates the numbers already computed in generate_report(); it does not
    recompute or alter any of them."""
    if not summary_results:
        return "<p>No results to summarize.</p>"

    recommended = _pick_recommendation(summary_results)
    cheapest_now = min(summary_results, key=lambda s: s["cost_per_1k_now"])
    fastest = min(summary_results, key=lambda s: s["p50_latency"])

    paragraphs = []

    lead = (
        f"<strong>{_esc(recommended['model_name'])}</strong> ({_esc(recommended['tier'])}) is the pick "
        f"from this run: {recommended['accuracy_pct']}% accuracy at ${recommended['cost_per_1k_now']:.4f} "
        f"per 1k requests, with a {recommended['p50_latency']:.0f} ms median latency."
    )
    if recommended["model_name"] != cheapest_now["model_name"]:
        lead += (
            f" {_esc(cheapest_now['model_name'])} is cheaper per request "
            f"(${cheapest_now['cost_per_1k_now']:.4f}/1k) but trails on accuracy "
            f"({cheapest_now['accuracy_pct']}%), which is why it isn't the top choice here."
        )
    if fastest["model_name"] != recommended["model_name"]:
        lead += (
            f" {_esc(fastest['model_name'])} has the lowest median latency "
            f"({fastest['p50_latency']:.0f} ms) if response time matters more than the accuracy gap."
        )
    paragraphs.append(lead)

    if local_row:
        match = next((b for b in breakeven_notes if b["vs"] == recommended["model_name"]), None)
        if local_row["model_name"] == recommended["model_name"]:
            paragraphs.append(
                f"The self-hosted {_esc(local_row['model_name'])} is already the strongest option on "
                f"accuracy and cost alike, so there's no API break-even to weigh here."
            )
        elif local_row["accuracy_pct"] < 70:
            note = (
                f" At scale its cost per request would fall toward ${local_row['cost_per_1k_100x']:.4f}/1k, "
                f"but that only matters once its accuracy ({local_row['accuracy_pct']}%) is fixed, "
                f"for instance through fine-tuning or a larger local model."
            )
            paragraphs.append(
                f"Self-hosting {_esc(local_row['model_name'])} isn't viable yet: accuracy "
                f"({local_row['accuracy_pct']}%) is the blocker, not cost." + note
            )
        elif match:
            paragraphs.append(
                f"Self-hosting {_esc(local_row['model_name'])} only pays off above roughly "
                f"{match['requests_per_hour']:.0f} requests/hour against {_esc(recommended['model_name'])}'s "
                f"pricing; below that volume the fixed hardware cost isn't spread over enough requests "
                f"to beat the API."
            )
    paragraphs.append(
        "Revisit this call if traffic volume, problem difficulty, or model pricing shifts enough to "
        "change any of the numbers above."
    )
    return "".join(f"<p>{p}</p>" for p in paragraphs)


def _build_html(summary_results, breakeven_notes, local_row):
    recommended = _pick_recommendation(summary_results)
    recommended_name = recommended["model_name"] if recommended else None

    # --- Section: setup table (tier / model / provider) ---
    setup_rows = "".join(
        f"""<tr>
<td>{_esc(s['tier'])}</td>
<td><code>{_esc(s['model_name'])}</code></td>
<td>{_esc(MODEL_PROVIDERS.get(s['model_name'], '—'))}</td>
</tr>"""
        for s in summary_results
    )
    run_param_rows = "".join(
        f'<div class="kv-row"><span class="kv-key">{_esc(k)}</span><span class="kv-val">{_esc(v)}</span></div>'
        for k, v in RUN_PARAMETERS.items()
    )
    dataset_rows = "".join(
        f'<div class="kv-row"><span class="kv-key">{_esc(k)}</span><span class="kv-val">{_esc(v)}</span></div>'
        for k, v in DATASET_INFO.items()
    )

    # --- Section: results table ---
    rows_html = ""
    for s in summary_results:
        acc = s["accuracy_pct"]
        acc_color = "#6FAE7C" if acc >= 80 else ("#C9A24B" if acc >= 50 else "#D9645F")
        is_winner = s["model_name"] == recommended_name
        tokens_cell = f"{s['tokens_per_sec']} tok/s" if s.get("tokens_per_sec") else "—"
        rows_html += f"""
<tr class="{'winner-row' if is_winner else ''}">
<td>{_esc(s['tier'])}{' <span class="winner-tag">recommended</span>' if is_winner else ''}</td>
<td><code>{_esc(s['model_name'])}</code></td>
<td>{s['correct']}/{s['total']}</td>
<td><span class="pill" style="background:{acc_color}22;color:{acc_color};border:1px solid {acc_color}55;">{acc}%</span></td>
<td>{s['p50_latency']:.0f} ms</td>
<td>{s['p95_latency']:.0f} ms</td>
<td class="num">${s['cost_per_1k_now']:.4f}</td>
<td class="num">${s['cost_per_1k_100x']:.4f}</td>
<td class="num">{tokens_cell}</td>
</tr>"""

    # --- Section: failure analysis ---
    failure_cards_html = ""
    has_failures = False
    for s in summary_results:
        for ex in s["wrong_examples"]:
            has_failures = True
            failure_cards_html += f"""
<div class="failure-card">
<div class="card-header">
<span class="item-id">Item {_esc(ex['id'])}</span>
<span class="tier-tag">{_esc(s['tier'])} · {_esc(s['model_name'])}</span>
</div>
<p class="failure-q">{_esc(ex['question'])}</p>
<div class="comparison-grid">
<div><span class="label">Expected</span><code class="code-target">{_esc(ex['expected'])}</code></div>
<div><span class="label">Model output</span><code class="code-err">{_esc(ex['actual'])}</code></div>
</div>
</div>"""
    if not has_failures:
        failure_cards_html = '<p class="no-failures">No failures recorded across any benchmarked model.</p>'

    # --- Section: break-even ---
    if breakeven_notes:
        breakeven_html = "".join(
            f"""<div class="kv-row">
<span class="kv-key">vs. {_esc(b['vs'])}</span>
<span class="kv-val">~{b['requests_per_hour']:.0f} requests/hour</span>
</div>"""
            for b in breakeven_notes
        )
        breakeven_section = f"""
<p>Local hosting's cost per request falls as more traffic is spread over the same fixed
hardware. Below each threshold, the API is cheaper because you're paying for idle
hardware time; above it, self-hosting wins per request.</p>
<div class="kv-panel">{breakeven_html}</div>"""
    else:
        breakeven_section = "<p>No self-hosted model in this run, so there's no break-even volume to compute.</p>"

    # --- Section: hardware note (only if a local model was run) ---
    if local_row:
        hardware_rows = "".join(
            f'<div class="kv-row"><span class="kv-key">{_esc(k)}</span><span class="kv-val">{_esc(v)}</span></div>'
            for k, v in HARDWARE_INFO.items()
        )
        hardware_rows += f"""
<div class="kv-row"><span class="kv-key">Measured p50 latency</span><span class="kv-val">{local_row['p50_latency']:.0f} ms</span></div>
<div class="kv-row"><span class="kv-key">Measured throughput</span><span class="kv-val">{local_row['tokens_per_sec'] or '—'} tok/s</span></div>"""
        hardware_section = f"""
<section>
<h2><span class="sec-no">05</span> Hardware note — {_esc(local_row['model_name'])}</h2>
<div class="kv-panel">{hardware_rows}</div>
</section>"""
        conclusion_no = "06"
    else:
        hardware_section = ""
        conclusion_no = "05"

    conclusion_html = _build_conclusion(summary_results, breakeven_notes, local_row)
    team_line = ", ".join(TEAM_MEMBERS)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(REPORT_TITLE)}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Newsreader:ital,wght@0,400;0,600;1,400&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root {{
  --bg: #0b0f17;
  --panel: #121826;
  --panel-alt: #0e1420;
  --border: #223049;
  --text: #e4e9f2;
  --text-dim: #8c97ab;
  --gold: #c9a24b;
  --teal: #4f9c93;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font-family: 'IBM Plex Sans', -apple-system, sans-serif;
  line-height: 1.6;
  padding: 56px 24px 96px;
}}
.container {{ max-width: 760px; margin: 0 auto; }}
header.title-block h1 {{
  font-family: 'Newsreader', Georgia, serif;
  font-weight: 600;
  font-size: 2.1rem;
  margin: 0 0 10px;
  letter-spacing: -0.01em;
}}
header.title-block .subtitle {{
  color: var(--text-dim);
  font-family: 'Newsreader', Georgia, serif;
  font-style: italic;
  font-size: 1.05rem;
  margin: 0 0 24px;
}}
.meta-row {{
  display: flex;
  gap: 32px;
  flex-wrap: wrap;
  padding-top: 18px;
  border-top: 1px solid var(--border);
  font-size: 0.85rem;
}}
.meta-row div span.k {{ color: var(--text-dim); display: block; margin-bottom: 2px; }}
hr.divider {{ border: none; border-top: 1px solid var(--border); margin: 44px 0; }}
section {{ margin-bottom: 48px; }}
h2 {{
  font-family: 'Newsreader', Georgia, serif;
  font-weight: 600;
  font-size: 1.3rem;
  margin: 0 0 16px;
}}
.sec-no {{
  font-family: 'IBM Plex Mono', monospace;
  color: var(--gold);
  font-size: 0.85rem;
  margin-right: 10px;
}}
p {{ margin: 0 0 14px; }}
.example-box {{
  background: var(--panel-alt);
  border-left: 3px solid var(--teal);
  border-radius: 4px;
  padding: 14px 18px;
  font-size: 0.92rem;
  margin: 18px 0;
}}
.example-box .label {{ color: var(--text-dim); font-size: 0.78rem; display: block; margin-bottom: 4px; }}
table {{ width: 100%; border-collapse: collapse; font-size: 0.88rem; }}
th {{
  text-align: left;
  color: var(--text-dim);
  font-weight: 500;
  font-size: 0.78rem;
  padding: 10px 12px;
  border-bottom: 1px solid var(--border);
}}
td {{ padding: 12px; border-bottom: 1px solid var(--border); vertical-align: middle; }}
td.num {{ font-family: 'IBM Plex Mono', monospace; }}
tr.winner-row td {{ background: rgba(201, 162, 75, 0.06); }}
tr.winner-row td:first-child {{ border-left: 2px solid var(--gold); }}
.winner-tag {{
  font-size: 0.72rem;
  color: var(--gold);
  border: 1px solid rgba(201,162,75,0.4);
  border-radius: 3px;
  padding: 1px 6px;
  margin-left: 6px;
}}
code {{ font-family: 'IBM Plex Mono', monospace; color: #9ecbff; background: var(--panel-alt); padding: 1px 5px; border-radius: 3px; }}
.pill {{ padding: 3px 9px; border-radius: 5px; font-size: 0.82rem; font-weight: 500; }}
.kv-panel {{ background: var(--panel); border: 1px solid var(--border); border-radius: 6px; padding: 4px 18px; }}
.kv-row {{ display: flex; justify-content: space-between; gap: 20px; padding: 11px 0; border-bottom: 1px solid var(--border); font-size: 0.9rem; }}
.kv-row:last-child {{ border-bottom: none; }}
.kv-key {{ color: var(--text-dim); }}
.kv-val {{ text-align: right; font-family: 'IBM Plex Mono', monospace; font-size: 0.85rem; }}
.failure-card {{ background: var(--panel); border: 1px solid var(--border); border-left: 3px solid #D9645F; border-radius: 6px; padding: 16px 18px; margin-bottom: 14px; }}
.card-header {{ display: flex; justify-content: space-between; font-size: 0.82rem; color: var(--text-dim); margin-bottom: 10px; }}
.failure-q {{ margin-bottom: 12px; font-size: 0.92rem; }}
.comparison-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
.comparison-grid .label {{ display: block; font-size: 0.76rem; color: var(--text-dim); margin-bottom: 4px; }}
.code-target {{ background: rgba(111,174,124,0.12); color: #8fca9c; display: block; }}
.code-err {{ background: rgba(217,100,95,0.12); color: #eb9691; display: block; }}
.no-failures {{ color: var(--text-dim); font-style: italic; }}
.conclusion-box {{ background: var(--panel); border: 1px solid var(--border); border-radius: 6px; padding: 20px 22px; }}
footer {{ margin-top: 60px; color: var(--text-dim); font-size: 0.78rem; border-top: 1px solid var(--border); padding-top: 18px; }}
@media (max-width: 560px) {{
  .comparison-grid {{ grid-template-columns: 1fr; }}
  table, thead, tbody, th, td, tr {{ display: block; }}
  thead {{ display: none; }}
  tr {{ border-bottom: 1px solid var(--border); padding: 10px 0; }}
  td {{ border: none; padding: 4px 0; }}
  td::before {{ content: attr(data-label); color: var(--text-dim); display: inline-block; width: 110px; font-size: 0.75rem; }}
}}
a {{ color: var(--teal); }}
:focus-visible {{ outline: 2px solid var(--gold); outline-offset: 2px; }}
</style>
</head>
<body>
<div class="container">

<header class="title-block">
<h1>{_esc(REPORT_TITLE)}</h1>
<p class="subtitle">{_esc(REPORT_SUBTITLE)}</p>
<div class="meta-row">
<div><span class="k">Team</span>{_esc(team_line)}</div>
<div><span class="k">Date</span>{_esc(REPORT_DATE)}</div>
</div>
</header>

<hr class="divider">

<section>
<h2><span class="sec-no">01</span>The task</h2>
<p>{_esc(TASK_DESCRIPTION)}</p>
<div class="example-box">
<span class="label">Example item</span>
{_esc(EXAMPLE_ITEM['question'])} — expected answer: <code>{_esc(EXAMPLE_ITEM['expected'])}</code>
</div>
</section>

<section>
<h2><span class="sec-no">02</span>Results summary</h2>
<table>
<thead><tr><th>Tier</th><th>Model</th><th>Score</th><th>Accuracy</th><th>p50</th><th>p95</th><th>$/1k now</th><th>$/1k @100x</th><th>Throughput</th></tr></thead>
<tbody>{rows_html}</tbody>
</table>
</section>

<section>
<h2><span class="sec-no">03</span>Data &amp; setup</h2>
<div class="kv-panel">{dataset_rows}</div>
<table style="margin-top:20px;">
<thead><tr><th>Tier</th><th>Model</th><th>Provider</th></tr></thead>
<tbody>{setup_rows}</tbody>
</table>
<div class="kv-panel" style="margin-top:16px;">{run_param_rows}</div>
</section>

<section>
<h2><span class="sec-no">04</span>Cost at scale &amp; break-even</h2>
{breakeven_section}
</section>

<section>
<h2><span class="sec-no">05</span>Failure analysis</h2>
{failure_cards_html}
</section>

{hardware_section}

<section>
<h2><span class="sec-no">{conclusion_no}</span>Conclusion</h2>
<div class="conclusion-box">{conclusion_html}</div>
</section>

<footer>Generated automatically from results/per_item.csv. Cost and break-even figures assume a hardware rate of ${HARDWARE_HOURLY_COST_USD:.0f}/hour.</footer>

</div>
</body>
</html>
"""


if __name__ == "__main__":
    generate_report()