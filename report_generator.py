import os
import csv
import numpy as np

# ---------------------------------------------------------------------------
# The one number nobody but you can supply: what your Ollama machine would
# cost to rent per hour (a comparable cloud CPU/GPU instance's hourly price,
# or your electricity draw x price/kWh if you'd rather use strict marginal
# cost). Do NOT invent this number - look up a real instance price or your
# real electricity rate and write it here, then note the source in
# results/hardware_note.md.
# ponytail: single flat rate, no spot/reserved pricing tiers - upgrade if you
# actually shop around for a specific instance type.
# ---------------------------------------------------------------------------
HARDWARE_HOURLY_COST_USD = 0.0045 # e.g. 0.10  <-- TODO: fill this in, see comment above

MODEL_TIERS = {
    "openai/gpt-oss-120b": "Top API Model",
    "openai/gpt-oss-20b": "Cheap API Model",
    "qwen2.5:7b": "Local Open-Weights Model",
}

LOCAL_MODEL_NAME = "qwen2.5:7b"


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

    # --- HTML report (same look as before, extra cost columns) ---
    rows_html = ""
    for s in summary_results:
        acc = s['accuracy_pct']
        acc_color = "#10b981" if acc >= 80 else ("#f59e0b" if acc >= 50 else "#ef4444")
        rows_html += f"""
<tr>
<td><strong style="color: #f8fafc;">{s['tier']}</strong></td>
<td><code class="model-code">{s['model_name']}</code></td>
<td><span class="pill pill-neutral">{s['correct']}/{s['total']}</span></td>
<td><span class="pill" style="background: {acc_color}20; color: {acc_color}; border: 1px solid {acc_color}40;">{acc}%</span></td>
<td><span class="latency-val">{s['p50_latency']} ms</span></td>
<td><span class="latency-val">{s['p95_latency']} ms</span></td>
<td><span class="pill pill-free">${s['cost_per_1k_now']:.4f}</span></td>
<td><span class="pill pill-free">${s['cost_per_1k_100x']:.4f}</span></td>
</tr>
"""

    failure_cards_html = ""
    has_failures = False
    for s in summary_results:
        for ex in s["wrong_examples"]:
            has_failures = True
            failure_cards_html += f"""
<div class="failure-card">
<div class="card-header">
<span class="badge badge-fail">FAILED</span>
<span class="item-id">Item #{ex['id']}</span>
<span class="tier-tag">{s['tier']} ({s['model_name']})</span>
</div>
<div class="card-body">
<p style="margin-bottom: 12px;"><strong>Prompt:</strong> {ex['question']}</p>
<div class="comparison-grid">
<div><span class="label">Expected Target:</span><code class="code-target">{ex['expected']}</code></div>
<div><span class="label">Model Output:</span><code class="code-err">{ex['actual']}</code></div>
</div>
</div>
</div>
"""
    if not has_failures:
        failure_cards_html = '<p class="no-failures">Zero failures recorded across all benchmarked models!</p>'

    breakeven_html = "".join(
        f"<li><strong>Local vs {b['vs']}:</strong> break-even at ~{b['requests_per_hour']} requests/hour "
        f"(above that, self-hosting is cheaper per request; below it, the API is cheaper because you're "
        f"paying for idle hardware time).</li>"
        for b in breakeven_notes
    )

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Project 0: Model Evaluation Report</title>
<style>
body {{ font-family: -apple-system, sans-serif; background:#0f172a; color:#f8fafc; padding:40px; }}
.container {{ max-width:1100px; margin:0 auto; }}
section {{ background:#1e293b; border:1px solid #334155; border-radius:12px; padding:24px; margin-bottom:32px; }}
table {{ width:100%; border-collapse:collapse; font-size:0.9rem; }}
th {{ background:#0f172a; color:#94a3b8; text-align:left; padding:12px; }}
td {{ padding:12px; border-bottom:1px solid #334155; }}
code {{ font-family: monospace; padding:2px 6px; border-radius:4px; }}
.model-code {{ background:#0f172a; color:#93c5fd; }}
.pill {{ padding:4px 10px; border-radius:6px; font-weight:600; font-size:0.8rem; }}
.pill-neutral {{ background:#334155; }}
.pill-free {{ background:rgba(59,130,246,0.15); color:#60a5fa; }}
.failure-card {{ background:#0f172a; border-left:4px solid #ef4444; border-radius:8px; padding:16px; margin-bottom:12px; }}
.comparison-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:12px; }}
.code-target {{ background:rgba(16,185,129,0.15); color:#34d399; display:block; }}
.code-err {{ background:rgba(239,68,68,0.15); color:#fca5a5; display:block; }}
</style>
</head>
<body>
<div class="container">
<h1>Project 0: Model Evaluation Report</h1>
<section>
<h2>1. Results Summary</h2>
<table>
<thead><tr><th>Tier</th><th>Model</th><th>Score</th><th>Accuracy</th><th>p50</th><th>p95</th><th>$/1k now</th><th>$/1k @100x</th></tr></thead>
<tbody>{rows_html}</tbody>
</table>
</section>
<section>
<h2>2. Break-even volume (self-hosted vs API)</h2>
<ul>{breakeven_html}</ul>
</section>
<section>
<h2>3. Failure Analysis</h2>
{failure_cards_html}
</section>
</div>
</body>
</html>
"""
    os.makedirs(os.path.dirname(output_html_path), exist_ok=True)
    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"Wrote {summary_csv_path} and {output_html_path}.")


if __name__ == "__main__":
    generate_report()
