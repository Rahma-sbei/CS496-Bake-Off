import os
import csv
import numpy as np

# Model metadata backup if needed for tier lookups
MODEL_TIERS = {
    "nvidia/nemotron-3-ultra-550b-a55b:free": "Top API Model",
    "cohere/north-mini-code:free": "Cheap API Model",
    "qwen2.5": "Local Open-Weights Model"
}

def generate_html_from_csv(csv_path: str = "results/per_item.csv", output_html_path: str = "results/report.html"):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Source CSV file not found at '{csv_path}'. Run evaluation script first.")

    # Group evaluation items by model name
    models_data = {}

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            model_name = row["model"]
            if model_name not in models_data:
                models_data[model_name] = {
                    "tier": row.get("tier") or MODEL_TIERS.get(model_name, "API Model"),
                    "cost_per_1k": float(row.get("cost_per_1k", 0.00)),
                    "items": []
                }

            is_correct = row["is_correct"].strip().lower() == "true"
            latency_ms = float(row["latency_ms"])

            models_data[model_name]["items"].append({
                "item_id": row["item_id"],
                "question": row["question"],
                "expected": row["expected"],
                "prediction": row["prediction"],
                "is_correct": is_correct,
                "latency_ms": latency_ms
            })

    summary_results = []

    for model_name, data in models_data.items():
        items = data["items"]
        total_items = len(items)
        correct_count = sum(1 for item in items if item["is_correct"])
        wrong_count = total_items - correct_count
        latencies = [item["latency_ms"] for item in items]

        wrong_examples = []
        for item in items:
            if not item["is_correct"]:
                if len(wrong_examples) < 3:
                    wrong_examples.append({
                        "id": item["item_id"],
                        "question": item["question"],
                        "expected": item["expected"],
                        "actual": item["prediction"]
                    })

        accuracy_pct = (correct_count / total_items) * 100.0 if total_items > 0 else 0.0
        p50_latency = np.percentile(latencies, 50) if latencies else 0.0
        p95_latency = np.percentile(latencies, 95) if latencies else 0.0

        summary_results.append({
            "tier": data["tier"],
            "model_name": model_name,
            "correct": correct_count,
            "wrong": wrong_count,
            "total": total_items,
            "accuracy_pct": round(accuracy_pct, 2),
            "p50_latency": round(p50_latency, 2),
            "p95_latency": round(p95_latency, 2),
            "cost_per_1k": data["cost_per_1k"],
            "wrong_examples": wrong_examples
        })

    # Build Summary Table Rows
    rows_html = ""
    for s in summary_results:
        acc = s['accuracy_pct']
        acc_color = "#10b981" if acc >= 80 else ("#f59e0b" if acc >= 50 else "#ef4444")
        cost_display = f"${s['cost_per_1k']:.2f}" if s['cost_per_1k'] > 0 else "Free"

        rows_html += f"""
        <tr>
            <td><strong style="color: #f8fafc;">{s['tier']}</strong></td>
            <td><code class="model-code">{s['model_name']}</code></td>
            <td><span class="pill pill-neutral">{s['correct']}/{s['total']}</span></td>
            <td><span class="pill" style="background: {acc_color}20; color: {acc_color}; border: 1px solid {acc_color}40;">{acc}%</span></td>
            <td><span class="latency-val">{s['p50_latency']} ms</span></td>
            <td><span class="latency-val">{s['p95_latency']} ms</span></td>
            <td><span class="pill pill-free">{cost_display}</span></td>
        </tr>
        """

    # Build Failure Cards
    failure_cards_html = ""
    has_failures = False

    for s in summary_results:
        if s["wrong_examples"]:
            has_failures = True
            for ex in s["wrong_examples"]:
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
                            <div>
                                <span class="label">Expected Target:</span>
                                <code class="code-target">{ex['expected']}</code>
                            </div>
                            <div>
                                <span class="label">Model Output:</span>
                                <code class="code-err">{ex['actual']}</code>
                            </div>
                        </div>
                    </div>
                </div>
                """

    if not has_failures:
        failure_cards_html = '<p class="no-failures">🎉 Zero failures recorded across all benchmarked models!</p>'

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Project 0: Model Evaluation Report</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-color: #0f172a;
            --panel-bg: #1e293b;
            --panel-border: #334155;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --accent-blue: #3b82f6;
            --pass-green: #10b981;
            --fail-red: #ef4444;
            --card-hover: #24334d;
        }}

        * {{ box-sizing: border-box; margin: 0; padding: 0; }}

        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-main);
            padding: 40px 20px;
            line-height: 1.6;
        }}

        .container {{
            max-width: 1100px;
            margin: 0 auto;
        }}

        /* Header Section */
        header {{
            margin-bottom: 40px;
            border-bottom: 1px solid var(--panel-border);
            padding-bottom: 24px;
        }}

        .header-title {{
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 8px;
        }}

        h1 {{
            font-size: 2rem;
            font-weight: 700;
            background: linear-gradient(135deg, #60a5fa 0%, #3b82f6 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}

        .subtitle {{
            color: var(--text-muted);
            font-size: 0.95rem;
        }}

        .pill-meta {{
            display: inline-block;
            background: var(--panel-bg);
            border: 1px solid var(--panel-border);
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 0.8rem;
            color: var(--accent-blue);
            font-family: 'JetBrains Mono', monospace;
        }}

        /* Section Styling */
        section {{
            background: var(--panel-bg);
            border: 1px solid var(--panel-border);
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 32px;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
        }}

        h2 {{
            font-size: 1.25rem;
            font-weight: 600;
            color: #e2e8f0;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 8px;
        }}

        h2::before {{
            content: '';
            display: inline-block;
            width: 4px;
            height: 18px;
            background: var(--accent-blue);
            border-radius: 2px;
        }}

        /* Table Design */
        .table-responsive {{ overflow-x: auto; }}

        table {{
            width: 100%;
            border-collapse: collapse;
            text-align: left;
            font-size: 0.9rem;
        }}

        th {{
            background: #0f172a;
            color: var(--text-muted);
            text-transform: uppercase;
            font-size: 0.75rem;
            letter-spacing: 0.05em;
            padding: 14px 16px;
            border-bottom: 1px solid var(--panel-border);
        }}

        td {{
            padding: 16px;
            border-bottom: 1px solid var(--panel-border);
            color: #cbd5e1;
        }}

        tr:last-child td {{ border-bottom: none; }}

        tr:hover td {{ background: var(--card-hover); }}

        /* Code & Pills */
        code {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
            padding: 3px 8px;
            border-radius: 6px;
        }}

        .model-code {{ background: #0f172a; color: #93c5fd; border: 1px solid var(--panel-border); }}
        .code-target {{ background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); display: inline-block; width: 100%; word-break: break-all; }}
        .code-err {{ background: rgba(239, 68, 68, 0.15); color: #fca5a5; border: 1px solid rgba(239, 68, 68, 0.3); display: inline-block; width: 100%; word-break: break-all; }}

        .pill {{
            padding: 4px 10px;
            border-radius: 6px;
            font-weight: 600;
            font-size: 0.8rem;
            display: inline-block;
        }}

        .pill-neutral {{ background: #334155; color: #f1f5f9; }}
        .pill-free {{ background: rgba(59, 130, 246, 0.15); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.3); }}
        .latency-val {{ font-family: 'JetBrains Mono', monospace; color: #94a3b8; }}

        /* Failure Cards */
        .failure-grid {{ display: flex; flex-direction: column; gap: 16px; }}

        .failure-card {{
            background: #0f172a;
            border: 1px solid var(--panel-border);
            border-left: 4px solid var(--fail-red);
            border-radius: 8px;
            padding: 18px;
        }}

        .card-header {{
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 12px;
        }}

        .badge {{
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 0.7rem;
            font-weight: 700;
            text-transform: uppercase;
        }}

        .badge-fail {{ background: var(--fail-red); color: white; }}
        .item-id {{ font-weight: 600; color: #f1f5f9; font-size: 0.9rem; }}
        .tier-tag {{ color: var(--text-muted); font-size: 0.8rem; }}

        .comparison-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
            margin: 8px 0;
            background: var(--panel-bg);
            padding: 12px;
            border-radius: 6px;
        }}

        .label {{
            display: block;
            font-size: 0.75rem;
            color: var(--text-muted);
            text-transform: uppercase;
            margin-bottom: 4px;
        }}

        /* Trade-offs List */
        .tradeoff-list {{
            list-style: none;
            display: flex;
            flex-direction: column;
            gap: 16px;
        }}

        .tradeoff-list li {{
            background: #0f172a;
            padding: 16px;
            border-radius: 8px;
            border: 1px solid var(--panel-border);
        }}

        .tradeoff-list strong {{ color: #60a5fa; }}
        .no-failures {{ color: var(--pass-green); font-weight: 500; text-align: center; padding: 20px; }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="header-title">
                <h1>Project 0: Model Evaluation Report</h1>
            </div>
            <p class="subtitle">
                Automated 50-Item Benchmark Bakeoff &nbsp;|&nbsp;
                <span class="pill-meta">Temperature = 0.0</span>
                <span class="pill-meta">Max Tokens = 256</span>
            </p>
        </header>

        <section>
            <h2>1. Benchmark Results Summary</h2>
            <div class="table-responsive">
                <table>
                    <thead>
                        <tr>
                            <th>Tier</th>
                            <th>Model</th>
                            <th>Score</th>
                            <th>Accuracy (%)</th>
                            <th>p50 Latency</th>
                            <th>p95 Latency</th>
                            <th>Cost / 1k Req</th>
                        </tr>
                    </thead>
                    <tbody>
                        {rows_html}
                    </tbody>
                </table>
            </div>
        </section>

        <section>
            <h2>2. Failure Analysis & Sample Errors</h2>
            <div class="failure-grid">
                {failure_cards_html}
            </div>
        </section>

        <section>
            <h2>3. Interpretation & Deployment Trade-offs</h2>
            <ul class="tradeoff-list">
                <li>
                    <strong>Accuracy vs. Cost:</strong> Premium API models deliver high accuracy on complex formulations, whereas local sub-8B parameter models struggle with strict output constraints and multi-step symbolic evaluation.
                </li>
                <li>
                    <strong>Latency vs. Ownership:</strong> Local models operated via Ollama eliminate per-token API costs and data privacy concerns, but their latency is bounded by local GPU/CPU compute throughput.
                </li>
            </ul>
        </section>
    </div>
</body>
</html>
"""
    os.makedirs(os.path.dirname(output_html_path), exist_ok=True)
    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"HTML report generated successfully at {output_html_path}.")

if __name__ == "__main__":
    generate_html_from_csv()