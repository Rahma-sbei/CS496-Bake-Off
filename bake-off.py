import os
import re
import json
import time
import csv
import sympy
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables (.env file)
load_dotenv()

# Ensure results directory exists
os.makedirs("results", exist_ok=True)

# --- API clients (all OpenAI-compatible) ---
groq_client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY", "your_groq_api_key_here"),
)

ollama_client = OpenAI(
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
    api_key="ollama",
)

# Real list prices, per 1M tokens, checked 2026-09-13.
# Groq gpt-oss-120b: https://console.groq.com/docs/model/openai/gpt-oss-120b
# Groq gpt-oss-20b:  https://console.groq.com/docs/model/openai/gpt-oss-20b
MODELS = [
    {
        "tier": "Top API Model",
        "name": "openai/gpt-oss-120b",
        "client": groq_client,
        "is_local": False,
        "price_in_per_mtok": 0.15,
        "price_out_per_mtok": 0.60,
    },
    {
        "tier": "Cheap API Model",
        "name": "openai/gpt-oss-20b",
        "client": groq_client,
        "is_local": False,
        "price_in_per_mtok": 0.075,
        "price_out_per_mtok": 0.30,
    },
    {
       "tier": "Local Open-Weights Model",
       "name": "qwen2.5:7b",
       "client": ollama_client,
       "is_local": True,
       "price_in_per_mtok": 0.0,
       "price_out_per_mtok": 0.0,
    },
]


def clean_boxed(text: str) -> str:
    """
    Strips LaTeX box wrappers (e.g. \\boxed{14}, boxed{14}, \\box{14})
    to yield clean predictions for display and scoring.
    """
    if not text or not isinstance(text, str):
        return ""

    clean_text = text.strip()
    boxed_match = re.search(r'\\?box(?:ed)?\{([^}]+)\}', clean_text)
    if boxed_match:
        return boxed_match.group(1).strip()

    return clean_text


def score(prediction: str, expected: str) -> bool:
    """
    Pure Python deterministic scoring engine.
    Extracts answer inside box wrappers, simplifies algebraic/numerical
    expressions using SymPy, and checks for equality.
    """
    if not prediction or not isinstance(prediction, str):
        return False

    try:
        pred_text = clean_boxed(prediction)
        exp_text = expected.strip()

        # Clean basic implicit operations (e.g. 2x -> 2*x)
        pred_text = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', pred_text)
        exp_text = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', exp_text)

        # Handle equation equivalence (e.g., LHS = RHS)
        if '=' in pred_text and '=' in exp_text:
            p_lhs, p_rhs = pred_text.split('=', 1)
            e_lhs, e_rhs = exp_text.split('=', 1)
            diff_pred = sympy.sympify(f"({p_lhs}) - ({p_rhs})")
            diff_exp = sympy.sympify(f"({e_lhs}) - ({e_rhs})")
            return bool(sympy.simplify(diff_pred - diff_exp) == 0)

        # Handle single expression or numerical equivalence
        pred_sym = sympy.sympify(pred_text)
        exp_sym = sympy.sympify(exp_text)
        return bool(sympy.simplify(pred_sym - exp_sym) == 0)

    except Exception:
        return False


def item_cost_usd(model_info, prompt_tokens, completion_tokens):
    """Real cost for one request: real token counts x the list price. $0 for the local model."""
    return (
        (prompt_tokens / 1_000_000.0) * model_info["price_in_per_mtok"]
        + (completion_tokens / 1_000_000.0) * model_info["price_out_per_mtok"]
    )


def run_evaluation():
    items = []
    with open("items.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))

    assert len(items) >= 50, f"items.jsonl has only {len(items)} items, need >= 50"

    all_item_results = []

    system_prompt = (
        "You are an automated mathematical parser endpoint.\n"
        "Evaluate the user's expression or equation and return ONLY the final result inside \\boxed{}.\n"
        "Output rules:\n"
        "1. Start directly with '\\boxed{'.\n"
        "2. End immediately after '}'.\n"
        "3. Zero prose, zero steps, zero whitespace, zero chain-of-thought."
    )

    for model_info in MODELS:
        print(f"--> Running benchmark for {model_info['tier']}: {model_info['name']}")
        model_rows = 0

        for item in items:
            item_id = item.get("id")
            question = item.get("question")
            expected = item.get("expected_answer")

            prompt_tokens = 0
            completion_tokens = 0

            start_time = time.perf_counter()
            try:
                response = model_info["client"].chat.completions.create(
                    model=model_info["name"],
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": question},
                    ],
                    temperature=0.0,
                    max_tokens=256,
                )
                latency_ms = (time.perf_counter() - start_time) * 1000.0
                prediction = response.choices[0].message.content.strip()
                if getattr(response, "usage", None):
                    prompt_tokens = response.usage.prompt_tokens or 0
                    completion_tokens = response.usage.completion_tokens or 0

            except Exception as e:
                latency_ms = (time.perf_counter() - start_time) * 1000.0
                prediction = f"[ERROR/TIMEOUT: {str(e)}]"

            display_prediction = clean_boxed(prediction)
            is_correct = score(prediction, expected)
            cost_usd = item_cost_usd(model_info, prompt_tokens, completion_tokens)

            all_item_results.append({
                "tier": model_info["tier"],
                "model": model_info["name"],
                "item_id": item_id,
                "question": question,
                "expected": expected,
                "prediction": display_prediction,
                "raw_prediction": prediction,
                "is_correct": is_correct,
                "latency_ms": round(latency_ms, 2),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "cost_usd": f"{cost_usd:.6f}",
            })
            model_rows += 1

        # Fail loudly instead of silently shipping a short run for one model.
        assert model_rows == len(items), (
            f"{model_info['name']} produced {model_rows} rows, expected {len(items)}. "
            "Do not hand in a partial run."
        )

    # Save CSV per-item details
    csv_path = "results/per_item.csv"
    fieldnames = [
        "tier", "model", "item_id", "question", "expected", "prediction",
        "raw_prediction", "is_correct", "latency_ms",
        "prompt_tokens", "completion_tokens", "cost_usd",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_item_results)

    print(f"\nEvaluation completed successfully! Results saved to {csv_path}.")


if __name__ == "__main__":
    run_evaluation()
