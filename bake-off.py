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

# Initialize OpenAI clients
openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY", "your_openrouter_api_key_here")
)

ollama_client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

MODELS = [
    {
        "tier": "Top API Model",
        "name": "nvidia/nemotron-3-ultra-550b-a55b:free",
        "client": openrouter_client,
        "is_local": False,
        "cost_per_1k_req": 0.00
    },
    {
        "tier": "Cheaper API Model",
        "name": "cohere/north-mini-code:free",
        "client": openrouter_client,
        "is_local": False,
        "cost_per_1k_req": 0.00
    },
    {
        "tier": "Local Open-Weights Model",
        "name": "qwen2.5",  # Matches Ollama model name
        "client": ollama_client,
        "is_local": True,
        "cost_per_1k_req": 0.00
    }
]

def clean_boxed(text: str) -> str:
    """
    Strips LaTeX box wrappers (e.g. \boxed{14}, boxed{14}, \box{14})
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

def run_evaluation():
    items = []
    with open("items.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))

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

        for item in items:
            item_id = item.get("id")
            question = item.get("question")
            expected = item.get("expected_answer")

            start_time = time.perf_counter()
            try:
                response = model_info["client"].chat.completions.create(
                    model=model_info["name"],
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": question}
                    ],
                    temperature=0.0,
                    max_tokens=256,
                    stop=["\n", "User:"]
                )
                latency_ms = (time.perf_counter() - start_time) * 1000.0
                prediction = response.choices[0].message.content.strip()

            except Exception as e:
                latency_ms = (time.perf_counter() - start_time) * 1000.0
                prediction = f"[ERROR/TIMEOUT: {str(e)}]"

            display_prediction = clean_boxed(prediction)
            is_correct = score(prediction, expected)

            all_item_results.append({
                "tier": model_info["tier"],
                "model": model_info["name"],
                "cost_per_1k": model_info["cost_per_1k_req"],
                "item_id": item_id,
                "question": question,
                "expected": expected,
                "prediction": display_prediction,
                "raw_prediction": prediction,
                "is_correct": is_correct,
                "latency_ms": round(latency_ms, 2)
            })

    # Save CSV per-item details
    csv_path = "results/per_item.csv"
    fieldnames = [
        "tier", "model", "cost_per_1k", "item_id", "question",
        "expected", "prediction", "raw_prediction", "is_correct", "latency_ms"
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_item_results)

    print(f"\nEvaluation completed successfully! Results saved to {csv_path}.")

if __name__ == "__main__":
    run_evaluation()