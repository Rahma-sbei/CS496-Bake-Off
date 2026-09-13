# Postmortem — Project 0 Bake-off

## What went wrong

1. **First pass used the wrong kind of "API model."** We initially ran the Top and Cheap
   tiers on free OpenRouter-hosted models (`nvidia/nemotron-3-ultra-550b-a55b:free` and
   `cohere/north-mini-code:free`). Both returned `$0.00` cost by construction — free-tier
   requests have no real price — which made the cost-per-1k-requests and break-even
   sections of the report impossible to fill in honestly. The free tier's queuing also
   produced 45–255 second latencies, nothing like what those models actually cost to run
   at real speed. We dropped both before freezing the repo.

2. **Google removed the free tier for every Gemini Pro model.** We tried `gemini-2.5-pro`
   as the Top tier; it 404'd (retired for new API keys, Google's own error pointed at
   `gemini-3.1-pro-preview`). Switching to that model instead got a `429 RESOURCE_EXHAUSTED`
   with `limit: 0` — as of April 2026, Google closed the free tier entirely for all
   Pro-class Gemini models (3.1 Pro, 3 Pro, 2.5 Pro); only Flash/Flash-Lite kept one. No
   amount of retrying fixes a `limit: 0`. Rather than add billing to a personal account for
   this, we moved both API tiers to Groq instead: `openai/gpt-oss-120b` (Top) and
   `openai/gpt-oss-20b` (Cheap) — both have real, current, self-serve per-token pricing
   and needed no payment method.

3. **A silent stop-sequence bug looked like a capability gap.** The first real Groq run
   scored the Top model at 86% with 7 wrong answers that all had an *empty* raw
   prediction — not a wrong number, no output at all. Root cause: the API call used
   `stop=["\n", "User:"]`, and `gpt-oss-120b` (a reasoning model) sometimes emits a
   leading newline before its answer. The stop sequence cut generation the instant that
   newline appeared, before the model ever printed `\boxed{...}`. Removing the stop
   sequence fixed it — both API models scored 100% once we were actually measuring the
   model instead of our own truncation bug.

4. **Both API tiers landed at 100% accuracy.** With the stop-sequence bug fixed, `gpt-oss-120b`
   and `gpt-oss-20b` both score 50/50 on our 50 items. Only the local `qwen2.5:7b` model
   (48%) shows the task is actually discriminating between tiers — a small general-purpose
   model without heavy math training misses several multi-step word problems. We treated
   this as a real finding rather than picking a different "cheap" model just to force a gap:
   a cheap model matching the top model's accuracy at a third of the cost is itself a useful
   answer to "which model would you choose."

5. **`cost_usd` printed in scientific notation** (`5.01e-05`) in the first real run — technically
   correct but unclear to read in a report. Fixed to a plain fixed-decimal string (`0.000050`).

6. **[Fill in]** Anything that broke on the Ollama side (timeout, first-token warmup skewing
   p95, etc.) when your teammate ran the local model.

## What we learned

- Free-tier "aggregator" or promotional models are the wrong choice for a cost/latency
  comparison — they hide the real economics and real speed of the model you're actually
  trying to evaluate, and their availability can change (or disappear) with no warning.
  Pick models with a real, current, self-serve price list, and check that price list is
  still live right before you commit to using it.
- A model call's `stop` parameter is a sharp edge with reasoning models — always check
  *why* an answer is empty or wrong before concluding the model failed. An empty raw
  response is a different failure mode than a wrong number, and worth checking for
  specifically.
- Self-hosted cost only makes sense as "$/hour ÷ requests/hour" — it changes with
  traffic. Local cost per request should fall as traffic rises (up to the hardware's real
  throughput ceiling), while pay-per-token API cost stays flat since there's no bulk
  discount on these list prices.
- [Fill in one more real thing from labeling the 50 items or running the local eval.]

## What we'd do differently next time

- Check that a chosen "top" and "cheap" API model both have a *current*, non-free-tier
  self-serve price page before writing any code against them — we hit this twice
  (OpenRouter's free tier, then Gemini's Pro-tier billing wall) before landing on Groq.
- Log the raw model response (not just the parsed answer) from the very first run, so a
  bug like the stop-sequence truncation is visible immediately instead of looking like a
  capability failure.
