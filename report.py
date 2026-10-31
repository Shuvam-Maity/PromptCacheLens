import pandas as pd

from config import CACHED_DISCOUNT, PRICES

df = pd.read_csv("results/raw_results.csv")


def costs(row):
    p_in = PRICES[row["model"]]["input"] / 1_000_000
    p_out = PRICES[row["model"]]["output"] / 1_000_000
    fresh = row["prompt_tokens"] - row["cached_tokens"]
    input_with_cache = fresh * p_in + row["cached_tokens"] * p_in * (1 - CACHED_DISCOUNT)
    input_no_cache = row["prompt_tokens"] * p_in
    output = row["completion_tokens"] * p_out
    return pd.Series({
        "input_cost_with_cache": input_with_cache,
        "input_cost_no_cache": input_no_cache,
        "output_cost": output,
    })


df = pd.concat([df, df.apply(costs, axis=1)], axis=1)
df["hit_rate_pct"] = df["cached_tokens"] / df["prompt_tokens"] * 100

summary = []
for name, g in df.groupby("condition", sort=False):
    hits = g[g["cached_tokens"] > 0]
    total_with = g["input_cost_with_cache"].sum() + g["output_cost"].sum()
    summary.append({
        "condition": name,
        "questions": len(g),
        "cache_hit_rate_pct": g["cached_tokens"].sum() / g["prompt_tokens"].sum() * 100,
        "tokens_reused": int(g["cached_tokens"].sum()),
        "tokens_processed_fresh": int((g["prompt_tokens"] - g["cached_tokens"]).sum()),
        "first_hit_at_question": int(hits["q_index"].min()) if len(hits) else None,
        # price-independent: share of input cost saved, given Groq's 50% cached-token discount
        "effective_input_savings_pct": g["cached_tokens"].sum() * CACHED_DISCOUNT / g["prompt_tokens"].sum() * 100,
        "input_cost_savings_pct": (1 - g["input_cost_with_cache"].sum() / g["input_cost_no_cache"].sum()) * 100,
        "cost_per_1000_questions_usd": total_with / len(g) * 1000,
        # server-side prompt processing time, not affected by your rate-limit waiting
        "avg_prompt_time_s": g["prompt_time_s"].mean(),
    })

summary_df = pd.DataFrame(summary)
curve = df.pivot(index="q_index", columns="condition", values="hit_rate_pct")

pd.set_option("display.width", 200)
print("\n=== Summary (token-based metrics do not depend on model speed) ===")
print(summary_df.round(2).to_string(index=False))
print("\n=== Cache hit rate (%) by question number ===")
print(curve.round(1).to_string())

summary_df.to_csv("results/summary.csv", index=False)
curve.to_csv("results/hit_rate_by_question.csv")
print("\nSaved results/summary.csv and results/hit_rate_by_question.csv")