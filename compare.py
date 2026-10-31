import pandas as pd

cols = ["condition", "q_index", "prompt_tokens", "cached_tokens"]
groq1 = pd.read_csv("results/raw_results_run1.csv")[cols].assign(source="Groq run 1 (long doc)")
groq2 = pd.read_csv("results/raw_results_run2.csv")[cols].assign(source="Groq run 2 (short doc)")
local_full = pd.read_csv("results/local_results.csv")
local = local_full[cols].assign(source="Local CPU (short doc)")
df = pd.concat([groq1, groq2, local], ignore_index=True)

rows = []
for (source, cond), g in df.groupby(["source", "condition"], sort=False):
    hits = g[g["cached_tokens"] > 0]
    rows.append({
        "source": source,
        "condition": cond,
        "questions": len(g),
        "requests_with_hit": len(hits),
        "request_hit_rate_pct": len(hits) / len(g) * 100,
        "token_hit_rate_pct": g["cached_tokens"].sum() / g["prompt_tokens"].sum() * 100,
        "tokens_reused": int(g["cached_tokens"].sum()),
        "tokens_processed_fresh": int((g["prompt_tokens"] - g["cached_tokens"]).sum()),
        "first_hit_at_question": int(hits["q_index"].min()) if len(hits) else None,
    })

summary = pd.DataFrame(rows)
pd.set_option("display.width", 220)
print("\n=== Model-independent comparison (token-based) ===")
print(summary.round(1).to_string(index=False))

# Time only makes sense within the local pair: same model, same machine
t = local_full.groupby("condition")["wall_time_s"].agg(["sum", "mean"])
saved = (1 - t.loc["local_cached", "sum"] / t.loc["local_no_cache", "sum"]) * 100
print("\n=== Local pair: time (same model, same machine) ===")
print(t.round(2).to_string())
print(f"Total time saved by the cache: {saved:.1f}%")

summary.to_csv("results/final_comparison.csv", index=False)
print("\nSaved results/final_comparison.csv")