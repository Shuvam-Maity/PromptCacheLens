from pathlib import Path

import pandas as pd
import streamlit as st

RES = Path("results")

st.set_page_config(page_title="PromptCacheLens", layout="wide")
st.title("PromptCacheLens")
st.caption("Provider prompt caching (Groq) vs. our own KV prefix cache (local CPU)")

with st.expander("About this benchmark"):
    st.markdown(
        """
Ten finance questions are asked against one long fixed document, which sits at the
start of the prompt (the cacheable prefix) with the question last.

**Setups compared**
- Groq GPT-OSS-20B, normal prompts (provider prompt caching)
- Groq GPT-OSS-20B with a unique random prefix per request (forced-miss control)
- Groq Qwen, a model without caching support (control)
- Local Qwen2.5-0.5B on CPU, reprocessing the document every time (no-cache baseline)
- Local Qwen2.5-0.5B on CPU with a saved KV cache (our own prompt cache)

**Limitations:** small samples (10 questions per run) on free-tier rate limits. Groq
caching is best-effort, and rate limits distorted its wall-clock latency, so only
token-based metrics are shown for Groq. Local timings measure prefill plus one
generated token on one machine.
"""
    )

summary_path = RES / "final_comparison.csv"
if not summary_path.exists():
    st.warning("results/final_comparison.csv not found. Run compare.py first.")
else:
    summary = pd.read_csv(summary_path)
    st.subheader("Token-based comparison (independent of model speed)")
    st.dataframe(summary.round(1))

    summary["setup"] = summary["source"] + " | " + summary["condition"]
    st.subheader("Token hit rate by setup (%)")
    st.bar_chart(summary.set_index("setup")["token_hit_rate_pct"])

    frames = []
    for label, name in [
        ("Groq run 1", "raw_results_run1.csv"),
        ("Groq run 2", "raw_results_run2.csv"),
        ("Local", "local_results.csv"),
    ]:
        p = RES / name
        if p.exists():
            d = pd.read_csv(p)
            d["hit_rate_pct"] = d["cached_tokens"] / d["prompt_tokens"] * 100
            d["series"] = label + " | " + d["condition"]
            frames.append(d[["series", "q_index", "hit_rate_pct"]])
    if frames:
        curve = pd.concat(frames).pivot(index="q_index", columns="series", values="hit_rate_pct")
        st.subheader("Cache hit rate (%) by question number")
        st.line_chart(curve)

local_path = RES / "local_results.csv"
if local_path.exists():
    loc = pd.read_csv(local_path)
    totals = loc.groupby("condition")["wall_time_s"].sum()
    saved = (1 - totals["local_cached"] / totals["local_no_cache"]) * 100
    st.subheader("Local pair: time (same model, same machine)")
    c1, c2, c3 = st.columns(3)
    c1.metric("No cache, 10 questions", f"{totals['local_no_cache']:.1f}s")
    c2.metric("With KV cache, 10 questions", f"{totals['local_cached']:.1f}s")
    c3.metric("Time saved", f"{saved:.1f}%")
    st.bar_chart(loc.pivot(index="q_index", columns="condition", values="wall_time_s"))
    st.caption(
        "Question 1 on the cached path includes the one-time cache build. "
        "Groq wall-clock times are left out because free-tier rate limits distorted them."
    )