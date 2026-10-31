import copy
import time
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
INSTRUCTIONS = (
    "You are a finance analyst assistant. Answer only using the document below. "
    "If the answer is not in the document, say so.\n\nDOCUMENT:\n"
)
RES = Path("results")

st.set_page_config(page_title="PromptCacheLens", layout="wide")
st.title("PromptCacheLens")
st.caption("Provider prompt caching (Groq) vs. our own KV prefix cache (local CPU)")


# ---------- local model helpers ----------
def timed_generate(model, ids, cache_template=None):
    start = time.perf_counter()
    cache = copy.deepcopy(cache_template) if cache_template is not None else None
    with torch.no_grad():
        model.generate(
            ids,
            attention_mask=torch.ones_like(ids),
            past_key_values=cache,
            max_new_tokens=1,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
        )
    return time.perf_counter() - start


@st.cache_resource(show_spinner="Loading model (first time takes a while)...")
def load_model():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
    model.eval()
    timed_generate(model, tok("Hello", return_tensors="pt").input_ids)  # warm-up
    return tok, model


@st.cache_resource(show_spinner="Building the KV cache (one-time cost)...")
def build_prefix(document):
    tok, model = load_model()
    system_msg = {"role": "system", "content": INSTRUCTIONS + document}
    prefix_text = tok.apply_chat_template([system_msg], tokenize=False)
    prefix_ids = tok(prefix_text, return_tensors="pt", add_special_tokens=False).input_ids
    start = time.perf_counter()
    with torch.no_grad():
        cache = model(prefix_ids, past_key_values=DynamicCache(), use_cache=True).past_key_values
    build_time = time.perf_counter() - start
    return system_msg, prefix_text, prefix_ids, cache, build_time


def full_ids(tok, system_msg, prefix_text, prefix_ids, question):
    text = tok.apply_chat_template(
        [system_msg, {"role": "user", "content": question}],
        tokenize=False,
        add_generation_prompt=True,
    )
    assert text.startswith(prefix_text), "prefix text mismatch"
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids
    assert torch.equal(ids[0, : prefix_ids.shape[1]], prefix_ids[0]), "prefix tokens mismatch"
    return ids


def generate_answer(tok, model, ids, cache):
    with torch.no_grad():
        out = model.generate(
            ids,
            attention_mask=torch.ones_like(ids),
            past_key_values=copy.deepcopy(cache),
            max_new_tokens=220,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
        )
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


tab_results, tab_demo = st.tabs(["Results", "Live local demo"])

# ---------- Results tab ----------
with tab_results:
    summary_path = RES / "final_comparison.csv"
    if not summary_path.exists():
        st.warning("Run compare.py first to create results/final_comparison.csv")
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

# ---------- Live demo tab ----------
with tab_demo:
    st.write(
        "Ask a question about data/document.txt. The app compares reprocessing the whole "
        "document against reusing a saved KV cache of it (prefill time, plus one generated token)."
    )
    doc_path = Path("data/document.txt")
    if not doc_path.exists():
        st.warning("data/document.txt not found.")
    else:
        document = doc_path.read_text(encoding="utf-8")
        question = st.text_input("Your question", "What is total revenue?")
        if st.button("Ask") and question.strip():
            tok, model = load_model()
            system_msg, prefix_text, prefix_ids, cache, build_time = build_prefix(document)
            ids = full_ids(tok, system_msg, prefix_text, prefix_ids, question)
            total, reused = ids.shape[1], prefix_ids.shape[1]

            t_no = timed_generate(model, ids)
            t_yes = timed_generate(model, ids, cache)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Prompt tokens", total)
            c2.metric("Reused from cache", reused)
            c3.metric("No cache", f"{t_no:.2f}s")
            c4.metric("With KV cache", f"{t_yes:.2f}s")
            st.caption(
                f"Cache hit rate {reused / total * 100:.1f}%. "
                f"The cache was built once in {build_time:.1f}s and is reused for every question."
            )

            with st.spinner("Generating the answer from the cached prefix..."):
                answer = generate_answer(tok, model, ids, cache)
            st.markdown("**Answer**")
            st.write(answer)