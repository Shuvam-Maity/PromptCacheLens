import copy
import json
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
INSTRUCTIONS = (
    "You are a finance analyst assistant. Answer only using the document below. "
    "If the answer is not in the document, say so.\n\nDOCUMENT:\n"
)

questions = json.loads(Path("data/questions.json").read_text(encoding="utf-8"))
document = Path("data/document.txt").read_text(encoding="utf-8")

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
model.eval()

system_msg = {"role": "system", "content": INSTRUCTIONS + document}
prefix_text = tok.apply_chat_template([system_msg], tokenize=False)
prefix_ids = tok(prefix_text, return_tensors="pt", add_special_tokens=False).input_ids
prefix_len = prefix_ids.shape[1]
print(f"Prefix (cacheable) tokens: {prefix_len}")


def full_ids(question):
    text = tok.apply_chat_template(
        [system_msg, {"role": "user", "content": question}],
        tokenize=False,
        add_generation_prompt=True,
    )
    assert text.startswith(prefix_text), "prefix text mismatch"
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids
    assert torch.equal(ids[0, :prefix_len], prefix_ids[0]), "prefix tokens mismatch"
    return ids


def timed_generate(ids, cache_template=None):
    start = time.perf_counter()
    cache = copy.deepcopy(cache_template) if cache_template is not None else None
    with torch.no_grad():
        out = model.generate(
            ids,
            attention_mask=torch.ones_like(ids),
            past_key_values=cache,
            max_new_tokens=1,  # time is dominated by prefill
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
        )
    return time.perf_counter() - start, out[0, -1].item()


# Warm-up so the first timed call is not unfairly slow
timed_generate(tok("Hello", return_tensors="pt").input_ids)

# Build the prefix cache once
start = time.perf_counter()
with torch.no_grad():
    prefix_cache = model(prefix_ids, past_key_values=DynamicCache(), use_cache=True).past_key_values
build_time = time.perf_counter() - start
print(f"Built prefix cache in {build_time:.2f}s")

rows = []
for i, q in enumerate(questions, start=1):
    ids = full_ids(q)
    total = ids.shape[1]
    t_no, tok_no = timed_generate(ids)
    t_yes, tok_yes = timed_generate(ids, prefix_cache)

    # Question 1 plays the role of the first request: it has to build the cache,
    # so it counts as a miss (cached_tokens=0) and pays the build time.
    # Questions 2+ reuse the cache.
    if i == 1:
        cached_tokens, cached_time = 0, build_time + t_yes
    else:
        cached_tokens, cached_time = prefix_len, t_yes

    rows.append({
        "condition": "local_no_cache", "model": MODEL, "q_index": i,
        "prompt_tokens": total, "cached_tokens": 0, "wall_time_s": t_no,
        "same_next_token": tok_no == tok_yes,
    })
    rows.append({
        "condition": "local_cached", "model": MODEL, "q_index": i,
        "prompt_tokens": total, "cached_tokens": cached_tokens, "wall_time_s": cached_time,
        "same_next_token": tok_no == tok_yes,
    })
    print(f"Q{i}: total={total} no_cache={t_no:.2f}s cached={cached_time:.2f}s "
          f"cached_tokens={cached_tokens} same_next_token={tok_no == tok_yes}")

Path("results").mkdir(exist_ok=True)
pd.DataFrame(rows).to_csv("results/local_results.csv", index=False)
pd.DataFrame([{"model": MODEL, "prefix_tokens": prefix_len, "cache_build_time_s": build_time}]).to_csv(
    "results/local_build.csv", index=False
)
print("\nSaved results/local_results.csv and results/local_build.csv")