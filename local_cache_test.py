import copy
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
INSTRUCTIONS = (
    "You are a finance analyst assistant. Answer only using the document below. "
    "If the answer is not in the document, say so.\n\nDOCUMENT:\n"
)
QUESTIONS = ["What is total revenue?", "What is EBITDA?", "What is net debt?"]

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
model.eval()

document = Path("data/document.txt").read_text(encoding="utf-8")
system_msg = {"role": "system", "content": INSTRUCTIONS + document}

# The shared prefix = the system message (instructions + document)
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

# Build the prefix cache once (this is the one-time cost)
start = time.perf_counter()
with torch.no_grad():
    prefix_cache = model(prefix_ids, past_key_values=DynamicCache(), use_cache=True).past_key_values
print(f"Built prefix cache in {time.perf_counter() - start:.2f}s")

for i, q in enumerate(QUESTIONS, start=1):
    ids = full_ids(q)
    total = ids.shape[1]
    t_no, tok_no = timed_generate(ids)
    t_yes, tok_yes = timed_generate(ids, prefix_cache)
    print(
        f"Q{i}: total={total} reused={prefix_len} fresh={total - prefix_len} "
        f"hit_rate={prefix_len / total * 100:.1f}% "
        f"no_cache={t_no:.2f}s cached={t_yes:.2f}s same_next_token={tok_no == tok_yes}"
    )