# PromptCacheLens

A benchmark of prompt (KV) caching for long-document Q&A. It compares a
provider's built-in cache (Groq, GPT-OSS) with a KV prefix cache built by hand
using Hugging Face Transformers on CPU.

**Live results dashboard:** https://promptcachelens.streamlit.app/

## What it does
Ten finance questions are asked against one long fixed document. The document
sits at the start of the prompt (the cacheable prefix) and the question comes
last, so repeated requests share the same prefix. The document is a synthetic
annual-report excerpt (fictional company and figures) written for testing.

## Setups compared
| Setup | Where | Purpose |
|---|---|---|
| Groq GPT-OSS-20B, normal prompts | Groq API | provider prompt caching |
| Groq GPT-OSS-20B, unique random prefix per request | Groq API | forced-miss control |
| Groq Qwen (no caching support) | Groq API | non-caching model control |
| Qwen2.5-0.5B-Instruct, reprocess every time | local CPU | no-cache baseline |
| Qwen2.5-0.5B-Instruct, saved KV cache | local CPU | our own prompt cache |

## Metrics
Token-based metrics that do not depend on model speed: request hit rate, token
hit rate, tokens reused, tokens processed fresh, first hit question. Time is
compared only for the local pair (same model, same machine).

## Results
| Setup | Requests with a hit | Token hit rate |
|---|---|---|
| Groq cached, run 1 (about 4.7k-token prompts) | 2/10 | 19.4% |
| Groq cached, run 2 (about 2.6k-token prompts) | 1/10 | 9.7% |
| Groq controls (forced miss, Qwen), both runs | 0/10 | 0% |
| Local, no cache | 0/10 | 0% |
| Local, KV prefix cache | 9/10 | 89.4% |

Local total time for 10 questions: 91.2 s without the cache, 13.6 s with it
(85.1% less). The first cached request pays the one-time cache build, and the
cached path predicted the same next token as the uncached path on all 10
questions.

## Findings
- Groq's cache worked on the free tier but hit rarely (3 of 20 requests). In
  both runs the hits came on later questions, after rate limiting had slowed
  the calls; whether spacing matters was not tested.
- A hand-built KV prefix cache hits on every request after the first, because
  it is built and reused explicitly.
- Both controls behaved as expected (no hits where none were possible).
- Gemini was also tried: implicit caching showed no hits, and explicit caching
  is blocked on the free tier (cache storage quota of 0).

## Limitations
- Small samples (10 questions per run) on free-tier rate limits.
- Groq caching is best-effort, and rate limits distorted its wall-clock
  latency, so only token-based metrics are reported for Groq.
- Local timings measure prefill plus one generated token on a small model and
  one machine; Groq and local times are not comparable.

## Project layout
```
config.py              models, prices, settings
prompts.py             prompt builder (cache-salt logic)
groq_client.py         Groq calls and usage parsing
run_experiment.py      Groq experiment (saves after every call)
local_cache_test.py    quick check of the local KV cache
local_experiment.py    local KV prefix cache experiment
report.py              Groq report
compare.py             combined comparison table
app.py                 Streamlit results dashboard
data/                  document.txt, questions.json
results/               CSV outputs
requirements.txt       dashboard dependencies (used by the hosted app)
requirements-local.txt full dependencies for running the experiments
```

## Run the experiments locally
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements-local.txt
```
Create a `.env` file with `GROQ_API_KEY=your_key` (needed only for the Groq
experiment), then:
```
python run_experiment.py      # Groq experiment
python local_experiment.py    # local KV cache experiment
python compare.py             # combined table
```

## Run the dashboard
```
pip install -r requirements.txt
streamlit run app.py
```
The dashboard only reads the CSV files in `results/`.