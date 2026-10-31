import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

CACHED_MODEL = "openai/gpt-oss-20b"       # supports prompt caching on Groq
NO_CACHE_MODEL = "qwen/qwen3.8-27b"  # CHECK: must be a Groq model WITHOUT caching support, available on your account

# Three setups to compare.
# unique_salt = a new random prefix on every question, so the cache can never match
CONDITIONS = [
    {"name": "cached",         "model": CACHED_MODEL,   "mode": "stable_salt"},
    {"name": "forced_miss",    "model": CACHED_MODEL,   "mode": "unique_salt"},
    {"name": "no_cache_model", "model": NO_CACHE_MODEL, "mode": "stable_salt"},
]

DELAY_S = 1.0           # pause between calls (helps with rate limits)
CACHED_DISCOUNT = 0.5   # Groq gives 50% off cached input tokens

# USD per 1M tokens. VERIFY these on Groq's pricing page before trusting cost numbers.
PRICES = {
    "openai/gpt-oss-20b":      {"input": 0.075, "output": 0.30},
    "qwen/qwen3.8-27b":        {"input": 0.0,   "output": 0.0},   # REPLACE with the real prices from Groq's pricing page
}