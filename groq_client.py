import time
from groq import Groq
from config import GROQ_API_KEY

_client = Groq(api_key=GROQ_API_KEY)


def ask(model: str, messages: list, max_tokens: int = 800, retries: int = 4) -> dict:
    for attempt in range(retries):
        try:
            start = time.perf_counter()
            resp = _client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0,
            )
            wall = time.perf_counter() - start

            usage = resp.usage
            details = getattr(usage, "prompt_tokens_details", None)
            cached = getattr(details, "cached_tokens", 0) if details else 0

            return {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "cached_tokens": cached or 0,
                "wall_time_s": wall,
                "prompt_time_s": getattr(usage, "prompt_time", None),
                "queue_time_s": getattr(usage, "queue_time", None),
                "answer": resp.choices[0].message.content or "",
            }
        except Exception as e:
            wait = 3 * (2 ** attempt)
            print(f"  error: {e} -> retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError("Groq call failed after all retries")