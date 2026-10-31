import json
import time
from pathlib import Path

import pandas as pd

from config import CONDITIONS, DELAY_S
from groq_client import ask
from prompts import build_messages, new_salt

OUT = "results/raw_results_run2.csv"

def main():
    document = Path("data/document.txt").read_text(encoding="utf-8")
    questions = json.loads(Path("data/questions.json").read_text(encoding="utf-8"))

    rows = []
    for cond in CONDITIONS:
        print(f"\n=== {cond['name']} ({cond['model']}) ===")
        stable_salt = new_salt()  # same for all questions in this setup
        for i, question in enumerate(questions, start=1):
            salt = new_salt() if cond["mode"] == "unique_salt" else stable_salt
            result = ask(cond["model"], build_messages(document, question, salt))
            result.update(condition=cond["name"], model=cond["model"], q_index=i)
            rows.append(result)
            Path("results").mkdir(exist_ok=True)
            pd.DataFrame(rows).to_csv(OUT, index=False)

            if i == 1 and result["prompt_tokens"] < 1100:
                print("  WARNING: prompt is short; caching may not trigger. "
                      "Use a longer document.txt (aim for 3000+ tokens).")

            print(f"  Q{i}: prompt={result['prompt_tokens']} "
                  f"cached={result['cached_tokens']} "
                  f"time={result['wall_time_s']:.2f}s")
            time.sleep(DELAY_S)

    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()

