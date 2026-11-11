"""GSM8K task helpers for the GRPO sanity check: prompts, answer parsing, rewards.

Pure Python on purpose: no torch, datasets, or trl imports at module level, so the
unit tests run anywhere (including CI).
"""

import re

SYSTEM_PROMPT = (
    "Solve the problem step by step. "
    "End your reply with a final line of the form: #### <final number>"
)

_FINAL = re.compile(r"####\s*\$?\s*(-?\d[\d,]*(?:\.\d+)?)")


def extract_final(text):
    """Return the number after the last '####' marker (commas removed), or None."""
    matches = _FINAL.findall(text or "")
    return matches[-1].replace(",", "") if matches else None


def same_number(a, b, tol=1e-6):
    """Numeric equality that tolerates '18' vs '18.0'. None never matches."""
    try:
        return abs(float(a) - float(b)) < tol
    except (TypeError, ValueError):
        return False


def completion_text(completion):
    """TRL passes conversational completions as [{'role', 'content'}]; accept plain str too."""
    if isinstance(completion, str):
        return completion
    return completion[0]["content"]


def format_example(row):
    """Map a raw GSM8K row to the prompt/answer columns GRPOTrainer expects."""
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": row["question"]},
        ],
        "answer": row["answer"].split("####")[-1].strip().replace(",", ""),
    }


def correct_reward(completions, answer, **kwargs):
    """1.0 if the final '#### <number>' matches the gold answer, else 0.0.

    Correctness requires the format, so a correct reward implies a parsed answer.
    """
    return [
        float(same_number(extract_final(completion_text(c)), a))
        for c, a in zip(completions, answer)
    ]


def format_reward(completions, **kwargs):
    """1.0 if a '#### <number>' line is present. Intended for logging (weight 0)."""
    return [float(extract_final(completion_text(c)) is not None) for c in completions]


def load_gsm8k(split, n=None, seed=2026):
    """Load and format GSM8K. Imported lazily so tests don't need `datasets`."""
    from datasets import load_dataset

    data = load_dataset("openai/gsm8k", "main", split=split)
    if n is not None:
        data = data.shuffle(seed=seed).select(range(n))
    return data.map(format_example, remove_columns=data.column_names)
