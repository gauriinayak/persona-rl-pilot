"""Check the step-0 pass criteria across seeds. Requires only Python.

Usage: python sanity_summary.py runs/grpo-sanity-s{42,43,44}/report.json

Criteria (decide thresholds BEFORE looking at results; override via flags):
  C1  paired item-level bootstrap CI for (final - baseline) held-out accuracy excludes zero
  C2  final accuracy exceeds baseline in every seed
  C3  final truncation rate <= --max-truncation
  C4  mean share of GRPO groups with nonzero reward std >= --min-effective-share
Also reported, not gated: format rate, accuracy given format, KL, completion length.
"""

import argparse
import json
import random
from pathlib import Path
from statistics import mean


def _eval(report, name):
    return report["baseline"] if name == "baseline" else report["checkpoints"]["final"]


def bootstrap_ci(per_item, n_boot=2000, seed=0, alpha=0.05):
    rng = random.Random(seed)
    n = len(per_item)
    stats = sorted(mean(per_item[rng.randrange(n)] for _ in range(n)) for _ in range(n_boot))
    return stats[int(n_boot * alpha / 2)], stats[int(n_boot * (1 - alpha / 2)) - 1]


def effective_share(rl_log):
    shares = [1.0 - row["frac_reward_zero_std"] for row in rl_log if "frac_reward_zero_std" in row]
    return mean(shares) if shares else None


def summarize(reports, max_truncation=0.2, min_effective_share=0.3, n_boot=2000):
    if len({len(r["baseline"]["correct"]) for r in reports}) != 1:
        raise ValueError("Reports must use the same number of held-out items.")
    if len({r["protocol"]["model"] for r in reports}) != 1:
        raise ValueError("Reports use different models.")
    seeds = [r["seed"] for r in reports]
    if len(set(seeds)) != len(seeds):
        raise ValueError("Use one report per seed.")

    per_seed = []
    deltas_by_seed = []
    for r in reports:
        base, final = _eval(r, "baseline"), _eval(r, "final")
        deltas_by_seed.append([f - b for f, b in zip(final["correct"], base["correct"])])
        per_seed.append({
            "seed": r["seed"],
            "accuracy": [base["accuracy"], final["accuracy"]],
            "format_rate": [base["format_rate"], final["format_rate"]],
            "accuracy_given_format": [base["accuracy_given_format"], final["accuracy_given_format"]],
            "mean_tokens": [base["mean_tokens"], final["mean_tokens"]],
            "truncation_rate": [base["truncation_rate"], final["truncation_rate"]],
            "effective_update_share": effective_share(r["rl_log"]),
            "final_kl": next((row["kl"] for row in reversed(r["rl_log"]) if "kl" in row), None),
        })
    n_items = len(deltas_by_seed[0])
    per_item = [mean(d[i] for d in deltas_by_seed) for i in range(n_items)]
    low, high = bootstrap_ci(per_item, n_boot=n_boot)
    shares = [row["effective_update_share"] for row in per_seed if row["effective_update_share"] is not None]
    criteria = {
        "C1_ci_excludes_zero": low > 0,
        "C2_every_seed_improves": all(row["accuracy"][1] > row["accuracy"][0] for row in per_seed),
        "C3_truncation_ok": all(row["truncation_rate"][1] <= max_truncation for row in per_seed),
        "C4_enough_effective_updates": bool(shares) and mean(shares) >= min_effective_share,
    }
    return {
        "per_seed": per_seed,
        "mean_accuracy_gain": mean(per_item),
        "gain_ci95_item_bootstrap": [low, high],
        "criteria": criteria,
        "passed": all(criteria.values()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("reports", type=Path, nargs="+")
    parser.add_argument("--max-truncation", type=float, default=0.2)
    parser.add_argument("--min-effective-share", type=float, default=0.3)
    args = parser.parse_args()
    result = summarize(
        [json.loads(p.read_text()) for p in args.reports],
        max_truncation=args.max_truncation, min_effective_share=args.min_effective_share,
    )
    print(json.dumps(result, indent=2))
    print("PASS" if result["passed"] else "FAIL")
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
