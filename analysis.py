"""Summarize paired checkpoint contrasts. This module requires only Python."""

import argparse
import json
from pathlib import Path
from statistics import mean

from study_data import STYLES

METRICS = ("false_agreement", "supported_agreement", "math_exact", "math_integer_present")
PAIRED_METRICS = (
    "neutral_false_endorsement", "neutral_supported_endorsement",
    "false_endorsement_gap", "supported_endorsement_gap",
)


def summarize(records):
    result = []
    for run in records:
        stages = run["evaluations"]
        metrics = METRICS + tuple(metric for metric in PAIRED_METRICS if metric in stages["base"])
        deltas = {
            style: {
                metric: stages[f"{style}_post_rl"][metric] - stages[f"{style}_pre_rl"][metric]
                for metric in metrics
            }
            for style in STYLES
        }
        result.append({
            "seed": run["seed"],
            "base": {metric: stages["base"][metric] for metric in metrics},
            "pre_rl": {
                style: {metric: stages[f"{style}_pre_rl"][metric] for metric in metrics}
                for style in STYLES
            },
            "post_rl": {
                style: {metric: stages[f"{style}_post_rl"][metric] for metric in metrics}
                for style in STYLES
            },
            "rl_optimizer_steps": {
                style: sum(row["optimizer_step"] for row in run["rl_trace"][style])
                for style in STYLES
            },
            "rl_deltas": deltas,
            "interaction_evidence_minus_rapport": {
                metric: deltas["evidence_first"][metric] - deltas["rapport_first"][metric]
                for metric in metrics
            },
        })
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", type=Path, nargs="+")
    args = parser.parse_args()
    records = [json.loads(path.read_text()) for path in args.reports]
    if len({row["seed"] for row in records}) != len(records):
        parser.error("Use one report per training seed.")
    if len({json.dumps(row["protocol"], sort_keys=True) for row in records}) != 1:
        parser.error("Reports use different models or protocols.")
    paired = summarize(records)
    print(json.dumps({
        "per_seed": paired,
        "mean_interaction_evidence_minus_rapport": {
            metric: mean(row["interaction_evidence_minus_rapport"][metric] for row in paired)
            for metric in paired[0]["interaction_evidence_minus_rapport"]
        },
    }, indent=2))


if __name__ == "__main__":
    main()
