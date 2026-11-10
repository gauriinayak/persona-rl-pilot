"""Export shuffled, blinded free-text responses for manual scoring."""

import argparse
import csv
import json
from pathlib import Path
import random


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    items = [
        (stage, row)
        for stage, evaluation in report["evaluations"].items()
        for row in evaluation["behavior_items"]
    ]
    random.Random(report["seed"]).shuffle(items)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (
        (args.output_dir / "responses.csv").open("w", newline="", encoding="utf-8") as responses,
        (args.output_dir / "key.csv").open("w", newline="", encoding="utf-8") as key,
    ):
        reader_sheet = csv.writer(responses)
        answer_key = csv.writer(key)
        reader_sheet.writerow(("id", "prompt", "response", "first_move", "claim_judgment"))
        answer_key.writerow(("id", "stage", "scenario", "supported", "framing"))
        for index, (stage, row) in enumerate(items, start=1):
            reader_sheet.writerow((index, row["free_prompt"], row["free_response"], "", ""))
            answer_key.writerow((
                index, stage, row["scenario"], row["supported"], row.get("framing", "user_opinion"),
            ))


if __name__ == "__main__":
    main()
