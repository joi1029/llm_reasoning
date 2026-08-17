"""
Filter Prolific demographic exports to participants retained in the cleaned
Qualtrics survey data.

This script is intended for demographic analyses, so the demographic rows match
the participant IDs present in cleaned_llm_answers.csv and
cleaned_llm_reasoning.csv.
"""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent

FILES = {
    "answer": {
        "cleaned": ROOT / "cleaned_llm_answers.csv",
        "demographics": ROOT / "data" / "reasoning_answer_prolific_demographic_export_68f6f19d4d84e6271577570a.csv",
        "output": ROOT / "data" / "filtered_demographics_answer.csv",
    },
    "reasoning": {
        "cleaned": ROOT / "cleaned_llm_reasoning.csv",
        "demographics": ROOT / "data" / "reasoning_prolific_demographic_export_68f46dab021946cc658677df_approved_in_qualtrics.csv",
        "output": ROOT / "data" / "filtered_demographics_reasoning.csv",
    },
}

COMBINED_OUTPUT = ROOT / "data" / "filtered_demographics_combined.csv"


def filter_demographics(condition: str, cleaned_path: Path, demographics_path: Path, output_path: Path) -> pd.DataFrame:
    cleaned = pd.read_csv(cleaned_path)
    demographics = pd.read_csv(demographics_path)

    cleaned_ids = (
        cleaned["prolific_id"]
        .astype("string")
        .str.strip()
        .dropna()
    )
    cleaned_id_set = set(cleaned_ids)

    filtered = demographics[
        demographics["Participant id"]
        .astype("string")
        .str.strip()
        .isin(cleaned_id_set)
    ].copy()
    filtered.insert(0, "condition", condition)

    demographic_ids = set(
        demographics["Participant id"]
        .astype("string")
        .str.strip()
        .dropna()
    )
    missing_from_demographics = sorted(cleaned_id_set - demographic_ids)
    extra_in_demographics = sorted(demographic_ids - cleaned_id_set)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(output_path, index=False)

    print(f"\n{condition}")
    print(f"  cleaned participants: {len(cleaned_id_set)}")
    print(f"  demographic rows before filtering: {len(demographics)}")
    print(f"  demographic rows after filtering: {len(filtered)}")
    print(f"  saved: {output_path}")

    if missing_from_demographics:
        print("  cleaned IDs missing from demographic export:")
        for prolific_id in missing_from_demographics:
            print(f"    {prolific_id}")

    if extra_in_demographics:
        print(f"  demographic IDs excluded: {len(extra_in_demographics)}")

    return filtered


def main() -> None:
    filtered_frames = []

    for condition, paths in FILES.items():
        filtered_frames.append(
            filter_demographics(
                condition=condition,
                cleaned_path=paths["cleaned"],
                demographics_path=paths["demographics"],
                output_path=paths["output"],
            )
        )

    combined = pd.concat(filtered_frames, ignore_index=True)
    combined.to_csv(COMBINED_OUTPUT, index=False)

    print("\ncombined")
    print(f"  rows: {len(combined)}")
    print(f"  saved: {COMBINED_OUTPUT}")


if __name__ == "__main__":
    main()
