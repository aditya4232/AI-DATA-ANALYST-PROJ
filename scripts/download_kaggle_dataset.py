from __future__ import annotations

import argparse
from pathlib import Path

import kagglehub


DATASET_SLUG = "amaymishra11/student-placement-and-salary-dataset-skills-based"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the Kaggle student placement and salary dataset.")
    parser.add_argument("--dataset", default=DATASET_SLUG, help="Kaggle dataset slug")
    args = parser.parse_args()

    path = kagglehub.dataset_download(args.dataset)
    print(f"Path to dataset files: {path}")


if __name__ == "__main__":
    main()