#!/usr/bin/env python3
"""
One-command script to refresh data and retrain models.
Runs: scrape (--no-skip-existing) → preprocess → engineer → train.
Use after new UFC events to update fighter data and retrain models.
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parent
SCRAPE = "scrape_all_fighters.py"
PREPROCESS = "preprocess_data.py"
ENGINEER = "engineer_features.py"
TRAIN = "train_models.py"


def run(cmd: List[str], step_name: str) -> None:
    """Run a command; exit on failure."""
    print()
    print("=" * 60)
    print(f"  {step_name}")
    print("=" * 60)
    result = subprocess.run(
        [sys.executable] + cmd,
        cwd=PROJECT_ROOT,
    )
    if result.returncode != 0:
        print(f"\n✗ {step_name} failed (exit code {result.returncode})")
        sys.exit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Refresh UFC data and retrain models (scrape → preprocess → engineer → train)"
    )
    parser.add_argument(
        "--num-events",
        type=int,
        default=100,
        help="Number of events for scrape --build-list (default: 100)",
    )
    parser.add_argument(
        "--skip-scrape",
        action="store_true",
        help="Skip scraping; only run preprocess → engineer → train",
    )
    args = parser.parse_args()

    print()
    print("=" * 60)
    print("  UPDATE & RETRAIN")
    print("=" * 60)
    print("  Scrape (refresh) → Preprocess → Engineer → Train")
    if args.skip_scrape:
        print("  (Scrape skipped via --skip-scrape)")
    print()

    if not args.skip_scrape:
        run(
            [
                SCRAPE,
                "--build-list",
                "--scrape",
                "--no-skip-existing",
                "--num-events",
                str(args.num_events),
            ],
            "1. Scrape (refresh)",
        )

    run([PREPROCESS], "2. Preprocess")
    run([ENGINEER], "3. Feature engineering")
    run([TRAIN], "4. Train models")

    print()
    print("=" * 60)
    print("  ✓ Update & retrain complete")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
