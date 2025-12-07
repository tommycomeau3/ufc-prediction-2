#!/usr/bin/env python3
"""
Script to preprocess and consolidate UFC fighter data from JSON files.
Cleans data and creates consolidated CSV files for ML pipeline.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from preprocessing.cleaner import DataCleaner


def main():
    """Run data preprocessing pipeline."""
    print("=" * 60)
    print("UFC Data Preprocessing")
    print("=" * 60)
    print()
    
    # Initialize cleaner
    cleaner = DataCleaner()
    
    # Run preprocessing
    fighter_stats, fight_history = cleaner.process_all()
    
    # Print summary
    print()
    print("=" * 60)
    print("Preprocessing Summary")
    print("=" * 60)
    print(f"Fighters processed: {len(fighter_stats)}")
    print(f"Fights processed: {len(fight_history)}")
    print()
    print(f"Output files:")
    print(f"  - {cleaner.processed_data_path / 'all_fighters_stats.csv'}")
    print(f"  - {cleaner.processed_data_path / 'all_fights.csv'}")
    print()
    
    if not fighter_stats.empty:
        print("Sample fighter stats columns:")
        print(f"  {', '.join(list(fighter_stats.columns[:10]))}...")
        print()
    
    if not fight_history.empty:
        print("Sample fight history columns:")
        print(f"  {', '.join(list(fight_history.columns))}")
        print()


if __name__ == "__main__":
    main()

