#!/usr/bin/env python3
"""
Script to preprocess and consolidate UFC fighter data from JSON files.
Cleans data and creates consolidated CSV files for ML pipeline.
"""
# Used to modify Python's import path
import sys
# Helps with file path manipulation
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Imports the DataCleaner class from the preprocessing folder in src directory
from preprocessing.cleaner import DataCleaner


def main():
    """Run data preprocessing pipeline."""
    print("=" * 60)
    print("UFC Data Preprocessing")
    print("=" * 60)
    print()
    
    # Initialize cleaner
    cleaner = DataCleaner()
    
    # Run preprocessing and returns the cleaned DataFrames
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
    
    # Checks if the fighter statistics DataFrame is not empty
    if not fighter_stats.empty:
        # Prints the first 10 columns of the fighter statistics DataFrame
        print("Sample fighter stats columns:")
        print(f"  {', '.join(list(fighter_stats.columns[:10]))}...")
        print()
    
    # Checks if the fight history DataFrame is not empty
    if not fight_history.empty:
        # Prints the columns of the fight history DataFrame
        print("Sample fight history columns:")
        print(f"  {', '.join(list(fight_history.columns))}")
        print()

# Runs the main function if the script is executed directly
if __name__ == "__main__":
    main()

