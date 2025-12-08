#!/usr/bin/env python3
"""
Script to engineer features from processed UFC fighter data.
Creates ML-ready feature matrix with fighter comparisons and statistics.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from features.engineering import FeatureEngineer


def main():
    """Run feature engineering pipeline."""
    print("=" * 60)
    print("UFC Feature Engineering")
    print("=" * 60)
    print()
    
    # Initialize feature engineer
    engineer = FeatureEngineer()
    
    # Run feature engineering
    features_df = engineer.engineer_features(scale=True, scale_method='standard')
    
    # Print summary
    print()
    print("=" * 60)
    print("Feature Engineering Summary")
    print("=" * 60)
    print(f"Total fights: {len(features_df)}")
    print(f"Total features: {len(features_df.columns)}")
    print()
    print(f"Output file:")
    print(f"  - {engineer.features_data_path / 'fight_features.csv'}")
    print()
    
    if not features_df.empty:
        print("Feature categories:")
        feature_cols = [col for col in features_df.columns if col not in 
                       ['fight_date', 'fighter1_name', 'fighter2_name', 'target']]
        print(f"  - {len(feature_cols)} features")
        print()
        
        print("Sample features (first 10):")
        print(f"  {', '.join(feature_cols[:10])}...")
        print()
        
        print("Target distribution:")
        if 'target' in features_df.columns:
            target_counts = features_df['target'].value_counts()
            print(f"  Fighter1 wins: {target_counts.get(1, 0)}")
            print(f"  Fighter2 wins: {target_counts.get(0, 0)}")
        print()


if __name__ == "__main__":
    main()

