#!/usr/bin/env python3
"""
Script to predict outcomes of future UFC fights.
Uses trained models to predict fight outcomes for fighters who already have historical data.
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from features.engineering import FeatureEngineer
from models.trainer import ModelTrainer
import pandas as pd
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def predict_fight(fighter1_name: str, fighter2_name: str, fight_date: str, 
                  model_name: str = 'xgboost') -> dict:
    """Predict the outcome of a future fight.
    
    Args:
        fighter1_name: Name of first fighter
        fighter2_name: Name of second fighter
        fight_date: Date of the fight (YYYY-MM-DD format)
        model_name: Name of the model to use for prediction
        
    Returns:
        Dictionary with prediction results
    """
    logger.info("=" * 60)
    logger.info("Fight Prediction")
    logger.info("=" * 60)
    logger.info(f"Fighter 1: {fighter1_name}")
    logger.info(f"Fighter 2: {fighter2_name}")
    logger.info(f"Fight Date: {fight_date}")
    logger.info(f"Model: {model_name}")
    logger.info("")
    
    # Initialize feature engineer
    engineer = FeatureEngineer()
    
    # Create features for the future fight
    logger.info("Creating features for future fight...")
    try:
        fight_features_df = engineer.create_single_fight_features(
            fighter1_name, fighter2_name, fight_date
        )
    except ValueError as e:
        logger.error(f"Error creating features: {e}")
        return {'error': str(e)}
    
    logger.info("Features created successfully")
    
    # Load the scaler
    logger.info("Loading feature scaler...")
    try:
        engineer.load_scaler()
    except FileNotFoundError as e:
        logger.error(f"Scaler not found: {e}")
        logger.error("Please run engineer_features.py first to create and save the scaler.")
        return {'error': 'Scaler not found. Please run feature engineering first.'}
    
    # Transform features using the scaler
    logger.info("Transforming features...")
    fight_features_scaled = engineer.transform_features(fight_features_df)
    
    # Prepare features for prediction (exclude metadata columns)
    exclude_cols = ['fight_date', 'fighter1_name', 'fighter2_name', 'target']
    feature_cols = [col for col in fight_features_scaled.columns if col not in exclude_cols]
    
    # Ensure we have all required features (load training features to check column order)
    training_features_file = engineer.features_data_path / "fight_features.csv"
    if training_features_file.exists():
        training_df = pd.read_csv(training_features_file)
        training_feature_cols = [col for col in training_df.columns if col not in exclude_cols]
        
        # Ensure column order matches training data
        if set(feature_cols) != set(training_feature_cols):
            missing = set(training_feature_cols) - set(feature_cols)
            extra = set(feature_cols) - set(training_feature_cols)
            if missing:
                logger.warning(f"Missing features: {missing}")
            if extra:
                logger.warning(f"Extra features: {extra}")
        
        # Reorder columns to match training data
        feature_cols = [col for col in training_feature_cols if col in fight_features_scaled.columns]
    
    X = fight_features_scaled[feature_cols]
    
    # Load the trained model
    logger.info(f"Loading model: {model_name}...")
    trainer = ModelTrainer()
    try:
        model = trainer.load_model(model_name)
    except FileNotFoundError as e:
        logger.error(f"Model not found: {e}")
        logger.error(f"Please train the {model_name} model first using train_models.py")
        return {'error': f'Model {model_name} not found. Please train it first.'}
    
    # Make predictions
    logger.info("Making prediction...")
    prediction = model.predict(X)[0]  # 1 = fighter1 wins, 0 = fighter2 wins
    
    # Get prediction probabilities if available
    proba = None
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X)[0]
        fighter1_prob = proba[1] if len(proba) > 1 else proba[0]
        fighter2_prob = proba[0] if len(proba) > 1 else 1 - proba[0]
    else:
        fighter1_prob = None
        fighter2_prob = None
    
    # Determine winner
    predicted_winner = fighter1_name if prediction == 1 else fighter2_name
    confidence = fighter1_prob if prediction == 1 else fighter2_prob
    
    result = {
        'fighter1': fighter1_name,
        'fighter2': fighter2_name,
        'fight_date': fight_date,
        'predicted_winner': predicted_winner,
        'prediction': 'Fighter 1' if prediction == 1 else 'Fighter 2',
        'prediction_numeric': int(prediction),
        'confidence': float(confidence) if confidence is not None else None,
        'fighter1_win_probability': float(fighter1_prob) if fighter1_prob is not None else None,
        'fighter2_win_probability': float(fighter2_prob) if fighter2_prob is not None else None,
        'model_used': model_name
    }
    
    # Print results
    print()
    print("=" * 60)
    print("Prediction Results")
    print("=" * 60)
    print(f"Predicted Winner: {predicted_winner}")
    if confidence is not None:
        print(f"Confidence: {confidence:.2%}")
        print(f"  {fighter1_name} win probability: {fighter1_prob:.2%}")
        print(f"  {fighter2_name} win probability: {fighter2_prob:.2%}")
    print(f"Model: {model_name}")
    print("=" * 60)
    print()
    
    return result


def main():
    """Main entry point for prediction script."""
    parser = argparse.ArgumentParser(
        description='Predict outcome of a future UFC fight'
    )
    
    parser.add_argument(
        'fighter1',
        type=str,
        help='Name of first fighter'
    )
    
    parser.add_argument(
        'fighter2',
        type=str,
        help='Name of second fighter'
    )
    
    parser.add_argument(
        '--date',
        type=str,
        required=True,
        help='Date of the fight (YYYY-MM-DD format)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='xgboost',
        choices=['random_forest', 'gradient_boosting', 'logistic_regression',
                 'svm', 'xgboost', 'lightgbm'],
        help='Model to use for prediction (default: xgboost - best accuracy 78.56%%)'
    )
    
    args = parser.parse_args()
    
    # Make prediction
    result = predict_fight(
        fighter1_name=args.fighter1,
        fighter2_name=args.fighter2,
        fight_date=args.date,
        model_name=args.model
    )
    
    # Exit with error code if prediction failed
    if 'error' in result:
        sys.exit(1)


if __name__ == "__main__":
    main()

