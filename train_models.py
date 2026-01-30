#!/usr/bin/env python3
"""
Script to train ML models for UFC fight outcome prediction.
Loads features, trains multiple models with hyperparameter tuning, and saves results.
Appends accuracy, F1, ROC-AUC (and timestamp) to logs/training_metrics.json for history tracking.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import yaml
import logging

from models.trainer import ModelTrainer
from models.models import get_model_creator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Train all configured models."""
    print("=" * 60)
    print("UFC Model Training")
    print("=" * 60)
    print()
    
    # Initialize trainer
    trainer = ModelTrainer()
    
    # Load features
    print("Loading features...")
    df = trainer.load_features()
    print()
    
    # Prepare data
    print("Preparing data...")
    X, y = trainer.prepare_data(df)
    print()
    
    # Split data
    print("Splitting data...")
    trainer.split_data(X, y, df)
    print()
    
    # Load model configuration
    models_config = trainer.config.get('models', {})
    
    # Train each enabled model
    model_configs = {
        'random_forest': models_config.get('random_forest', {}),
        'gradient_boosting': models_config.get('gradient_boosting', {}),
        'logistic_regression': models_config.get('logistic_regression', {}),
        'svm': models_config.get('svm', {}),
        'xgboost': models_config.get('xgboost', {}),
        'lightgbm': models_config.get('lightgbm', {})
    }
    
    print("=" * 60)
    print("Training Models")
    print("=" * 60)
    print()
    
    trained_count = 0
    
    for model_name, config in model_configs.items():
        if not config.get('enabled', False):
            logger.info(f"Skipping {model_name} (disabled in config)")
            continue
        
        try:
            # Build hyperparameter grid (exclude 'enabled')
            hyperparams = {k: v for k, v in config.items() if k != 'enabled'}
            
            # Train model
            model = trainer.train_model(model_name, hyperparameters=hyperparams if hyperparams else None)
            
            if model:
                # Save model
                trainer.save_model(model_name, model)
                trained_count += 1
                print()
            
        except Exception as e:
            logger.error(f"Error training {model_name}: {e}")
            print()
    
    # Print summary
    print()
    print("=" * 60)
    print("Training Summary")
    print("=" * 60)
    print(f"Models trained: {trained_count}")
    print()
    print("Model Performance:")
    print("-" * 60)
    
    for model_name, scores in trainer.model_scores.items():
        print(f"{model_name}:")
        print(f"  Accuracy:  {scores['accuracy']:.4f}")
        print(f"  Precision: {scores['precision']:.4f}")
        print(f"  Recall:    {scores['recall']:.4f}")
        print(f"  F1 Score:  {scores['f1_score']:.4f}")
        if scores['roc_auc']:
            print(f"  ROC-AUC:   {scores['roc_auc']:.4f}")
        print()

    trainer.log_metrics_history()

    print(f"Models saved to: {trainer.models_path}")


if __name__ == "__main__":
    main()

