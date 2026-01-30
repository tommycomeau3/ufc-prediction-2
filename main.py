#!/usr/bin/env python3
"""
Main pipeline orchestrator for UFC fight outcome prediction system.
Coordinates the entire workflow: data collection -> preprocessing -> feature engineering -> training -> evaluation.
"""

import sys
import argparse
from pathlib import Path
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from preprocessing.cleaner import DataCleaner
from features.engineering import FeatureEngineer
from models.trainer import ModelTrainer
from evaluation.metrics import ModelEvaluator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_preprocessing():
    """Run data preprocessing pipeline."""
    print("=" * 60)
    print("Step 1: Data Preprocessing")
    print("=" * 60)
    print()
    
    cleaner = DataCleaner()
    cleaner.process_all()
    print()


def run_feature_engineering():
    """Run feature engineering pipeline."""
    print("=" * 60)
    print("Step 2: Feature Engineering")
    print("=" * 60)
    print()
    
    engineer = FeatureEngineer()
    engineer.engineer_features()
    print()


def run_training():
    """Run model training pipeline."""
    print("=" * 60)
    print("Step 3: Model Training")
    print("=" * 60)
    print()
    
    trainer = ModelTrainer()
    
    # Load features
    df = trainer.load_features()
    
    # Prepare data
    X, y = trainer.prepare_data(df)
    
    # Split data
    trainer.split_data(X, y, df)
    
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
    
    trained_count = 0
    
    for model_name, config in model_configs.items():
        if not config.get('enabled', False):
            logger.info(f"Skipping {model_name} (disabled in config)")
            continue
        
        try:
            # Build hyperparameter grid
            hyperparams = {k: v for k, v in config.items() if k != 'enabled'}
            
            # Train model
            model = trainer.train_model(model_name, hyperparameters=hyperparams if hyperparams else None)
            
            if model:
                trainer.save_model(model_name, model)
                trained_count += 1
        
        except Exception as e:
            logger.error(f"Error training {model_name}: {e}")

    trainer.log_metrics_history()

    print(f"Trained {trained_count} models")
    print()


def run_evaluation():
    """Run model evaluation pipeline."""
    print("=" * 60)
    print("Step 4: Model Evaluation")
    print("=" * 60)
    print()
    
    trainer = ModelTrainer()
    evaluator = ModelEvaluator()
    
    # Load features
    df = trainer.load_features()
    
    # Prepare data
    X, y = trainer.prepare_data(df)
    feature_names = list(X.columns)
    
    # Split data
    trainer.split_data(X, y, df)
    
    # Find all trained models
    model_files = list(trainer.models_path.glob("*_model.pkl"))
    
    if not model_files:
        logger.warning("No trained models found! Skipping evaluation.")
        return
    
    all_model_scores = {}
    
    # Evaluate each trained model
    for model_file in model_files:
        model_name = model_file.stem.replace('_model', '')
        
        try:
            # Load model
            model = trainer.load_model(model_name)
            
            # Evaluate model
            metrics = evaluator.evaluate_model(
                model,
                trainer.X_test,
                trainer.y_test,
                feature_names,
                model_name
            )
            
            all_model_scores[model_name] = metrics
        
        except Exception as e:
            logger.error(f"Error evaluating {model_name}: {e}")
    
    # Create comparison plots
    if len(all_model_scores) > 1:
        metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        
        for metric in metrics_to_compare:
            if all(metric in scores for scores in all_model_scores.values()):
                evaluator.plot_model_comparison(all_model_scores, metric)
    
    # Print summary
    print()
    print("Evaluation Summary:")
    print("-" * 60)
    for model_name, scores in all_model_scores.items():
        print(f"\n{model_name}:")
        print(f"  Accuracy:  {scores['accuracy']:.4f}")
        print(f"  Precision: {scores['precision']:.4f}")
        print(f"  Recall:    {scores['recall']:.4f}")
        print(f"  F1 Score:  {scores['f1_score']:.4f}")
        if 'roc_auc' in scores:
            print(f"  ROC-AUC:   {scores['roc_auc']:.4f}")
    print()


def main():
    """Main pipeline entry point."""
    parser = argparse.ArgumentParser(
        description='UFC Fight Outcome Prediction Pipeline'
    )
    
    parser.add_argument(
        '--steps',
        nargs='+',
        choices=['preprocess', 'features', 'train', 'evaluate', 'all'],
        default=['all'],
        help='Pipeline steps to run (default: all)'
    )
    
    parser.add_argument(
        '--skip-preprocessing',
        action='store_true',
        help='Skip preprocessing step'
    )
    
    parser.add_argument(
        '--skip-features',
        action='store_true',
        help='Skip feature engineering step'
    )
    
    parser.add_argument(
        '--skip-training',
        action='store_true',
        help='Skip model training step'
    )
    
    parser.add_argument(
        '--skip-evaluation',
        action='store_true',
        help='Skip model evaluation step'
    )
    
    args = parser.parse_args()
    
    # Determine which steps to run
    if 'all' in args.steps:
        steps_to_run = ['preprocess', 'features', 'train', 'evaluate']
        
        # Apply skip flags
        if args.skip_preprocessing:
            steps_to_run.remove('preprocess')
        if args.skip_features:
            steps_to_run.remove('features')
        if args.skip_training:
            steps_to_run.remove('train')
        if args.skip_evaluation:
            steps_to_run.remove('evaluate')
    else:
        steps_to_run = args.steps
    
    print()
    print("=" * 60)
    print("UFC Fight Outcome Prediction Pipeline")
    print("=" * 60)
    print()
    print(f"Running steps: {', '.join(steps_to_run)}")
    print()
    
    # Run pipeline steps
    if 'preprocess' in steps_to_run:
        run_preprocessing()
    
    if 'features' in steps_to_run:
        run_feature_engineering()
    
    if 'train' in steps_to_run:
        run_training()
    
    if 'evaluate' in steps_to_run:
        run_evaluation()
    
    print()
    print("=" * 60)
    print("Pipeline Complete!")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()

