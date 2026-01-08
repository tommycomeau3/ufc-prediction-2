#!/usr/bin/env python3
"""
Script to evaluate trained ML models for UFC fight outcome prediction.
Loads trained models and generates metrics and visualizations.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from models.trainer import ModelTrainer
from evaluation.metrics import ModelEvaluator
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Evaluate all trained models."""
    print("=" * 60)
    print("UFC Model Evaluation")
    print("=" * 60)
    print()
    
    # Initialize trainer and evaluator
    trainer = ModelTrainer()
    evaluator = ModelEvaluator()
    
    # Load features
    print("Loading features...")
    df = trainer.load_features()
    print()
    
    # Prepare data
    print("Preparing data...")
    X, y = trainer.prepare_data(df)
    
    # Get feature names (X already excludes metadata columns)
    feature_names = list(X.columns)
    
    print()
    
    # Split data (same split as training)
    print("Splitting data...")
    trainer.split_data(X, y, df)
    print()
    
    # Load model configuration
    models_config = trainer.config.get('models', {})
    
    # Find all trained models
    model_files = list(trainer.models_path.glob("*_model.pkl"))
    
    if not model_files:
        print("No trained models found!")
        print(f"Looking in: {trainer.models_path}")
        print("Please train models first using: python train_models.py")
        return
    
    print("=" * 60)
    print("Evaluating Models")
    print("=" * 60)
    print()
    
    all_model_scores = {}
    
    # Evaluate each trained model
    for model_file in model_files:
        model_name = model_file.stem.replace('_model', '')
        
        try:
            print(f"Evaluating {model_name}...")
            
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
            
            print(f"✓ {model_name} evaluation complete")
            print()
            
        except Exception as e:
            logger.error(f"Error evaluating {model_name}: {e}")
            print()
    
    # Create comparison plots
    if len(all_model_scores) > 1:
        print("=" * 60)
        print("Creating Comparison Plots")
        print("=" * 60)
        print()
        
        metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        
        for metric in metrics_to_compare:
            # Check if all models have this metric
            if all(metric in scores for scores in all_model_scores.values()):
                evaluator.plot_model_comparison(all_model_scores, metric)
                print(f"✓ Created comparison plot for {metric}")
        
        print()
    
    # Print summary
    print()
    print("=" * 60)
    print("Evaluation Summary")
    print("=" * 60)
    print()
    print(f"Models evaluated: {len(all_model_scores)}")
    print()
    print("Model Performance:")
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
    print(f"Plots saved to: {evaluator.plots_path}")


if __name__ == "__main__":
    main()

