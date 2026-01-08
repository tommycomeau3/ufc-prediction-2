"""
Evaluation metrics and visualizations for UFC fight outcome prediction models.
Provides classification metrics, confusion matrix, ROC curves, and feature importance.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging
import yaml

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluate ML models with metrics and visualizations."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize evaluator with configuration.
        
        Args:
            config_path: Path to configuration YAML file
        """
        self.config = self._load_config(config_path)
        
        # Get paths from config
        paths_config = self.config.get('paths', {})
        self.plots_path = Path(self.config.get('evaluation', {}).get('plots_path', 'plots'))
        self.plots_path.mkdir(parents=True, exist_ok=True)
        
        # Get evaluation settings
        eval_config = self.config.get('evaluation', {})
        self.save_plots = eval_config.get('save_plots', True)
        self.show_feature_importance = eval_config.get('show_feature_importance', True)
        self.top_n_features = eval_config.get('top_n_features', 20)
        
        # Set style for plots
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (10, 6)
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file.
        
        Args:
            config_path: Path to config file
            
        Returns:
            Dictionary containing configuration
        """
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file not found at {config_path}, using defaults")
            return {}
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {}
    
    def calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Calculate classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (optional, for ROC-AUC)
            
        Returns:
            Dictionary of metric names and values
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred)
        }
        
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
        
        return metrics
    
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str,
        save: bool = True
    ) -> None:
        """Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name of the model
            save: Whether to save the plot
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Loss', 'Win'],
            yticklabels=['Loss', 'Win']
        )
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        if save and self.save_plots:
            plot_file = self.plots_path / f"{model_name}_confusion_matrix.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            logger.info(f"Saved confusion matrix to {plot_file}")
        
        plt.close()
    
    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        model_name: str,
        save: bool = True
    ) -> None:
        """Plot ROC curve.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            model_name: Name of the model
            save: Whether to save the plot
        """
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        auc = roc_auc_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.4f})', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save and self.save_plots:
            plot_file = self.plots_path / f"{model_name}_roc_curve.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            logger.info(f"Saved ROC curve to {plot_file}")
        
        plt.close()
    
    def plot_feature_importance(
        self,
        model,
        feature_names: list,
        model_name: str,
        top_n: Optional[int] = None,
        save: bool = True
    ) -> None:
        """Plot feature importance (for tree-based models).
        
        Args:
            model: Trained model with feature_importances_ attribute
            feature_names: List of feature names
            model_name: Name of the model
            top_n: Number of top features to show (None for all)
            save: Whether to save the plot
        """
        if not hasattr(model, 'feature_importances_'):
            logger.warning(f"{model_name} does not support feature importance")
            return
        
        top_n = top_n or self.top_n_features
        importances = model.feature_importances_
        
        # Sort features by importance
        indices = np.argsort(importances)[::-1][:top_n]
        
        plt.figure(figsize=(10, max(6, top_n * 0.4)))
        plt.barh(range(len(indices)), importances[indices], align='center')
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Feature Importance - {model_name}')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        if save and self.save_plots:
            plot_file = self.plots_path / f"{model_name}_feature_importance.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            logger.info(f"Saved feature importance to {plot_file}")
        
        plt.close()
    
    def plot_model_comparison(
        self,
        model_scores: Dict[str, Dict[str, float]],
        metric: str = 'f1_score',
        save: bool = True
    ) -> None:
        """Plot comparison of multiple models on a metric.
        
        Args:
            model_scores: Dictionary mapping model names to their scores
            metric: Metric to compare (e.g., 'accuracy', 'f1_score')
            save: Whether to save the plot
        """
        model_names = list(model_scores.keys())
        scores = [model_scores[name].get(metric, 0) for name in model_names]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(model_names, scores, color='steelblue')
        plt.ylabel(metric.replace('_', ' ').title())
        plt.title(f'Model Comparison - {metric.replace("_", " ").title()}')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.,
                height,
                f'{height:.4f}',
                ha='center',
                va='bottom'
            )
        
        plt.tight_layout()
        
        if save and self.save_plots:
            plot_file = self.plots_path / f"model_comparison_{metric}.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            logger.info(f"Saved model comparison to {plot_file}")
        
        plt.close()
    
    def evaluate_model(
        self,
        model,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_names: list,
        model_name: str
    ) -> Dict[str, float]:
        """Comprehensive model evaluation.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            feature_names: List of feature names
            model_name: Name of the model
            
        Returns:
            Dictionary of metrics
        """
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        metrics = self.calculate_metrics(y_test, y_pred, y_pred_proba)
        
        # Create visualizations
        self.plot_confusion_matrix(y_test, y_pred, model_name)
        
        if y_pred_proba is not None:
            self.plot_roc_curve(y_test, y_pred_proba, model_name)
        
        if self.show_feature_importance:
            self.plot_feature_importance(model, feature_names, model_name)
        
        # Print classification report
        print(f"\n{classification_report(y_test, y_pred, target_names=['Loss', 'Win'])}")
        
        return metrics

