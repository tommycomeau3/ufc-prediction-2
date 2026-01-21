"""
Model training pipeline for UFC fight outcome prediction.
Handles data loading, train/test splitting, hyperparameter tuning, and model training.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import yaml
import joblib
from datetime import datetime

from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from .models import get_model_creator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Trains ML models for UFC fight outcome prediction."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize model trainer with configuration.
        
        Args:
            config_path: Path to configuration YAML file
        """
        self.config = self._load_config(config_path)
        
        # Get paths from config
        paths_config = self.config.get('paths', {})
        self.features_data_path = Path(paths_config.get('features_data', 'data/features'))
        self.models_path = Path(paths_config.get('models', 'models'))
        self.models_path.mkdir(parents=True, exist_ok=True)
        
        # Get model settings
        models_config = self.config.get('models', {})
        self.test_size = models_config.get('test_size', 0.2)
        self.random_state = models_config.get('random_state', 42)
        self.use_time_based_split = models_config.get('use_time_based_split', True)
        self.cv_folds = models_config.get('cv_folds', 5)
        self.tune_hyperparameters = models_config.get('tune_hyperparameters', True)
        self.n_iter = models_config.get('n_iter', 50)
        
        # Data storage
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.trained_models = {}
        self.model_scores = {}
        
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
    
    def load_features(self, filename: str = "fight_features.csv") -> pd.DataFrame:
        """Load feature matrix from CSV file.
        
        Args:
            filename: Name of feature CSV file
            
        Returns:
            DataFrame with features
        """
        features_file = self.features_data_path / filename
        
        if not features_file.exists():
            raise FileNotFoundError(f"Features file not found: {features_file}")
        
        logger.info(f"Loading features from {features_file}")
        df = pd.read_csv(features_file)
        
        # Convert date column if it exists
        if 'fight_date' in df.columns:
            df['fight_date'] = pd.to_datetime(df['fight_date'], errors='coerce')
        
        logger.info(f"Loaded {len(df)} fight records with {len(df.columns)} columns")
        return df
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data for training by separating features and target.
        
        Args:
            df: DataFrame with features and target
            
        Returns:
            Tuple of (X, y) where X is features and y is target
        """
        # Exclude metadata columns from features
        exclude_cols = ['fight_date', 'fighter1_name', 'fighter2_name', 'target']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols]
        y = df['target']
        
        logger.info(f"Prepared {len(X)} samples with {len(feature_cols)} features")
        logger.info(f"Target distribution: {y.value_counts().to_dict()}")
        
        return X, y
    
    def split_data(self, X: pd.DataFrame, y: pd.Series, df: Optional[pd.DataFrame] = None) -> None:
        """Split data into training and testing sets.
        
        Args:
            X: Feature matrix
            y: Target vector
            df: Optional full DataFrame for time-based splitting
        """
        if self.use_time_based_split and df is not None and 'fight_date' in df.columns:
            # Time-based split: train on older fights, test on newer fights
            logger.info("Using time-based train/test split")
            df_sorted = df.sort_values('fight_date')
            split_idx = int(len(df_sorted) * (1 - self.test_size))
            
            train_df = df_sorted.iloc[:split_idx]
            test_df = df_sorted.iloc[split_idx:]
            
            train_indices = train_df.index
            test_indices = test_df.index
            
            self.X_train = X.loc[train_indices]
            self.X_test = X.loc[test_indices]
            self.y_train = y.loc[train_indices]
            self.y_test = y.loc[test_indices]
            
            logger.info(f"Train set: {len(self.X_train)} fights (dates: {train_df['fight_date'].min()} to {train_df['fight_date'].max()})")
            logger.info(f"Test set: {len(self.X_test)} fights (dates: {test_df['fight_date'].min()} to {test_df['fight_date'].max()})")
        else:
            # Random split
            logger.info("Using random train/test split")
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=y
            )
            
            logger.info(f"Train set: {len(self.X_train)} fights")
            logger.info(f"Test set: {len(self.X_test)} fights")
    
    def train_model(self, model_name: str, hyperparameters: Optional[Dict] = None) -> Optional:
        """Train a single model with optional hyperparameter tuning.
        
        Args:
            model_name: Name of the model to train
            hyperparameters: Optional hyperparameter grid for tuning
            
        Returns:
            Trained model or None if training failed
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("Data not prepared. Call split_data() first.")
        
        logger.info(f"Training {model_name}...")
        
        # Get model creator function
        model_creator = get_model_creator(model_name)
        
        # Create base model
        if hyperparameters and self.tune_hyperparameters:
            # Hyperparameter tuning with RandomizedSearchCV
            logger.info(f"Tuning hyperparameters for {model_name}...")
            base_model = model_creator()
            
            search = RandomizedSearchCV(
                base_model,
                hyperparameters,
                n_iter=self.n_iter,
                cv=self.cv_folds,
                scoring='accuracy',
                random_state=self.random_state,
                n_jobs=-1,
                verbose=1
            )
            
            search.fit(self.X_train, self.y_train)
            model = search.best_estimator_
            logger.info(f"Best parameters: {search.best_params_}")
            logger.info(f"Best CV score: {search.best_score_:.4f}")
        else:
            # Train without tuning
            model = model_creator(**(hyperparameters or {}))
            model.fit(self.X_train, self.y_train)
        
        # Evaluate on test set
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        roc_auc = roc_auc_score(self.y_test, y_pred_proba) if y_pred_proba is not None else None
        
        self.model_scores[model_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc
        }
        
        logger.info(f"{model_name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f if roc_auc else 'N/A'}")
        
        self.trained_models[model_name] = model
        return model
    
    def save_model(self, model_name: str, model) -> None:
        """Save trained model to disk.
        
        Args:
            model_name: Name of the model
            model: Trained model object
        """
        model_file = self.models_path / f"{model_name}_model.pkl"
        joblib.dump(model, model_file)
        logger.info(f"Saved {model_name} model to {model_file}")
    
    def load_model(self, model_name: str):
        """Load trained model from disk.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Loaded model object
        """
        model_file = self.models_path / f"{model_name}_model.pkl"
        
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")
        
        model = joblib.load(model_file)
        logger.info(f"Loaded {model_name} model from {model_file}")
        return model


def main():
    """Example usage of the model trainer."""
    trainer = ModelTrainer()
    
    # Load features
    df = trainer.load_features()
    
    # Prepare data
    X, y = trainer.prepare_data(df)
    
    # Split data
    trainer.split_data(X, y, df)
    
    # Train a model
    model = trainer.train_model('random_forest')
    
    # Save model
    if model:
        trainer.save_model('random_forest', model)


if __name__ == "__main__":
    main()

