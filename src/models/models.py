"""
Model definitions for UFC fight outcome prediction.
Contains model configurations and initialization functions.
"""

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import logging

logger = logging.getLogger(__name__)

# Try importing optional models
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost not available. Install with: pip install xgboost")

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logger.warning("LightGBM not available. Install with: pip install lightgbm")


def create_random_forest(**kwargs):
    """Create Random Forest classifier.
    
    Args:
        **kwargs: Hyperparameters for Random Forest
        
    Returns:
        RandomForestClassifier instance
    """
    default_params = {
        'n_estimators': 100,
        'max_depth': None,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'random_state': 42,
        'n_jobs': -1
    }
    default_params.update(kwargs)
    return RandomForestClassifier(**default_params)


def create_gradient_boosting(**kwargs):
    """Create Gradient Boosting classifier.
    
    Args:
        **kwargs: Hyperparameters for Gradient Boosting
        
    Returns:
        GradientBoostingClassifier instance
    """
    default_params = {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 3,
        'random_state': 42
    }
    default_params.update(kwargs)
    return GradientBoostingClassifier(**default_params)


def create_logistic_regression(**kwargs):
    """Create Logistic Regression classifier.
    
    Args:
        **kwargs: Hyperparameters for Logistic Regression
        
    Returns:
        LogisticRegression instance
    """
    default_params = {
        'C': 1.0,
        'penalty': 'l2',
        'solver': 'lbfgs',
        'max_iter': 1000,
        'random_state': 42
    }
    default_params.update(kwargs)
    return LogisticRegression(**default_params)


def create_svm(**kwargs):
    """Create Support Vector Machine classifier.
    
    Args:
        **kwargs: Hyperparameters for SVM
        
    Returns:
        SVC instance
    """
    default_params = {
        'C': 1.0,
        'kernel': 'rbf',
        'gamma': 'scale',
        'probability': True,
        'random_state': 42
    }
    default_params.update(kwargs)
    return SVC(**default_params)


def create_xgboost(**kwargs):
    """Create XGBoost classifier.
    
    Args:
        **kwargs: Hyperparameters for XGBoost
        
    Returns:
        XGBClassifier instance or None if not available
    """
    if not XGBOOST_AVAILABLE:
        logger.warning("XGBoost not available, skipping...")
        return None
    
    default_params = {
        'n_estimators': 100,
        'max_depth': 3,
        'learning_rate': 0.1,
        'subsample': 1.0,
        'random_state': 42,
        'eval_metric': 'logloss',
        'use_label_encoder': False
    }
    default_params.update(kwargs)
    return XGBClassifier(**default_params)


def create_lightgbm(**kwargs):
    """Create LightGBM classifier.
    
    Args:
        **kwargs: Hyperparameters for LightGBM
        
    Returns:
        LGBMClassifier instance or None if not available
    """
    if not LIGHTGBM_AVAILABLE:
        logger.warning("LightGBM not available, skipping...")
        return None
    
    default_params = {
        'n_estimators': 100,
        'max_depth': 3,
        'learning_rate': 0.1,
        'num_leaves': 31,
        'random_state': 42,
        'verbose': -1
    }
    default_params.update(kwargs)
    return LGBMClassifier(**default_params)


def get_model_creator(model_name: str):
    """Get model creation function by name.
    
    Args:
        model_name: Name of the model ('random_forest', 'xgboost', etc.)
        
    Returns:
        Model creation function
    """
    creators = {
        'random_forest': create_random_forest,
        'gradient_boosting': create_gradient_boosting,
        'logistic_regression': create_logistic_regression,
        'svm': create_svm,
        'xgboost': create_xgboost,
        'lightgbm': create_lightgbm
    }
    
    if model_name not in creators:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(creators.keys())}")
    
    return creators[model_name]

