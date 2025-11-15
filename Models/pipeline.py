from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import inspect

def get_pipeline(model_name: str, numerical_features: list, categorical_features: list = None,
                 scaler='auto', **extra_model_params):
    """
    Returns a ready-to-use sklearn pipeline with optional preprocessing and the chosen model.
    Extra model parameters can be provided as kwargs. Invalid ones are ignored.

    Args:
        model_name: str, one of ['logistic', 'random_forest', 'xgboost', 'lightgbm', 'catboost', 'gradient_boost']
        numerical_features: list of numeric column names
        categorical_features: list of categorical column names (optional)
        scaler: 'standard', 'robust', 'auto', or 'false' (skip numeric scaling)
        **extra_model_params: optional extra parameters for the chosen model

    Returns:
        sklearn Pipeline
    """

    # Choose scaler or skip
    if scaler == 'standard':
        scaler_instance = StandardScaler()
    elif scaler == 'robust':
        scaler_instance = RobustScaler()
    elif scaler == 'auto':
        scaler_instance = StandardScaler() if model_name.lower() in ['lightgbm', 'catboost'] else RobustScaler()
    elif scaler == 'false':
        scaler_instance = None
    else:
        raise ValueError("Scaler must be 'standard', 'robust', 'auto', or 'false'.")

    # Numeric transformer
    transformers = []
    if numerical_features:
        if scaler_instance:
            transformers.append(('num', scaler_instance, numerical_features))
        else:
            transformers.append(('num', 'passthrough', numerical_features))

    # Categorical transformer
    if categorical_features:
        transformers.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features))

    preprocessor = ColumnTransformer(transformers) if transformers else 'passthrough'

    # Define base model
    model_name_lower = model_name.lower()
    if model_name_lower == 'logistic':
        model_cls = LogisticRegression
        default_params = {'max_iter': 8000, 'solver': 'liblinear', 'random_state': 42}
    elif model_name_lower == 'random_forest':
        model_cls = RandomForestClassifier
        default_params = {'random_state': 42, 'n_jobs': -1}
    elif model_name_lower == 'xgboost':
        model_cls = XGBClassifier
        default_params = {'eval_metric': 'logloss', 'tree_method': 'hist', 'random_state': 42}
    elif model_name_lower == 'lightgbm':
        model_cls = LGBMClassifier
        default_params = {'boosting_type': 'dart', 'random_state': 42}
    elif model_name_lower == 'catboost':
        model_cls = CatBoostClassifier
        default_params = {'task_type': 'CPU', 'random_seed': 42, 'verbose': 0}
    elif model_name_lower == 'gradient_boost':
        model_cls = GradientBoostingClassifier
        default_params = {'random_state': 42}
    else:
        raise ValueError(f"Model '{model_name}' not supported.")

    # Filter extra_model_params to only valid ones
    valid_params = inspect.signature(model_cls).parameters
    filtered_params = {k: v for k, v in extra_model_params.items() if k in valid_params}

    # Combine default and extra params
    model_instance = model_cls(**default_params, **filtered_params)

    # Build pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('remove_constant_features', VarianceThreshold(threshold=0)),
        ('selectkbest', SelectKBest(score_func=f_classif)),
        ('classifier', model_instance)
    ])

    return pipeline
