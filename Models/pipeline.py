from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


def get_pipeline(model_name: str, numerical_features: list, categorical_features: list = None, scaler='auto'):
    """
    Returns a ready-to-use sklearn pipeline with optional preprocessing and the chosen model.

    Args:
        model_name: str, one of ['logistic', 'random_forest', 'xgboost', 'lightgbm', 'catboost', 'gradient_boost']
        numerical_features: list of numeric column names
        categorical_features: list of categorical column names (optional)
        scaler: 'standard', 'robust', 'auto', or 'false' (skip numeric scaling)

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
        transformers.append(('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features))

    if transformers:
        preprocessor = ColumnTransformer(transformers)
    else:
        preprocessor = 'passthrough'

    # Define model
    model_name_lower = model_name.lower()
    if model_name_lower == 'logistic':
        model = LogisticRegression(max_iter=8000, solver='liblinear', random_state=42)
    elif model_name_lower == 'random_forest':
        model = RandomForestClassifier(random_state=42, n_jobs=-1)
    elif model_name_lower == 'xgboost':
        model = XGBClassifier(eval_metric='logloss', tree_method='hist', random_state=42)
    elif model_name_lower == 'lightgbm':
        model = LGBMClassifier(boosting_type='dart', random_state=42)
    elif model_name_lower == 'catboost':
        model = CatBoostClassifier(task_type='CPU', random_seed=42, verbose=0)
    elif model_name_lower == 'gradient_boost':
        model = GradientBoostingClassifier(random_state=42)
    else:
        raise ValueError(f"Model '{model_name}' not supported.")
    
    # Build pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    return pipeline
