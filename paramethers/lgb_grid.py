# params/lgb_params.py

param_grid = {
    'boosting_type': ['dart'],  # fixed
    'n_estimators': [400, 500, 600, 700, 800],
    'learning_rate': [0.01, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2],
    'num_leaves': [31, 63, 127, 255],
    'max_depth': [-1, 3, 5, 8, 10, 12],
    'feature_fraction': [0.7, 0.8, 0.9, 1.0],
    'bagging_fraction': [0.7, 0.8, 0.9, 1.0],
    'min_child_samples': [5, 10, 20, 30, 50],
    'lambda_l1': [0.001, 0.01, 0.1, 1.0, 5.0, 10.0],
    'lambda_l2': [0.001, 0.01, 0.1, 1.0, 5.0, 10.0]
}
