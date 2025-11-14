# params/gradientboost_params.py
param_grid = {
    'n_estimators': [200, 300, 400, 500],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 4, 5, 6],
    'min_samples_split': [2, 4, 6, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 0.5, 0.7, None],
    'subsample': [0.6, 0.8, 1.0]  # optional, for stochastic gradient boosting
}
