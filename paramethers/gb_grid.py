# params/gradientboost_params.py
param_grid = {
    "classifier__n_estimators": [100, 300, 500],
    "classifier__learning_rate": [0.01, 0.05, 0.1],
    "classifier__max_depth": [3, 5, 7],
    "classifier__min_samples_split": [2, 5, 10],
    "classifier__min_samples_leaf": [1, 2, 4],
    "classifier__max_features": [0.5, 0.7, 1.0],
    "classifier__subsample": [0.6, 0.8, 1.0]
}

param_grid_optuna = {
    'classifier__n_estimators': (200, 500),                   # int range → trial.suggest_int
    'classifier__learning_rate': (0.01, 0.2),                 # float range → trial.suggest_float
    'classifier__max_depth': (3, 6),                           # int range
    'classifier__min_samples_split': (2, 10),                 # int range
    'classifier__min_samples_leaf': (1, 4),                   # int range
    'classifier__subsample': (0.6, 1.0),                      # float range
    'classifier__max_features': ['sqrt', 0.5, 0.7, None],     # categorical → trial.suggest_categorical
    'classifier__random_state': 42                             # fixed
}

