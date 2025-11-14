# params/xgb_params.py
param_grid = {
    'classifier__max_depth': [3, 4, 5, 6, 7, 8],
    'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
    'classifier__subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'classifier__colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'classifier__min_child_weight': [1, 2, 3, 5, 7, 10],
    'classifier__gamma': [0, 0.5, 1, 2, 3, 5],
    'classifier__reg_alpha': [0.001, 0.01, 0.1, 1, 5, 10],
    'classifier__reg_lambda': [0.001, 0.01, 0.1, 1, 5, 10],
    'classifier__n_estimators': [100, 200, 300, 400, 500]
}
