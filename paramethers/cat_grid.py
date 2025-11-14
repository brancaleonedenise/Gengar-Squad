# params/catboost_params_pipeline.py

param_grid = {
    'classifier__depth': [4, 5, 6, 7, 8, 9, 10],
    'classifier__learning_rate': [0.01, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2],
    'classifier__iterations': [200, 400, 600, 800, 1000, 1200],
    'classifier__l2_leaf_reg': [1, 3, 5, 7, 10, 12],
    'classifier__random_seed': [42],  # fixed
    'classifier__task_type': ['CPU'],  # fixed
    'classifier__verbose': [0]  # optional: silence output during training
}
