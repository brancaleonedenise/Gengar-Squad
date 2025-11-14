# params/catboost_params.py

param_grid = {
    'depth': [4, 5, 6, 7, 8, 9, 10],
    'learning_rate': [0.01, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2],
    'iterations': [200, 400, 600, 800, 1000, 1200],
    'l2_leaf_reg': [1, 3, 5, 7, 10, 12],
    'random_seed': [42],  # fixed
    'task_type': ['CPU'],  # fixed
}
