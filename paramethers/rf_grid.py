# params/random_forest_params.py

param_grid = {
    'classifier__n_estimators': [100, 200, 300, 400, 500],  # number of trees
    'classifier__max_depth': [None, 10, 20, 30],            # max depth of each tree
    'classifier__min_samples_split': [2, 5, 10],            # min samples to split a node
    'classifier__min_samples_leaf': [1, 2, 4, 6],           # min samples per leaf
    'classifier__max_features': ['sqrt', 'log2', 0.5, 0.7]  # number of features considered per split
}
