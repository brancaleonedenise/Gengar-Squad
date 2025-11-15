
import numpy as np
# params/logistic_params.
# Import selectkbest for use into the param_grid


param_grid = {
    'selectkbest__k': [30, 45, "all"],
    'classifier__C': np.logspace(1,3, 8),
    'classifier__penalty': ['l1','l2'],
    'classifier__class_weight': [None, 'balanced'],
    'classifier__solver': ['liblinear']
}
