
import numpy as np
# params/logistic_params.
# Import selectkbest for use into the param_grid


param_grid = {
    'selectkbest__k': [30, 45, "all"],
    'classifier__C': [100, 500, 1000],
    'classifier__penalty': ['l1'],
    'classifier__class_weight': [None, 'balanced'],
    'classifier__solver': ['liblinear']
}
