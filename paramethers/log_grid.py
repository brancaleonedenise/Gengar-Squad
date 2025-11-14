
import numpy as np
# params/logistic_params.py

param_grid = {
    'classifier__C': np.logspace(0, 3, 10),  # regularization strength
    'classifier__penalty': ['l1', 'l2'],      # type of regularization
    'classifier__class_weight': [None, 'balanced'],  # handle class imbalance
    'classifier__solver': ['liblinear', 'saga']       # compatible solvers for L1/L2
}
