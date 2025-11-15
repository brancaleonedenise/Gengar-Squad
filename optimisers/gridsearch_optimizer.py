from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score
import joblib
import os
from datetime import datetime

def run_grid_search(pipeline, X_train, y_train, param_grid, cv=5, scoring='accuracy', save_dir='submissions'):
    """
    Runs GridSearchCV on a pipeline and saves the best estimator and parameters.

    Args:
        pipeline: sklearn Pipeline object (preprocessor + model)
        X_train, y_train: training data
        param_grid: dictionary of hyperparameters for GridSearch
        cv: cross-validation folds
        scoring: metric to optimize
        save_dir: folder to save results

    Returns:
        best_pipeline, best_params, best_score
    """
    scorer = make_scorer(accuracy_score) if scoring == 'accuracy' else scoring

    grid = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring=scorer,
        n_jobs=-1,
        verbose=3
    )
    grid.fit(X_train, y_train)

    best_pipeline = grid.best_estimator_
    best_params = grid.best_params_
    best_score = grid.best_score_

    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(save_dir, f"{pipeline.steps[-1][1].__class__.__name__}_gridsearch_{timestamp}.joblib")
    joblib.dump(best_pipeline, filename)

    print(f"[INFO] GridSearch completed. Best score: {best_score:.4f}. Pipeline saved as {filename}")

    return best_pipeline, best_params, best_score
