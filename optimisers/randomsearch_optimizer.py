from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, accuracy_score
import joblib
import os
from datetime import datetime

def run_random_search(pipeline, X_train, y_train, param_distributions,
                               n_iter=50, cv=5, scoring='accuracy', 
                               save_dir='Models', random_state=42):
    """
    Runs RandomizedSearchCV on a pipeline and saves the best estimator and parameters.

    Args:
        pipeline: sklearn Pipeline object (preprocessor + model)
        X_train, y_train: training data
        param_distributions: dictionary of hyperparameter distributions for RandomizedSearch
        n_iter: number of random combinations
        cv: cross-validation folds
        scoring: metric to optimize
        save_dir: folder to save results
        random_state: reproducibility seed

    Returns:
        best_pipeline, best_params, best_score
    """
    scorer = make_scorer(accuracy_score) if scoring == 'accuracy' else scoring

    random_search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring=scorer,
        n_jobs=-1,
        random_state=random_state,
        verbose=1
    )
    random_search.fit(X_train, y_train)

    best_pipeline = random_search.best_estimator_
    best_params = random_search.best_params_
    best_score = random_search.best_score_

    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(save_dir, f"{pipeline.steps[-1][1].__class__.__name__}_randomsearch_{timestamp}.joblib")
    joblib.dump(best_pipeline, filename)

    print(f"[INFO] RandomizedSearch completed. Best score: {best_score:.4f}. Pipeline saved as {filename}")

    return best_pipeline, best_params, best_score
