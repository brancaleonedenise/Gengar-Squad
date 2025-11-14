import optuna
from sklearn.metrics import accuracy_score
import logging
from datetime import datetime

# --- Setup logging ---
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)

def optimize_optuna(pipeline_factory, X_train, y_train, X_val, y_val, param_space, n_trials=50):
    """
    Optimizes a scikit-learn pipeline using Optuna.

    Args:
        pipeline_factory: callable that returns a new pipeline instance
        X_train, y_train: training data
        X_val, y_val: validation data
        param_space: dict of parameter search space (keys must match pipeline parameter names)
        n_trials: number of Optuna trials

    Returns:
        best_params: dictionary of best hyperparameters
        best_score: validation accuracy of best trial
    """

    def objective(trial):
        # Build a fresh pipeline
        pipeline = pipeline_factory()
        
        # Generate trial-specific parameters
        trial_params = {}
        for param_name, param_values in param_space.items():
            if isinstance(param_values, list):
                trial_params[param_name] = trial.suggest_categorical(param_name, param_values)
            elif isinstance(param_values, tuple) and len(param_values) == 2:  # numeric range
                low, high = param_values
                if all(isinstance(x, int) for x in (low, high)):
                    trial_params[param_name] = trial.suggest_int(param_name, low, high)
                else:
                    trial_params[param_name] = trial.suggest_float(param_name, low, high)
            else:
                trial_params[param_name] = param_values  # fixed value

        # Set trial parameters
        pipeline.set_params(**trial_params)

        # Fit and evaluate
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_val)
        acc = accuracy_score(y_val, preds)

        logging.info(f"Trial {trial.number+1}/{n_trials} - Accuracy: {acc:.4f} - Params: {trial_params}")
        return acc

    logging.info(f"Starting Optuna optimization for {n_trials} trials...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    best_score = study.best_value

    logging.info(f"Optimization finished. Best Accuracy: {best_score:.4f}")
    logging.info(f"Best Parameters: {best_params}")

    logging.info("Fitting the best pipeline on the provided training data...")
    best_pipeline = pipeline_factory()
    best_pipeline.set_params(**best_params)
    best_pipeline.fit(X_train, y_train)

    return best_pipeline, best_params, best_score
