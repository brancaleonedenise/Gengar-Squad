import os
import pandas as pd
from datetime import datetime

def save_submission(X_test, model, submissions_dir="submissions"):
    """
    Predict using a trained model and save a CSV submission with timestamp and model name.
    Predictions are saved as integers (1 for True, 0 for False).

    Args:
        X_test (pd.DataFrame): DataFrame of features for prediction.
        model: Trained model with a .predict() method.
        submissions_dir (str): Folder to save submissions.
    
    Returns:
        str: Path to the saved CSV file.
    """
    if not os.path.exists(submissions_dir):
        os.makedirs(submissions_dir)

    # Generate predictions and convert boolean to integer (True->1, False->0)
    y_pred = model.predict(X_test).astype(int)

    # Build filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = type(model).__name__
    filename = f"{model_name}_{timestamp}.csv"
    filepath = os.path.join(submissions_dir, filename)

    # Save CSV
    submission_df = pd.DataFrame({"battle_id": X_test.index, "pred": y_pred})
    submission_df.to_csv(filepath, index=False)

    print(f"[INFO] Submission created: {filepath}")
    return filepath