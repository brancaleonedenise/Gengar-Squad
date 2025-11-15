import os
import pandas as pd
from datetime import datetime

def save_submission(
    X_test, 
    model, 
    submissions_dir="submissions",
    name=None
):
    """
    Predict using a trained model and save a CSV submission.

    Args:
        X_test (pd.DataFrame): Test features.
        model: Trained model with .predict().
        submissions_dir (str): Folder to save outputs.
        name (str, optional): Custom filename (without .csv).
    
    Returns:
        str: Path to the saved CSV.
    """
    if not os.path.exists(submissions_dir):
        os.makedirs(submissions_dir)

    # Predict and convert boolean to int
    y_pred = model.predict(X_test).astype(int)

    # Filename logic
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = type(model).__name__

    if name is None:
        filename = f"{model_name}_{timestamp}.csv"
    else:
        filename = f"{name}.csv"

    filepath = os.path.join(submissions_dir, filename)

    # Save file
    submission_df = pd.DataFrame({
        "battle_id": X_test.index,
        "pred": y_pred
    })
    submission_df.to_csv(filepath, index=False)

    print(f"[INFO] Submission created: {filepath}")
    return filepath
