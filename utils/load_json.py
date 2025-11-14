import os
import pandas as pd


def load_jsonl(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_json(path, lines=True)