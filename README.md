# Gengar-Squad: Pokémon Battle Outcome Prediction

This project builds a complete machine-learning pipeline for predicting the outcome of Generation 2 Pokémon battles using structured JSON battle logs. It includes feature engineering, multiple model pipelines, hyperparameter optimization, and ensemble methods.

## 📁 Project Structure

```
Gengar-Squad/
│
├── Features/
│   ├── features_denise.py
│   ├── features_kayo.py
│   └── features_olya.py
│
├── Models/
│   └── pipeline.py
│
├── optimisers/
│   ├── gridsearch_optimizer.py
│   ├── optuna_optimizer.py
│   └── randomsearch_optimizer.py
│
├── paramethers/
│   ├── cat_grid.py
│   ├── gb_grid.py
│   ├── lgb_grid.py
│   ├── log_grid.py
│   ├── rf_grid.py
│   └── xgb_grid.py
│
├── utils/
│   ├── extra.py
│   ├── functions.py
│   └── load_json.py
│
├── Submission/
│   └── submit.py
│
├── main.py
├── Notebook_denise.ipynb
├── Notebook_kayo.ipynb
├── Notebook_olya.ipynb
└── requirements.txt
```

---

## 🧩 Problem Description

The goal is to predict whether **Player 1 wins the battle** (`player_won`) based on:

* Player 1 team composition (stats, types, levels)
* Player 2 lead Pokémon
* A detailed per-turn battle timeline (moves, statuses, damage trends, boosts, effects)
* Battle metadata

The data comes in deeply nested JSON structures and requires extensive feature engineering before model training.

---

## 🔧 Core Components

### **1. Feature Engineering**

Located in the `Features/` directory.

Three versions of feature engineering exist:

* `features_denise.py`
* `features_kayo.py`
* `features_olya.py`

They convert raw nested battle logs into a flattened, model-ready DataFrame, extracting:

* Base stat features
* Type encodings
* Timeline statistics (damage dealt, turn count, boosts, statuses)
* Team and lead summary features

---

### **2. Model Pipelines**

`Models/pipeline.py`

Provides unified pipelines for all ML models used in the project:

* Logistic Regression
* Random Forest
* XGBoost
* LightGBM
* CatBoost
* Gradient Boosting

Each pipeline supports:

* Automatic or manual scaler configuration
* Plug-and-play integration with optimizers
* Consistent feature preprocessing

---

### **3. Hyperparameter Optimization**

Located in the `optimisers/` directory.

Available optimizers:

* **Grid Search** (`gridsearch_optimizer.py`)
* **Random Search** (`randomsearch_optimizer.py`)
* **Optuna Bayesian Optimization** (`optuna_optimizer.py`)

Each optimizer:

* Tunes any pipeline
* Uses predefined search spaces from `paramethers/*.py`
* Returns best parameters and validation performance

---

### **4. Ensemble Learning**

Ensembles tested at the time of development:

* **StackingClassifier** with logistic regression meta-learner
* **VotingClassifier** (soft/hard)
* Model blending

Used to combine the best optimized versions of:

* LightGBM
* XGBoost
* CatBoost
* Random Forest
* Logistic Regression
* Gradient Boosting

---

### **5. Submission Pipeline**

`Submission/submit.py`
Generates the final CSV predictions for competition submission.

---

## 🧪 Running the Project

### **1. Install dependencies**

```
pip install -r requirements.txt
```

### **2. Run feature extraction**



### **3. Train a model pipeline**

```python
from Models.pipeline import get_pipeline

pipeline = get_pipeline("xgboost", numerical_features=feature_list, scaler="false")
pipeline.fit(X_train, y_train)
```

### **4. Optimize using Optuna**

```python
from optimisers.optuna_optimizer import optimize_optuna
from paramethers.xgb_grid import param_grid

best_params, best_score = optimize_optuna(
    lambda: pipeline,
    X_train,
    y_train,
    X_val,
    y_val,
    param_grid,
    n_trials=50
)
```

### **5. Build an ensemble**

```python
stacking_model.fit(X_train, y_train)
preds = stacking_model.predict(X_val)
```

### **6. Submit results**

```
python Submission/submit.py
```

---

## 📊 Performance Evaluation

All models are evaluated using:

* Accuracy


---

## 🧠 Notebooks

Three exploratory notebooks demonstrate each team member’s feature and model experimentation:

* `Notebook_denise.ipynb`
* `Notebook_kayo.ipynb`
* `Notebook_olya.ipynb`

---

## 🤝 Contributors

* **Olya**
* **Denise**
* **Kayo**

Each contributor built part of the feature engineering, model selection, and experimentation workflow.

---
