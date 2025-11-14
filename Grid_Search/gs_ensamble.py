param_grid = {
    'classifier__penalty': ['l1', 'l2'], 
    'classifier__C': [ 0.01, 0.1, 1, 10, 100],
    'classifier__solver': ['liblinear'] 
}

grid_search = GridSearchCV(
    tuning_pipeline_reg, 
    param_grid, 
    cv=10, 
    scoring='accuracy', 
    n_jobs=-1,
    verbose=1 # Mostra i log
)

grid_search.fit(X_train_features, y_train)

best_index = grid_search_log.best_index_
std_dev_performance = grid_search_log.cv_results_['std_test_score'][best_index]

print("\n--- Results GridSearchCV ---")
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best Accuracy (mean CV): {grid_search.best_score_:.4f}")
print(f"Standard Deviation: {std_dev_performance * 100:.2f}%")
print("------------------------------")