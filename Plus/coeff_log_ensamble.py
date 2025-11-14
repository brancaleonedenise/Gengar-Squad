final_model = grid_search.best_estimator_

try:
    categorical_names = final_model.named_steps['preprocessor'] \
                                     .named_transformers_['cat'] \
                                     .named_steps['onehot'] \
                                     .get_feature_names_out(categorical_features)
except AttributeError:
    categorical_names = final_model.named_steps['preprocessor'] \
                                     .named_transformers_['cat'] \
                                     .named_steps['onehot'] \
                                     .get_feature_names_out()

all_feature_names = numeric_features + list(categorical_names)

coefficients = final_model.named_steps['classifier'].coef_[0]

importance_df = pd.DataFrame({
    'Feature': all_feature_names,
    'Coefficient': coefficients
})

importance_df['Impact'] = importance_df['Coefficient'].abs()
importance_df = importance_df.sort_values(by='Impact', ascending=False)

print("--- Importance of Features (model coefficients) ---")
print(importance_df.to_string()) # .to_string() stampa tutto il DataFrame senza troncamenti