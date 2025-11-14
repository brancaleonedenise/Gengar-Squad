final_model = grid_search.best_estimator_

test_predictions = final_model.predict(X_test_features)
test_battle_ids = X_test_features.index

submission_df = pd.DataFrame({
    'battle_id': test_battle_ids,
    'player_won': test_predictions 
})


submission_df.to_csv('submission_predictions.csv', index=False)

print("\n-------------------------------------------------")
print("submission_predictions.csv!")
print("-------------------------------------------------")
print(submission_df.head())