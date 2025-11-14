preprocessor_xgb = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ],
    remainder='passthrough'
)

clf_xgb = XGBClassifier(random_state=42, 
                        eval_metric='logloss',
                        n_estimators=600,
                        learning_rate=0.03,
                        max_depth=5,
                        subsample=0.8,
                        colsample_bytree=0.9,
                        gamma=0.2)
pipeline_xgb = Pipeline(steps=[('preprocessor', preprocessor_xgb),
                               ('classifier', clf_xgb)])

numeric_transformer_logreg = Pipeline(steps=[('scaler', RobustScaler())])
categorical_transformer_logreg = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
preprocessor_logreg = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer_logreg, numeric_features),
        ('cat', categorical_transformer_logreg, categorical_features)
    ],
    remainder='passthrough'
)

clf_logreg = LogisticRegression(C=10, solver='liblinear', random_state=42, penalty='l1') 
pipeline_logreg = Pipeline(steps=[('preprocessor', preprocessor_logreg),
                                  ('classifier', clf_logreg)])

meta_model = LogisticRegression(C=1.0, random_state=42)

stacking_clf = StackingClassifier(
    estimators=[
        ('xgb', pipeline_xgb), 
        ('logreg', pipeline_logreg) # Combina i due modelli DIVERSI
    ], 
    final_estimator=meta_model, # Questo è il tuo "Meta-Modello"
    passthrough=False, # Diamo al "Capo" solo le previsioni (corregge l'errore 'starmie')
    cv=5,              # Cruciale per un addestramento robusto
    n_jobs=-1,
    verbose=1
)

scores = cross_val_score(
    stacking_clf,                
    X_train_features,     
    y_train,              
    cv=5,                 
    scoring='accuracy',
    n_jobs=-1
)

print("\n--- ✅ Risultati della Cross-Validation ---")
print(f"Accuratezza per ognuno dei 5 test: {scores}")
print(f"Accuracy MEDIA FINALE: {scores.mean() * 100:.2f}%")
print(f"Deviazione Standard: {scores.std() * 100:.2f}%")

# --- 4. Addestramento Finale (per la Submission) ---
print("\n🚀 Inizio addestramento finale (Meta-Modello) sull'intero set di training...")
stacking_clf.fit(X_train_features, y_train)
print("Addestramento completato.")

# --- 5. Creazione File CSV Finale (submission_predictions.csv) ---
print("\n📄 Creazione file 'submission_predictions.csv'...")

final_model = stacking_clf # Il Meta-Modello è il modello finale
test_predictions = final_model.predict(X_test_features)
test_battle_ids = X_test_features.index

submission_df = pd.DataFrame({
    'battle_id': test_battle_ids,
    'player_won': test_predictions.astype(int)
})

submission_df.to_csv('submission_predictions.csv', index=False)

print("-------------------------------------------------")
print("File 'submission_predictions.csv' creato con successo!")
print("Lo troverai nel pannello 'Data' -> 'Output' sulla destra.")
print("-------------------------------------------------")
print("\nAnteprima del file:")
print(submission_df.head())