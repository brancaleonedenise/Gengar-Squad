X_train_features = create_advanced_features_1(train_df)
X_test_features = create_advanced_features_1(test_df)

y_train = train_df.set_index('battle_id')['player_won']
y_train = y_train.loc[X_train_features.index]

numeric_transformer = Pipeline(steps=[('scaler', RobustScaler())])
categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough'
)

X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train_features, 
    y_train, 
    test_size=0.2, 
    random_state=42,
    stratify=y_train
)

tuning_pipeline_reg = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42, max_iter=1000, solver='liblinear'))
])

tuning_pipeline_reg.fit(X_train_split, y_train_split)

y_val_pred = tuning_pipeline_reg.predict(X_val_split)
val_accuracy = accuracy_score(y_val_split, y_val_pred)

print(f"\n--- Model Results Baseline ---")
print(f"Accuracy on Validation Set: {val_accuracy:.4f}")
print("---------------------------------")