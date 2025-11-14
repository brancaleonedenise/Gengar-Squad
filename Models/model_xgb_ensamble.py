X_train_features = create_advanced_features_2(train_df)
X_test_features = create_advanced_features_2(test_df)
y_train = train_df.set_index('battle_id')['player_won'].loc[X_train_features.index]

numeric_transformer = Pipeline(steps=[('scaler', 'passthrough')]) 
categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough'
)

tuning_pipeline_xgb = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(random_state=42, eval_metric='logloss')) 
])

