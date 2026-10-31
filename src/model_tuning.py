
import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)

# 1. Load dataset
df = pd.read_csv(
    "data/processed/customer_churn_clean.csv"
)

# 2. Separate features and target
X = df.drop(
    columns=['Churn', 'customerID'],
    errors='ignore'
)

y = df['Churn'].map({
    'No': 0,
    'Yes': 1
})

# 3. Identify feature types
numerical_features = X.select_dtypes(
    include=['number']
).columns.tolist()

categorical_features = X.select_dtypes(
    include=['object', 'category', 'string']
).columns.tolist()

# 4. Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'),
         categorical_features)
    ]
)

# 5. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training Data:", X_train.shape)
print("Testing Data:", X_test.shape)

# 6. Create Logistic Regression pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(
        max_iter=2000,
        random_state=42
    ))
])

# 7. Define hyperparameter combinations
param_grid = {
    'classifier__C': [0.01, 0.1, 1, 10, 100],

    'classifier__class_weight': [
        None,
        'balanced'
    ],

    'classifier__solver': [
        'lbfgs',
        'liblinear'
    ]
}

# 8. Apply GridSearchCV
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring='f1',
    cv=5,
    n_jobs=-1,
    verbose=1,
    refit=True
)

print("\nStarting Hyperparameter Tuning...")

grid_search.fit(X_train, y_train)

# 9. Display best parameters
print("\nBEST PARAMETERS:")
print(grid_search.best_params_)

print("\nBEST CROSS-VALIDATION F1 SCORE:")
print(grid_search.best_score_)

# 10. Evaluate the best model on test data
best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

print("\nFINAL MODEL EVALUATION")

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=['No Churn', 'Churn']
))

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_prob))

# 11. Save tuned model
os.makedirs('models', exist_ok=True)

joblib.dump(
    best_model,
    'models/tuned_churn_model.pkl'
)

print("\nTuned model saved successfully!")

# 12. Save tuning results
os.makedirs('reports', exist_ok=True)

results_df = pd.DataFrame(grid_search.cv_results_)

results_df.to_csv(
    'reports/hyperparameter_tuning_results.csv',
    index=False
)

print("Tuning results saved successfully!")
