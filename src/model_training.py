
import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

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
def create_preprocessor():
    return ColumnTransformer(
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

X_train.to_csv("data/processed/X_train.csv", index=False)
X_test.to_csv("data/processed/X_test.csv", index=False)

y_train.to_csv("data/processed/y_train.csv", index=False)
y_test.to_csv("data/processed/y_test.csv", index=False)

# 6. Define machine learning models
models = {

    'Logistic Regression': LogisticRegression(
        max_iter=1000,
        random_state=42
    ),

    'Balanced Logistic Regression': LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=42
    ),

    'Decision Tree': DecisionTreeClassifier(
        random_state=42
    ),

    'Random Forest': RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    ),

    'Gradient Boosting': GradientBoostingClassifier(
        random_state=42
    )
}

results = []
trained_models = {}

# 7. Train and evaluate models
for name, model in models.items():

    print("\nTraining:", name)

    pipeline = Pipeline([
        ('preprocessor', create_preprocessor()),
        ('classifier', model)
    ])

    pipeline.fit(X_train, y_train)

    # Predictions
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    # Evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    results.append({
        'Model': name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'ROC AUC': roc_auc
    })

    trained_models[name] = pipeline

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred,
        target_names=['No Churn', 'Churn']
    ))

# 8. Compare model performance
results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by='F1 Score',
    ascending=False
)

print("\nMODEL PERFORMANCE COMPARISON")
print(results_df.to_string(index=False))

# 9. Save results
os.makedirs('reports', exist_ok=True)

results_df.to_csv(
    'reports/model_comparison.csv',
    index=False
)

# 10. Save the best model
best_model_name = results_df.iloc[0]['Model']
best_model = trained_models[best_model_name]

os.makedirs('models', exist_ok=True)

joblib.dump(
    best_model,
    'models/best_churn_model.pkl'
)

print("\nBest Model:", best_model_name)
print("Model saved successfully!")
