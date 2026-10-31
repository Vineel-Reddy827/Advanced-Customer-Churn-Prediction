
import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    precision_recall_curve,
    confusion_matrix
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

# 3. Recreate the original train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 4. Load tuned model
model = joblib.load(
    "models/tuned_churn_model.pkl"
)

print("Tuned model loaded successfully!")

# 5. Generate out-of-fold probabilities
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

print("\nGenerating cross-validation predictions...")

oof_probabilities = cross_val_predict(
    model,
    X_train,
    y_train,
    cv=cv,
    method='predict_proba',
    n_jobs=-1
)[:, 1]

# 6. Find the optimal threshold using F1 score
precision, recall, thresholds = precision_recall_curve(
    y_train,
    oof_probabilities
)

f1_scores = (
    2 * precision[:-1] * recall[:-1]
    / (precision[:-1] + recall[:-1] + 1e-10)
)

best_index = np.argmax(f1_scores)

best_threshold = thresholds[best_index]

print("\nTHRESHOLD OPTIMIZATION RESULTS")

print("Optimal Threshold:", best_threshold)
print("Cross-validation F1 Score:", f1_scores[best_index])

# 7. Get test probabilities
y_prob = model.predict_proba(X_test)[:, 1]

# 8. Evaluate default threshold
y_pred_default = (y_prob >= 0.5).astype(int)

# 9. Evaluate optimized threshold
y_pred_optimized = (
    y_prob >= best_threshold
).astype(int)


def evaluate_model(y_true, y_pred, y_prob, title):

    print("\n", title)

    print("\nClassification Report:")
    print(classification_report(
        y_true,
        y_pred,
        target_names=['No Churn', 'Churn']
    ))

    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall:", recall_score(y_true, y_pred))
    print("F1 Score:", f1_score(y_true, y_pred))
    print("ROC AUC:", roc_auc_score(y_true, y_prob))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))


# 10. Compare both thresholds
evaluate_model(
    y_test,
    y_pred_default,
    y_prob,
    "DEFAULT THRESHOLD (0.5)"
)

evaluate_model(
    y_test,
    y_pred_optimized,
    y_prob,
    "OPTIMIZED THRESHOLD"
)

# 11. Save optimal threshold
os.makedirs('models', exist_ok=True)

with open(
    'models/optimal_threshold.txt',
    'w'
) as file:
    file.write(str(best_threshold))

# 12. Save comparison results
comparison = pd.DataFrame([
    {
        'Threshold': 0.5,
        'Accuracy': accuracy_score(y_test, y_pred_default),
        'Precision': precision_score(y_test, y_pred_default),
        'Recall': recall_score(y_test, y_pred_default),
        'F1 Score': f1_score(y_test, y_pred_default)
    },
    {
        'Threshold': best_threshold,
        'Accuracy': accuracy_score(y_test, y_pred_optimized),
        'Precision': precision_score(y_test, y_pred_optimized),
        'Recall': recall_score(y_test, y_pred_optimized),
        'F1 Score': f1_score(y_test, y_pred_optimized)
    }
])

os.makedirs('reports', exist_ok=True)

comparison.to_csv(
    'reports/threshold_comparison.csv',
    index=False
)

print("\nThreshold saved successfully!")
print("Comparison report saved successfully!")
print("\nTHRESHOLD OPTIMIZATION COMPLETED!")
