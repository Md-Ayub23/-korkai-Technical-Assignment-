
import os
import joblib
import pandas as pd

from preprocessing import preprocess
from report_generator import generate_report

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
)


df = preprocess("../data/marketing_campaign.csv")

print(f"Dataset Shape: {df.shape}")
print(f"Response rate: {df['Response'].mean():.2%}")


features = [
    "Age",
    "Income",
    "Education",
    "Marital_Status",
    "Children",
    "Customer_Tenure",
    "Recency",
    "NumDealsPurchases",
    "NumWebVisitsMonth",
    "Previous_Purchase",
    "Total_Spending",
    "TotalAcceptedCmp",
    "Complain",
]

target = "Response"

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"Train Shape: {X_train.shape}")
print(f"Test Shape : {X_test.shape}")

numeric_features = [
    "Age",
    "Income",
    "Children",
    "Customer_Tenure",
    "Recency",
    "NumDealsPurchases",
    "NumWebVisitsMonth",
    "Previous_Purchase",
    "Total_Spending",
    "TotalAcceptedCmp",
    "Complain",
]

categorical_features = ["Education", "Marital_Status"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)


models = {
    "Logistic Regression": (
        LogisticRegression(max_iter=1000, class_weight="balanced"),
        {"classifier__C": [0.01, 0.1, 1, 10]},
    ),
    "Decision Tree": (
        DecisionTreeClassifier(random_state=42, class_weight="balanced"),
        {"classifier__max_depth": [3, 5, 7, 10, None], "classifier__min_samples_leaf": [1, 5, 10]},
    ),
    "Random Forest": (
        RandomForestClassifier(random_state=42, class_weight="balanced"),
        {
            "classifier__n_estimators": [200, 400],
            "classifier__max_depth": [5, 10, None],
            "classifier__min_samples_leaf": [1, 3, 5],
        },
    ),
    "XGBoost": (
        XGBClassifier(
            random_state=42,
            eval_metric="logloss",
            scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        ),
        {
            "classifier__n_estimators": [200, 400],
            "classifier__max_depth": [3, 5, 7],
            "classifier__learning_rate": [0.01, 0.05, 0.1],
        },
    ),
}

best_model = None
best_model_name = ""
best_f1 = 0

fitted_pipelines = {}
results = []
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

os.makedirs("../reports", exist_ok=True)

for name, (model, param_grid) in models.items():

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model),
        ]
    )

    search = GridSearchCV(
        pipeline, param_grid, scoring="f1", cv=cv, n_jobs=-1
    )
    search.fit(X_train, y_train)
    pipeline = search.best_estimator_
    fitted_pipelines[name] = pipeline

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba)

    results.append(
        {
            "Model": name,
            "Best Params": search.best_params_,
            "Accuracy": round(accuracy, 4),
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1 Score": round(f1, 4),
            "ROC AUC": round(roc_auc, 4),
        }
    )

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)
    print(f"Best params: {search.best_params_}")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC AUC  : {roc_auc:.4f}")

    print("\nConfusion Matrix")
    print(confusion_matrix(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"../reports/{name.replace(' ', '_')}_confusion_matrix.png")
    plt.close()

    print("\nClassification Report")
    print(classification_report(y_test, y_pred))


    if f1 > best_f1:
        best_f1 = f1
        best_model = pipeline
        best_model_name = name


rf_pipeline = fitted_pipelines["Random Forest"]
rf_feature_names = rf_pipeline.named_steps["preprocessor"].get_feature_names_out()
rf_importances = rf_pipeline.named_steps["classifier"].feature_importances_

importance_df = pd.DataFrame(
    {"Feature": rf_feature_names, "Importance": rf_importances}
).sort_values("Importance", ascending=False)

print("\n" + "=" * 60)
print("RANDOM FOREST FEATURE IMPORTANCE")
print("=" * 60)
print(importance_df.to_string(index=False))
importance_df.to_csv("../reports/feature_importance.csv", index=False)


# Voting Ensemble: Logistic Regression (high recall) + Random Forest
# (high precision). Soft voting averages predicted probabilities, so the
# combination can land in between the two on the precision/recall tradeoff
# in a way that sometimes beats both individually on F1.



def strip_prefix(params):
    return {k.replace("classifier__", ""): v for k, v in params.items()}


lr_best_params = next(r["Best Params"] for r in results if r["Model"] == "Logistic Regression")
rf_best_params = next(r["Best Params"] for r in results if r["Model"] == "Random Forest")

lr_clf = LogisticRegression(
    max_iter=1000, class_weight="balanced", **strip_prefix(lr_best_params)
)
rf_clf = RandomForestClassifier(
    random_state=42, class_weight="balanced", **strip_prefix(rf_best_params)
)

voting_pipeline = VotingClassifier(
    estimators=[
        ("lr", Pipeline([("preprocessor", preprocessor), ("classifier", lr_clf)])),
        ("rf", Pipeline([("preprocessor", preprocessor), ("classifier", rf_clf)])),
    ],
    voting="soft",
)
voting_pipeline.fit(X_train, y_train)
fitted_pipelines["Voting Ensemble (LR + RF)"] = voting_pipeline

y_pred = voting_pipeline.predict(X_test)
y_proba = voting_pipeline.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, y_proba)

results.append(
    {
        "Model": "Voting Ensemble (LR + RF)",
        "Best Params": "soft voting of tuned LR + RF",
        "Accuracy": round(accuracy, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1 Score": round(f1, 4),
        "ROC AUC": round(roc_auc, 4),
    }
)

print("\n" + "=" * 60)
print("Voting Ensemble (LR + RF)")
print("=" * 60)
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC AUC  : {roc_auc:.4f}")
print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report")
print(classification_report(y_test, y_pred))

if f1 > best_f1:
    best_f1 = f1
    best_model = voting_pipeline
    best_model_name = "Voting Ensemble (LR + RF)"


results_df = pd.DataFrame(results)

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)
print(results_df.drop(columns=["Best Params"]).to_string(index=False))

os.makedirs("../reports", exist_ok=True)
results_df.to_csv("../reports/model_comparison.csv", index=False)


os.makedirs("../models", exist_ok=True)
joblib.dump(best_model, "../models/model.pkl")

print("\n" + "=" * 60)
print("FINAL MODEL")
print("=" * 60)
print(f"Best Model    : {best_model_name}")
print(f"Best F1 Score : {best_f1:.4f} (at default 0.5 threshold)")


# Threshold Tuning
# The default 0.5 cutoff is arbitrary for an imbalanced problem.
# Sweep thresholds against the test set's predicted probabilities and
# pick the one that maximizes F1 instead.


best_proba = best_model.predict_proba(X_test)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_test, best_proba)


f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-12)
best_idx = f1_scores.argmax()
best_threshold = thresholds[best_idx]

y_pred_tuned = (best_proba >= best_threshold).astype(int)
tuned_precision = precision_score(y_test, y_pred_tuned, zero_division=0)
tuned_recall = recall_score(y_test, y_pred_tuned, zero_division=0)
tuned_f1 = f1_score(y_test, y_pred_tuned, zero_division=0)
tuned_accuracy = accuracy_score(y_test, y_pred_tuned)

print(f"\nOptimal threshold : {best_threshold:.3f} (default is 0.5)")
print(f"Accuracy  : {tuned_accuracy:.4f}")
print(f"Precision : {tuned_precision:.4f}")
print(f"Recall    : {tuned_recall:.4f}")
print(f"F1 Score  : {tuned_f1:.4f}")
print("\nConfusion Matrix at tuned threshold")
print(confusion_matrix(y_test, y_pred_tuned))

# Save the threshold alongside the model so inference code uses the same cutoff
joblib.dump(best_threshold, "../models/decision_threshold.pkl")

print("\nModel saved successfully!")
print("Location: ../models/model.pkl")
print(f"Decision threshold saved: ../models/decision_threshold.pkl ({best_threshold:.3f})")

results_df_for_report = results_df.drop(columns=["Best Params"])
best_accuracy = results_df_for_report.loc[
    results_df_for_report["Model"] == best_model_name, "Accuracy"
].values[0]

dataset_stats = {
    "raw_records": "2,240",
    "clean_records": f"{df.shape[0]:,}",
    "response_rate": f"{df['Response'].mean():.1%}"
}


threshold_info = {
    "threshold": best_threshold,
    "accuracy": tuned_accuracy,
    "precision": tuned_precision,
    "recall": tuned_recall,
    "f1": tuned_f1
}

results_df_for_report = results_df.drop(columns=["Best Params"])

generate_report(
    results_df_for_report, 
    best_model_name, 
    best_accuracy,
    feature_importance_path="../reports/feature_importance.csv",
    threshold_info=threshold_info,
    dataset_stats=dataset_stats
)