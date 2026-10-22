
# ============================================================
# CREDIT SCORING MODEL
# Train and compare three machine learning classification models
# ============================================================

import os
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

# ------------------------------------------------------------
# 1. FILE PATHS
# ------------------------------------------------------------

DATA_PATH = os.path.join("dataset", "credit_data.csv")
MODEL_DIR = "models"
RESULTS_DIR = "results"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ------------------------------------------------------------
# 2. LOAD DATASET
# ------------------------------------------------------------

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}. "
        "Put credit_data.csv inside the dataset folder."
    )

df = pd.read_csv(DATA_PATH)

print("Dataset columns:", df.columns.tolist())
print("\nFirst 5 rows:")
print(df.head())

print("\nDataset shape:", df.shape)





# Remove accidental spaces in column names
df.columns = df.columns.str.strip()

# Check capitalization and column names
print("\nOriginal dataset shape:", df.shape)
print("\nDataset columns:")
print(df.columns.tolist())

# Required columns for this project
required_columns = [
    "Age",
    "Housing",
    "Saving accounts",
    "Checking account",
    "Credit amount",
    "Duration",
    "Purpose",
    "Risk"
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns: {missing_columns}\n"
        f"Your CSV contains: {df.columns.tolist()}\n"
        "Check the downloaded dataset and column names."
    )

# Keep only the selected features and target
df = df[required_columns].copy()

# ------------------------------------------------------------
# 3. CLEAN DATA
# ------------------------------------------------------------

# Convert numeric columns safely
numeric_columns = ["Age", "Credit amount", "Duration"]

for column in numeric_columns:
    df[column] = pd.to_numeric(
        df[column], errors="coerce"
    )

# Normalize target labels
df["Risk"] = df["Risk"].astype("string").str.strip().str.lower()

# Remove rows without a usable target label
df = df[df["Risk"].isin(["good", "bad"])].copy()

# Convert target:
# 0 = good historical credit risk
# 1 = bad historical credit risk
df["Risk"] = df["Risk"].map({
    "good": 0,
    "bad": 1
})

# Remove rows missing critical numeric inputs
df = df.dropna(
    subset=["Age", "Credit amount", "Duration", "Risk"]
)

# Avoid division by zero
df = df[df["Duration"] > 0].copy()

print("\nCleaned dataset shape:", df.shape)
print("\nTarget distribution:")
print(df["Risk"].value_counts())

if df["Risk"].nunique() != 2:
    raise ValueError(
        "The cleaned dataset must contain both good and bad labels."
    )

# ------------------------------------------------------------
# 4. FEATURE ENGINEERING
# ------------------------------------------------------------

# A simple credit amount per month proxy
df["Monthly credit amount proxy"] = (
    df["Credit amount"] / df["Duration"]
)

print("\nNew engineered feature created:")
print("Monthly credit amount proxy")

# ------------------------------------------------------------
# 5. SEPARATE FEATURES AND TARGET
# ------------------------------------------------------------

X = df.drop(columns=["Risk"])
y = df["Risk"].astype(int)

numeric_features = [
    "Age",
    "Credit amount",
    "Duration",
    "Monthly credit amount proxy"
]

categorical_features = [
    "Housing",
    "Saving accounts",
    "Checking account",
    "Purpose"
]

# ------------------------------------------------------------
# 6. SPLIT INTO TRAINING AND TESTING DATA
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# ------------------------------------------------------------
# 7. PREPROCESSING
# ------------------------------------------------------------

# Numerical features:
# - Fill missing values using the training median
# - Standardize numeric values

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical features:
# - Fill missing values
# - Convert text categories into numeric indicator columns

categorical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="constant", fill_value="Unknown")
    ),
    (
        "encoder",
        OneHotEncoder(handle_unknown="ignore")
    )
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

# ------------------------------------------------------------
# 8. DEFINE THREE CLASSIFICATION MODELS
# ------------------------------------------------------------

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=5,
        class_weight="balanced",
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
}

# ------------------------------------------------------------
# 9. TRAIN, PREDICT AND EVALUATE
# ------------------------------------------------------------

trained_models = {}
results = []

for name, classifier in models.items():

    print("\n" + "=" * 60)
    print("Training:", name)

    # Preprocessing is fitted on training data only
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])

    model.fit(X_train, y_train)

    # Predict labels: 0 = good, 1 = bad
    y_pred = model.predict(X_test)

    # Predict probability of bad historical risk
    y_probability = model.predict_proba(X_test)[:, 1]

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test, y_pred, zero_division=0
    )

    recall = recall_score(
        y_test, y_pred, zero_division=0
    )

    f1 = f1_score(
        y_test, y_pred, zero_division=0
    )

    auc = roc_auc_score(y_test, y_probability)

    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1-score:", round(f1, 4))
    print("ROC-AUC:", round(auc, 4))

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=["Good risk", "Bad risk"],
            zero_division=0
        )
    )

    # Save the model in memory
    trained_models[name] = model

    # Store performance results
    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision_Bad_Risk": precision,
        "Recall_Bad_Risk": recall,
        "F1_Bad_Risk": f1,
        "ROC_AUC": auc
    })

# ------------------------------------------------------------
# 10. COMPARE MODELS
# ------------------------------------------------------------

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    by="ROC_AUC", ascending=False
).reset_index(drop=True)

print("\nMODEL COMPARISON")
print(results_df.round(4).to_string(index=False))

results_df.to_csv(
    os.path.join(RESULTS_DIR, "model_comparison.csv"),
    index=False
)

# Select the highest test ROC-AUC for this educational example
best_model_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_model_name]

# ------------------------------------------------------------
# 11. SAVE THE BEST MODEL
# ------------------------------------------------------------

model_path = os.path.join(
    MODEL_DIR, "credit_scoring_model.joblib"
)

joblib.dump(best_model, model_path)

print("\nBest model:", best_model_name)
print("Saved model:", model_path)

# Save model selection and metrics for the report
summary = {
    "best_model": best_model_name,
    "selection_metric": "ROC-AUC on the held-out test set",
    "positive_class": "Bad historical credit risk (1)",
    "test_size": 0.20,
    "random_state": 42,
    "dataset_rows_after_cleaning": int(len(df)),
    "training_rows": int(len(X_train)),
    "testing_rows": int(len(X_test)),
    "metrics": results_df.to_dict(orient="records")
}

with open(
    os.path.join(RESULTS_DIR, "evaluation_summary.json"),
    "w",
    encoding="utf-8"
) as file:
    json.dump(summary, file, indent=4)

# ------------------------------------------------------------
# 12. SAVE A CONFUSION MATRIX FOR THE BEST MODEL
# ------------------------------------------------------------

best_predictions = best_model.predict(X_test)

cm = confusion_matrix(
    y_test,
    best_predictions,
    labels=[0, 1]
)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Good risk", "Bad risk"],
    yticklabels=["Good risk", "Bad risk"]
)

plt.title(f"Confusion Matrix - {best_model_name}")
plt.xlabel("Predicted label")
plt.ylabel("Actual label")
plt.tight_layout()

plt.savefig(
    os.path.join(RESULTS_DIR, "confusion_matrix.png"),
    dpi=150
)

plt.close()

# ------------------------------------------------------------
# 13. SAVE FINAL CLASSIFICATION REPORT
# ------------------------------------------------------------

report_text = classification_report(
    y_test,
    best_predictions,
    labels=[0, 1],
    target_names=["Good risk", "Bad risk"],
    zero_division=0
)

with open(
    os.path.join(RESULTS_DIR, "classification_report.txt"),
    "w",
    encoding="utf-8"
) as file:
    file.write(f"Best model: {best_model_name}\n\n")
    file.write(report_text)

print("\nAll training and evaluation steps completed.")
print("Check the models and results folders.")
