# === Imports ===
import pandas as pd
import numpy as np
import os
import kagglehub
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# === Step 1: Load the dataset using KaggleHub ===

# Automatically downloads (or loads cached version)
path = kagglehub.dataset_download("benroshan/factors-affecting-campus-placement")

# Load CSV
df = pd.read_csv(os.path.join(path, "Placement_Data_Full_Class.csv"))

# === Step 2: Clean and prepare ===
df = df.drop(columns=["sl_no"])
df["status"] = df["status"].map({"Placed": 1, "Not Placed": 0})
df = df[df["status"].notna()]  # keep all placed and not placed students

# Define output folder
output_dir = "ml_output"
os.makedirs(output_dir, exist_ok=True)

# Define features and target
target = "status"
X = df.drop(columns=[target, "salary"])
y = df[target]

# === Step 3: Stratified split ===
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

# === Step 4: Preprocessing pipeline ===
categorical_features = X_train.select_dtypes(include=["object"]).columns.tolist()
numeric_features = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

X_train_processed = preprocessor.fit_transform(X_train)
X_val_processed = preprocessor.transform(X_val)
X_test_processed = preprocessor.transform(X_test)

# === Step 5: Train models ===
lr_model = LogisticRegression(max_iter=500, random_state=42)
lr_model.fit(X_train_processed, y_train)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_processed, y_train)

# === Step 6: Save models ===
joblib.dump(lr_model, os.path.join(output_dir, "logistic_model.joblib"))
joblib.dump(rf_model, os.path.join(output_dir, "random_forest_model.joblib"))

# === Step 7: Evaluate and save plots ===
models = {"Logistic Regression": lr_model, "Random Forest": rf_model}

for name, model in models.items():
    print(f"\n=== {name} ===")
    y_pred = model.predict(X_val_processed)
    y_prob = model.predict_proba(X_val_processed)[:, 1]

    print("Classification Report:")
    print(classification_report(y_val, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_val, y_pred)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{name.lower().replace(' ', '_')}_confusion_matrix.png"))
    plt.close()

    # ROC AUC
    auc = roc_auc_score(y_val, y_prob)
    print(f"ROC AUC: {auc:.3f}")

    fpr, tpr, _ = roc_curve(y_val, y_prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.2f})")

# Combined ROC curve
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.title("ROC Curve Comparison")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "roc_curve_comparison.png"))
plt.close()

# === Step 8: Write results to a .txt file ===
results_path = os.path.join(output_dir, "evaluation_results.txt")
with open(results_path, "w") as f:
    for name, model in models.items():
        y_pred = model.predict(X_val_processed)
        y_prob = model.predict_proba(X_val_processed)[:, 1]
        auc = roc_auc_score(y_val, y_prob)
        report = classification_report(y_val, y_pred)
        cm = confusion_matrix(y_val, y_pred)

        f.write(f"=== {name} ===\n")
        f.write(f"ROC AUC: {auc:.3f}\n\n")
        f.write("Classification Report:\n")
        f.write(report + "\n")
        f.write("Confusion Matrix:\n")
        f.write(np.array2string(cm))
        f.write("\n\n" + "="*50 + "\n\n")