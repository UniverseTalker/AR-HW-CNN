import os, re
import numpy as np
from PIL import Image

import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import joblib


# مسارات مجلدات  التدريب و الفحص
TRAIN_DIR = r"C:\Users\Lenovo\Desktop\BIg academy files\IP\ip\Train Images 13440x32x32\train"
TEST_DIR  = r"C:\Users\Lenovo\Desktop\BIg academy files\IP\ip\Test Images 3360x32x32\test"

IMG_SIZE = (32, 32)
label_re = re.compile(r"label_(\d+)")

def load_folder_flat(folder):
    """
    Loads images from folder.
    Extracts label from filename (label_#).
    Converts image to grayscale 32x32, normalizes, then flattens to 1024 features.
    """
    X, y = [], []
    files = [f for f in os.listdir(folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

    for f in files:
        m = label_re.search(f)
        if not m:
            continue
        lab = int(m.group(1))

        path = os.path.join(folder, f)
        img = Image.open(path).convert("L").resize(IMG_SIZE)
        arr = np.array(img, dtype=np.float32) / 255.0  # normalize 0..1
        X.append(arr.flatten())                        # 1024 features
        y.append(lab)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    return X, y

def evaluate_and_row(model_name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)

    print(f"\n=== {model_name} ===")
    print("Accuracy:", acc)
    print("Precision (macro):", prec)
    print("Recall (macro):", rec)
    print("F1 (macro):", f1)
    print("\nClassification report:\n", classification_report(y_true, y_pred, zero_division=0))
    print("\nConfusion matrix:\n", confusion_matrix(y_true, y_pred))

    return {
        "Model": model_name,
        "Accuracy": acc,
        "Precision_macro": prec,
        "Recall_macro": rec,
        "F1_macro": f1
    }

print("Loading train/test images...")
X_train, y_train = load_folder_flat(TRAIN_DIR)
X_test, y_test = load_folder_flat(TEST_DIR)

print("X_train:", X_train.shape, "y_train:", y_train.shape, "Unique labels:", len(np.unique(y_train)))
print("X_test :", X_test.shape,  "y_test :", y_test.shape)

# === 2) موديل 1: Logistic Regression (Baseline قوي) ===
logreg = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=2000, n_jobs=-1))
])

# === 3) موديل 2: Linear SVM ===
svm = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LinearSVC(dual=False, max_iter=5000, tol=1e-3))
])


models = {
    "LogisticRegression": logreg,
    "LinearSVM": svm
}

os.makedirs("models", exist_ok=True)
results = []

for name, model in models.items():
    print(f"\nTraining {name} ...")
    model.fit(X_train, y_train)
    print(f"{name} training finished")

    y_pred = model.predict(X_test)
    results.append(evaluate_and_row(name, y_test, y_pred))

    # Save model for later use (optional)
    joblib.dump(model, f"models/{name}.joblib")
    print(f"Saved: models/{name}.joblib")

# === 4) حفظ النتائج في CSV للمقارنة ===
df = pd.DataFrame(results)
df.to_csv("ml_models_results.csv", index=False)
print("\nSaved: ml_models_results.csv")
print(df)
