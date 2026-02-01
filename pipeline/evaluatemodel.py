import os
import json
from sklearn.metrics import accuracy_score, classification_report

METRICS_PATH = "artifacts/metrics/metrics.json"


def evaluate_model(model, X_test, y_test):
    """
    Evaluates model, compares with previous metrics, and saves new metrics.
    """
    # ---------------- Predict ----------------
    y_pred = model.predict(X_test)
    new_accuracy = accuracy_score(y_test, y_pred)

    print("📊 New Model Performance")
    print("Accuracy:", new_accuracy)
    print(classification_report(y_test, y_pred))

    # ---------------- Load old accuracy ----------------
    old_accuracy = None
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r") as f:
            old_metrics = json.load(f)
            old_accuracy = old_metrics.get("accuracy")

    # ---------------- Compare ----------------
    if old_accuracy is None:
        print("🆕 No previous model found. Treating this as first model.")
        is_better = True
    elif new_accuracy > old_accuracy:
        print(f"📈 Improvement detected: {old_accuracy} → {new_accuracy}")
        is_better = True
    else:
        print(f"📉 No improvement: {old_accuracy} → {new_accuracy}")
        is_better = False

    # ---------------- Save metrics ----------------
    os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)

    metrics = {
        "accuracy": new_accuracy
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=4)

    print("💾 Metrics saved")

    return is_better, new_accuracy, old_accuracy
