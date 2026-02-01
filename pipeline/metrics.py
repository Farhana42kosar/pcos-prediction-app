import os
import json
from datetime import datetime

def save_metrics(metrics: dict, path="artifacts/metrics.json"):
    os.makedirs("artifacts", exist_ok=True)

    metrics["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"✅ Metrics saved at: {path}")
