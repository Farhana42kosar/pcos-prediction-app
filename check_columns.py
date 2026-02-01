import joblib
import os

ARTIFACTS_DIR = "artifacts"

# Get latest preprocessor
preprocessors = [
    f for f in os.listdir(ARTIFACTS_DIR)
    if f.startswith("pcos_preprocessor") and f.endswith(".pkl")
]

if not preprocessors:
    raise FileNotFoundError("❌ No preprocessor found in artifacts folder")

latest_preprocessor = sorted(preprocessors)[-1]

preprocessor = joblib.load(
    os.path.join(ARTIFACTS_DIR, latest_preprocessor)
)

print("\n📌 Columns used during TRAINING:\n")
print(preprocessor.feature_names_)
print(f"\nTotal features: {len(preprocessor.feature_names_)}")
