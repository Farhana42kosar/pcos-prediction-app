from fastapi import FastAPI
import joblib
import pandas as pd

from app.schema import PCOSInput

app = FastAPI(
    title="PCOS Prediction API",
    version="1.0"
)

# ---------- Paths ----------
MODEL_PATH = "artifacts/pcos_model_latest.pkl"
PREPROCESSOR_PATH = "artifacts/preprocessor.pkl"

# ---------- Load artifacts ----------
model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)

# ---------- Blood group mapping ----------
BLOOD_GROUP_MAP = {
    "A+": 0, "A-": 1,
    "B+": 2, "B-": 3,
    "O+": 4, "O-": 5,
    "AB+": 6, "AB-": 7
}

DEFAULT_BLOOD_GROUP = "O+"


@app.get("/")
def health():
    return {"status": "PCOS API running"}


@app.post("/predict")
def predict_pcos(data: PCOSInput):

    input_dict = data.dict()

    # -------- blood group handling --------
    bg = input_dict.get("blood_group")
    if bg is None:
        bg = DEFAULT_BLOOD_GROUP

    input_dict["blood_group"] = BLOOD_GROUP_MAP.get(
        bg,
        BLOOD_GROUP_MAP[DEFAULT_BLOOD_GROUP]
    )

    # Convert to DataFrame
    df = pd.DataFrame([input_dict])

    # -------- Feature Engineering (MATCH TRAINING) --------
    df["bmi"] = df["weight_kg"] / ((df["heightcm"] / 100) ** 2)

    df["age_group"] = pd.cut(
        df["age_yrs"],
        bins=[0, 25, 35, 50, 100],
        labels=["<25", "25-35", "35-50", "50+"]
    )

    # -------- Preprocess & Predict --------
    X = preprocessor.transform(df)
    prediction = model.predict(X)[0]

    return {
        "prediction": int(prediction),
        "result": "PCOS Detected" if prediction == 1 else "No PCOS"
    }
