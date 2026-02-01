import joblib
import pandas as pd

MODEL_PATH = "artifacts/model.pkl"
PREPROCESSOR_PATH = "artifacts/preprocessor.pkl"

def predict(input_data: dict):
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    df = pd.DataFrame([input_data])
    X = preprocessor.transform(df)

    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0].max()

    return {"prediction": int(pred), "confidence": float(prob)}
