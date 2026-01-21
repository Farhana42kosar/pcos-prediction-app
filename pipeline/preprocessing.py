import pandas as pd
import numpy as np
import os
import joblib

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE


class DataPreprocessor:

    def __init__(self):
        # ---------------- Target ----------------
        self.target_column = "pcos_yn"

        # ---------------- Features (STATIC) ----------------
        self.numeric_features = [
            "age_yrs",
            "weight_kg",
            "heightcm",
            "cycleri",
            "cycle_lengthdays",
            "hbgdl",
            "bmi"
        ]

        self.categorical_features = [
            "blood_group",
            "marraige_status_yrs",
            "age_group"
        ]

        self.preprocessor_path = "artifacts/preprocessor.pkl"

    # ---------------- Normalize Column Names ----------------
    def normalize_columns(self, df):
        df.columns = (
            df.columns
            .str.strip()
            .str.lower()
            .str.replace(" ", "_")
            .str.replace(r"[()\/]", "", regex=True )
        )
        return df
    
    def rename_columns(self, df):
        column_map = {
        "age_yrs": "age_yrs",
        "weight_kg": "weight_kg",
        "heightcm": "heightcm",

        # cycle column (MAIN ISSUE)
        "cycler_i": "cycle_ri",
        "cycle_r_i": "cycle_ri",
        "cycle_ri": "cycle_ri",

        "cycle_lengthdays": "cycle_lengthdays",
        "hbgdl": "hbgdl",

        "blood_group": "blood_group",
        "marraige_status_yrs": "marraige_status_yrs",

        "pcos_yn": "pcos_yn",
        "pcos_y_n": "pcos_yn",
    }
        df = df.rename(columns=column_map)
        return df
    
      
        
        



    # ---------------- Schema Validation ----------------
    def validate_schema(self, df):
        required_columns = set(
            self.numeric_features +
            self.categorical_features +
            [self.target_column]
        )

        missing = required_columns - set(df.columns)
        if missing:
            raise Exception(f"Missing required columns: {missing}")

    # ---------------- Outlier Handling ----------------
    def handle_outlier(self, df):
        for col in self.numeric_features:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            df[col] = df[col].clip(lower, upper)
        return df

    # ---------------- Feature Engineering ----------------
    def feature_engineering(self, df):
        # BMI
        df["bmi"] = df["weight_kg"] / ((df["heightcm"] / 100) ** 2)

        # Age group
        df["age_group"] = pd.cut(
            df["age_yrs"],
            bins=[0, 25, 35, 50, 100],
            labels=["<25", "25-35", "35-50", "50+"]
        )

        return df

    # ---------------- Build Preprocessing Pipeline ----------------
    def build_preprocessor(self):
        numeric_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        categorical_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ])

        return ColumnTransformer(transformers=[
            ("num", numeric_pipeline, self.numeric_features),
            ("cat", categorical_pipeline, self.categorical_features)
        ])

    # ---------------- Fit & Transform (Training) ----------------
    def fit_transform(self, df, balance_target=True):
        df = self.normalize_columns(df)

        print("Normalized columns:")
        print(df.columns.tolist())

        df = self.rename_columns(df)   # ✅ ADD HERE

        df = self.feature_engineering(df)
        self.validate_schema(df)
        df = self.handle_outlier(df)


        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]

        preprocessor = self.build_preprocessor()
        X_transformed = preprocessor.fit_transform(X)

        os.makedirs("artifacts", exist_ok=True)
        joblib.dump(preprocessor, self.preprocessor_path)

        if balance_target:
            smote = SMOTE(random_state=42)
            X_transformed, y = smote.fit_resample(X_transformed, y)

        return X_transformed, y

    # ---------------- Transform New Data (Inference) ----------------
    def transform(self, df):
        df = self.normalize_columns(df)
        df = self.rename_columns(df)   # ✅ SAME PLACE

        df = self.feature_engineering(df)
        self.validate_schema(df)
        df = self.handle_outlier(df)

        X = df.drop(columns=[self.target_column])
        preprocessor = joblib.load(self.preprocessor_path)

        return preprocessor.transform(X)
