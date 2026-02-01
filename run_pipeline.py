import os
import glob
import shutil
from datetime import datetime

import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

from pipeline.data_ingestion import DataIngestion
from pipeline.preprocessing import DataPreprocessor
from pipeline.train import train_model
from pipeline.evaluatemodel import evaluate_model
from pipeline.save_model import SaveArtifacts
from pipeline.data_drift import detect_data_drift


# ---------------- Folders ----------------
NEW_DATA_FOLDER = "data/new/"
PROCESSED_DATA_FOLDER = "data/processed/"
ARTIFACTS_FOLDER = "artifacts/"


def run_pipeline():
    print("🚀 Starting PCOS ML pipeline automation...\n")

    # Get all new CSV files
    new_files = glob.glob(os.path.join(NEW_DATA_FOLDER, "*.csv"))

    if not new_files:
        print("⚠️ No new datasets found. Exiting pipeline.")
        return

    # Process each dataset
    for file_path in new_files:
        try:
            print(f"\n📥 Processing dataset: {file_path}")

            # ----------------- Data Ingestion -----------------
            ingestion = DataIngestion(file_path)
            df = ingestion.load_data()
            ingestion.save_raw_data(df)
            print(f"✅ Data loaded | Shape: {df.shape}")

            # ----------------- Data Drift Detection -----------------
            processed_files = glob.glob(os.path.join(PROCESSED_DATA_FOLDER, "*.csv"))

            if processed_files:
                old_df = pd.read_csv(processed_files[-1])  # latest processed data

                drift_detected, drift_report = detect_data_drift(
                    old_df.select_dtypes(include="number"),
                    df.select_dtypes(include="number")
                )

                if drift_detected:
                    print("⚠️ Data drift detected! Retraining will proceed.")
                else:
                    print("✅ No significant data drift detected.")
            else:
                print("ℹ️ No previous data found. Skipping drift check.")

            # ----------------- Preprocessing -----------------
            preprocess = DataPreprocessor()
            X, y = preprocess.fit_transform(df, balance_target=False)
            print("✅ Preprocessing complete")
            print("Training feature",X.columns.tolist())

            # ----------------- Train/Test Split -----------------
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            # ----------------- Apply SMOTE (TRAIN ONLY) -----------------
            smote = SMOTE(random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            print("✅ SMOTE applied on training data only")

            # ----------------- Model Training -----------------
            model = train_model(X_train, y_train)
            print("✅ Model training complete")

            # ----------------- Evaluation -----------------
            metrics = evaluate_model(model, X_test, y_test)
            print("✅ Model evaluation complete")

            # ----------------- Save Artifacts -----------------
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            saver = SaveArtifacts(save_dir=ARTIFACTS_FOLDER)

            model_name = f"pcos_model_{timestamp}.pkl"
            saver.save_model(model, model_name)

            print(f"✅ Artifacts saved with timestamp: {timestamp}")

            # ----------------- Move Processed Dataset -----------------
            os.makedirs(PROCESSED_DATA_FOLDER, exist_ok=True)
            dest_file = os.path.join(
                PROCESSED_DATA_FOLDER, os.path.basename(file_path)
            )
            shutil.move(file_path, dest_file)

            print(f"📦 Dataset moved to processed folder: {dest_file}")

        except Exception as e:
            print(f"❌ Error processing {file_path}")
            print(e)

    print("\n🎉 Pipeline execution completed successfully!")


if __name__ == "__main__":
    run_pipeline()
