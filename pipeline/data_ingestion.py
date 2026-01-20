import pandas as pd
import os
from datetime import datetime

class DataIngestion:
#start
    def __init__(self, raw_data_path="data/PCOS_data.csv"):
        self.raw_data_path=raw_data_path
#function to load new dataset
    def load_data(self):
        print("[Data ingestion]loading dataset")
        if not os.path.exists(self.raw_data_path):
            raise FileNotFoundError(f"dataset not found at{self.raw_data_path}")
        

        df=pd.read_csv(self.raw_data_path)
        print(f"dataset loaded successfully. Shape: {df.shape}")

        return df
    
#To save new dataset
    def save_raw_data(self,df):
        os.makedirs("data", exist_ok=True)


        timestamp=datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path=f"data/PCOS_data_{timestamp}.csv"
        df.to_csv(file_path,index=False)
        print(f"Raw data saved at: {file_path}")

        return file_path


    #     self.raw_data_path = raw_data_path

    # def load_data(self):
    #     print("📥 [Data Ingestion] Loading dataset...")

    #     if not os.path.exists(self.raw_data_path):
    #         raise FileNotFoundError(f"Dataset not found at {self.raw_data_path}")

    #     df = pd.read_csv(self.raw_data_path)

    #     print(f"✅ Dataset loaded successfully. Shape: {df.shape}")
    #     return df

    # def save_raw_data(self, df):
    #     os.makedirs("data/raw", exist_ok=True)

    #     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #     file_path = f"data/raw/pcos_{timestamp}.csv"

    #     df.to_csv(file_path, index=False)

    #     print(f"💾 Raw data saved at: {file_path}")

    #     return file_path
