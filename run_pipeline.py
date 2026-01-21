from pipeline.data_ingestion import DataIngestion
from pipeline.preprocessing import DataPreprocessor
from pipeline.train import train_model
from pipeline.evaluatemodel import evaluate_model
from sklearn.model_selection import train_test_split

def run_pipeline():
    print("starting Pcos Ml pipeline")
    ingestion=DataIngestion("data/PCOS_data.csv")
    df=ingestion.load_data()
    ingestion.save_raw_data(df)
    print("data ingestion done")

    #Preprocess
    preprocess=DataPreprocessor()
    X,y=preprocess.fit_transform(df)
    print("processing complete")

    #train test
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
    model=train_model(X_train,y_train)

    #evaluate
    evaluate_model(model,X_test,y_test)
    print("model evaluation")


if __name__ == "__main__":
    run_pipeline()
