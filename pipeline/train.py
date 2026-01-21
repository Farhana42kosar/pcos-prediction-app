from sklearn.linear_model import LogisticRegression
import os
import joblib

def train_model(X_train,y_train):
    print("Training started")


    #train model
    model=LogisticRegression(max_iter=1000)
    model.fit(X_train,y_train)
    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(model, "artifacts/model.pkl")
    return model

    




    




