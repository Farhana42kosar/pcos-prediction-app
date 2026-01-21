from sklearn.metrics import accuracy_score,classification_report


def evaluate_model(model,X_test,y_test):
    pred=model.predict(X_test)

    print("Accuracy",accuracy_score(y_test,pred))
    print(classification_report(y_test,pred))