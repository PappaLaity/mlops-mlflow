import mlflow
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from functions.iris_functions import evaluate_iris_lr_model, load_iris_data, train_iris_lr_model

if __name__ == "__main__":
    mlflow.autolog()
    mlflow.set_tracking_uri(uri="http://localhost:2500")
    with mlflow.start_run():
        # X_train, X_test, y_train, y_test = load_iris_data()

        # lr_model = train_iris_lr_model(X_train,y_train)

        # y_pred = lr_model.predict(X_test)

        # accuracy = evaluate_iris_lr_model(y_test,y_pred)

        # print(f"Model Accuracy: {accuracy:.4f}")
        db = load_diabetes()
        X_train, X_test, y_train, y_test = train_test_split(db.data, db.target)

        rf = RandomForestRegressor(n_estimators=100, max_depth=6, max_features=3)
        # MLflow triggers logging automatically upon model fitting
        rf.fit(X_train, y_train)
