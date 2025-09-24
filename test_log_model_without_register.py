import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Example dataset
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

mlflow.set_experiment("Test Log without Register")

input_example = X[:5]

with mlflow.start_run():
    mlflow.sklearn.log_model(
        sk_model=model,
        name="log_reg_model",
        input_example=input_example
    )
