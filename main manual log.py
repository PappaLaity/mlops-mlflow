import mlflow
from functions.iris_functions import (
    evaluate_iris_lr_model,
    load_iris_data,
    plot_confusion_matrix,
    train_iris_dt_model,
    train_iris_lr_model,
)
from mlflow.models import infer_signature


if __name__ == "__main__":
    lr_params = {
        "solver": "lbfgs",
        # "max_iter": 1000,
        "max_iter": 30,
        "random_state": 8888,
        # "random_state": 42,
    }
    dt_params = {
        "criterion":"entropy",
        "random_state": 8888,
        # "random_state": 42,
    }
    
    X_train, X_test, y_train, y_test = load_iris_data()

    models = [train_iris_lr_model, train_iris_dt_model]
    models_params = [lr_params,dt_params]
    models_names = ["iri_logistic_regression_model","iris_decision_trees_model"]
    mlflow.set_tracking_uri(uri="http://localhost:5000")

    for idx, model in enumerate(models):

        res_model = model(X_train, y_train, models_params[idx])

        y_pred = res_model.predict(X_test)

        accuracy = evaluate_iris_lr_model(y_test, y_pred)

        plot_path = plot_confusion_matrix(y_test, y_pred, models_names[idx])

        print(f"Model Accuracy: {accuracy:.4f}")

        mlflow.set_experiment("Experiment Models Manual-log v2")

        with mlflow.start_run():
            # Log the hyperparameters
            mlflow.log_params(models_params[idx])

            # Log the loss metric
            mlflow.log_metric("accuracy", accuracy)

            # Set a tag that we can use to remind ourselves what this run was for
            mlflow.set_tag("Training Info", models_names[idx])

            mlflow.log_artifact(plot_path, artifact_path="plots")

            # Infer the model signature
            signature = infer_signature(X_train, res_model.predict(X_train))

            # Log the model
            model_info = mlflow.sklearn.log_model(
                sk_model=res_model,
                name=models_names[idx],
                signature=signature,
                input_example=X_train,
                registered_model_name="tracking-model-"+models_names[idx],
            )    


    # ==== Reload model from MLflow ====
    model_name = "tracking-model-iris_decision_trees_model"   # mets le même nom que dans registered_model_name
    model_version = 1 #6

    # Chargement par nom et version
    loaded_model = mlflow.sklearn.load_model(
        model_uri=f"models:/{model_name}/{model_version}"
    )

    # ==== Make predictions ====
    sample = X_test[:5]
    print("Original predictions:", res_model.predict(sample))
    print("Reloaded model predictions:", loaded_model.predict(sample))

    
