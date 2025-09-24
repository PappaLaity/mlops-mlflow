from mlflow import MlflowClient

client = MlflowClient()

# Ajouter description to the model globally
# client.update_registered_model(
#     name="tracking-model-iris_decision_trees_model",
#     description="Decision Trees model trained on Iris dataset"
# )

# Ajouter tags à une version spécifique
# client.set_model_version_tag(
#     name="tracking-model-iris_decision_trees_model",
#     version=6,
#     key="framework",
#     value="scikit-learn"
# )

client.set_model_version_tag(
    name="tracking-model-iris_decision_trees_model",
    version=5,
    key="stage",
    value="Staging"
)
client.set_model_version_tag(
    name="tracking-model-iris_decision_trees_model",
    version=6,
    key="stage",
    value="Production"
)
