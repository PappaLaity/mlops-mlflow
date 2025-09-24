import requests
import json

url = "http://127.0.0.1:5001/invocations"

# Exemple d’inputs (Iris dataset)
data = {
    "columns": ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"],
    "data": [[5.1, 3.5, 1.4, 0.2],
             [6.2, 3.4, 5.4, 2.3]]
}

response = requests.post(url, json=data)
print("Predictions:", response.json())

# mlflow models serve -m "runs:/e8dbffb0b1954492bf4bb8457a3c919a/tracking-model-iris_decision_trees_model" -p 5000 --no-conda

