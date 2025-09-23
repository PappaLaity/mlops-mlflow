import pandas as pd
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def load_iris_data():
        
    iris = datasets.load_iris()

    data = pd.DataFrame(iris.data, columns=iris.feature_names)
    target = iris.target

    return train_test_split(data, target, test_size=0.3, random_state=42)



def train_iris_lr_model(X_train,y_train,params):
    
    lr_model = LogisticRegression(**params)

    lr_model.fit(X_train,y_train)
    
    return lr_model


def evaluate_iris_lr_model(y_test, y_pred):
    return accuracy_score(y_test, y_pred) 