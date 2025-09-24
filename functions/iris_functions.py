import os
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt


def load_iris_data():
        
    iris = datasets.load_iris()

    data = pd.DataFrame(iris.data, columns=iris.feature_names)
    target = iris.target

    return train_test_split(data, target, test_size=0.3, random_state=42)



def train_iris_lr_model(X_train,y_train,params):
    
    lr_model = LogisticRegression(**params)

    lr_model.fit(X_train,y_train)
    
    return lr_model

def train_iris_dt_model(X_train,y_train,params):
    
    model = DecisionTreeClassifier(**params)

    model.fit(X_train,y_train)
    
    return model



def evaluate_iris_lr_model(y_test, y_pred):
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    return accuracy_score(y_test, y_pred) 


def plot_confusion_matrix(y_test, y_pred,model_name):
    """Function to plot the confusion matrix."""
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8,6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(set(y_test)))
    plt.xticks(tick_marks, set(y_test))
    plt.yticks(tick_marks, set(y_test))

    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    # plt.show()

    os.makedirs("plots", exist_ok=True)
    plot_path = "plots/"+model_name+"_confusion_matrix.png"
    plt.savefig(plot_path)
    plt.close()
    return plot_path