from os import error

import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression, Perceptron
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline

SCORES = ["Model", "Accuracy", "Precision", "Recall", "F-Score"]
MATRIX_COL = ["True (-)", "False (+)"]
MATRIX_IDX = ["False (-)", "True (+)"]

def matrix_displayer(matrix):

    for key, data in matrix.items():
        print(key + "\'s confusion matrix:")
        try:
            matrix_array = pd.DataFrame([data[0], data[1]], index=MATRIX_IDX, columns=MATRIX_COL)
            print(matrix_array, end='\n'*2)
        except IndexError:
            error("Index out of range")
            exit(84)

def model_trainer():
    pass

train_X = [
    "i have a stomach ache, i'm in pain",
    "i love my honey",
    "i don't like the rain",
    "i'm so happy because i'm with my love",
    "i don't like spain, because this country smell bad",
    "The Christmas markets are beautiful"
]

train_Y = [-1, 1, -1, 1, -1, 1]

test_X = [
    "i don't like to be sick",
    "i like to discover new countries"
]

test_Y = [-1, 1]

models_datas = []
confusion_matrix_container = {} 

######## Model: Linear Regression ########
print("INFOS: Linear Regression training...")

pipe = make_pipeline(
    CountVectorizer(stop_words='english'),
    LinearRegression()
)

pipe.fit(train_X, train_Y)
test_predictions = pipe.predict(test_X)
test_predictions = list(map(lambda x: x.round() if x.round() == -1 else 1, test_predictions))

confusion_matrix_container["LinearRegression"] = confusion_matrix(test_Y, test_predictions)

models_datas.append([
    "LinearRegression",
    accuracy_score(test_Y, test_predictions),
    precision_score(test_Y, test_predictions, average='micro'),
    recall_score(test_Y, test_predictions, average='micro'),
    f1_score(test_Y, test_predictions, average='micro')
])

######## Model: Perceptron ########
print("INFOS: Perceptron training...")

pipe = make_pipeline(
    CountVectorizer(stop_words='english'),
    Perceptron()
)

pipe.fit(train_X, train_Y)
test_predictions = pipe.predict(test_X)

confusion_matrix_container["Perceptron"] = confusion_matrix(test_Y, test_predictions)

models_datas.append([
    "Perceptron",
    accuracy_score(test_Y, test_predictions),
    precision_score(test_Y, test_predictions, average='micro'),
    recall_score(test_Y, test_predictions, average='micro'),
    f1_score(test_Y, test_predictions, average='micro')
])

######## Model: KNeighborsClassifier ########

print("INFOS: KNeighborsClassifier training...")

pipe = make_pipeline(
    CountVectorizer(stop_words='english'),
    KNeighborsClassifier(n_neighbors=3) #Try with 1, 3, 5, 7, 9 to compare
)

pipe.fit(train_X, train_Y)
test_predictions = pipe.predict(test_X)

confusion_matrix_container["KNeighborsClassifier"] = confusion_matrix(test_Y, test_predictions)

models_datas.append([
    "KNeighborsClassifier",
    accuracy_score(test_Y, test_predictions),
    precision_score(test_Y, test_predictions, average='micro'),
    recall_score(test_Y, test_predictions, average='micro'),
    f1_score(test_Y, test_predictions, average='micro')
])

######## Model: DecisionTreeClassifier ########

print("INFOS: DecisionTreeClassifier training...")

pipe = make_pipeline(
    CountVectorizer(stop_words='english'),
    tree.DecisionTreeClassifier()
)

pipe.fit(train_X, train_Y)
test_predictions = pipe.predict(test_X)

confusion_matrix_container["DecisionTreeClassifier"] = confusion_matrix(test_Y, test_predictions)

models_datas.append([
    "KNeighborsClassifier",
    accuracy_score(test_Y, test_predictions),
    precision_score(test_Y, test_predictions, average='micro'),
    recall_score(test_Y, test_predictions, average='micro'),
    f1_score(test_Y, test_predictions, average='micro')
])

tree.DecisionTreeClassifier()


# Displaying

comp_array = pd.DataFrame(models_datas, index=None, columns=SCORES)

print("###### MODELS SCORES ######", end='\n'*2)
print(comp_array, end='\n'*2)

print("###### CONFUSIONS MATRIX ######", end='\n'*2)
matrix_displayer(confusion_matrix_container)
