from os import error

import pandas as pd
from sklearn import preprocessing, svm, tree
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression, Perceptron
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.dummy import DummyClassifier

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

def model_trainer(
    model_name,
    model, trainingX,
    trainingY,
    testX,
    testY,
    confusion_matrix_container,
    models_datas):

    print(f"INFOS: {model_name} training...")

    model.fit(trainingX, trainingY)
    predictions = model.predict(testX)

    if model_name == "Linear Regression":
        predictions = list(map(lambda x: x.round() if x.round() == -1 else 1, predictions))

    confusion_matrix_container[model_name] = confusion_matrix(testY, predictions)

    models_datas.append([
        model_name,
        accuracy_score(test_Y, predictions),
        precision_score(test_Y, predictions, average='micro'),
        recall_score(test_Y, predictions, average='micro'),
        f1_score(test_Y, predictions, average='micro')
    ])

train_X = [
    "i have a stomach ache, i'm in pain",
    "i love my honey",
    "i don't like the rain",
    "i'm so happy because i'm with my love",
    "i don't like spain, because this country smell bad",
    "the Christmas markets are beautiful",
    "she did not like Bikhram yoga.",
    "this book was very bad",
    "i really love this book",
    "i don't like cats",
    "you're so cool !, you're my best friend",
    "i don't like to be sick",
    "i like to discover new countries",
    "i don't like dogs",
    "i dit not like vegetable",
    "i love you"
]

train_Y = [-1, 1, -1, 1, -1, 1, -1, -1, 1, -1, 1, -1, 1, -1, -1, 1]

train_X, test_X, train_Y, test_Y = train_test_split(train_X, train_Y, test_size=0.20)

models_datas = []
confusion_matrix_container = {}

vectorizer = CountVectorizer(stop_words='english', analyzer='word', ngram_range=(1, 2))
train_X_vectorized = vectorizer.fit_transform(train_X)
test_X_vectorized = vectorizer.transform(test_X)

######## Model: Baseline (DummyClassifier) ########
model_trainer("Baseline", DummyClassifier(strategy="most_frequent"), train_X_vectorized, train_Y, test_X_vectorized, test_Y, confusion_matrix_container, models_datas)

######## BONUS Model: Linear Regression ########
model_trainer("Linear Regression", LinearRegression(), train_X_vectorized, train_Y, test_X_vectorized, test_Y, confusion_matrix_container, models_datas)

######## Model: Perceptron ######## max_iter: the number of time the model will train
model_trainer("Perceptron_MaxIter1", Perceptron(max_iter = 1), train_X_vectorized, train_Y, test_X_vectorized, test_Y, confusion_matrix_container, models_datas)
model_trainer("Perceptron_MaxIter5", Perceptron(max_iter = 5), train_X_vectorized, train_Y, test_X_vectorized, test_Y, confusion_matrix_container, models_datas)
model_trainer("Perceptron_MaxIter15", Perceptron(max_iter = 15), train_X_vectorized, train_Y, test_X_vectorized, test_Y, confusion_matrix_container, models_datas)
model_trainer("Perceptron_MaxIter30", Perceptron(max_iter = 30), train_X_vectorized, train_Y, test_X_vectorized, test_Y, confusion_matrix_container, models_datas)
model_trainer("Perceptron_MaxIter50", Perceptron(max_iter = 50), train_X_vectorized, train_Y, test_X_vectorized, test_Y, confusion_matrix_container, models_datas)

######## Model: KNeighborsClassifier ########
model_trainer("KNeighborsClassifier_Neighbors1", KNeighborsClassifier(n_neighbors=1), train_X_vectorized, train_Y, test_X_vectorized, test_Y, confusion_matrix_container, models_datas)
model_trainer("KNeighborsClassifier_Neighbors3", KNeighborsClassifier(n_neighbors=3), train_X_vectorized, train_Y, test_X_vectorized, test_Y, confusion_matrix_container, models_datas)
model_trainer("KNeighborsClassifier_Neighbors5", KNeighborsClassifier(n_neighbors=5), train_X_vectorized, train_Y, test_X_vectorized, test_Y, confusion_matrix_container, models_datas)
model_trainer("KNeighborsClassifier_Neighbors7", KNeighborsClassifier(n_neighbors=7), train_X_vectorized, train_Y, test_X_vectorized, test_Y, confusion_matrix_container, models_datas)
model_trainer("KNeighborsClassifier_Neighbors9", KNeighborsClassifier(n_neighbors=9), train_X_vectorized, train_Y, test_X_vectorized, test_Y, confusion_matrix_container, models_datas)

######## Model: DecisionTreeClassifier ########

model_trainer("DecisionTreeClassifier_MAXDepth1", tree.DecisionTreeClassifier(max_depth = 1), train_X_vectorized, train_Y, test_X_vectorized, test_Y, confusion_matrix_container, models_datas)
model_trainer("DecisionTreeClassifier_MAXDepth50", tree.DecisionTreeClassifier(max_depth = 50), train_X_vectorized, train_Y, test_X_vectorized, test_Y, confusion_matrix_container, models_datas)
model_trainer("DecisionTreeClassifier_MAXDepth100", tree.DecisionTreeClassifier(max_depth = 100), train_X_vectorized, train_Y, test_X_vectorized, test_Y, confusion_matrix_container, models_datas)

model_trainer("DecisionTreeClassifier_MSLeaf1", tree.DecisionTreeClassifier(min_samples_leaf = 2), train_X_vectorized, train_Y, test_X_vectorized, test_Y, confusion_matrix_container, models_datas)
model_trainer("DecisionTreeClassifier_MSLeaf10", tree.DecisionTreeClassifier(min_samples_leaf = 10), train_X_vectorized, train_Y, test_X_vectorized, test_Y, confusion_matrix_container, models_datas)
model_trainer("DecisionTreeClassifier_MSLeaf25", tree.DecisionTreeClassifier(min_samples_leaf = 25), train_X_vectorized, train_Y, test_X_vectorized, test_Y, confusion_matrix_container, models_datas)

model_trainer("DecisionTreeClassifier_Criterion.Gini", tree.DecisionTreeClassifier(criterion = "gini"), train_X_vectorized, train_Y, test_X_vectorized, test_Y, confusion_matrix_container, models_datas)
model_trainer("DecisionTreeClassifier_Criterion.Entropy", tree.DecisionTreeClassifier(criterion = "entropy"), train_X_vectorized, train_Y, test_X_vectorized, test_Y, confusion_matrix_container, models_datas)

######## Model: Support Vector Machines ########
model_trainer("SupportVectorMachines_LINEAR1", svm.SVC(kernel='linear', C = 1), train_X_vectorized, train_Y, test_X_vectorized, test_Y, confusion_matrix_container, models_datas)
model_trainer("SupportVectorMachines_LINEAR100", svm.SVC(kernel='linear', C = 100), train_X_vectorized, train_Y, test_X_vectorized, test_Y, confusion_matrix_container, models_datas)
model_trainer("SupportVectorMachines_LINEAR1000", svm.SVC(kernel='linear', C = 1000), train_X_vectorized, train_Y, test_X_vectorized, test_Y, confusion_matrix_container, models_datas)

model_trainer("SupportVectorMachines_RBF1", svm.SVC(kernel='rbf', C = 1), train_X_vectorized, train_Y, test_X_vectorized, test_Y, confusion_matrix_container, models_datas)
model_trainer("SupportVectorMachines_RBF100", svm.SVC(kernel='rbf', C = 100), train_X_vectorized, train_Y, test_X_vectorized, test_Y, confusion_matrix_container, models_datas)
model_trainer("SupportVectorMachines_RBF1000", svm.SVC(kernel='rbf', C = 1000), train_X_vectorized, train_Y, test_X_vectorized, test_Y, confusion_matrix_container, models_datas)

model_trainer("SupportVectorMachines_RBF.GAMMA1", svm.SVC(kernel='rbf', gamma = 1), train_X_vectorized, train_Y, test_X_vectorized, test_Y, confusion_matrix_container, models_datas)
model_trainer("SupportVectorMachines_RBF.GAMMA100", svm.SVC(kernel='rbf', gamma = 100), train_X_vectorized, train_Y, test_X_vectorized, test_Y, confusion_matrix_container, models_datas)
model_trainer("SupportVectorMachines_RBF.GAMMA1000", svm.SVC(kernel='rbf', gamma = 1000), train_X_vectorized, train_Y, test_X_vectorized, test_Y, confusion_matrix_container, models_datas)


# ######## Model: Naive Bayes ########

# from sklearn.naive_bayes import MultinomialNB

# print("INFOS: Naive Bayes training...")
# # https://www.cours-gratuit.com/tutoriel-python/tutoriel-python-matriser-la-classification-nave-baysienne-avec-scikit-learn
# #Take 20% of the datasets to train the model
# x_train, x_test, y_train, y_test = train_test_split(train_X, train_Y, test_size=0.20)


# confusion_matrix_container["Naive_Bayes"] = confusion_matrix(y_test, test_predictions)

# models_datas.append([
#     "Naive_Bayes",
#     accuracy_score(test_Y, test_predictions),
#     precision_score(test_Y, test_predictions, average='micro'),
#     recall_score(test_Y, test_predictions, average='micro'),
#     f1_score(test_Y, test_predictions, average='micro')
# ])

# Knowledge transfer for domain-specific classifiers

# Displaying

comp_array = pd.DataFrame(models_datas, index=None, columns=SCORES)

print("###### MODELS SCORES ######", end='\n'*2)
print(comp_array, end='\n'*2)

print("###### CONFUSIONS MATRIX ######", end='\n'*2)
matrix_displayer(confusion_matrix_container)
