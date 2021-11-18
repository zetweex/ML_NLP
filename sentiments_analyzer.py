import re
from os import error

import pandas as pd
from sklearn import svm, tree
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression, Perceptron
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier

from review_parser import review_file_to_list

SCORES = ["Model", "Accuracy", "Precision.Macro", "Recall.Macro", "F-Score.Macro", "Precision.Micro", "Recall.Micro", "F-Score.Micro"]
MATRIX_TF = ["True", "False"]
MATRIX_POLARITY = ["positive", "negative"]
MATRIX_RATING = ["1.0", "2.0", "4.0", "5.0"]
MATRIX_DOMAIN = ["books", "dvd", "kitchen_&_housewares", "electronics"]

FOLDER_RESULTS_PATH = "results/"
RESULTS_PATH = "results.txt"
MATRIX_PATH = "confusion_matrix.txt"

#     ______                 __  _                 
#    / ____/_  ______  _____/ /_(_)___  ____  _____
#   / /_  / / / / __ \/ ___/ __/ / __ \/ __ \/ ___/
#  / __/ / /_/ / / / / /__/ /_/ / /_/ / / / (__  ) 
# /_/    \__,_/_/ /_/\___/\__/_/\____/_/ /_/____/  
                                                 

def matrix_to_file(matrix, filename):

    with open(filename, 'w') as f:
        for key, data in matrix.items():
            print(key + "\'s confusion matrix:", file=f)
            try:
                matrix_array = pd.DataFrame([elem for elem in data["matrix"]], index=data["IDX"], columns=data["IDX"])
                print(matrix_array, end='\n'*2, file=f)
            except IndexError:
                error("Index out of range")

def model_trainer(
    model_name,
    model,
    trainingX,
    trainingY,
    testX,
    testY,
    confusion_matrix_container,
    models_datas,
    idx):

    print(f"INFOS: {model_name} training...")

    model.fit(trainingX, trainingY)
    predictions = model.predict(testX)

    if model_name == "Linear Regression":
        predictions = list(map(lambda x: x.round() if x.round() == -1 else 1, predictions))

    confusion_matrix_container[model_name] = {}
    confusion_matrix_container[model_name]["matrix"] = confusion_matrix(testY, predictions)
    confusion_matrix_container[model_name]["IDX"] = idx

    models_datas.append([
        model_name,
        accuracy_score(testY, predictions),
        precision_score(testY, predictions, average='macro'),
        recall_score(testY, predictions, average='macro'),
        f1_score(testY, predictions, average='macro'),
        precision_score(testY, predictions, average='micro'),
        recall_score(testY, predictions, average='micro'),
        f1_score(testY, predictions, average='micro'),
    ])

def my_tokenizer(text):
    # Create a space between special characters 
    text = re.sub("(\\W)"," \\1 ", text)
    # Split based on whitespace
    return re.split("\\s+", text)

#     ____            __     
#    / __ )____  ____/ /_  __
#   / __  / __ \/ __  / / / /
#  / /_/ / /_/ / /_/ / /_/ / 
# /_____/\____/\__,_/\__, /  
#                   /____/   

models_datas = []
confusion_matrix_container = {}
train_Y = {}
train_X = {}
test_Y = {}
test_X = {}
train_X_vectorized = {}
test_X_vectorized = {}
dico, datas_training = {}, {}

vectorizer = CountVectorizer(stop_words='english', analyzer='word', ngram_range=(1, 2), tokenizer=my_tokenizer, lowercase=True)

parsed_files = review_file_to_list("archive")
sentences = [reviews["<review_text>"] for reviews in parsed_files]

# Get different kind of training data (by rating, domain and polarity)
train_Y["polarity"] = [reviews["<polarity>"] for reviews in parsed_files]
train_Y["domain"] = [reviews["<domain>"] for reviews in parsed_files]
train_Y["rating"] = [reviews["<rating>"] for reviews in parsed_files]

train_X["polarity"], test_X["polarity"], train_Y["polarity"], test_Y["polarity"] = train_test_split(sentences, train_Y["polarity"], test_size=0.20)
train_X["domain"], test_X["domain"], train_Y["domain"], test_Y["domain"] = train_test_split(sentences, train_Y["domain"], test_size=0.20)
train_X["rating"], test_X["rating"], train_Y["rating"], test_Y["rating"] = train_test_split(sentences, train_Y["rating"], test_size=0.20)

train_X_vectorized["polarity"] = vectorizer.fit_transform(train_X["polarity"])
test_X_vectorized["polarity"] = vectorizer.transform(test_X["polarity"])

train_X_vectorized["domain"] = vectorizer.fit_transform(train_X["domain"])
test_X_vectorized["domain"] = vectorizer.transform(test_X["domain"])

train_X_vectorized["rating"] = vectorizer.fit_transform(train_X["rating"])
test_X_vectorized["rating"] = vectorizer.transform(test_X["rating"])

# Get a dict like this, to split the training datas depending on the domain:
# dict{
#   books: {sentences: [], polarity: []},
#   dvd: {sentences: [], polarity: []},
#   kitchen_&_housewares: {sentences: [], polarity: []},
#   electronics: {sentences: [], polarity: []},
# }
for review in parsed_files:
    try:
        dico[review["<domain>"]].append(review)
    except KeyError:
        dico[review["<domain>"]] = []

for key, data in dico.items():
    datas_training[key] = {}
    datas_training[key]["text"] = []
    datas_training[key]["polarity"] = []
    for review in data:
        datas_training[key]["text"].append(review["<review_text>"])
        datas_training[key]["polarity"].append(review["<polarity>"])

######## Model: Baseline (DummyClassifier) ########
model_trainer("Baseline_Polarity", DummyClassifier(strategy="most_frequent"), train_X_vectorized["polarity"], train_Y["polarity"], test_X_vectorized["polarity"], test_Y["polarity"], confusion_matrix_container, models_datas, MATRIX_POLARITY)
model_trainer("Baseline_Domain", DummyClassifier(strategy="most_frequent"), train_X_vectorized["domain"], train_Y["domain"], test_X_vectorized["domain"], test_Y["domain"], confusion_matrix_container, models_datas, MATRIX_DOMAIN)
model_trainer("Baseline_Rating", DummyClassifier(strategy="most_frequent"), train_X_vectorized["rating"], train_Y["rating"], test_X_vectorized["rating"], test_Y["rating"], confusion_matrix_container, models_datas, MATRIX_RATING)

######## BONUS Model: Linear Regression ########
model_trainer("Linear Regression", LinearRegression(), train_X_vectorized["polarity"], train_Y["polarity"], test_X_vectorized["polarity"], test_Y["polarity"], confusion_matrix_container, models_datas, MATRIX_POLARITY)

######## Model: Perceptron ######## max_iter: the number of time the model will train

# This loop will browse the dict domain by domain (4 times) and train a Perceptron model for each domain
for domain, datas in datas_training.items():
    datas_training[domain]["train_X"] = []
    datas_training[domain]["train_Y"] = []
    datas_training[domain]["test_X"] = []
    datas_training[domain]["test_Y"] = []

    datas_training[domain]["train_X"], datas_training[domain]["test_X"], datas_training[domain]["train_Y"], datas_training[domain]["test_Y"] = train_test_split(datas["text"], datas["polarity"], test_size=0.20)

    datas_training[domain]["train_X"] = vectorizer.fit_transform(datas_training[domain]["train_X"])
    datas_training[domain]["test_X"] = vectorizer.transform(datas_training[domain]["test_X"])

    model_trainer("Perceptron_MaxIter1." + domain, Perceptron(max_iter = 1), datas_training[domain]["train_X"], datas_training[domain]["train_Y"], datas_training[domain]["test_X"], datas_training[domain]["test_Y"], confusion_matrix_container, models_datas, MATRIX_POLARITY)
    model_trainer("Perceptron_MaxIter5." + domain, Perceptron(max_iter = 5), datas_training[domain]["train_X"], datas_training[domain]["train_Y"], datas_training[domain]["test_X"], datas_training[domain]["test_Y"], confusion_matrix_container, models_datas, MATRIX_POLARITY)
    model_trainer("Perceptron_MaxIter30." + domain, Perceptron(max_iter = 30), datas_training[domain]["train_X"], datas_training[domain]["train_Y"], datas_training[domain]["test_X"], datas_training[domain]["test_Y"], confusion_matrix_container, models_datas, MATRIX_POLARITY)
    model_trainer("Perceptron_MaxIter100." + domain, Perceptron(max_iter = 100), datas_training[domain]["train_X"], datas_training[domain]["train_Y"], datas_training[domain]["test_X"], datas_training[domain]["test_Y"], confusion_matrix_container, models_datas, MATRIX_POLARITY)

######## Model: KNeighborsClassifier ########
model_trainer("KNeighborsClassifier_Neighbors1", KNeighborsClassifier(n_neighbors=1), train_X_vectorized["domain"], train_Y["domain"], test_X_vectorized["domain"], test_Y["domain"], confusion_matrix_container, models_datas, MATRIX_DOMAIN)
model_trainer("KNeighborsClassifier_Neighbors5", KNeighborsClassifier(n_neighbors=5), train_X_vectorized["domain"], train_Y["domain"], test_X_vectorized["domain"], test_Y["domain"], confusion_matrix_container, models_datas, MATRIX_DOMAIN)
model_trainer("KNeighborsClassifier_Neighbors20", KNeighborsClassifier(n_neighbors=20), train_X_vectorized["domain"], train_Y["domain"], test_X_vectorized["domain"], test_Y["domain"], confusion_matrix_container, models_datas, MATRIX_DOMAIN)
model_trainer("KNeighborsClassifier_Neighbors100", KNeighborsClassifier(n_neighbors=100), train_X_vectorized["domain"], train_Y["domain"], test_X_vectorized["domain"], test_Y["domain"], confusion_matrix_container, models_datas, MATRIX_DOMAIN)
model_trainer("KNeighborsClassifier_Neighbors1000", KNeighborsClassifier(n_neighbors=1000), train_X_vectorized["domain"], train_Y["domain"], test_X_vectorized["domain"], test_Y["domain"], confusion_matrix_container, models_datas, MATRIX_DOMAIN)

######## Model: DecisionTreeClassifier ########

model_trainer("DecisionTreeClassifier_MAXDepth1", tree.DecisionTreeClassifier(max_depth = 1), train_X_vectorized["polarity"], train_Y["polarity"], test_X_vectorized["polarity"], test_Y["polarity"], confusion_matrix_container, models_datas, MATRIX_POLARITY)
model_trainer("DecisionTreeClassifier_MAXDepth50", tree.DecisionTreeClassifier(max_depth = 50), train_X_vectorized["polarity"], train_Y["polarity"], test_X_vectorized["polarity"], test_Y["polarity"], confusion_matrix_container, models_datas, MATRIX_POLARITY)
model_trainer("DecisionTreeClassifier_MAXDepth100", tree.DecisionTreeClassifier(max_depth = 100), train_X_vectorized["polarity"], train_Y["polarity"], test_X_vectorized["polarity"], test_Y["polarity"], confusion_matrix_container, models_datas, MATRIX_POLARITY)

model_trainer("DecisionTreeClassifier_MSLeaf1", tree.DecisionTreeClassifier(min_samples_leaf = 2), train_X_vectorized["polarity"], train_Y["polarity"], test_X_vectorized["polarity"], test_Y["polarity"], confusion_matrix_container, models_datas, MATRIX_POLARITY)
model_trainer("DecisionTreeClassifier_MSLeaf10", tree.DecisionTreeClassifier(min_samples_leaf = 50), train_X_vectorized["polarity"], train_Y["polarity"], test_X_vectorized["polarity"], test_Y["polarity"], confusion_matrix_container, models_datas, MATRIX_POLARITY)
model_trainer("DecisionTreeClassifier_MSLeaf25", tree.DecisionTreeClassifier(min_samples_leaf = 100), train_X_vectorized["polarity"], train_Y["polarity"], test_X_vectorized["polarity"], test_Y["polarity"], confusion_matrix_container, models_datas, MATRIX_POLARITY)

model_trainer("DecisionTreeClassifier_Criterion.Gini", tree.DecisionTreeClassifier(criterion = "gini"), train_X_vectorized["polarity"], train_Y["polarity"], test_X_vectorized["polarity"], test_Y["polarity"], confusion_matrix_container, models_datas, MATRIX_POLARITY)
model_trainer("DecisionTreeClassifier_Criterion.Entropy", tree.DecisionTreeClassifier(criterion = "entropy"), train_X_vectorized["polarity"], train_Y["polarity"], test_X_vectorized["polarity"], test_Y["polarity"], confusion_matrix_container, models_datas, MATRIX_POLARITY)

######## Model: Support Vector Machines ########
model_trainer("SupportVectorMachines_LINEAR1", svm.SVC(kernel='linear', C = 1), train_X_vectorized["polarity"], train_Y["polarity"], test_X_vectorized["polarity"], test_Y["polarity"], confusion_matrix_container, models_datas, MATRIX_POLARITY)
model_trainer("SupportVectorMachines_LINEAR100", svm.SVC(kernel='linear', C = 100), train_X_vectorized["polarity"], train_Y["polarity"], test_X_vectorized["polarity"], test_Y["polarity"], confusion_matrix_container, models_datas, MATRIX_POLARITY)
model_trainer("SupportVectorMachines_LINEAR1000", svm.SVC(kernel='linear', C = 1000), train_X_vectorized["polarity"], train_Y["polarity"], test_X_vectorized["polarity"], test_Y["polarity"], confusion_matrix_container, models_datas, MATRIX_POLARITY)

model_trainer("SupportVectorMachines_RBF1", svm.SVC(kernel='rbf', C = 1), train_X_vectorized["polarity"], train_Y["polarity"], test_X_vectorized["polarity"], test_Y["polarity"], confusion_matrix_container, models_datas, MATRIX_POLARITY)
model_trainer("SupportVectorMachines_RBF100", svm.SVC(kernel='rbf', C = 100), train_X_vectorized["polarity"], train_Y["polarity"], test_X_vectorized["polarity"], test_Y["polarity"], confusion_matrix_container, models_datas, MATRIX_POLARITY)
model_trainer("SupportVectorMachines_RBF1000", svm.SVC(kernel='rbf', C = 1000), train_X_vectorized["polarity"], train_Y["polarity"], test_X_vectorized["polarity"], test_Y["polarity"], confusion_matrix_container, models_datas, MATRIX_POLARITY)

model_trainer("SupportVectorMachines_RBF.GAMMA0.01", svm.SVC(kernel='rbf', gamma = 0.01), train_X_vectorized["polarity"], train_Y["polarity"], test_X_vectorized["polarity"], test_Y["polarity"], confusion_matrix_container, models_datas, MATRIX_POLARITY)
model_trainer("SupportVectorMachines_RBF.GAMMA10", svm.SVC(kernel='rbf', gamma = 10), train_X_vectorized["polarity"], train_Y["polarity"], test_X_vectorized["polarity"], test_Y["polarity"], confusion_matrix_container, models_datas, MATRIX_POLARITY)
model_trainer("SupportVectorMachines_RBF.GAMMA200", svm.SVC(kernel='rbf', gamma = 200), train_X_vectorized["polarity"], train_Y["polarity"], test_X_vectorized["polarity"], test_Y["polarity"], confusion_matrix_container, models_datas, MATRIX_POLARITY)

######## Model: Naive Bayes (Multinominal) ########
model_trainer("NaiveBayes_MultinomialNB.Rating", MultinomialNB(), train_X_vectorized["rating"], train_Y["rating"], test_X_vectorized["rating"], test_Y["rating"], confusion_matrix_container, models_datas, MATRIX_RATING)
model_trainer("NaiveBayes_MultinomialNB.Polarity", MultinomialNB(), train_X_vectorized["polarity"], train_Y["polarity"], test_X_vectorized["polarity"], test_Y["polarity"], confusion_matrix_container, models_datas, MATRIX_POLARITY)

######## Knowledge transfer for domain-specific classifiers ########

vectorizer = CountVectorizer(stop_words='english', analyzer='word', ngram_range=(1, 2), tokenizer=my_tokenizer, lowercase=True)

dico, datas_training = {}, {}

for review in parsed_files:
    try:
        dico[review["<domain>"]].append(review)
    except KeyError:
        dico[review["<domain>"]] = []

for key, data in dico.items():
    datas_training[key] = {}
    datas_training[key]["text"] = []
    datas_training[key]["polarity"] = []
    for review in data:
        datas_training[key]["text"].append(review["<review_text>"])
        datas_training[key]["polarity"].append(review["<polarity>"])

for domain, datas in datas_training.items():
    datas_training[domain]["train_X"] = []
    datas_training[domain]["train_Y"] = []
    datas_training[domain]["test_X"] = []
    datas_training[domain]["test_Y"] = []

    datas_training[domain]["train_X"], datas_training[domain]["test_X"], datas_training[domain]["train_Y"], datas_training[domain]["test_Y"] = train_test_split(datas["text"], datas["polarity"], test_size=0.20)

# Train a model with the training data of a specific domain, as well as with the test data of the other domains
for domain, _datas in datas_training.items():
    for s_domain, _s_datas in datas_training.items():
        model_trainer("Perceptron_MaxIter50." + domain + ".TestSet." + s_domain, Perceptron(max_iter = 50), vectorizer.fit_transform(datas_training[domain]["train_X"]), datas_training[domain]["train_Y"], vectorizer.transform(datas_training[s_domain]["test_X"]), datas_training[s_domain]["test_Y"], confusion_matrix_container, models_datas, MATRIX_POLARITY)


### Send datas to the folder results (results.txt and confusion_matrix.txt will be inside)
comp_array = pd.DataFrame(models_datas, index=None, columns=SCORES)

# Models Scores
with open(FOLDER_RESULTS_PATH + RESULTS_PATH, 'w') as f:
    print(comp_array, file=f)

# Confusion Matrix
matrix_to_file(confusion_matrix_container, FOLDER_RESULTS_PATH + MATRIX_PATH)