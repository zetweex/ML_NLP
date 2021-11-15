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

SCORES = ["Model", "Accuracy", "Precision", "Recall", "F-Score"]
MATRIX_TF = ["True", "False"]
MATRIX_POLARITY = ["positive", "negative"]
MATRIX_RATING = ["1.0", "2.0", "4.0", "5.0"]
MATRIX_DOMAIN = ["books", "dvd", "kitchen_&_housewares", "electronics"]

#     ______                 __  _                 
#    / ____/_  ______  _____/ /_(_)___  ____  _____
#   / /_  / / / / __ \/ ___/ __/ / __ \/ __ \/ ___/
#  / __/ / /_/ / / / / /__/ /_/ / /_/ / / / (__  ) 
# /_/    \__,_/_/ /_/\___/\__/_/\____/_/ /_/____/  
                                                 

def matrix_displayer(matrix):

    for key, data in matrix.items():
        print(key + "\'s confusion matrix:")
        try:
            matrix_array = pd.DataFrame([elem for elem in data["matrix"]], index=data["IDX"], columns=data["IDX"])
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
        precision_score(testY, predictions, average='micro'),
        recall_score(testY, predictions, average='micro'),
        f1_score(testY, predictions, average='micro')
    ])

def my_tokenizer(text):
    # Create a space between special characters 
    text=re.sub("(\\W)"," \\1 ", text)

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

vectorizer = CountVectorizer(stop_words='english', analyzer='word', ngram_range=(1, 2), tokenizer=my_tokenizer)

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

######## Model: Baseline (DummyClassifier) ########
model_trainer("Baseline_Polarity", DummyClassifier(strategy="most_frequent"), train_X_vectorized["polarity"], train_Y["polarity"], test_X_vectorized["polarity"], test_Y["polarity"], confusion_matrix_container, models_datas, MATRIX_POLARITY)
model_trainer("Baseline_Domain", DummyClassifier(strategy="most_frequent"), train_X_vectorized["domain"], train_Y["domain"], test_X_vectorized["domain"], test_Y["domain"], confusion_matrix_container, models_datas, MATRIX_DOMAIN)
model_trainer("Baseline_Rating", DummyClassifier(strategy="most_frequent"), train_X_vectorized["rating"], train_Y["rating"], test_X_vectorized["rating"], test_Y["rating"], confusion_matrix_container, models_datas, MATRIX_RATING)
######## BONUS Model: Linear Regression ########
model_trainer("Linear Regression", LinearRegression(), train_X_vectorized["polarity"], train_Y["polarity"], test_X_vectorized["polarity"], test_Y["polarity"], confusion_matrix_container, models_datas, MATRIX_POLARITY)

######## Model: Perceptron ######## max_iter: the number of time the model will train
model_trainer("Perceptron_MaxIter1", Perceptron(max_iter = 1), train_X_vectorized["polarity"], train_Y["polarity"], test_X_vectorized["polarity"], test_Y["polarity"], confusion_matrix_container, models_datas, MATRIX_POLARITY)
model_trainer("Perceptron_MaxIter5", Perceptron(max_iter = 5), train_X_vectorized["polarity"], train_Y["polarity"], test_X_vectorized["polarity"], test_Y["polarity"], confusion_matrix_container, models_datas, MATRIX_POLARITY)
model_trainer("Perceptron_MaxIter15", Perceptron(max_iter = 15), train_X_vectorized["polarity"], train_Y["polarity"], test_X_vectorized["polarity"], test_Y["polarity"], confusion_matrix_container, models_datas, MATRIX_POLARITY)
model_trainer("Perceptron_MaxIter30", Perceptron(max_iter = 30), train_X_vectorized["polarity"], train_Y["polarity"], test_X_vectorized["polarity"], test_Y["polarity"], confusion_matrix_container, models_datas, MATRIX_POLARITY)
model_trainer("Perceptron_MaxIter50", Perceptron(max_iter = 50), train_X_vectorized["polarity"], train_Y["polarity"], test_X_vectorized["polarity"], test_Y["polarity"], confusion_matrix_container, models_datas, MATRIX_POLARITY)

######## Model: KNeighborsClassifier ########
model_trainer("KNeighborsClassifier_Neighbors1", KNeighborsClassifier(n_neighbors=1), train_X_vectorized["domain"], train_Y["domain"], test_X_vectorized["domain"], test_Y["domain"], confusion_matrix_container, models_datas, MATRIX_DOMAIN)
model_trainer("KNeighborsClassifier_Neighbors3", KNeighborsClassifier(n_neighbors=3), train_X_vectorized["domain"], train_Y["domain"], test_X_vectorized["domain"], test_Y["domain"], confusion_matrix_container, models_datas, MATRIX_DOMAIN)
model_trainer("KNeighborsClassifier_Neighbors5", KNeighborsClassifier(n_neighbors=5), train_X_vectorized["domain"], train_Y["domain"], test_X_vectorized["domain"], test_Y["domain"], confusion_matrix_container, models_datas, MATRIX_DOMAIN)
model_trainer("KNeighborsClassifier_Neighbors7", KNeighborsClassifier(n_neighbors=7), train_X_vectorized["domain"], train_Y["domain"], test_X_vectorized["domain"], test_Y["domain"], confusion_matrix_container, models_datas, MATRIX_DOMAIN)
model_trainer("KNeighborsClassifier_Neighbors9", KNeighborsClassifier(n_neighbors=9), train_X_vectorized["domain"], train_Y["domain"], test_X_vectorized["domain"], test_Y["domain"], confusion_matrix_container, models_datas, MATRIX_DOMAIN)

######## Model: DecisionTreeClassifier ########

model_trainer("DecisionTreeClassifier_MAXDepth1", tree.DecisionTreeClassifier(max_depth = 1), train_X_vectorized["polarity"], train_Y["polarity"], test_X_vectorized["polarity"], test_Y["polarity"], confusion_matrix_container, models_datas, MATRIX_POLARITY)
model_trainer("DecisionTreeClassifier_MAXDepth50", tree.DecisionTreeClassifier(max_depth = 50), train_X_vectorized["polarity"], train_Y["polarity"], test_X_vectorized["polarity"], test_Y["polarity"], confusion_matrix_container, models_datas, MATRIX_POLARITY)
model_trainer("DecisionTreeClassifier_MAXDepth100", tree.DecisionTreeClassifier(max_depth = 100), train_X_vectorized["polarity"], train_Y["polarity"], test_X_vectorized["polarity"], test_Y["polarity"], confusion_matrix_container, models_datas, MATRIX_POLARITY)

model_trainer("DecisionTreeClassifier_MSLeaf1", tree.DecisionTreeClassifier(min_samples_leaf = 2), train_X_vectorized["polarity"], train_Y["polarity"], test_X_vectorized["polarity"], test_Y["polarity"], confusion_matrix_container, models_datas, MATRIX_POLARITY)
model_trainer("DecisionTreeClassifier_MSLeaf10", tree.DecisionTreeClassifier(min_samples_leaf = 10), train_X_vectorized["polarity"], train_Y["polarity"], test_X_vectorized["polarity"], test_Y["polarity"], confusion_matrix_container, models_datas, MATRIX_POLARITY)
model_trainer("DecisionTreeClassifier_MSLeaf25", tree.DecisionTreeClassifier(min_samples_leaf = 25), train_X_vectorized["polarity"], train_Y["polarity"], test_X_vectorized["polarity"], test_Y["polarity"], confusion_matrix_container, models_datas, MATRIX_POLARITY)

model_trainer("DecisionTreeClassifier_Criterion.Gini", tree.DecisionTreeClassifier(criterion = "gini"), train_X_vectorized["polarity"], train_Y["polarity"], test_X_vectorized["polarity"], test_Y["polarity"], confusion_matrix_container, models_datas, MATRIX_POLARITY)
model_trainer("DecisionTreeClassifier_Criterion.Entropy", tree.DecisionTreeClassifier(criterion = "entropy"), train_X_vectorized["polarity"], train_Y["polarity"], test_X_vectorized["polarity"], test_Y["polarity"], confusion_matrix_container, models_datas, MATRIX_POLARITY)

######## Model: Support Vector Machines ########
model_trainer("SupportVectorMachines_LINEAR1", svm.SVC(kernel='linear', C = 1), train_X_vectorized["polarity"], train_Y["polarity"], test_X_vectorized["polarity"], test_Y["polarity"], confusion_matrix_container, models_datas, MATRIX_POLARITY)
model_trainer("SupportVectorMachines_LINEAR100", svm.SVC(kernel='linear', C = 100), train_X_vectorized["polarity"], train_Y["polarity"], test_X_vectorized["polarity"], test_Y["polarity"], confusion_matrix_container, models_datas, MATRIX_POLARITY)
model_trainer("SupportVectorMachines_LINEAR1000", svm.SVC(kernel='linear', C = 1000), train_X_vectorized["polarity"], train_Y["polarity"], test_X_vectorized["polarity"], test_Y["polarity"], confusion_matrix_container, models_datas, MATRIX_POLARITY)

model_trainer("SupportVectorMachines_RBF1", svm.SVC(kernel='rbf', C = 1), train_X_vectorized["polarity"], train_Y["polarity"], test_X_vectorized["polarity"], test_Y["polarity"], confusion_matrix_container, models_datas, MATRIX_POLARITY)
model_trainer("SupportVectorMachines_RBF100", svm.SVC(kernel='rbf', C = 100), train_X_vectorized["polarity"], train_Y["polarity"], test_X_vectorized["polarity"], test_Y["polarity"], confusion_matrix_container, models_datas, MATRIX_POLARITY)
model_trainer("SupportVectorMachines_RBF1000", svm.SVC(kernel='rbf', C = 1000), train_X_vectorized["polarity"], train_Y["polarity"], test_X_vectorized["polarity"], test_Y["polarity"], confusion_matrix_container, models_datas, MATRIX_POLARITY)

model_trainer("SupportVectorMachines_RBF.GAMMA1", svm.SVC(kernel='rbf', gamma = 1), train_X_vectorized["polarity"], train_Y["polarity"], test_X_vectorized["polarity"], test_Y["polarity"], confusion_matrix_container, models_datas, MATRIX_POLARITY)
model_trainer("SupportVectorMachines_RBF.GAMMA100", svm.SVC(kernel='rbf', gamma = 100), train_X_vectorized["polarity"], train_Y["polarity"], test_X_vectorized["polarity"], test_Y["polarity"], confusion_matrix_container, models_datas, MATRIX_POLARITY)
model_trainer("SupportVectorMachines_RBF.GAMMA1000", svm.SVC(kernel='rbf', gamma = 1000), train_X_vectorized["polarity"], train_Y["polarity"], test_X_vectorized["polarity"], test_Y["polarity"], confusion_matrix_container, models_datas, MATRIX_POLARITY)


######## Model: Naive Bayes (Multinominal) ########
model_trainer("NaiveBayes_MultinomialNB", MultinomialNB(), train_X_vectorized["rating"], train_Y["rating"], test_X_vectorized["rating"], test_Y["rating"], confusion_matrix_container, models_datas, MATRIX_RATING)

######## Knowledge transfer for domain-specific classifiers ########

# Get a dict like this, to split the training datas depending on the domain:
# dict{
#   books: {sentences: [], polarity: []},
#   dvd: {sentences: [], polarity: []},
#   kitchen_&_housewares: {sentences: [], polarity: []},
#   electronics: {sentences: [], polarity: []},
# }

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


# This loop will browse the dict domain by domain (4 times) and train a Perceptron model for each domain
for domain, datas in datas_training.items():
    datas_training[domain]["train_X"] = []
    datas_training[domain]["train_Y"] = []
    datas_training[domain]["test_X"] = []
    datas_training[domain]["test_Y"] = []

    datas_training[domain]["train_X"], datas_training[domain]["test_X"], datas_training[domain]["train_Y"], datas_training[domain]["test_Y"] = train_test_split(datas["text"], datas["polarity"], test_size=0.20)

    datas_training[domain]["train_X"] = vectorizer.fit_transform(datas_training[domain]["train_X"])
    datas_training[domain]["test_X"] = vectorizer.transform(datas_training[domain]["test_X"])

    model_trainer("Perceptron_MaxIter50." + domain, Perceptron(max_iter = 50), datas_training[domain]["train_X"], datas_training[domain]["train_Y"], datas_training[domain]["test_X"], datas_training[domain]["test_Y"], confusion_matrix_container, models_datas, MATRIX_POLARITY)

### Displaying ###

comp_array = pd.DataFrame(models_datas, index=None, columns=SCORES)

print("###### MODELS SCORES ######", end='\n'*2)
print(comp_array, end='\n'*2)

print("###### CONFUSIONS MATRIX ######", end='\n'*2)
matrix_displayer(confusion_matrix_container)
