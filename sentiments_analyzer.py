import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
import csv
import re
#Need version 3.9 of Python

#Librairies used : pandas, sklearn
#pip install pandas, sklearn

#STEP 1 train our model

# DATASET_PATH = "archive/dataset.csv"

# # CSV parser will get the sentitment and the sentence associate, from the csv file (dataset).
# # We'll finally have an array of sentence and an array of sentiment?
# with open(DATASET_PATH, encoding='latin-1') as my_file:
#     csv_reader = csv.reader(my_file, delimiter=',')
#     sentiment, sentence = zip(*((1 if int(row[0]) == 0 else -1, re.sub(r'[^A-z]+', ' ',row[len(row) - 1])) for row in csv_reader))

#----------------------------------

# TEST #
# tmp = pd.DataFrame([model.coef_], columns=sorted(vectorizer.vocabulary_.keys()), index=['coef']).T

# Here we create the pipeline
# First we create our vectorizer, who is able to remove the stop_word
# We are using the LinearRegression() model from sklearn

# Tokenization / Vectorization
# This Countvectorizer will transform our sentences in vector
# will split each sentence in words (without stop_word),
# and will finally create a matrix like this.
#
#        amazing  annoying  beautiful  gift  hate  idiot  love  stupid  ugly
# text1        1         0          1     1     1      0     1       0     0
# text2        0         1          0     0     1      1     0       1     1

sentence = [
    "i have a stomach ache, i'm in pain",
    "i love my honey",
    "i don't like the rain",
    "i'm so happy because i'm with my love",
    "i don't like spain, because this country smell bad",
    "The Christmas markets are beautiful"
]

sentiment = [
    -1,
    1,
    -1,
    1,
    -1,
    1
]


pipe = make_pipeline(
    CountVectorizer(stop_words='english'),
    LinearRegression()
)

# Here we give to our model:
# - Our sentence array
# - Our sentiment array (composed with -1 and 1)
pipe.fit(sentence, sentiment)

#negative amazon commentary about an harry potter book
print(pipe.predict(["I am very disappointed, I received a book with damaged corners"]))

#positive amazon commentary about an harry potter book
print(pipe.predict(["I'm frankly not disappointed, it's quite the opposite, I could not have hoped for better."]))
print(pipe.predict(["i don't like to be sick"]))
print(pipe.predict(["i like to discover new countries"]))

#STEP 2 Evaluate this model