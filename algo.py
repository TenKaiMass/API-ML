from joblib import dump
import pandas as pd
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import re
from nltk.corpus import stopwords
# import nltk


# Importation du dataset
dt = pd.read_csv("dataset/labels.csv")
dt = dt.sample(100)
# Clean des données
# #Gerer les tweet
tweet = dt['tweet']
# # # on met tout en minuscule
tweet = tweet.str.lower()
# # # enleve les cara speciaux
tweet = tweet.apply(lambda x: re.sub("[^a-z\s]", "", x))
# # #enleve les #
tweet = tweet.str.replace("#", " ")
# # # Enleve les stop word
# nltk.download('stopwords')
stropwords = set(stopwords.words("english"))
tweet = tweet.apply(lambda x: " ".join(
    w for w in x.split() if w not in stropwords))
dt['tweet'] = tweet
# print(dt.head(5))
# selection features et target
y = dt['class']
X = tweet
# print(dt.info())
# Split des data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
clf = make_pipeline(
    TfidfVectorizer(stop_words=stopwords.words("english")),
    OneVsRestClassifier(svm.SVC(kernel='linear', probability=True))

)
# Entrainement
clf.fit(X_train, y_train)

# sauvegarde avec dumps dumodel entrainer
dump(clf, "algo.joblib")
