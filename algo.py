import pickle
import pandas as pd
from stop_words import get_stop_words
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn import datasets
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
# Clean des données
clf = make_pipeline(
    TfidfVectorizer(stop_words=get_stop_words('en')),
    OneVsRestClassifier(svm.SVC(kernel='linear', probability=True))
)
# Importation du dataset
dt = pd.read_csv("dataset/labels.csv")
# selection features et target
y = dt['class']
X = dt.drop('class', axis=1)


# Entrainement
clf.fit(X, y)

# sauvegarde avec dumps dumodel entrainer
s = pickle.dumps(clf)
