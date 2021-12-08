import pickle
import pandas as pd
from stop_words import get_stop_words
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn import datasets
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
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

# Split des data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
# Entrainement
clf.fit(X_train, y_train)

# sauvegarde avec dumps dumodel entrainer
s = pickle.dumps(clf)
