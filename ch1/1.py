__author__ = 'Alok'

import time

from sklearn import svm
from sklearn import datasets
from sklearn.externals import joblib

THETA_FILE = "thetas/theta_file.txt"


def train_model():
    digits = datasets.load_digits()
    clf = svm.SVC(gamma=0.001, C=100.)
    t1 = - time.time()
    clf.fit(digits.data[:-1], digits.target[:-1])
    print("time spent fitting model: ", t1 + time.time())
    joblib.dump(clf, THETA_FILE)

try:
    clf = joblib.load(THETA_FILE)
except FileNotFoundError:
    train_model()
    clf = joblib.load(THETA_FILE)

digits = datasets.load_digits()
t1 = -time.time()
x = clf.predict(digits.data[102])
print("prediction time: ", t1 + time.time())

print(x)
