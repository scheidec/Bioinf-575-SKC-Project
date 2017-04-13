# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 13:51:03 2017

@author: nstar
"""

from __future__ import print_function
import time
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import cross_val_score

print(__doc__)

# Loading the pfam data features
training = np.load('four_class_5678_train.npy')
testing = np.load('four_class_5678_test.npy')

training = training.astype(float)
testing = testing.astype(float)

x_train = training[:, 0:20]
y_train = training[:, 20]
x_test = testing[:, 0:20]
y_test = testing[:, 20]
y_train = y_train - 5
y_test = y_test - 5

C_range = np.logspace(-2, 3, 10)
gamma_range = np.logspace(-2, 3, 10)
param_grid = dict(gamma=gamma_range, C=C_range)

results = []

for C in C_range:
    for gamma in gamma_range:
        clf = OneVsRestClassifier(SVC(C=C, gamma=gamma))
#        clf.fit(x_train, y_train)
        score = cross_val_score(clf, training[:, 0:20], y_train, cv = 5)
        results.append((C, gamma, np.mean(score), np.std(score)))
        print()
        print("%0.3f (+/-%0.03f) for C = %f  gamma = %f" % (np.mean(score), np.std(score) * 2, C, gamma))

##clf = GridSearchCV(SVC(C = 0.01), tuned_parameters, cv = 5)
#clf = OneVsRestClassifier(SVC(C = 10, gamma = 0.1))
#
#clf.fit(x_train, y_train)
#cross_val_score(clf, training[:, 0:20], y_train, cv = 5)
#
#y_score = clf.predict(testing[:, 0:20])
##y_score_proba = clf.predict_proba(testing[:, 0:20])
#
#accuracy = (np.sum(y_score == y_test)/float(len(y_test)))

