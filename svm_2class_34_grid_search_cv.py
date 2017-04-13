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

print(__doc__)

# Loading the pfam data features
training = np.load('two_class_34_train.npy')
testing = np.load('two_class_34_test.npy')

training = training.astype(float)
testing = testing.astype(float)

x_train = training[:, 0:20]
y_train = training[:, 20]
x_test = testing[:, 0:20]
y_test = testing[:, 20]
y_train = y_train - 3
y_test = y_test - 3


# Set the parameters by cross-validation
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [50, 25, 20, 15, 10, 1],
                     'C': [10, 20, 50, 100, 125, 150]}]

#scores = ['precision', 'recall']
#
#for score in scores:
print("# Tuning hyper-parameters")
print()

clf = GridSearchCV(SVC(C = 0.01), tuned_parameters, cv = 5)
clf.fit(x_train, y_train)

print("Best parameters set found on development set:")
print()
print(clf.best_params_)
print()
print("Grid scores on development set:")
print()
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))
print()

print("Detailed classification report:")
print()
print("The model is trained on the full development set.")
print("The scores are computed on the full evaluation set.")
print()
y_true, y_pred = y_test, clf.predict(x_test)
print(classification_report(y_true, y_pred))
print()
print("accuracy: %0.4f" % (np.sum(y_pred == y_test)/float(len(y_test))))