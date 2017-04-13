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
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler  

print(__doc__)

# Loading the pfam data features
training = np.load('two_class_78_train.npy')
testing = np.load('two_class_78_test.npy')

training = training.astype(float)
testing = testing.astype(float)

x_train = training[:, 0:20]
y_train = training[:, 20]
x_test = testing[:, 0:20]
y_test = testing[:, 20]
y_train = y_train - 7
y_test = y_test - 7

scaler = StandardScaler()  
scaler.fit(x_train)  
x_train = scaler.transform(x_train)  
x_test = scaler.transform(x_test)

# Set the parameters by cross-validation
tuned_parameters = [{'alpha': 10.0 ** -np.arange(1, 7)}]

#scores = ['precision', 'recall']
#
#for score in scores:
print("# Tuning hyper-parameters")
print()

clf = GridSearchCV(MLPClassifier(solver = 'lbfgs', random_state = 1), tuned_parameters, cv = 5)
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