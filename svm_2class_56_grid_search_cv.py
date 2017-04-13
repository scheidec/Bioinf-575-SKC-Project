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

print(__doc__)

# Loading the pfam data features
training = np.load('two_class_56_train.npy')
testing = np.load('two_class_56_test.npy')

training = training.astype(float)
testing = testing.astype(float)

x_train = training[:, 0:20]
y_train = training[:, 20]
x_test = testing[:, 0:20]
y_test = testing[:, 20]
y_train = y_train - 5
y_test = y_test - 5


# Set the parameters by cross-validation
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1000, 500, 200, 100, 10, 1, 0.1, 1e-2],
                     'C': [1, 10, 20, 25, 30, 50]}]

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

#plt.figure(figsize=(8, 6))
#plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
#plt.imshow(means, interpolation='nearest', cmap=plt.cm.hot)
#plt.xlabel('gamma')
#plt.ylabel('C')
#plt.colorbar()
#plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
#plt.yticks(np.arange(len(C_range)), C_range)
#plt.title('Validation accuracy')
#plt.show()