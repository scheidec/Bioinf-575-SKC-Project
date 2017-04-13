# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 13:51:03 2017

@author: nstar
"""

from __future__ import print_function

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC

print(__doc__)

# Loading the pfam data features
training = np.load('two_class_12_train.npy')
testing = np.load('two_class_12_test.npy')


x_train = training[:, 0:20]
y_train = training[:, 20]
x_test = testing[:, 0:20]
y_test = testing[:, 20]

# Set the parameters by cross-validation
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [100, 10, 1, 0.1, 1e-2, 1e-3, 1e-4],
                     'C': [0.01, 0.1, 1, 10, 20, 50, 100]},
                    {'kernel': ['linear'], 'C': [0.01, 0.1, 1, 10, 20, 50, 100]}]

scores = ['precision', 'recall']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=5,
                       scoring='%s_macro' % score)
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
