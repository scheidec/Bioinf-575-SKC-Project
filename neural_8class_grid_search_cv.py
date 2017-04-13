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
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler  

print(__doc__)

# Loading the pfam data features
training = np.load('eight_class_train.npy')
testing = np.load('eight_class_test.npy')

training = training.astype(float)
testing = testing.astype(float)

x_train = training[:, 0:20]
y_train = training[:, 20]
x_test = testing[:, 0:20]
y_test = testing[:, 20]
y_train = y_train - 1
y_test = y_test - 1

scaler = StandardScaler()  
scaler.fit(x_train)  
x_train = scaler.transform(x_train)  
x_test = scaler.transform(x_test)

alpha_range = 10.0 ** -np.arange(1, 7)
param_grid = dict(alpha = alpha_range)

results = []

for alpha in alpha_range:
    clf = OneVsRestClassifier(MLPClassifier(solver = 'lbfgs', random_state = 1, alpha = alpha))
#        clf.fit(x_train, y_train)
    score = cross_val_score(clf, training[:, 0:20], y_train, cv = 5)
    results.append((alpha, np.mean(score), np.std(score)))
    print()
    print("%0.3f (+/-%0.03f) for alpha = %f" % (np.mean(score), np.std(score) * 2, alpha))
        
results_ary = np.asarray(results)
idx = np.argmax(results_ary[:,2])
print(results_ary[idx,:])

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

model = OneVsRestClassifier(MLPClassifier(solver = 'lbfgs', random_state = 1, alpha = results_ary[idx,0]))
model.fit(x_train, y_train)
y_score = model.predict(x_test)
accuracy = (np.sum(y_score == y_test)/float(len(y_test)))
print("accuracy: %0.4f" % accuracy)


