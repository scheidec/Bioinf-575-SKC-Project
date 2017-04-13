# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 13:14:48 2017

@author: nstar
"""
from __future__ import print_function
import numpy as np
from sklearn import svm
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score

n_classes = 2
model = svm.SVC(gamma = 21.544347, C = 278.25594, probability = True)

training = np.load('four_class_5678_train.npy')
testing = np.load('four_class_5678_test.npy')

training = training.astype(float)
testing = testing.astype(float)

y_train = training[:, 20]
y_train = y_train - 1
y_test = testing[:, 20]
y_test = y_test - 1


model.fit(training[:, 0:20], y_train)
cross_val_score(model, training[:, 0:20], y_train, cv = 5)

y_score = model.predict(testing[:, 0:20])
y_score_proba = model.predict_proba(testing[:, 0:20])

accuracy = np.sum(y_score == y_test) / float(len(y_test))
print("accuracy: %f" % accuracy)
#
## Compute ROC curve and ROC area for each class
#FP, TP, T = roc_curve(y_test, y_score_proba[:,1])
#roc_auc = auc(FP, TP)
#
##Plot of a ROC curve for a specific class
#plt.figure()
#lw = 2
#plt.plot(FP, TP, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
#plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
#plt.xlim([0.0, 1.0])
#plt.ylim([0.0, 1.05])
#plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate')
#plt.title('Receiver operating characteristic example')
#plt.legend(loc="lower right")
#plt.show()