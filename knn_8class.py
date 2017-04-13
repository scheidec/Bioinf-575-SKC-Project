#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 16:46:31 2017

@author: cscheidel
"""

from __future__ import print_function
import numpy as np
from sklearn import neighbors
from sklearn.cross_validation import  cross_val_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

model = neighbors.KNeighborsClassifier(n_neighbors=8)

training = np.load('eight_class_train.npy')
testing = np.load('eight_class_test.npy')

training = training.astype(float)
testing = testing.astype(float)

#training[:, 20] = training[:, 20] - 1
#testing[:, 20] = testing[:, 20] - 1

model = model.fit(training[:, 0:20], training[:, 20])
answers = model.predict(testing[:, 0:20])
answers_proba = model.predict_proba(testing[:, 0:20])

accuracy = np.sum(answers == testing[:, 20]) / float(len(testing[:,20]))

scores = cross_val_score(model, training[:, 0:20], training[:, 20] , cv=5)
print("mean: {:.4f} (std: {:.4f})".format(scores.mean(),
                                          scores.std()),
                                          end="\n\n" )

# Compute ROC curve and AUC for each class
FP, TP, T = roc_curve(testing[:, 20], answers_proba[:,1])
roc_auc = auc(FP, TP)

#Plot of a ROC curve for a specific class
plt.figure()
lw = 2
plt.plot(FP, TP, color='darkorange', lw=lw, label='ROC curve (area = %0.4f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic knn_4class_1234')
plt.legend(loc="lower right")
plt.show()