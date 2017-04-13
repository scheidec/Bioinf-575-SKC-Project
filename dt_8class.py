#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 16:30:55 2017

@author: cscheidel
"""

from __future__ import print_function
import numpy as np
from sklearn import tree
from sklearn.cross_validation import  cross_val_score

model = tree.DecisionTreeClassifier()

training = np.load('eight_class_train.npy')
testing = np.load('eight_class_test.npy')

model = model.fit(training[:, 0:20], training[:, 20])
answers = model.predict(testing[:, 0:20])

accuracy = np.sum(answers == testing[:, 20]) / float(len(testing[:,20]))

scores = cross_val_score(model, training[:, 0:20], training[:, 20] , cv=5)
print("mean: {:.4f} (std: {:.4f})".format(scores.mean(),
                                          scores.std()),
                                          end="\n\n" )