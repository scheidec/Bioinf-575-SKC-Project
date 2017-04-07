#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 16:45:20 2017

@author: cscheidel
"""

import numpy as np
from sklearn import neighbors


model = neighbors.KNeighborsClassifier(n_neighbors=5)

training = np.load('four_class_5678_train.npy')
testing = np.load('four_class_5678_test.npy')

model = model.fit(training[:, 0:20], training[:, 20])
answers = model.predict(testing[:, 0:20])

accuracy = np.sum(answers == testing[:, 20]) / float(len(testing[:,20]))