#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 16:24:27 2017

@author: cscheidel
"""

import numpy as np
from sklearn import tree


model = tree.DecisionTreeClassifier()

training = np.load('two_class_56_train.npy')
testing = np.load('two_class_56_test.npy')

model = model.fit(training[:, 0:20], training[:, 20])
answers = model.predict(testing[:, 0:20])

accuracy = np.sum(answers == testing[:, 20]) / float(len(testing[:,20]))