# -*- coding: utf-8 -*-
"""
Created on Wed Apr 05 15:27:20 2017

@author: linkw
"""

import numpy as np
from sklearn import svm

model = svm.SVC(gamma = 0.01, C = 100)

training = np.load('two_class_12_train.npy')
testing = np.load('two_class_12_test.npy')

model.fit(training[:, 0:20], training[:, 20])
answers = model.predict(testing[:, 0:20])

accuracy = np.sum(answers == testing[:, 20]) / float(len(testing[:,20]))