#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 15:41:31 2017

@author: cscheidel
"""

import numpy as np
from sklearn import tree
import pydotplus


model = tree.DecisionTreeClassifier()

training = np.load('two_class_12_train.npy')
testing = np.load('two_class_12_test.npy')

model = model.fit(training[:, 0:20], training[:, 20])
answers = model.predict(testing[:, 0:20])

accuracy = np.sum(answers == testing[:, 20]) / float(len(testing[:,20]))

dot_data = tree.export_graphviz(model, out_file=None) 
graph = pydotplus.graph_from_dot_data(dot_data) 
graph.write_pdf("dt_2class_12.pdf") 