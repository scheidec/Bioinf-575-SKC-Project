#!/urs/env/bin python

""" Bagging meta-estimator for 2 classes (1,2)
"""


__author__ = 'Nanxiang Zhao'
__email__ = 'samzhao@umich.edu'


import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import cross_val_score




# training the model
clf = BaggingClassifier(max_samples = 0.65)

# loading the data
training = np.load('two_class_12_train.npy')
testing = np.load('two_class_12_test.npy')

# cross-validation
scores = cross_val_score(clf, training[:, 0:20], training[:, 20], cv=5)
print(scores.mean())
print(scores.std())


# predict on testing data
clf = clf.fit(training[:, 0:20], training[:, 20])
answers = clf.predict(testing[:, 0:20])
accuracy = np.sum(answers == testing[:, 20]) / float(len(testing[:,20]))
score_prob = clf.predict_proba(testing[:, 0:20])
print(score_prob)
print(accuracy)