#!/urs/env/bin python

""" Ada Boost for 4 classes (5,6,7,8)
"""


__author__ = 'Nanxiang Zhao'
__email__ = 'samzhao@umich.edu'


import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score

training = np.load('four_class_5678_train.npy')
testing = np.load('four_class_5678_test.npy')

# train the model
clf = AdaBoostClassifier(n_estimators=200)


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