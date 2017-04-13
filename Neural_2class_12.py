# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 10:30:16 2017

@author: nstar
"""


import numpy as np
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler  

model = MLPClassifier(solver = 'lbfgs', random_state = 1)

training = np.load('two_class_12_train.npy')
testing = np.load('two_class_12_test.npy')

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
X_train = scaler.transform(x_train)  
X_test = scaler.transform(x_test)

model.fit(x_train, y_train)
answers = model.predict(x_test)

accuracy = np.sum(answers == testing[:, 20]) / float(len(testing[:,20]))

model.fit(x_train, y_train)
cv_score = cross_val_score(model, x_train, y_train, cv = 5)

y_score = model.predict(x_test)
y_score_proba = model.predict_proba(x_test)

accuracy = np.sum(y_score == y_test) / float(len(y_test))


# Compute ROC curve and ROC area for each class
FP, TP, T = roc_curve(y_test, y_score_proba[:,1])
roc_auc = auc(FP, TP)

#Plot of a ROC curve for a specific class
plt.figure()
lw = 2
plt.plot(FP, TP, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()