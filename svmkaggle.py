import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.grid_search import GridSearchCV
import csv

# get data
data = np.genfromtxt('./data/kaggle_train_wc.csv', delimiter = ',')
(row, col) = data.shape
# format data
x, y = data[:, :col-1], data[:, col-1]

# cross validation
xtest = np.genfromtxt('./data/kaggle_test_wc.csv', delimiter = ',')
# xtrain, xtest, ytrain, ytest = train_test_split(
#     x, y, test_size = 0.2)
svr = SVC(C=0.0001, kernel='linear')
model = svr.fit(x, y)
test_pred = list(model.predict(xtest))

myfile = open('svc_pred.csv', 'wb')
wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
wr.writerow(test_pred)

# final_prediction = []
# for i in range(len(predictions[0])):
#     count = 0;
#     for j in range(5):
#         if predictions[j][i] > 0:
#             count += 1
#     final_prediction.append(count)
    # if count > 2:
    #     final_prediction.append(1)
    # else:
    #     final_prediction.append(0)

# print final_prediction

# specify parameters
# C_range = (0.1, 1)
# gamma_range = 10.0 ** np.arange(-2, 4)
# kernel_options = ('rbf', 'poly')
# coef0_range = range(1,3)
# parameters = dict(C=C_range, gamma=gamma_range,
#                   kernel=kernel_options, coef0=coef0_range)

# SVM of 0.89124 is C = 1.0, coef0=1, kernel='linear'
# New Best VSM of 0.90024 is C = 0.0001, coef0=1, kernel='linear'
# run svm
# for i in range(10):
#     for j in range(8):
#         svr = SVC(C=1.0, coef0=i+1, degree=j+1, kernel='poly')
#         svr_scores = cross_val_score(svr, x, y)
#         print "poly: ", "coef0: ", i+1, "degree: ", j+1, "score: ", svr_scores.mean()

# for i in range(10):
#     for j in range(-2, 7):
#         svr = SVC(C=10**j, coef0=i+1, kernel='rbf')
#         svr_scores = cross_val_score(svr, x, y)
#         print "rbf: ", "coef0: ", i+1, "C: ", 10**j, "score: ", svr_scores.mean()
# clf = GridSearchCV(svr, parameters)
# model = svr.fit(xtrain, ytrain)
# test_pred = model.predict(xtest)
# tst_err_score = accuracy_score(ytest, test_pred)
# print(clf.best_params_)
# print(tst_err_score)
