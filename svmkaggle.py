import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.grid_search import GridSearchCV

# get data
data = np.genfromtxt('./data/kaggle_train_wc.csv', delimiter = ',')
(row, col) = data.shape
# format data
x, y = data[:, :col-1], data[:, col-1]
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.2)
# specify parameters
C_range = 10.0 ** np.arange(-2, 9)
gamma_range = 10.0 ** np.arange(-2, 4)
kernel_options = ('linear','poly')
coef0_range = range(1,11)
parameters = dict(C=C_range, gamma=gamma_range,
                  kernel=kernel_options, coef0=coef0_range)
# run svm
svr = SVC()
clf = GridSearchCV(svr, parameters)
model = clf.fit(xtrain, ytrain)
test_pred = model.predict(xtest)
tst_err_score = accuracy_score(ytest, test_pred)
print(clf.best_params_)
print(tst_err_score)
