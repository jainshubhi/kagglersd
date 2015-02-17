import numpy as np
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
parameters = {'kernel':('rbf','linear','poly','sigmoid'),
    'C':[0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000],
    }
# run svm
svr = SVC()
clf = GridSearchCV(svr, parameters)
model = clf.fit(xtrain, ytrain)
test_pred = model.predict(xtest)
tst_err_score = accuracy_score(ytest, test_pred)
print(clf.best_params_)
print(tst_err_score)
