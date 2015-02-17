import numpy as np
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier
import matplotlib.pyplot as plt
from sklearn.grid_search import GridSearchCV

# get data
data = np.genfromtxt('./data/kaggle_train_wc.csv', delimiter = ',')
(row, col) = data.shape
# format data
x, y = data[:, :col-1], data[:, col-1]
# Let's gridsearch first
n_estimators = [i+1 for i in range(1000)]
parameters = {'n_estimators':n_estimators}
# Adaboost the shit out of it
ada = AdaBoostClassifier()
clf = GridSearchCV(ada, parameters)
scores = cross_val_score(clf, x, y)
print(scores.mean())
