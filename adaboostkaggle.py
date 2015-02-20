import numpy as np
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
from sklearn.grid_search import GridSearchCV

# get data
data = np.genfromtxt('./data/kaggle_train_wc.csv', delimiter = ',')
(row, col) = data.shape

# format data
x, y = data[:, :col-1], data[:, col-1]

# Adaboost the shit out of it
n_estimators = [i*50 for i in range(1,10)]
parameters = {'n_estimators':n_estimators}
ada = AdaBoostClassifier()
ada_clf = GridSearchCV(ada, parameters)
ada_scores = cross_val_score(ada_clf, x, y)

print 'finished adaboost'

# Multinomial the shit out of it
alphas = [i/10 for i in range(10)]
parameters = {'alpha':alphas}
mul = MultinomialNB()
mul_clf = GridSearchCV(mul, parameters)
mul_scores = cross_val_score(mul_clf, x, y)

print 'finished multinomial'

# Decision Tree Classifier the shit out of it
leaves = [(i+1)*10 for i in range(10)]
nodes = [(i+1)*10 for i in range(10)]
parameters = {'min_samples_leaf':leaves, 'max_depth': nodes}
dtc = DecisionTreeClassifier()
dtc_clf = GridSearchCV(dtc, parameters)
dtc_scores = cross_val_score(dtc_clf, x, y)

print 'finished dtc'

# Extra Tree Classifier the shit out of it
parameters = {'min_samples_leaf':leaves, 'max_depth': nodes,
    'n_estimators': n_estimators}
etc = ExtraTreesClassifier()
etc_clf = GridSearchCV(etc, parameters)
etc_scores = cross_val_score(etc_clf, x, y)

print 'finished etc'

# Random Forest Classifier the shit out of it
rfc = RandomForestClassifier()
rfc_clf = GridSearchCV(rfc, parameters)
rfc_scores = cross_val_score(rfc_clf, x, y)

print 'finished rfc'

print("AdaBoost is: ",  ada_scores.mean())
print("Multinomial Naive Bayes is: ", mul_scores.mean())
print("Decision Trees is: ", dtc_scores.mean())
print("Extra Trees is: ", etc_scores.mean())
print("Random Forest is: ", rfc_scores.mean()) 
