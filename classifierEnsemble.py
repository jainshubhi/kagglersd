import numpy as np
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV

# Import the word count data for training.
data = np.genfromtxt('./data/kaggle_train_wc.csv', delimiter = ',')
(row, col) = data.shape
# format data
xwc, ywc = data[:, :col-1], data[:, col-1]

# Import the word count data for training.
data = np.genfromtxt('./data/kaggle_train_tf_idf.csv', delimiter = ',')
(row, col) = data.shape
# format data
xtf, ytf = data[:, :col-1], data[:, col-1]

# Import the test data
xtest_wc = np.genfromtxt('./data/kaggle_test_wc.csv', delimiter = ',')
xtest_tf = np.genfromtxt('./data/kaggle_test_tf_idf.csv', delimiter = ',')

# Train a MultinomialNB Model
mnb = MultinomialNB()
mnb_score_wc = cross_val_score(mnb, xwc, ywc)
mnb_score_tf = cross_val_score(mnb, xtf, ytf)
print('MNB WC: ', mnb_score_wc.mean())
print('MNB TF-IDF: ', mnb_score_tf.mean())
