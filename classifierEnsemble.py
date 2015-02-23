import numpy as np
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
import csv
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
'''mnb = MultinomialNB()
mnb_score_wc = cross_val_score(mnb, xwc, ywc)
mnb_score_tf = cross_val_score(mnb, xtf, ytf)
print('MNB WC: ', mnb_score_wc.mean())
print('MNB TF-IDF: ', mnb_score_tf.mean())

dtc = DecisionTreeClassifier()
dtc_score_wc = cross_val_score(dtc, xwc, ywc)
dtc_score_tf = cross_val_score(dtc, xtf, ytf)
print('DTC WC: ', dtc_score_wc.mean())
print('DTC TF-IDF: ', dtc_score_tf.mean())

etc = ExtraTreesClassifier()
etc_score_wc = cross_val_score(etc, xwc, ywc)
etc_score_tf = cross_val_score(etc, xtf, ytf)
print('ETC WC: ', etc_score_wc.mean())
print('ETC TF-IDF: ', etc_score_tf.mean())

svc = SVC(C = 0.0001, kernel = 'linear', coef0 = 1)
svc_score_wc = cross_val_score(svc, xwc, ywc)
svc_score_tf = cross_val_score(svc, xtf, ytf)
print('SVC WC: ', svc_score_wc.mean())
print('SVC TF-IDF: ', svc_score_tf.mean())

base = GradientBoostingClassifier()

# Best: GBC, 0.920, tf-idf
bag = BaggingClassifier(base_estimator = base)
bag_score_wc = cross_val_score(bag, xwc, ywc)
bag_score_tf = cross_val_score(bag, xtf, ytf)
print('BAG WC: ', bag_score_wc.mean())
print('BAG TF-IDF: ', bag_score_tf.mean())

# Best: DTC, 0.868, wc
ada = AdaBoostClassifier(base_estimator = base, algorithm = 'SAMME')
ada_score_wc = cross_val_score(ada, xwc, ywc)
ada_score_tf = cross_val_score(ada, xtf, ytf)
print('ADA WC: ', ada_score_wc.mean())
print('ADA TF-IDF: ', ada_score_tf.mean())'''

n_est = [i*10 for i in range(8, 21)]

for i in n_est:
    gbc = GradientBoostingClassifier(n_estimators = i)
    gbc_score_wc = cross_val_score(gbc, xwc, ywc, cv = 5)
    gbc_score_tf = cross_val_score(gbc, xtf, ytf, cv = 5)
    print('GBC WC: ', i, gbc_score_wc.mean())
    print('GBC TF-IDF: ',i,  gbc_score_tf.mean())
