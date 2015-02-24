import numpy as np
from sklearn.cross_validation import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (ExtraTreesClassifier,
							  GradientBoostingClassifier,
							  BaggingClassifier,
							  AdaBoostClassifier) 
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
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

# Train a KNeighbors clustering model
# Best Result was with TF-IDF and i = 20 => 0.8435
'''neighbors = [i*5 for i in range(1, 21)];
for i in neighbors:'''
'''Kn = KNeighborsClassifier(n_neighbors = 20, weights = 'distance')
Kn_score_wc = cross_val_score(Kn, xwc, ywc, cv = 5)
Kn_score_tf = cross_val_score(Kn, xtf, ytf, cv = 5)
print('Kneighbors WC: ', 20, Kn_score_wc.mean())
print('Kneighbors TF-IDF: ', 20, Kn_score_tf.mean())'''

# Try other GBC parameters
# Best score returned by these parameters: WC = 0.932, TF-IDF = 0.929
gbc = GradientBoostingClassifier(learning_rate = 0.05, n_estimators = 2000, max_depth = 5, min_samples_leaf = 45, verbose = 1)
gbc_score_wc = cross_val_score(gbc, xwc, ywc, cv = 5)
# gbc_score_tf = cross_val_score(gbc, xtf, ytf, cv = 5)
print('GBC WC: ', 2000, gbc_score_wc.mean())
# print('GBC TF-IDF: ', 2000,  gbc_score_tf.mean())

# modelKn = Kn.fit(xtf, ytf)
# Kn_pred = list(modelKn.predict(xtest_tf))
# modelGbc = gbc.fit(xtf, ytf)
modelGbc = gbc.fit(xwc, ywc)
# gbc_pred = list(modelGbc.predict(xtest_tf))
gbc_pred = list(modelGbc.predict(xtest_wc))

# Write output
# myfile1 = open('Kn_pred_r.csv', 'wb')
myfile2 = open('GBC_pred_wc.csv', 'wb')
# wr1 = csv.writer(myfile1)
wr2 = csv.writer(myfile2)
# wr1.writerow(Kn_pred)
wr2.writerow(gbc_pred)
# myfile1.close()
myfile2.close()

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

'''n_est = [i*10 for i in range(8, 21)]

for i in n_est:
    gbc = GradientBoostingClassifier(n_estimators = i)
    gbc_score_wc = cross_val_score(gbc, xwc, ywc, cv = 5)
    gbc_score_tf = cross_val_score(gbc, xtf, ytf, cv = 5)
    print('GBC WC: ', i, gbc_score_wc.mean())
    print('GBC TF-IDF: ',i,  gbc_score_tf.mean())'''
