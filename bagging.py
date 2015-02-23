import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
from sklearn.cross_validation import cross_val_score
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

bag = BaggingClassifier(base_estimator = base)
bag_score_wc = cross_val_score(bag, xwc, ywc)
bag_score_tf = cross_val_score(bag, xtf, ytf)
print('BAG WC: ', bag_score_wc.mean())
print('BAG TF-IDF: ', bag_score_tf.mean())


'''model = bag.fit(xtf, ytf)
test_pred = list(model.predict(xtest_tf))

myfile = open('bag_pred.csv', 'wb')
wr = csv.writer(myfile)
wr.writerow(test_pred)'''
