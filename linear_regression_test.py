import numpy as np
from numpy import genfromtxt
import sklearn

my_data = np.genfromtxt('./data/kaggle_train_wc.csv', delimiter=',')

print(type(my_data))
