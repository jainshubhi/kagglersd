library(lars)
d = read.csv("./Documents/Self/Caltech/CS155/HW3/data/kaggle_train_wc.csv")
dataX = data.matrix(d[,-502])
dataY = d[,502]
a = lars(dataX, dataY, type = "lasso")

