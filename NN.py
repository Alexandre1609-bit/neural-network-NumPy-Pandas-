import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


#Import et nettoyage du DataSet 
data = pd.read_csv(r"ds\Iris.csv")
print(data.head())
data.drop(['Id'], axis=1,inplace=True)


data = np.array(data)


m, n = data.shape
np.random.shuffle(data)

#Layer 
data_dev = data[0:150].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]

data_train = data[150:m].T
Y_train = data_train[0]
X_train = data_train[1:n]


