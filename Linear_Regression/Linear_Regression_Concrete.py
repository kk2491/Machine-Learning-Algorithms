# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 12:34:50 2018

@author: kishkuma
"""


import pandas as pd
import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

data = pd.read_csv("concrete.csv")
data.head()
data.corr()

colnames = data.columns.values

plt.hist(data)

ind_data = data.iloc[:,0:8]
dep_data = data['strength']

train_X, test_x, train_y, test_y = train_test_split(ind_data, dep_data)
del dep_data, ind_data

model = linear_model.LinearRegression()
model.fit(train_X, train_y)

predict_train = model.predict(train_X)

#Accuracy

print ("mean squared error for train :", mean_squared_error(train_y, predict_train))
print ("R squared value :", r2_score(train_y, predict_train))

predict_test = model.predict(test_x)

print ("mean squared error for train :", mean_squared_error(test_y, predict_test))
print ("R squared value :", r2_score(test_y, predict_test))



