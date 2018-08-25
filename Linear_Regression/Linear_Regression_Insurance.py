# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 06:48:40 2018

@author: kishkuma
"""

#Using R - https://www.kaggle.com/mirichoi0218/regression-how-much-will-the-premium-be

import pandas as pd
import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

#Read the CSV File
data = pd.read_csv("insurance.csv")
data.head()

data.describe()
data.info()
data.columns.values
plt.hist(data['age'])
plt.hist(data['sex'])
plt.hist(data['charges'])

#Convert str to factors
temp = pd.factorize(data['sex'])
data['sex'] = temp[0]
temp = pd.factorize(data['smoker'])
data['smoker'] = temp[0]

del data['region']

#Correlation is fine, none of the variables are highly correlated with one another
data.corr()

#Divide the data into train test

temp = data.columns.values
train_X = data.iloc[:,0:5]
train_y = data.iloc[:,5]

X_train, X_test, y_train, y_test = train_test_split(train_X, train_y)
del train_X, train_y
del temp

#Build the model

model = linear_model.LinearRegression()
model.fit(X_train, y_train)

#Print the coefficients
print ("Coefficients of the model :", model.coef_)
model.intercept_
model.rank_
model._residues
#Check how to get the significance level

model.predict(X_train)

#Check the accuracy of the model
print ("Mean squared error for training set :", mean_squared_error(y_train, model.predict(X_train)))
print ("R2 value for training set :", r2_score(y_train, model.predict(X_train)))

#Plot the residues

plt.scatter(model.predict(X_train),model.predict( X_train)- y_train, c= 'b', s=40, alpha = 0.5)
plt.scatter(model.predict(X_test), model.predict(X_test)-y_test, c='g', s=40)
plt.hlines(y= 0, xmin = 0, xmax= 50)
plt.title('Residual Plot')






















