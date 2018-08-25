# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 23:49:13 2018

@author: kishkuma
"""


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

del data['region']

backup = data[:]
#data = backup[:]

#Converting from cat to int 
#temp_data = pd.DataFrame({'sex': ['Male','Female']})
#print(pd.get_dummies(temp_data))

#Converting categorical variables to numeric type using One hot encoding
temp_data = pd.get_dummies(data['sex'])

data = pd.concat([data, temp_data], axis = 1)
del data['sex']

temp_data = pd.get_dummies(data['smoker'])
data = pd.concat([data, temp_data], axis = 1)
del data['smoker']

data.rename(columns = {'no' : 'no_smoker', 'yes' : 'yes_smoker'})

#Correlation check to avoid multi collinearity
#Correlation is fine, none of the variables are highly correlated with one another
data.corr()


#Normalize the data
from sklearn import preprocessing
scaler = preprocessing.Normalizer().fit(data)
normal_data = scaler.transform(data)

data = pd.DataFrame(normal_data)

#Divide the data into train test

temp = data.columns.values
train_X = data.iloc[:,0:3]
train_X = pd.concat([train_X,data.iloc[:,4:]], axis = 1)
#train_X = data.iloc[:,0:5]
train_y = data.iloc[:,(3)]


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

#Plot the residues

plt.scatter(model.predict(X_train),model.predict(X_train)- y_train, c= 'b', s=40, alpha = 0.01)

#plt.plot(model.predict(X_train))

plt.scatter(model.predict(X_test), model.predict(X_test)-y_test, c='g', s=40)
plt.hlines(y= 0, xmin = 0, xmax=5)
plt.title('Residual Plot')

#plt.plot(data['age'], data['charges'])
#plt.hist(data['charges'])

model_accuracy(y_train, model.predict(X_train))
model_accuracy(y_test, model.predict(X_test))


#Function to calculate the accuracy of the model
def model_accuracy(y_actual, y_predicted):
    print ("Mean squared error for training set :", mean_squared_error(y_actual, y_predicted))
    print ("R2 value for training set :", r2_score(y_actual, y_predicted))
    return 0


    















