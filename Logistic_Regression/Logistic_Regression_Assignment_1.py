# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 22:43:22 2018

@author: kishkuma
"""
#Missing values - https://pandas.pydata.org/pandas-docs/stable/missing_data.html

#This is without Normalization

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from sklearn.metrics import roc_curve


data = pd.read_csv("claimants (4).csv")
data.columns.values
data.head()
del data['CASENUM']

len(data)  #Length of the data set

data.describe()
data.info()

#NULL values are present in the dataset
np.sum(pd.isnull(data))
np.sum(pd.isna(data))

#Time to replace or remove the Null values
np.sum(pd.isnull(data['CLMSEX']))

backup = data[:]

#Replace the missing values with the mean
data = data.fillna(data.mean())

#Divide the data into train and test
target = data['ATTORNEY']
ind_data = data.iloc[:,1:]

train_X, test_X, train_y, test_y = train_test_split(ind_data, target, random_state = 3)

#Build the logistic regression model
model = LogisticRegression()
model.fit(train_X, train_y)

predict_train = predict_function(model, train_X)
predict_test = predict_function(model, test_X)

train_class = class_report(train_y, predict_train)
print(train_class)
test_class = class_report(test_y, predict_test)
print(test_class)

train_acc = accuracy(train_y, predict_train)
print("Accuracy on training dataset : ", train_acc)
test_acc = accuracy(test_y, predict_test)
print("Accuracy on testing dataset : ", test_acc)

train_cm = accuracy(train_y, predict_train)
test_cm = accuracy(test_y, predict_test)


def predict_function(model, data):
    predict_data = model.predict(data)
    return predict_data

def class_report(actual, predicted):
    return classification_report(actual, predicted)

def accuracy(actual, predicted):
    return accuracy_score(actual, predicted)

def cm(actual, predicted):
    return confusion_matrix(actual, predicted)

















