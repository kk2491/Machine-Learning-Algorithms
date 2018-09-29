# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 23:51:43 2018

@author: kishkuma
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler

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


#Lets try the same with Normalization
data_for_scale = ind_data[:]

scalar = StandardScaler().fit(data_for_scale)
scalar.transform(data_for_scale)

train_X, test_X, train_y, test_y = train_test_split(data_for_scale, target, random_state = 3)
model1 = LogisticRegression()
model1.fit(train_X, train_y)

predict_train = predict_function(model1, train_X)
predict_test = predict_function(model1, test_X)

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

