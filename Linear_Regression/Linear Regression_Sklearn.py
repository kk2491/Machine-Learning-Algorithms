#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 22:22:11 2017

@author: Kanth
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
# Load the diabetes dataset
diabetes = datasets.load_diabetes()


# Use only one feature

# Split the data into training/testing sets
X_train, X_test, y_train, y_test = train_test_split(diabetes.data, diabetes.target, 
                                                    random_state=3)

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(X_train, y_train)

# Make predictions using the testing set
diabetes_y_pred = regr.predict(X_test)

# The coefficients
print('Coefficients', regr.coef_)
print('Intercept',regr.intercept_)
# The mean squared error

# MSE = 1/n(yactual- ypredicted)2
print("Mean squared error: %.2f"
      % mean_squared_error(y_test, diabetes_y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, diabetes_y_pred))
r2_score(y_test, diabetes_y_pred)

plt.hist(y_test)
plt.hist(diabetes_y_pred)


# Plot outputs
plt.scatter(y_test, diabetes_y_pred,  color='black')

#Residual Plot

plt.scatter(regr.predict(X_train),regr.predict( X_train)- y_train, c= 'b', s=40, alpha = 0.5)
plt.scatter(regr.predict(X_test), regr.predict(X_test)-y_test, c='g', s=40)
plt.hlines(y= 0, xmin = 0, xmax= 50)
plt.title('Residual Plot')
































