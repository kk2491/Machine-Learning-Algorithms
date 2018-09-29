
#Problem statement : Based on the previous payment record, need to predict 
#whether the credit card customer will be default in the coming month

#Logistic regression used to achieve the solution

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import plotly.plotly as py
#import plotly.graph_objs as go
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression

data = pd.read_csv("UCI_Credit_Card.csv")
print ("Number of observation", len(data))

print (data.head())

#To fetch the column names of the data frame
headers = dataset_headers(data)
print ("Column names of the dataset ", headers)
#print "Dataset headers :: {headers}".format(headers = headers)
    
#Use the feature selection to select only important features which has 
#high probability of predicting the target

#Techniques to use for feature selection
#   1. Boruta Feature selection
#   2. Random Forest important variable selection
#   3. Forward or Backward Selection

#To plot and find the correlation between the independent variables
#If the correlation between 2 variables is > 0.75 or 0.8 one of them can be removed
#as it results in multi-collinearity problem

#Check if there are any NULL/NA values
np.sum(pd.isnull(data))
np.sum(pd.isna(data))

#Check for outliers using Box plot and describe
data.describe()
plt.boxplot(data["LIMIT_BAL"])

#Removing the outlier
data = data.loc[data["BILL_AMT6"] != -339603]
data = data.loc[data["PAY_AMT4"] != 0]

correlation_plot(data, headers)

#Some of the features can be removed if there is high correlation

#Model is built considering all the independent variables    
training_features = ['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']

temp_data = pd.get_dummies(data["SEX"], prefix = 'Sex')
data = pd.concat([data,temp_data], axis = 1)
del data["SEX"]

temp_data = pd.get_dummies(data["MARRIAGE"], prefix = 'MARRIAGE')
data = pd.concat([data,temp_data], axis = 1)
del data["MARRIAGE"]

#data = data.rename(columns = {'Sex1' : 'Male', 'Sex2' : 'Female'}, inplace = True)

target = 'default.payment.next.month'

training_features = ['ID', 'LIMIT_BAL', 'EDUCATION', 'AGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3','BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2','PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6','Sex_1', 'Sex_2', 'MARRIAGE_0','MARRIAGE_1', 'MARRIAGE_2', 'MARRIAGE_3']

#Divide the data into train and test
train_X, test_X, train_y, test_y = train_test_split(data[training_features],data[target],train_size = 0.7)

print ("Train X Size", train_X.shape)
print ("Train Y Size", train_y.shape)
print ("Test X Size", test_X.shape)
print ("Test Y Size", test_y.shape)

trained_logistic_regression_model = train_logistic_regression(train_X,train_y)
    
predict_train = trained_logistic_regression_model.predict(train_X)
    
#table(train_y, predict_train)
tn, fp, fn, tp = confusion_matrix(train_y,predict_train).ravel()
    
#Find the accuracy
train_accuracy = model_accuracy(trained_logistic_regression_model,train_X,train_y)
print ("Accuracy on Training set ", train_accuracy)
    
test_accuracy = model_accuracy(trained_logistic_regression_model,test_X,test_y)
print ("Accuracy on Test set ", test_accuracy)

#Got the accuracy around 80%, ROC curve can be drawn and based on the requirement threshold can be set

#Function to find the column names of the dataframe
def dataset_headers(dataset):
    return list(dataset.columns.values)

#Implementing the logistic regression model in python with scikit-learn
def train_logistic_regression(train_X, train_y):
    
    logistic_regression_model = LogisticRegression()
    logistic_regression_model.fit(train_X,train_y)
    return logistic_regression_model

def model_accuracy(trained_model,features,target):
    accuracy_score = trained_model.score(features,target)
    return accuracy_score


def correlation_plot(data,names):
    correlations = data.corr()
    # plot correlation matrix
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(correlations, vmin=-1, vmax=1)
    fig.colorbar(cax)
    ticks = np.arange(0,25,1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(names)
    ax.set_yticklabels(names)
    plt.savefig("Correlations.png")
    plt.show()
    
    return correlations

#This gives accuracy of 78 for both training and test sets... 

