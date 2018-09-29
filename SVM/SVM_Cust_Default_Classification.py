#In this standardize the data and test



import pandas as pd
import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("Credit_Card.csv")
print ("Number of observation", len(data))
    
training_features = ['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
    
target = 'default.payment.next.month'
    
independant_data = data[training_features]
dependant_data = data[target]
    
#Standardize the data
scaler = StandardScaler()
    
independant_data = np.asmatrix(independant_data)
    
std_independant_data = scaler.fit(independant_data)
std_independant_data = scaler.transform(independant_data)
std_independant_data = pd.DataFrame(std_independant_data)
    
dependant_data.head()
    
#train_X, test_X, train_y, test_y = train_test_split(data[training_features],data[target],train_size = 0.7)
    
train_X, test_X, train_y, test_y = train_test_split(std_independant_data, dependant_data, train_size = 0.8, random_state = 3)
    
print ("Train X Size", train_X.shape)
print ("Train Y Size", train_y.shape)
print ("Test X Size", test_X.shape)
print ("Test Y Size", test_y.shape)

trained_SVM_model = train_SVM(train_X,train_y)
    
predict_train = trained_SVM_model.predict(train_X)
    
#predict_train_cutoff = trained_logistic_regression_model.predict( )
    
#table(train_y, predict_train)
tn, fp, fn, tp = confusion_matrix(train_y,predict_train).ravel()

#Find the accuracy
train_accuracy = model_accuracy(trained_SVM_model,train_X,train_y)
print ("Accuracy on Training set ", train_accuracy)
    
#Accuracy = (TP+TN)/Total
    
test_accuracy = model_accuracy(trained_SVM_model,test_X,test_y)
print ("Accuracy on Test set ", test_accuracy)
    
    


#Function to find the column names of the dataframe
def dataset_headers(dataset):
    return list(dataset.columns.values)

#Implementing the logistic regression model in python with scikit-learn
def train_SVM(train_X, train_y):
    
    SVM_model = SVC(kernel = "poly", C = 2, gamma = 0.20)
    SVM_model.fit(train_X,train_y)
    return SVM_model

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
    ticks = numpy.arange(0,25,1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(names)
    ax.set_yticklabels(names)
    #plt.savefig("Correlations.png")
    plt.show()
    
    return correlations