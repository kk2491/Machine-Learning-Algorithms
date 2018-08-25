#This is to predict the insurance charge of the customer

import pandas as pd
import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

#Read the CSV File
data = pd.read_csv("insurance.csv")
data.head()
data.corr()

data.describe()
data.info()
data.columns.values
plt.hist(data['age'])
plt.hist(data['sex'])
plt.hist(data['charges'])


#Check how target is changing with column region
data.region.unique()
temp_data1 = data.loc[data['region'] == 'southwest']
temp_data2 = data.loc[data['region'] == 'southeast']
temp_data3 = data.loc[data['region'] == 'northwest']
temp_data4 = data.loc[data['region'] == 'northeast']

fig, ax = plt.subplots(nrows = 2, ncols = 2)

plt.subplot(2,2,1)
plt.plot(temp_data1['charges'])
plt.subplot(2,2,2)
plt.plot(temp_data2['charges'])
plt.subplot(2,2,3)
plt.plot(temp_data3['charges'])
plt.subplot(2,2,4)
plt.plot(temp_data4['charges'])
plt.show()

del temp_data1,temp_data2,temp_data3,temp_data4

backup = data[:]
#data = backup[:]

del data['region']

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

data.head()

#Divide the data into dependant and independant 
ind_data = data.iloc[:,[0,1,2,4,5,6,7]]
dep_data = data.iloc[:,3]

#Normalize the data
from sklearn import preprocessing
scaler = preprocessing.Normalizer().fit(ind_data)
normal_data = scaler.transform(ind_data)

ind_data = pd.DataFrame(normal_data)

#Divide the data into train test
X_train, X_test, y_train, y_test = train_test_split(ind_data, dep_data, train_size = 0.8, random_state = 3)


#Build the model
model = linear_model.LinearRegression()
model.fit(X_train, y_train)

#Print the coefficients
print ("Coefficients of the model :", model.coef_)
model.intercept_
model.rank_
model._residues

model.predict(X_train)

#Plot the residues

plt.scatter(model.predict(X_train),model.predict(X_train)- y_train, c= 'b', s=40, alpha = 0.1)

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







