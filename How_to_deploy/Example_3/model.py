# https://towardsdatascience.com/logistic-regression-a-simplified-approach-using-python-c4bc81a87c31

import numpy as np 
import pandas as pd 

train = pd.read_csv("titanic_train.csv")
print(train.head())
print(train.columns)
train.isnull()

def impute_age(cols):
	
	Age = cols[0]
	Pclass = cols[1]

	if pd.isnull(Age):
		if Pclass == 1:
			return 37
		elif Pclass == 2:
			return 29
		else:
			return 24
	else:
		return Age

train["Age"] = train[["Age", "Pclass"]].apply(impute_age, axis = 1)
train.drop('Cabin', axis = 1, inplace = True)

sex = pd.get_dummies(train["Sex"], drop_first = True)
embark = pd.get_dummies(train["Embarked"], drop_first = True)

train.drop(["Sex", "Embarked", "Name", "Ticket"], axis = 1, inplace = True)
train = pd.concat([train, sex, embark], axis = 1)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train.drop("Survived", axis = 1), train["Survived"], test_size = 0.30, random_state = 101)
print("Let copy this ..")
print(X_train.head())

from sklearn.linear_model import LogisticRegression 
logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)
predictions = logmodel.predict(X_test)

from sklearn.metrics import classification_report
print(classification_report(y_test, predictions))

# Saving the model to deploy
import pickle
#Saving the weights to pickle file
pickle.dump(logmodel, open("model.pkl", "wb"))
new_model = pickle.load(open("model.pkl", "rb"))
print("Model Loaded..!!")
print("Model Coefficients : {} \n Model Intercept : {}".format(new_model.coef_, new_model.intercept_))
