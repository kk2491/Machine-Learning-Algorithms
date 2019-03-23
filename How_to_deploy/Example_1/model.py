import pandas as pd 
import numpy as np 
from sklearn import linear_model
import pickle

df = pd.read_csv("winequality-red.csv", delimiter = ";")

print(df.head())
print(df.columns)

#Dependant Variable
label = df["quality"]

#Independant Variables
features = df.drop("quality", axis = 1)

model = linear_model.LinearRegression()
print(model)

model.fit(features, label)
print("Model Coefficients : {} \n Model Intercept : {} \n Model Score : {}".format(model.coef_, model.intercept_, model.score(features, label)))

#Use the trained model to predict the label for new data
new_data = [[6.0, 0.79, 0, 1.8, 0.060, 20, 33, 0.99, 4.11, 0.1, 4.0]]
output = model.predict(new_data)
print("Output : {}".format(output))

#Saving the weights to pickle file
pickle.dump(model, open("model.pkl", "wb"))

#Load the model
new_model = pickle.load(open("model.pkl", "rb"))
print("Model Loaded..!!")
print("Model Coefficients : {} \n Model Intercept : {}".format(new_model.coef_, new_model.intercept_))
