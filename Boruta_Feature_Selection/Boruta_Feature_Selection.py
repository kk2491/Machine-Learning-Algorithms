
#Boruta package - works well for classification and regression
#In this usage of Boruta package for variable selection is explained

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
import numpy as np

data = pd.read_csv("train_u6lujuX_CVtuZ9i.csv")

data2 = pd.read_csv("claimants (4).csv")
data2.head()
del data2["CASENUM"]

#Check it later
data.head()
data.info()
del data["Loan_ID"]
np.sum(pd.isna(data))
data1 = data.dropna()
data1 = pd.get_dummies(data1)
data1.columns.values
del data1["Loan_Status_N"]


#From line number 14
np.sum(pd.isna(data2))
data2 = data2.dropna()
data2.head()

#Prepare dependant and independant data
#Why this data2.iloc and data.iloc[:].values
ind_data = data2.iloc[:,1:].values

dep_data = data2.iloc[:,:1].values


#Define a random forest classifier 
rf = RandomForestClassifier(n_jobs = -1,class_weight = "balanced", max_depth = 2)

#Define Boruta feature selection method
feat_selector = BorutaPy(rf, n_estimators = "auto", verbose = 2, random_state = 1)

#Find the relevant features for the data set 
feat_selector.fit(ind_data, dep_data)

#This will give the index and boolean values
feat_selector.support_

#This gives the ranking
feat_selector.ranking_

#This transforms the original data such that only imp variables are taken into consideration
ind_filtered_data = feat_selector.transform(ind_data)

