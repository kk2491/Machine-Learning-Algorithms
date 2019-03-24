# Reference : https://towardsdatascience.com/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8

import pandas as pd 
import numpy as np 
from sklearn import preprocessing
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
import sys
sns.set(style = "white")
sns.set(style = "whitegrid", color_codes = True)

plot = sys.argv[1]

data = pd.read_csv("banking.csv", header = 0)
print(data.head())

data["education"].unique()

data["education"] = np.where(data["education"] == "basic.9y", "Basic", data["education"])
data["education"] = np.where(data["education"] == "basic.6y", "Basic", data["education"])
data["education"] = np.where(data["education"] == "basic.4y", "Basic", data["education"])

print(data["y"].value_counts())

if plot == "plot":
	sns.countplot(x="y", data=data, palette="hls")
	plt.show()
	plt.savefig("count_plot")
elif plot == "noplot":
	pass

count_no_sub = len(data[data["y"] == 0])
count_sub = len(data[data["y"] == 1])
pct_of_no_sub = count_no_sub/(count_no_sub + count_sub)
pct_of_sub = count_sub/(count_no_sub + count_sub)
print("Percentage of No Subscription : {} \n Subscription {}".format(pct_of_no_sub*100, pct_of_sub*100))


