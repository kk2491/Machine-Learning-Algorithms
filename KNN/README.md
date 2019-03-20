5. KNN 

Type		--	Supervised Learning
Target attribute -- 	Discrete Classification or Continuous variable
Pre-processing 	
a. Replace the null values 
b. Check for outliers and replace them accordingly
c. Divide the data into train and test
d. Covert the categorical variables to numeric variables
e. Perform feature selection using any of the feature selection method
f. Normalize or Standardize the train data set

Build KNN model (The model doesn’t learn from the data, target variable will be predicted based on the nearest neighbors / samples)
To determine the nearest K neighbors Euclidian distance metric is used (Manhattan or Minkowsi distance are the other popular methods)

Classification Problem
Here the target will be predicted by majority of the class in the K nearest neighbors.

Metrics to consider for evaluation
a. Confusion matrix
b. Classification accuracy (Accuracy = (TP+TN)/(TP+TN+FP+FN)
c. Classification error (misclassification rate = (FP+FN)/(TP+TN+FN+FP)
d. Recall / Sensitivity / True positive rate = (TP/(TP+FN))
e. Specificity = TN/(TN+FP)
f. False positive rate = FP/(TN+FP)	
g. Precision = TP/(TP+FP)

Regression Problems
Here are the target will be predicted by the mean of the K nearest neighbors/samples.
Metrics to consider for evaluation
g. Mean squared error
h. Root mean squared error
i. R squared value
j. Mean absolute error
k. Median absolute error
l. AIC and BIC values

Choosing value of K (in regression and classification)
Value of K is the point where cross validation error starts going up.
K is higher – Model is high biased or under fit
K is lower – Model is having high variance and over fit 
