6. Support Vector Machine

Type		--	Supervised Learning
Target attribute -- 	Discrete Classification or Continuous variable
Pre-processing 	
a. Replace the null values 
b. Check for outliers and replace them accordingly
c. Divide the data into train and test
d. Covert the categorical variables to numeric variables
e. Perform feature selection using any of the feature selection method
f. Normalize the train data set

Build Support vector machine model using the appropriate kernel function (This needs to be done Trial and error as it is hard to visualize if the data set is having more number of features).

Classification Problem

Here the hyper plane (kernel function) divides the distinct classes such that there is large margin between the hyper plane and the sample.

Metrics to consider for evaluation.
a. Confusion matrix
b. Classification accuracy (Accuracy = (TP+TN)/(TP+TN+FP+FN)
c. Classification error (misclassification rate = (FP+FN)/(TP+TN+FN+FP)
d. Recall / Sensitivity / True positive rate = (TP/(TP+FN))
e. Specificity = TN/(TN+FP)
f. False positive rate = FP/(TN+FP)	
g. Precision = TP/(TP+FP)




Regression 
a. Mean squared error
b. Root mean squared error
c. R squared value
d. Mean absolute error
e. Median absolute error
f. AIC and BIC values
