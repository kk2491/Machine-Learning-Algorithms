**Decision Tree Classification model (C5.0, C4.5 and CART)**
```
Type             --	Supervised Learning
Target attribute -- 	Discrete Classification 
Pre-processing 	
a. Replace the null values and outliers will be handled by the tree structure
b. Divide the data into train and test

Build Decision tree model on the training data
Terms to consider
a. Classification error = 1 – max(p1, p2)
b. Gini Index = 1 – summation(pi ^ 2)
c. Entropy = summation (- pi log2(pi))
d. Information gain

Metrics to consider for evaluation
a. Confusion matrix
b. Classification accuracy (Accuracy = (TP+TN)/(TP+TN+FP+FN)
c. Classification error (misclassification rate = (FP+FN)/(TP+TN+FN+FP)
d. Recall / Sensitivity / True positive rate = (TP/(TP+FN))
e. Specificity = TN/(TN+FP)
f. False positive rate = FP/(TN+FP)	
g. Precision = TP/(TP+FP)
Decision trees are prone to over fit the training data
Prune the tree appropriately and evaluate the model with the test data and compare the results.
```
==============================================================================================

**Decision Tree Regression model (CART)**
```
Type             --	Supervised Learning
Target Attribute -- 	Continuous variable

Pre-processing
a. Replace the null values 
b. Outliers needs to be replaced accordingly
c. Divide the data into train and test

Build decision tree algorithm on train data set
Metrics to consider for evaluation
a. Mean squared error
b. Root mean squared error
c. R squared value
d. Mean absolute error
e. Median absolute error
f. AIC and BIC values
Prune the tree appropriately and evaluate the model with the test data and compare the results.
```
