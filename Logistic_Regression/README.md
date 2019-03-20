2. Logistic Regression:

Type		   -- Supervised Learning 	
Target Attribute – Discrete variable / Classes
Pre-processing    
a. Remove the replace the NULL and NA values
b. Check for Outliers and replace.
c. Divide the data into train and test data set.
d. Check for multicollinearity.
e. Convert the categorical variables to numeric variables
f. Use feature selection techniques to select only the important features.
Forward selection
Backward selection
Hybrid feature selection


Build logistic regression model (without regularization term)

Metrics to consider for evaluation

a. Confusion matrix
b. Classification accuracy (Accuracy = (TP+TN)/(TP+TN+FP+FN)
c. Classification error (misclassification rate = (FP+FN)/(TP+TN+FN+FP)
d. Recall / Sensitivity / True positive rate = (TP/(TP+FN))
e. Specificity = TN/(TN+FP)
f. False positive rate = FP/(TN+FP)	
g. Precision = TP/(TP+FP)
h. AUC value
Depends on the business problem appropriate metric can be used to evaluate the model.
In Logistic regression threshold of the probability to classify the classes can be selected by plotting the ROC curve based on the metric requirement.
