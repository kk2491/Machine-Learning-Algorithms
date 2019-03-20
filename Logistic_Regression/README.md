**Logistic Regression:** <br />
Type		   -- Supervised Learning <br />	
Target Attribute – Discrete variable / Classes <br />
Pre-processing    <br />
a. Remove the replace the NULL and NA values <br />
b. Check for Outliers and replace. <br />
c. Divide the data into train and test data set. <br />
d. Check for multicollinearity. <br />
e. Convert the categorical variables to numeric variables <br />
f. Use feature selection techniques to select only the important features. <br />
Forward selection <br />
Backward selection <br />
Hybrid feature selection <br />


Build logistic regression model (without regularization term) <br />

Metrics to consider for evaluation <br />

a. Confusion matrix <br />
b. Classification accuracy (Accuracy = (TP+TN)/(TP+TN+FP+FN) <br />
c. Classification error (misclassification rate = (FP+FN)/(TP+TN+FN+FP) <br />
d. Recall / Sensitivity / True positive rate = (TP/(TP+FN)) <br />
e. Specificity = TN/(TN+FP) <br />
f. False positive rate = FP/(TN+FP)	 <br />
g. Precision = TP/(TP+FP) <br />
h. AUC value <br />
Depends on the business problem appropriate metric can be used to evaluate the model. <br />
In Logistic regression threshold of the probability to classify the classes can be selected by plotting the ROC curve based on the metric requirement. <br />
