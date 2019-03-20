**Support Vector Machine**

Type		--	Supervised Learning <br />
Target attribute -- 	Discrete Classification or Continuous variable <br />

Pre-processing 	 <br />
a. Replace the null values  <br /> 
b. Check for outliers and replace them accordingly <br />
c. Divide the data into train and test <br />
d. Covert the categorical variables to numeric variables <br />
e. Perform feature selection using any of the feature selection method <br />
f. Normalize the train data set <br />

Build Support vector machine model using the appropriate kernel function (This needs to be done Trial and error as it is hard to visualize if the data set is having more number of features). <br />

**Classification Problem**

Here the hyper plane (kernel function) divides the distinct classes such that there is large margin between the hyper plane and the sample.

Metrics to consider for evaluation. <br />
a. Confusion matrix <br />
b. Classification accuracy (Accuracy = (TP+TN)/(TP+TN+FP+FN) <br />
c. Classification error (misclassification rate = (FP+FN)/(TP+TN+FN+FP) <br />
d. Recall / Sensitivity / True positive rate = (TP/(TP+FN)) <br />
e. Specificity = TN/(TN+FP) <br />
f. False positive rate = FP/(TN+FP)	 <br />
g. Precision = TP/(TP+FP) <br />


**Regression** <br />

a. Mean squared error <br />
b. Root mean squared error <br />
c. R squared value <br />
d. Mean absolute error <br />
e. Median absolute error <br />
f. AIC and BIC values <br />
