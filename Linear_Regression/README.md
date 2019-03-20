**1. Linear Regression** <br />
Type		   -- Supervised Learning <br />  	
Target Attribute -- Continuous variable <br />
Pre-processing    <br />
a. Remove the replace the NULL and NA values <br />
b. Check for Outliers and replace <br />
c. Divide the data into train and test data set. <br />
d. Check for multicollinearity. <br />
e. Convert the categorical variables to numeric variables <br />
f. Use feature selection techniques to select only the important features. <br />
   Forward selection <br />
   Backward selection <br />
   Hybrid feature selection <br />
   Build linear regression model (without regularization) <br />
  
**Metrics to consider for evaluation** <br />
a. R square value – This is the proportion of the data explained by the model <br />
b. Adjusted R square – This takes account of number of features <br />
c. RMSE – Root Mean Squared Error – This gives the root of squared difference between the actual and predicted target variable <br />
d. Mean Absolute Error <br />
e. Mean Squared Error <br />
f. AIC and BIC values <br />
g. Residual Analysis – Error terms should be randomly distributed <br />

If the model is over fitting below approaches can be used <br />
a. Normalize the data and re-build the new model with regularization parameter. <br />
b. Build new model with only significant features by performing feature selection. <br />
c. Ask the customer to provide more samples of data. <br />

If the model is under fitting <br />
a. Build new model with polynomial feature or by using feature transformation and feature extraction. <br />
