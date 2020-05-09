I would like to Thank [Sentdex](https://github.com/sentdex/) for excellent tutorials on Machine Learning Algorithms. 

## Machine Learning - Flowchart

### 1. **Understand the client requirement / Problem statement** ###

### 2. **Data Understanding** ###

**a. Data Collection (CSV file, logs, sensor data, data from SQL etc)** <br />

**b. Data explore** <br />

**c. Data quality** : Analyze the data such that the sufficient information or data is available to build the model <br />

### 3. **Data Preparation** ###

**a. Cleaning the data** <br />
Check for NULL and NA values in the dataset, and take necessary actions <br />
Remove if dataset is huge and removing a samples doesn’t affect the quality of the data <br />
Impute the missing values with mean, median or KNN <br />

**b. Check for outliers in the dataset** <br />
Might be due to human error. This can be checked by using boxplot or the summary statistics of the dat**a. Remove or replace accordingly <br />

**c. Check how the features are distribute using histogram** <br />

**d. Divide the data into train and test data set.** <br />

**e. Feature Selection** <br />
This can be done using filter, wrapper and embedded method <br />
Filter – Using statistical methods. Correlation check. <br />
Wrapper – Subset, Forward, Backward, Hybrid selection, Boruta feature selection, Random forest important variable selection <br />
Embedded – Lasso and Ridge <br />

**f. Feature Engineering** <br />
New features are created from the existing features <br />
Feature transformations <br />
Perform normalization and standardization <br />

**g. Dimensionality reduction** <br />
The dimension of the dataset can be reduced by using techniques such as PCa. <br />

### 4. **Model Building** ###

**a. Based on the problem statement decide whether the problem belongs to supervised and unsupervised model.** <br /> 

**b. If supervised model, check whether the target variable is continuous or discrete.** <br /> 
If continuous – use regression model <br /> 
If discrete – use classification model <br /> 

**c. Build the model using appropriate machine learning algorithm on train data.** <br /> 

**d. Multiple models can be built to check which gives the better accuracy.** <br /> 

### 5. **Model Evaluation** ###

**a. Once the model is built check the accuracy/error rate on both training data and testing data using appropriate evaluation metrics.** <br />

**b. If the accuracy is good on Training and poor on Testing dataset then the model is overfitting** <br />
Build new model using regularization <br />
Ask the customer to provide more data samples <br />

**c. If the accuracy is poor on both training and test data set then the model is overfitting** <br />
Build new model by adding more features <br />
Build new model which includes feature transformation <br />




