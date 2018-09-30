rm(list = ls())

#In this the usage of CARET has been implemented

library(caret)
data = read.csv("Data.csv",stringsAsFactors = T)

str(data)

#==============================================================

#1. Pre processing using CARET
#Check for NA values
sum(is.na(data))

#use Caret to impute these missing values using KNN algorithm
impute_miss = preProcess(x = data,method = c("knnImpute","center","scale"))

library(RANN)
#All the NA values are replaced accordingly
processed_data = predict(preProValues,data)

sum(is.na(processed_data))

#=================================================================================

#2. One hot encoding using CARET

#Converting the target variable from categorical to numeric
processed_data$Loan_Status = ifelse(test = processed_data$Loan_Status == "N",0,1)
id = processed_data$Loan_ID
processed_data$Loan_ID = NULL
str(processed_data)

#Creating every categorical variable into numeric using one hot encoding
dmy = dummyVars(formula = " ~ .", data = processed_data, fullRank = T)
dmy$terms

transformed_data = data.frame(predict(object = dmy, newdata = processed_data))
str(transformed_data)

#Converting the depending variable back to categorical
transformed_data$Loan_Status = as.factor(transformed_data$Loan_Status)
str(transformed_data)

#=========================================================================================

#3. Splitting the data using CARET
#splitting the data set into 2 parts based on the outcome (75% and 25%)
index = createDataPartition(transformed_data$Loan_Status,p = 0.75,list = FALSE)
train_data = transformed_data[index,]
test_data = transformed_data[-index,]
str(train_data)

#=========================================================================================

#4. Feature selection using CARET
#First define a control function using random forest
control = rfeControl(functions = rfFuncs,method = "repeatedcv",repeats = 3,verbose = FALSE)
outcomeName = "Loan_Status"
predictors = names(train_data)[!names(train_data) %in% outcomeName]

#Apply the control function on train dataset using recursive feature elimination
Loan_Pred_Profile = rfe(train_data[,predictors],train_data[,outcomeName],rfeControl = control)

#We will get the top 5 variables in this case
print(Loan_Pred_Profile)

#=========================================================================================

#5. Training models using CARET

#To list all the models available in caret package
names(getModelInfo())

#Any algorithm can be used to train the data set
model_gbm = train(train_data[,predictors], train_data[,outcomeName], method = "gbm")
model_rf = train(train_data[,predictors], train_data[,outcomeName], method = "rf")
model_nnet = train(train_data[,predictors], train_data[,outcomeName], method = "nnet")
model_glm = train(train_data[,predictors], train_data[,outcomeName], method = "glm")

#=========================================================================================

#6. Parameters tuning using CARET

fitControl = trainControl(method = "repeatedcv", number = 5, repeats = 5)

#6.1 Using the tuneGrid

modelLookup(model = "gbm")

#Creating a grid
grid = expand.grid(n.trees = c(10,20,50,100,500,1000),interaction.depth = c(1,5,10),shrinkage = c(0.01,0.05,0.1,0.5),n.minobsinnode = c(3,5,10))

#Training the model
model_gbm = train(train_data[,predictors],train_data[,outcomeName],method = "gbm",trControl = fitControl,tuneGrid = grid)

print(model_gbm)

#Check which combinations gives the highest accuracy 

plot(model_gbm)

#6.2 Using the tuneLength

model_gbm = train(trainSet[,predictors],trainSet[,outcomeName],method = "gbm", trControl = fitControl, tuneLength = 10)
print(model_gbm)

plot(model_gbm)

#=====================================================================================

#7. Variable importance estimation using CARET

#Checking the variable importance for GBM model
varImp(object = model_gbm)
#Plotting the variable importance
plot(varImp(object = model_gbm), main = "GBM - Variable Importance")

#Checking the variable importance for Random forest model
varImp(object = model_rf)
plot(varImp(object = model_rf), main = "RF - Variable importance")

#Checking the variable importance for neural net
varImp(object = model_nnet)
plot(varImp(object = model_nnet), main = "NNET - Variable importance")

#Checking the variable importance for Logistic Regression
varImp(object = model_glm)
plot(varImp(object = model_glm), main = "GLM - Variable importance")

#=====================================================================================

#8. Predictions using CARET
#predict.train(Y) - Type - Raw - gives class, prob - gives probability

predictions = predict.train(object = model_gbm,newdata = testSet[,predictors],type = "raw")
table(predictions)

confusionMatrix(predictions,testSet[,outcomeName])

#=====================================================================================