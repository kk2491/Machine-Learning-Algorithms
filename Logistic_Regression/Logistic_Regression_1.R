rm(list = ls())
setwd("C:/Kishor/DoNotTouch/Lab_Room")

data = read.csv(file = "C:/Kishor/DoNotTouch/Videos/7th Set/20170430_Batch27_CSE7202c_Lab02_MultipleLinearReg/CustomerData_GLM.csv",
                col.names = c("cust_id","City","Num_Children","Min_child","Max_child","Tenure","Purchase_Freq","Units_purchased","Games_Freq","Games_played","Games_bought","Fav_channel","Fav_Game","Total_revenue","Churn"))

#consider only 3 attributes here 

#Min_Child, City and Churn - Consider only 3 attributes (just for this example)

data = data[,c("City","Min_child","Churn")]
summary(data)
str(data)
nrow(data)

data$Churn = as.factor(as.character(data$Churn))
data$City = as.factor(as.character(data$City))

#Remove the rows with mim_child is 113 - this is the outlier
data = data[data$Min_child != 113,]  

#Divide the dataset into test, train and eval data
set.seed(123)
IDS = 1:nrow(data)
train_data = sample(x = IDS,size = nrow(data)*0.6)
test_data = sample(x = setdiff(IDS,train_data),size = nrow(data)*0.2)
eval_data = sample(x = setdiff(IDS,c(train_data,test_data)),size = nrow(data)*0.2)

train_data = data[train_data,]
test_data = data[test_data,]
eval_data = data[eval_data,]

rm(IDS)

#Consider only the min age of the child (test for other data sets)
log_model = glm(formula = Churn ~ Min_child,data = train_data,family = "binomial")
summary(log_model)

#If null deviance and residual deviance is same or almost equal then the independant variable 
#does not have much impact on the target variable

#For 1 unit increase in Min_child the log of odds of churn is equal to one will be decreased by 0.04 unit
#Check the above statement

#Now predict the value for the train, test and eval 
#If no data frame is mentioned in the data = " " then by default training data will be considered
#Response is used for classification problems


train_predict = predict(object = log_model,newdata = train_data,type = "response")
head(train_predict)
head(train_data)

summary(train_predict)
#In logistic regression - First find the probability of each data point then draw the cut off
#Standard is 0.5

#--------- From the train_predict we have the probability for each data point

#This to segregate the pedicted prob based on the cut off 
predict_train = ifelse(test = (train_predict > 0.06), 1, 0)
predict_train = as.factor(as.character(predict_train))

summary(predict_train)

#Now decide the cut off using ROC curve

library(ROCR)

pred_data1 = prediction(predictions = train_predict, train_data$Churn)
pred_data2 = performance(prediction.obj = pred_data1,measure = "spec")

plot(pred_data2)

cm_train = table(train_data$Churn,predict_train)

#Recall = True positive / Actual number of positive
#Recall = True positive / (True positive + False negative)    -> when false negative has to be reduced (Sensitivity)
#Precision = True positive / (True positive + False positive) -> when false positive has to be reduced
#Specificity = True negative / Conditional negative

table(train_data$Churn)

max1 = which.max(pred2@y.values[[1]])
max2 = pred2@y.values[[1]][max1]
max3 = pred2@x.values[[1]][max1]




test_predict = predict(object = log_model,newdata = test_data,type = "response")
cm_test = table(test_data$Churn,test_predict)

eval_predict = predict(object = log_model,newdata = eval_data,type = "response")
cm_eval = table(eval_data$Churn,eval_predict)













