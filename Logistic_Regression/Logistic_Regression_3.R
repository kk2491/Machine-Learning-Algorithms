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

train_predict = predict(object = log_model,newdata = train_data,type = "response")
summary(train_predict)

#Just try with standard 0.5 (0.5 doesnt make any sense in this case)
predict_train = ifelse(test = (train_predict > 0.20), 1, 0)
summary(predict_train)

cm_before = table(train_data$Churn,predict_train)

#Precision = True positive / (True positive + False positive) -> when false positive has to be reduced
per_before = cm_before[2,2]/(cm_before[2,2]+cm_before[1,2])

#Now decide the cut off using ROC curve

library(ROCR)

pred_data1 = prediction(predictions = train_predict, train_data$Churn)
pred_data2 = performance(prediction.obj = pred_data1,measure = "sens")

plot(pred_data2)

max1 = which.max(pred_data2@y.values[[1]])
max2 = pred_data2@y.values[[1]][max1]
max3 = pred_data2@x.values[[1]][max1]

predict_train = ifelse(test = (train_predict > 0.01315682), 1, 0)

cm_train = table(train_data$Churn,predict_train)
per_after = cm_train[2,2]/(cm_train[2,2]+cm_train[1,2])

#Predict the values for test and eval data set
test_predict = predict(object = log_model,newdata = test_data,type = "response")
predict_test = ifelse(test = (test_predict > 0.01315682), 1, 0)
cm_test = table(test_data$Churn,predict_test)

eval_predict = predict(object = log_model,newdata = eval_data,type = "response")
predict_eval = ifelse(test = (eval_predict > 0.01315682), 1, 0)
cm_eval = table(eval_data$Churn,predict_eval)













