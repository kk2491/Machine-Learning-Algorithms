#C5.0 Decision Tree - For only Classification

rm(list = ls())
setwd("C:/Kishor/DoNotTouch/Lab_Room")

attr = c('id','age','exp','income','zip','family','ccavg','edu','mortgage','personal','security','cdact','online','creditcard')
data = read.csv(file = "UniversalBank.csv",col.names = attr, header = TRUE)

#Check the correlation between the attributes and remove the unwanted columns
cor(data)

#To check the NA values
sum(is.na(data))

#Remove ID, experience and zipcode

drop_attr = c('exp','zip','id')
attr = setdiff(x = attr, y = drop_attr)
data = data[,attr]

summary(data)
str(data)

#Convert the datatypes accordingly

cat_attr = c('family','edu','personal','security','cdact','online','creditcard')
num_attr = c('age','income','ccavg','mortgage')

cat_data = data.frame(sapply(X = data[,cat_attr],as.factor))
num_data = data.frame(sapply(X = data[,num_attr],as.numeric))

data = cbind(cat_data,num_data)
head(data)

str(data)

#Divide the dataset to train, test and eval

set.seed(123)
ids = 1:nrow(data)
train_id = sample(x = ids, length(ids)*0.6)
test_id = sample(x = setdiff(ids,train_id), length(ids)*0.2)
eval_id = setdiff(ids, c(train_id,test_id))

train_data = data[train_id,]
test_data = data[test_id,]
eval_data = data[eval_id,]

#----------------------------------------------------

#Decision tree using C5.0 

library(C50)

DCT = C5.0(personal ~ .,data = train_data,rules = TRUE)  
summary(DCT)
C5imp(object = DCT,pct = TRUE)   #This is to check the variable importance in decision tree

#Check the prediction on Train Data
pred_train = predict(object = DCT,newdata = train_data,type = "class")  
cm = table(pred_train,train_data$personal)
table(pred_train)
table(train_data$personal)
rcTrain = cm[1,1]/(sum(cm[1,1],cm[2,1]))   #2700/2708

#Check the prediction on Test data
pred_test = predict(object = DCT,newdata = test_data,type = "class")
cm_test = table(pred_test,test_data$personal)
rcTest = cm_test[1,1]/(sum(cm_test[1,1],cm_test[2,1]))

#Check the prediction on Eval data
pred_eval = predict(object = DCT,newdata = eval_data,type = "class")
cm_eval = table(pred_test,eval_data$personal)
rcEval = cm_eval[1,1]/(sum(cm_eval[1,1],cm_eval[2,1]))

#Print the results of all data sets
cat("Recall in Train Dataset", rcTrain, '\n',
    "Recall in Test Dataset" , rcTest, '\n',
    "Recall in Eval Dataset ", rcEval)





