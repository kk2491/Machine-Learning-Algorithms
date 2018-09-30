#KNN - Implementation of KNN algorithm for Classification and Regression problem

rm(list = ls())
setwd(dir = "C:/Kishor/DoNotTouch/Lab_Room")

#Load the required libraries (Dont know which one)
library(vegan)
library(dummies)
library(FNN)
library(DMwR)
library(class)

#redefine the columns in the dataset
attr = c('id','age','exp','income','zip','family','ccavg','education','mortgage','personal_loan','security','cd_act','online','credit_card')

data = read.csv(file = "Data_set.csv",col.names = attr, header = TRUE)

summary(data)
head(data)
str(data)
cor(data)

#remove the unwanted columns

drop_attr = c('id','exp','zip')
attr = setdiff(x = attr, y = drop_attr)
data = data[,attr]
head(data)

#Check for null values and outliers 
sum(is.na(data))

#Outliers can be found by using the boxplot or by analysing the summary statistics of the data set

#Convert the columns into appropriate datatypes

cat_attr = c('family','education','personal_loan','security','cd_act','online','credit_card')
num_attr = setdiff(x = attr, y = cat_attr)
cat_data = data.frame(sapply(X = data[,cat_attr],as.factor))
num_data = data.frame(sapply(X = data[,num_attr],as.numeric))

#Here we created 2 separate tables and converted into factor and numeric accordingly

head(cat_data)
str(cat_data)
head(num_data)
str(num_data)

#Add both the tables
data = cbind(num_data,cat_data)

# Check for missing values and outliers.
summary(data)
head(data)
str(data)

#---------------------------------------------------------------------------
#Build the classification model (Our target variable is personal_loan)

cls_num_attr = num_attr
cls_cat_attr = setdiff(x = cat_attr, y = 'personal_loan')

#Standardizing the numeric data
cls_data = decostand(x = data[,cls_num_attr],method = "range")  #This will standardize the data from 0 to 1 
head(cls_data)
summary(cls_data)

#Convert all the categorical data into numeric data (Using dummy and as.numeric function)

# 1: Using dummy function convert family and education to numeric variables
edu = dummy(x = data$education)
family = dummy(x = data$family)

head(edu)
head(data$education)

cls_data = cbind(cls_data,edu,family)

head(cls_data)

cls_cat_attr = setdiff(x = cls_cat_attr, y = c('education','family'))

#2 : Using as.numeric convert the remaining columns to categorical

cls_data = cbind(cls_data, sapply(data[,cls_cat_attr], as.numeric))

head(cls_data)
str(cls_data)

cls_attr = names(cls_data)

#Append the target variable (personal_loan)

cls_data = cbind(cls_data, loan = data[,"personal_loan"])

#Divide the dataset into train, test and eval

set.seed(123)

row_IDs = 1:nrow(cls_data)
train_row_IDs = sample(x = row_IDs, size = length(row_IDs)*0.6 )
test_rows_IDs = sample(x = setdiff(row_IDs,train_row_IDs), size = length(row_IDs)*0.2)
eval_rows_IDs = setdiff(row_IDs, c(train_row_IDs,test_rows_IDs)) 

train_data = cls_data[train_row_IDs,]
test_data = cls_data[test_rows_IDs,]
eval_data = cls_data[eval_rows_IDs,]

rm(train_row_IDs, test_rows_IDs, eval_rows_IDs, row_IDs)

#This is to check how many are 0s and 1s
table(cls_data$loan)
table(train_data$loan)
table(test_data$loan)
table(eval_data$loan)

#Build KNN model

#For K = 1
pred_train = knn(train = train_data[,cls_attr],test = train_data[,cls_attr],cl = train_data$loan, k=1)
cm_train = table(pred_train, train_data$loan)
accu_train = sum(diag(cm_train))/sum(cm_train)

pred_test = knn(train = train_data[,cls_attr],test = test_data[,cls_attr],cl = train_data$loan, k =1 )
cm_test = table(pred_test, test_data$loan)
accu_test = sum(diag(cm_test))/sum(cm_test)

accu_train
accu_test

rm(pred_test, pred_train, cm_train, cm_test)

#For k = 3

pred_train = knn(train = train_data[,cls_attr],test = train_data[,cls_attr],cl = train_data$loan, k=3)
cm_train = table(pred_train, train_data$loan)
accu_train = sum(diag(cm_train))/sum(cm_train)

pred_test = knn(train = train_data[,cls_attr],test = test_data[,cls_attr],cl = train_data$loan, k =3 )
cm_test = table(pred_test, test_data$loan)
accu_test = sum(diag(cm_test))/sum(cm_test)

accu_train
accu_test

rm(pred_test, pred_train, cm_train, cm_test)

#For k = 5

pred_train = knn(train = train_data[,cls_attr],test = train_data[,cls_attr],cl = train_data$loan, k=5)
cm_train = table(pred_train, train_data$loan)
accu_train = sum(diag(cm_train))/sum(cm_train)

pred_test = knn(train = train_data[,cls_attr],test = test_data[,cls_attr],cl = train_data$loan, k =5 )
cm_test = table(pred_test, test_data$loan)
accu_test = sum(diag(cm_test))/sum(cm_test)

accu_train
accu_test

rm(pred_test, pred_train, cm_train, cm_test)

#For k = 7

pred_train = knn(train = train_data[,cls_attr],test = train_data[,cls_attr],cl = train_data$loan, k=7)
cm_train = table(pred_train, train_data$loan)
accu_train = sum(diag(cm_train))/sum(cm_train)

pred_test = knn(train = train_data[,cls_attr],test = test_data[,cls_attr],cl = train_data$loan, k =7 )
cm_test = table(pred_test, test_data$loan)
accu_test = sum(diag(cm_test))/sum(cm_test)

accu_train
accu_test

rm(pred_test, pred_train, cm_train, cm_test)

#Condense training dataset  (Check this condense for reducing the train dataset)
keep = condense(train = train_data, class = train_data$loan)
length(keep)
nrow(train_data)
keep

#Build the model using the condensed train data

pred = knn(train = train_data[keep, cls_attr], test = test_data[,cls_attr], 
           cl = train_data$loan[keep], k = 5)

cm = table(pred,test_data$loan)
acu = sum(diag(cm))/sum(cm)

acu

#Evaluate the model on the eval data

pred_eval = knn(train = train_data[keep, cls_attr],test = eval_data[,cls_attr],
                cl = train_data$loan[keep], k = 5)
cm_eval = table(pred_eval, eval_data$loan)
acu_eval = sum(diag(cm_eval))/sum(cm_eval)
acu_eval

#--------------------------Classification Completed--------------------------------#

#Here the target variable is income
reg_num_attr = setdiff(num_attr,"income")
reg_cat_attr = cat_attr

#Standardizing the numeric values
reg_data = decostand(x = data[,reg_num_attr],method = "range")

#Convert all categorical variables to numeric variables

#1. Use dummies to convert family and education

edu = dummy(data$education)
family = dummy(data$family)
reg_data = cbind(reg_data, edu, family)
reg_cat_attr = setdiff(reg_cat_attr, c('education','family'))

#2. Using asnumeric convert the remaining into numeric data type
reg_data = cbind(reg_data, sapply(data[,reg_cat_attr],as.numeric))

reg_attr = names(reg_data)

#Now append the target attribute

reg_data = cbind(reg_data, inc = data$income)

str(reg_data)

#Divide the data into train, test and eval

set.seed(123)

rids = 1: nrow(reg_data)
train_ids = sample(x = rids,size = length(rids)*0.6)
test_ids = sample(setdiff(rids, train_ids), size = length(rids)*0.2)
eval_ids = setdiff(rids, c(train_ids,test_ids))

rm(rids)

rtrain_data = reg_data[train_ids,]
rtest_data = reg_data[test_ids,]
reval_data = reg_data[eval_ids,]

rm(train_ids,test_ids,eval_ids)

#Check how the data is split in above 3 wrt target attribute
summary(reg_data$inc)
summary(rtrain_data$inc)
summary(rtest_data$inc)
summary(reval_data$inc)

#Build KNN model

#For k = 1
pred_train = knn.reg(train = rtrain_data[,reg_attr],test = rtrain_data[,reg_attr],y = rtrain_data$inc,k = 1)
#See this pred_train has got many attributes not only the predicted target variables
pred_test = knn.reg(train = rtrain_data[,reg_attr], test = rtest_data[,reg_attr],y= rtrain_data$inc, k = 1)

regr.eval(trues = rtrain_data[,"inc"],preds = pred_train$pred)
regr.eval(trues = rtest_data[,"inc"],preds = pred_test$pred)

rm(pred_train,pred_test)

#For k = 3
pred_train = knn.reg(train = rtrain_data[,reg_attr],test = rtrain_data[,reg_attr],y = rtrain_data$inc,k = 3)
pred_test = knn.reg(train = rtrain_data[,reg_attr], test = rtest_data[,reg_attr],y= rtrain_data$inc, k = 3)
regr.eval(trues = rtrain_data[,"inc"],preds = pred_train$pred)
regr.eval(trues = rtest_data[,"inc"],preds = pred_test$pred)
rm(pred_train,pred_test)

#For k = 5
pred_train = knn.reg(train = rtrain_data[,reg_attr],test = rtrain_data[,reg_attr],y = rtrain_data$inc,k = 5)
pred_test = knn.reg(train = rtrain_data[,reg_attr], test = rtest_data[,reg_attr],y= rtrain_data$inc, k = 5)
regr.eval(trues = rtrain_data[,"inc"],preds = pred_train$pred)
regr.eval(trues = rtest_data[,"inc"],preds = pred_test$pred)
rm(pred_train,pred_test)

#For k = 7
pred_train = knn.reg(train = rtrain_data[,reg_attr],test = rtrain_data[,reg_attr],y = rtrain_data$inc,k = 7)
pred_test = knn.reg(train = rtrain_data[,reg_attr], test = rtest_data[,reg_attr],y= rtrain_data$inc, k = 7)
regr.eval(trues = rtrain_data[,"inc"],preds = pred_train$pred)
regr.eval(trues = rtest_data[,"inc"],preds = pred_test$pred)
rm(pred_train,pred_test)

#For k = 9
pred_train = knn.reg(train = rtrain_data[,reg_attr],test = rtrain_data[,reg_attr],y = rtrain_data$inc,k = 9)
pred_test = knn.reg(train = rtrain_data[,reg_attr], test = rtest_data[,reg_attr],y= rtrain_data$inc, k = 9)
regr.eval(trues = rtrain_data[,"inc"],preds = pred_train$pred)
regr.eval(trues = rtest_data[,"inc"],preds = pred_test$pred)
rm(pred_train,pred_test)

#For k = 11
pred_train = knn.reg(train = rtrain_data[,reg_attr],test = rtrain_data[,reg_attr],y = rtrain_data$inc,k = 11)
pred_test = knn.reg(train = rtrain_data[,reg_attr], test = rtest_data[,reg_attr],y= rtrain_data$inc, k = 11)
regr.eval(trues = rtrain_data[,"inc"],preds = pred_train$pred)
regr.eval(trues = rtest_data[,"inc"],preds = pred_test$pred)
rm(pred_train,pred_test)


#Test the final model and report the results

pred_final = knn.reg(train = rtrain_data[,reg_attr], test = reval_data[,reg_attr],y= rtrain_data$inc, k = 5)
regr.eval(trues = reval_data[,"inc"],preds = pred_final$pred)










