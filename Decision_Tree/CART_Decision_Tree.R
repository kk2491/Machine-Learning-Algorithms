#CART Decision Tree - Can be used for both classification and regression - Video 51

rm(list = ls())
setwd("C:/Kishor/DoNotTouch/Lab_Room")
par(mfrow=c(1,1))
#CART decision tree for Regression - Target variable is Income

attr = c('id','age','exp','income','zip','family','ccavg','edu','mortagage','loan','security','cdact','online','ccard')
data = read.csv(file = "UnivBank.csv",header = TRUE,col.names = attr)
colnames(data)
str(data)

#Check the correlation and remove the unwanted columns
cor(data)
drop_attr = c('id','exp','zip') 
attr = setdiff(attr,drop_attr)

data = data[,attr]   #This is the updated dataset after removing unwanted columns

str(data)

#Change the datatype of the columns accordingly
cat_attr = c('family','edu','loan','security','cdact','online','ccard')
num_attr = c('age','ccavg','mortagage')

cat_data = data.frame(sapply(X = data[,cat_attr], as.factor))
num_data = data.frame(sapply(X = data[,num_attr], as.numeric))

data1 = cbind(cat_data,num_data)
data = cbind(data1,inc = data$income)

rm(data1,cat_data,num_data)
str(data)

#Divide the dataset into train, test and eval
set.seed(123)
ids = 1:nrow(data)
train_ids = sample(x = ids,size = length(ids)*0.6)
test_ids = sample(x = setdiff(ids,train_ids), size = length(ids)*0.2)
eval_ids = setdiff(x = ids,y = c(train_ids,test_ids))

train_data = data[train_ids,]
test_data = data[test_ids,]
eval_data = data[eval_ids,]

rm(train_ids,test_ids,eval_ids)

#Load rpart library

library(rpart)        # This is to use rpart (CART algo)
library(rpart.plot)   # This is to plot

#Build the model using train data
cart_model = rpart(formula = inc ~.,data = train_data,method = "anova")  #Now the model is ready
summary(cart_model)

#plot the cart_model 
plot(cart_model,main = "Decision Tree for income",uniform = TRUE)
text(cart_model,cex = .7,use.n = TRUE,xpd =TRUE)    #cex - text size in plot
prp(cart_model, faclen = 0, cex = 0.5, extra = 0)   #faclen = 0 use full name of the factor labels
prp(cart_model, faclen = 0, cex = 0.5, extra = 1)   #extra = 1 add number of observations at each node

printcp(cart_model)    #Check the complexity parameter for each iteration, and 
                       
#Predict the values for train data using the model obtained

pred_train = predict(object = cart_model,newdata = train_data,type = "vector")
pred_test = predict(object = cart_model,newdata = test_data,type = "vector")
pred_eval = predict(object = cart_model,newdata = eval_data,type = "vector")


#Library DMWR to use regr.eval function

library(DMwR)
regr.eval(trues = train_data$inc,preds = pred_train, train.y = train_data$inc)
regr.eval(trues = test_data$inc,preds = pred_test, train.y = test_data$inc)
regr.eval(trues = eval_data$inc,preds = pred_eval, train.y = eval_data$inc)

#Select the appropriate complexity parameter to avoid over fitting of training data

printcp(cart_model)  

#select the one with the least xerror
bestcp = cart_model$cptable[which.min(cart_model$cptable[,'xerror']),"CP"]  

#We can directly prune the built model or we can build the new model using the right complexity parameter

#cart_model_pruned = prune(tree = cart_model,cp = bestcp)
#prp(cart_model_pruned,faclen = 0, cex = 0.5, extra = 1)

#Build 2 model using 2 different CP and check the error matrix
cart_model1 = rpart(formula = inc ~.,data = train_data,method = "anova",cp = 0.01)
prp(x = cart_model1,faclen = 0,extra = 1,cex = 0.5)

cart_model2 = rpart(formula = inc ~.,data = train_data,method = "anova",cp = 0.03)
prp(x = cart_model2,faclen = 0,extra = 1,cex = 0.5)

#Predict the values for all 3 datasets
#For cp = 0.01
pred_train = predict(object = cart_model1,newdata = train_data,type = "vector")
pred_test = predict(object = cart_model1,newdata = test_data,type = "vector")
pred_eval = predict(object = cart_model1,newdata = eval_data,type = "vector")

#Check the error matrix
regr.eval(trues = train_data$inc,preds = pred_train,train.y = train_data$inc)
regr.eval(trues = test_data$inc,preds = pred_test,train.y = train_data$inc)
regr.eval(trues = eval_data$inc,preds = pred_eval,train.y = train_data$inc)

#For cp = 0.03
pred_train = predict(object = cart_model2,newdata = train_data,type = "vector")
pred_test = predict(object = cart_model2,newdata = test_data,type = "vector")
pred_eval = predict(object = cart_model2,newdata = eval_data,type = "vector")

#Check the error matrix
regr.eval(trues = train_data$inc,preds = pred_train,train.y = train_data$inc)
regr.eval(trues = test_data$inc,preds = pred_test,train.y = train_data$inc)
regr.eval(trues = eval_data$inc,preds = pred_eval,train.y = train_data$inc)

#============================================================================

#CART - decision tree for Classification problem

cart_class = rpart(formula = loan ~.,data = train_data,method = "class")
plot(x = cart_class,main="Classification tree for Loan",uniform = TRUE)
text(x = cart_class,use.n = TRUE, cex = 0.7)
prp(x = cart_class,extra = 1,faclen = 0,cex = 0.5)

summary(cart_class)

pred_train = predict(object = cart_class,newdata = train_data,type = "class")
pred_test = predict(object = cart_class,newdata = test_data,type = "class")
pred_eval = predict(object = cart_class,newdata = eval_data,type = "class")

cm_train = table(train_data$loan,pred_train)
rc_train = cm_train[1,1]/sum(cm_train[1,1],cm_train[1,2])
cm_test = table(test_data$loan,pred_test)
rc_test = cm_test[1,1]/sum(cm_test[1,1],cm_test[1,2])
cm_eval = table(eval_data$loan,pred_eval)
rc_eval = cm_eval[1,1]/sum(cm_eval[1,1],cm_eval[1,2])

printcp(cart_class)

cart_class_prune = rpart(formula = loan ~.,data = train_data,method = "class",cp = 0.01)

pred_train = predict(object = cart_class_prune,newdata = train_data,type = "class")
pred_test = predict(object = cart_class_prune,newdata = test_data,type = "class")
pred_eval = predict(object = cart_class_prune,newdata = eval_data,type = "class")

cm_train = table(train_data$loan,pred_train)
rc_train = cm_train[1,1]/sum(cm_train[1,1],cm_train[1,2])
cm_test = table(test_data$loan,pred_test)
rc_test = cm_test[1,1]/sum(cm_test[1,1],cm_test[1,2])
cm_eval = table(eval_data$loan,pred_eval)
rc_eval = cm_eval[1,1]/sum(cm_eval[1,1],cm_eval[1,2])

#-------------------------------------------------------------------------










