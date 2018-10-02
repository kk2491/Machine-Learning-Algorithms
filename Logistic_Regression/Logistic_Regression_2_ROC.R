rm(list = ls())
setwd("C:/Kishor/DoNotTouch/Lab_Room")

data = read.csv(file = "C:/Kishor/DoNotTouch/Videos/7th Set/20170430_Batch27_CSE7202c_Lab02_MultipleLinearReg/admission.csv")
str(data)
data$admit = as.factor(as.character(data$admit))
data$prestige = as.factor(as.character(data$prestige))

lgmodel = glm(formula = admit ~ .,data = data, family = "binomial")
summary(lgmodel)

#This calculates the probability of each data point
predict_data = predict(object = lgmodel,newdata = data,type = "response")
head(predict_data)

data_predict = ifelse((predict_data > 0.5),1,0)
head(data_predict)

cm_data = table(data$admit,data_predict)
accu_before = (cm_data[1,1]+cm_data[2,2])/(cm_data[1,1]+cm_data[2,2]+cm_data[1,2]+cm_data[2,1])

library(ROCR)
pred1 = prediction(predictions = predict_data, data$admit)

#For each cut off value it calculates the accuracy
pred2 = performance(prediction.obj = pred1,measure = "acc")

plot(pred2)

summary(pred2)

#Now we need to find the value with heighest Y value (accuracy)
#Then find the corresponding X value which is the cut off value

sn = slotNames(pred2)        #This is to find the slot names associated with the prediction output
sapply(sn, function(x) length(slot(pred2,x)))   #This will give the length of each
sapply(sn, function(x) class(slot(pred2,x))) 
slot(object = pred2,name = "y.name")

max1 = which.max(pred2@y.values[[1]])
max2 = pred2@y.values[[1]][max1]
max3 = pred2@x.values[[1]][max1]


data_predict = ifelse((predict_data > max3),1,0)
cm_data = table(data$admit,data_predict)

accu_after = (cm_data[1,1]+cm_data[2,2])/(cm_data[1,1]+cm_data[2,2]+cm_data[1,2]+cm_data[2,1])
