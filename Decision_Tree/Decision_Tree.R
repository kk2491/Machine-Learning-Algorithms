#C5.0 Decision Tree - Video 51, Looks like KNN is not done check again

rm(list = ls())
getwd()
setwd("C:/Kishor/DoNotTouch/Lab_Room")

#------------------------------------------
#Check what is train, test and eval dataset
#Check what is require() command
#------------------------------------------

# Rename the columns which has space or dots
attr = c('id', 'age', 'exp', 'inc', 'zip', 'family','ccavg', 'edu', 'mortgage', 'loan', 
         'securities', 'cd', 'online', 'cc') 

# Read the data using csv file
data = read.csv(file = "C:/Kishor/DoNotTouch/Videos/8th Set/UniversalBank.csv",header = TRUE, col.names = attr)

#Remove the unwanted columns (age, zip and id)

cor(data)

data = data[, -c(1,2,5)]   #Try other ways dropping columns

head(data)

summary(data)
str(data)

data$family=as.factor(data$family)
data$edu=as.factor(data$edu)
#data$mortgage=as.factor(data$mortgage)
data$loan=as.factor(data$loan)
data$securities=as.factor(data$securities)
data$cd=as.factor(data$cd)
data$online=as.factor(data$online)
data$cc=as.factor(data$cc)

#convert mortgage as numeric
data$mortgage=as.numeric(data$mortgage)

set.seed(13)
rows=seq(1,5000,1)
trainRows=sample(rows,3000)
remainingRows=rows[-(trainRows)]
testRows=sample(remainingRows, 1000)
evalRows=rows[-c(trainRows,testRows)]

train = data[trainRows,] 
test=data[testRows,] 
eval=data[evalRows,]

rm(evalRows,remainingRows,rows,testRows,trainRows)

#Decision Trees using C5.0 (For Classification Problem)

library(C50)
library(partykit)
#C5.0(x=train,y = train$loan,rules = TRUE)    #Check this why there is no data

dtC50= C5.0(loan ~ ., data = train, rules=TRUE)
summary(dtC50)
C5imp(dtC50, pct=TRUE)

plot(dtC50)

a = table(train$loan, predict(dtC50, newdata=train, type="class"))
rcTrain=(a[2,2])/(a[2,1]+a[2,2])*100
a=table(test$loan, predict(dtC50, newdata=test, type="class"))
rcTest=(a[2,2])/(a[2,1]+a[2,2])*100

#Experiment for best results
a=table(eval$loan, predict(dtC50, newdata=eval, type="class"))
rcEval=(a[2,2])/(a[2,1]+a[2,2])*100

cat("Recall in Training", rcTrain, '\n',
    "Recall in Testing", rcTest, '\n',
    "Recall in Evaluation", rcEval)

rm(a,rcEval,rcTest,rcTrain)

c5.0()


version
