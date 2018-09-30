#KNN - Vidoe 51 (Not explained in the video)

rm(list = ls())

setwd(dir = "C:/Kishor/DoNotTouch/Lab_Room")

#Load the required libraries (Dont know which one)




#redefine the columns in the dataset
attr = c('id','age','exp','income','zip','family','ccavg','education','mortgage','personal_loan','security','cd_act','online','credit_card')

data = read.csv(file = "C:/Kishor/DoNotTouch/Videos/8th Set/UniversalBank.csv")

summary(data)

head(data)