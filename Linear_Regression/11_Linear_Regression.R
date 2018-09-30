#Linear Regression - Video 43

rm(list = ls())
getwd()
setwd("C:/Kishor/DoNotTouch/Lab_Room")

bigmac = read.csv("BigMac-NetHourlyWage.csv", header = T, sep = ",")

head(bigmac)

class(bigmac)

str(bigmac)

names(bigmac) = c("Country", "Big_Mac_Price", "Net_Hourly_Wage")

head(bigmac)

lm(data = bigmac, x = bigmac$Big_Mac_Price, y = bigmac$Net_Hourly_Wage)
bigmac_lm = lm(Net_Hourly_Wage~Big_Mac_Price, data = bigmac)

summary(lin)

par(mfrow=c(1,1))
plot(bigmac$Big_Mac_Price, bigmac$Net_Hourly_Wage)
abline(bigmac_lm)  # Regression line



par(mfrow=c(1,1))
plot(bigmac$Big_Mac_Price, bigmac$Net_Hourly_Wage)
abline(bigmac_lm)  # Regression line

# Check the Linear regression assumption using plots
par(mfrow=c(2,2))
plot(bigmac_lm)
str(bigmac_lm)

#-------------------- Calculations ------------------------

x = bigmac$Big_Mac_Price
y = bigmac$Net_Hourly_Wage

xbar = mean(x)
ybar = mean(y)

#---Computing Estimates---

# Computing Slope 
num = sum((x-xbar)*(y-ybar))
denom = sum((x-xbar)^2)
slope = num/denom
slope

# Computing Intercept
intercept = ybar-(slope*xbar)
intercept

# Computing covariance
# One step in R
cov_R = cov(x,y)

# Manual Calculations
n = nrow(bigmac)  
cov_manual = sum((x-xbar)*(y-ybar))/(n-1)
cov_manual
cov_R

#Computing correlation coefficient, r
# One step in R
cor_R = cor(x,y)  #r
# Manual Calculations
r = cov_manual/(sd(x)*sd(y))
r
cor_R


# Computing coefficient of determination
R2 = r^2
R2   

# Computing Std.Error
yhat = predict(bigmac_lm)   #Check this
sse = sum((y-yhat)^2) #sum of sq.errors or sum of sq. residuals
mse = sse/25          #degrees of freedom = n-k-1 i.e. k = 1
se = sqrt(mse) # Standard Error
se

se_var_x = se/(sqrt(sum((x-xbar)^2)))
se_var_x
se_intercept = se_var_x * sqrt(mean(x^2))
se_intercept

###Computing t-Value
#tValue <- ESTIMATE/se
tval_intercept = intercept/se_intercept
tval_var_x = slope/se_var_x
tval_intercept; tval_var_x

pval_intercept = pt(q=1.697, df = 26, lower.tail = F)*2
pval_var_x = pt(q=5.144, df = 25, lower.tail = F)*2
pval_intercept;pval_var_x

#Computing fitted values - yhat
yhat = as.numeric(bigmac_lm$fitted.values)
coef(bigmac_lm)[1]+coef(bigmac_lm)[2]*x[1]

#computing rSquared
ssx = sum((x-xbar)^2)
ssy = sum((y-ybar)^2)
ssxy = sum((x-xbar)*(y-ybar))

rSquared = cov((x-xbar),(y-ybar))^2/(var(x)*var(y))
summary(bigmac_lm)$r.squared

rSquared = sum((x-xbar)*(y-ybar))^2/(ssx*ssy)
summary(bigmac_lm)$r.squared

ss_tot = sum((y-ybar)^2)
ss_reg = sum((yhat-ybar)^2)
ss_res = sum((y-yhat)^2)

rSquared = 1-(ss_res/ss_tot)
