rm(list = ls())
cat("\014")
library(ggplot2)
library(dplyr)
library(glmnet)
library(car)
library(randomForest)
library(gridExtra)
set.seed(1)

#setwd("C:\\Users\\Aron\\Data Mining")
dja = read.csv("DJA.csv")
dja = dja[, 2]

#create the data matrix
M = 100
p = 40
n = length(dja) - p - 1
nX = matrix(0, nrow = n, ncol = p)
y = matrix(0, nrow = n, ncol = 1)
k = 1
for (i in (p+1):(n+p)) {
  X[k,]  = dja[(i-1) : (i-p)] 
  y[k,1] = dja[i]
  k      = k + 1
}

#set n.train and n.test
n.train  =   floor(0.8 * n)
n.test =   n  - n.train

#set up vectors to store the R squared errors
Rsq.train.las  =     rep(0,M)
Rsq.train.en   =     rep(0,M)
Rsq.train.rid  =     rep(0,M)
Rsq.train.rf   =     rep(0,M)

Rsq.test.las   =     rep(0,M)
Rsq.test.en    =     rep(0,M)  
Rsq.test.rid   =     rep(0,M)  
Rsq.test.rf    =     rep(0,M)  

#get times in order to compare run times for each model
start.time  = rep(NA, M)
end.time    = rep(NA, M)
time.las    = rep(NA, M)
time.en     = rep(NA, M)
time.rid    = rep(NA, M)
time.rf     = rep(NA, M)


#run cross validation for lasso, elastic net, ridge, and random forest
for (m in c(1:M)) {
  shuffled_indexes =     sample(n)
  train            =     shuffled_indexes[1:n.train]
  test             =     shuffled_indexes[(1+n.train):n]
  X.train          =     X[train, ]
  y.train          =     y[train]
  X.test           =     X[test, ]
  y.test           =     y[test]
  
  #fit lasso 
  
  start.time[m]    =     Sys.time()
  cv.fit.las       =     cv.glmnet(X.train, y.train, alpha = 1, nfolds = 10, type.measure = "mae")
  fit.las          =     glmnet(X.train, y.train, alpha = 1, lambda = cv.fit.las$lambda.min)
  y.train.hat.las  =     predict(fit.las, newx = X.train, type = "response") # y.train.hat=X.train %*% fit$beta + fit$a0
  y.test.hat.las   =     predict(fit.las, newx = X.test, type = "response") # y.test.hat=X.test %*% fit$beta  + fit$a0
  Rsq.test.las[m]  =     1-mean((y.test - y.test.hat.las)^2)/mean((y - mean(y))^2)
  Rsq.train.las[m] =     1-mean((y.train - y.train.hat.las)^2)/mean((y - mean(y))^2)  
  end.time[m]      =     Sys.time()
  time.las[m]      =     end.time[m] - start.time[m]
  
  #fit elastic net 
  
  start.time[m]    =     Sys.time()
  cv.fit.en        =     cv.glmnet(X.train, y.train, alpha = 0.5, nfolds = 10, type.measure = "mae")
  fit.en           =     glmnet(X.train, y.train, alpha = 0.5, lambda = cv.fit.en$lambda.min)
  y.train.hat.en   =     predict(fit.en, newx = X.train, type = "response") # y.train.hat=X.train %*% fit$beta + fit$a0
  y.test.hat.en    =     predict(fit.en, newx = X.test, type = "response") # y.test.hat=X.test %*% fit$beta  + fit$a0
  Rsq.test.en[m]   =     1-mean((y.test - y.test.hat.en)^2)/mean((y - mean(y))^2)
  Rsq.train.en[m]  =     1-mean((y.train - y.train.hat.en)^2)/mean((y - mean(y))^2)  
  end.time[m]      =     Sys.time()
  time.en[m]       =     end.time[m] - start.time[m]
  
  #fit ridge 
  
  start.time[m]    =     Sys.time()
  cv.fit.rid       =     cv.glmnet(X.train, y.train, alpha = 0, nfolds = 10, type.measure = "mae")
  fit.rid          =     glmnet(X.train, y.train, alpha = 0, lambda = cv.fit.rid$lambda.min)
  y.train.hat.rid  =     predict(fit.rid, newx = X.train, type = "response") 
  y.test.hat.rid   =     predict(fit.rid, newx = X.test, type = "response") 
  Rsq.test.rid[m]  =     1-mean((y.test - y.test.hat.rid)^2)/mean((y - mean(y))^2)
  Rsq.train.rid[m] =     1-mean((y.train - y.train.hat.rid)^2)/mean((y - mean(y))^2)  
  end.time[m]      =     Sys.time()
  time.rid[m]      =     end.time[m] - start.time[m]
  
  
  #fit random forest
  
  start.time[m]    =     Sys.time()
  fit.rf           =     randomForest(X.train, y.train, mtry = sqrt(p), importance = TRUE)
  y.test.hat.rf    =     predict(fit.rf, X.test)
  y.train.hat.rf   =     predict(fit.rf, X.train)
  Rsq.test.rf[m]   =     1-mean((y.test - y.test.hat.rf)^2)/mean((y - mean(y))^2)
  Rsq.train.rf[m]  =     1-mean((y.train - y.train.hat.rf)^2)/mean((y - mean(y))^2)  
  end.time[m]      =     Sys.time()
  time.rf[m]       =     end.time[m] - start.time[m]
  
  cat(sprintf("m=%3.f| Rsq.test.las=%.2f,  Rsq.train.las=%.2f| Rsq.test.en=%.2f,  Rsq.train.en=%.2f| Rsq.test.rid=%.2f,  Rsq.train.rid=%.2f| Rsq.test.rf=%.2f,  Rsq.train.rf=%.2f| 
              \n", m,  Rsq.test.las[m], Rsq.train.las[m],  Rsq.test.en[m], Rsq.train.en[m], Rsq.test.rid[m], Rsq.train.rid[m],  Rsq.test.rf[m], Rsq.train.rf[m]))
  
  
}

#boxplots of R squared train and test errors 

par(mfrow=c(1,2))
boxplot(Rsq.train.las, Rsq.train.en, Rsq.train.rid, Rsq.train.rf,
        main = "Boxplots for R^2 Train Error",
        at = c(1,2,3,4),
        names = c("Lasso", "Elastic Net", "Ridge", "Random Forest"),
        las = 2,
        col = c("red","yellow","blue","green"),
        border = "black",
        horizontal = FALSE,
        notch = TRUE,
        ylim = c(0.85, 1)
)

boxplot(Rsq.test.las, Rsq.test.en, Rsq.test.rid, Rsq.test.rf,
        main = "Boxplots for R^2 Test Error",
        at = c(1,2,3,4),
        names = c("Lasso", "Elastic Net", "Ridge", "Random Forest"),
        las = 2,
        col = c("red","yellow","blue","green"),
        border = "black",
        horizontal = FALSE,
        notch = TRUE,
        ylim = c(0.85, 1)
)

#plot 10-fold cross validation curve for one of the 100 samples

par(mfrow=c(3,1))
plot(cv.fit.las, main = "10-fold CV Curve for Lasso") #lasso only uses the 2 previous days
plot(cv.fit.en, main = "10-fold CV Curve for Elastic Net") #ridge chooses 5 parameters
plot(cv.fit.rid, main = "10-fold CV Curve for Ridge") #ridge uses all 40 days to predict


#plot residuals

res.train.las     =     c(y.train.hat.las - y.train)
res.train.en      =     c(y.train.hat.en - y.train)
res.train.rid     =     c(y.train.hat.rid - y.train)
res.train.rf      =     c(y.train.hat.rf - y.train)

res.test.las      =     c(y.test.hat.las - y.test)
res.test.en       =     c(y.test.hat.en - y.test)
res.test.rid      =     c(y.test.hat.rid - y.test)
res.test.rf       =     c(y.test.hat.rf - y.test)

res.lasso            =     data.frame(c(rep("train", n.train),rep("test", n.test)), 
                                                  c(1:n),
                                                  c(y.train.hat.las - y.train, y.test.hat.las - y.test))
colnames(res.lasso)  =     c("state", "time", "residual")
res.lasso.plot       =     ggplot(res.lasso, aes(x=time, y=residual, colour=state)) + geom_line()
res.lasso.plot 


par(mfrow=c(1,2))

boxplot(res.train.las, res.train.en, res.train.rid, res.train.rf,
        main = "Boxplots of Train Residuals",
        at = c(1,2,3,4),
        names = c("Lasso", "Elastic Net", "Ridge", "Random Forest"),
        las = 2,
        col = c("red","yellow","blue","green"),
        border = "black",
        horizontal = FALSE,
        notch = TRUE,
        ylim = c(-1000,1000)
)

boxplot(res.test.las, res.test.en, res.test.rid, res.test.rf,
        main = "Boxplots of Test Residuals",
        at = c(1,2,3,4),
        names = c("Lasso", "Elastic Net", "Ridge", "Random Forest"),
        las = 2,
        col = c("red","yellow","blue","green"),
        border = "black",
        horizontal = FALSE,
        notch = TRUE,
        ylim = c(-1000,1000)
)

#create bootstrap bar plots using all the data

bootstrapSamples  =     100
beta.bs.las       =     matrix(0, nrow = p, ncol = bootstrapSamples)    
beta.bs.en        =     matrix(0, nrow = p, ncol = bootstrapSamples)    
beta.bs.rid       =     matrix(0, nrow = p, ncol = bootstrapSamples)    
beta.bs.rf        =     matrix(0, nrow = p, ncol = bootstrapSamples)         

for (m in 1:bootstrapSamples){
  bs_indexes       =     sample(n, replace=T)
  X.bs             =     X[bs_indexes, ]
  y.bs             =     y[bs_indexes]
  
  #fit bs for lasso
  cv.fit.bs.las    =     cv.glmnet(X.bs, y.bs, alpha = 1, nfolds = 10, type.measure = "mae")
  fit.bs.las       =     glmnet(X.bs, y.bs, alpha = 1, lambda = cv.fit.bs.las$lambda.min)
  beta.bs.las[,m]  =     as.vector(fit.bs.las$beta)
  
  #fit bs for elastic net
  cv.fit.bs.en     =     cv.glmnet(X.bs, y.bs, alpha = 0.5, nfolds = 10, type.measure = "mae")
  fit.bs.en        =     glmnet(X.bs, y.bs, alpha = 0.5, lambda = cv.fit.bs.en$lambda.min)
  beta.bs.en[,m]   =     as.vector(fit.bs.en$beta)
  
  #fit bs for ridge
  cv.fit.bs.rid    =     cv.glmnet(X.bs, y.bs, alpha = 0, nfolds = 10, type.measure = "mae")
  fit.bs.rid       =     glmnet(X.bs, y.bs, alpha = 0, lambda = cv.fit.bs.rid$lambda.min)
  beta.bs.rid[,m]  =     as.vector(fit.bs.rid$beta)
  
  #fit bs rf
  fit.bs.rf        =     randomForest(X.bs, y.bs, mtry = sqrt(p), importance = TRUE)
  beta.bs.rf[,m]   =     as.vector(fit.bs.rf$importance[,1])
  
  cat(sprintf("Bootstrap Sample %3.f \n", m))
  
}
#calculate bootstrapped standard errors 
bs.sd.las    = apply(beta.bs.las, 1, "sd")
bs.sd.en     = apply(beta.bs.en, 1, "sd")
bs.sd.rid    = apply(beta.bs.rid, 1, "sd")
bs.sd.rf     = apply(beta.bs.rf, 1, "sd")


#fit lasso to the whole data
cv.fit.full.las  =     cv.glmnet(X, y, alpha = 1, nfolds = 10)
fit.full.las     =     glmnet(X, y, alpha = 1, lambda = cv.fit.full.las$lambda.min)

#fit elastic net to the whole data
cv.fit.full.en   =     cv.glmnet(X, y, alpha = 0.5, nfolds = 10)
fit.full.en      =     glmnet(X, y, alpha = 0.5, lambda = cv.fit.full.en$lambda.min)

#fit ridge to the whole data
cv.fit.full.rid  =     cv.glmnet(X, y, alpha = 0, nfolds = 10)
fit.full.rid     =     glmnet(X, y, alpha = 0, lambda = cv.fit.full.rid$lambda.min)

#fit rf to the whole data
#something is wrong here.
fit.full.rf           =     randomForest(X, as.vector(y), mtry = sqrt(p), importance = TRUE)


betaS.las              =     data.frame(c(1:p), as.vector(fit.full.las$beta), 2*bs.sd.las)
colnames(betaS.las)    =     c( "feature", "value", "err")

betaS.en               =     data.frame(c(1:p), as.vector(fit.full.en$beta), 2*bs.sd.en)
colnames(betaS.en)     =     c( "feature", "value", "err")

betaS.rid              =     data.frame(c(1:p), as.vector(fit.full.rid$beta), 2*bs.sd.rid)
colnames(betaS.rid)    =     c( "feature", "value", "err")

betaS.rf               =     data.frame(c(1:p), as.vector(fit.full.rf$importance[,1]), 2*bs.sd.rf)
colnames(betaS.rf)     =     c( "feature", "value", "err")


plot.las =  ggplot(betaS.las, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")    +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.2) +
  ggtitle("Lasso")
  
plot.en =  ggplot(betaS.en, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")    +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.2) +
  ggtitle("Elastic Net")

plot.rid =  ggplot(betaS.rid, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")    +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.2) +
  ggtitle("Ridge")

plot.rf =  ggplot(betaS.rf, aes(x=feature, y=value)) +
  geom_bar(stat = "identity", fill="white", colour="black")    +
  geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.2) +
  ggtitle("Random Forest")


grid.arrange(plot.las, plot.en, plot.rid, plot.rf, nrow = 4)



#barplot of each model showing which coefficients are used and their weights
par(mfrow=c(2,2))
barplot(as.vector(fit.las$beta), main = "Barplot of Lasso")
barplot(as.vector(fit.en$beta), main = "Barplot of Elastic Net")
barplot(as.vector(fit.rid$beta), main = "Barplot of Ridge")


#barplot showing three models together
betaS           =     data.frame(c(rep("lasso", p),  rep("elastic net", p), rep("ridge", p) ), 
                                  c(1:p, 1:p, 1:p), c(betaS.las[,2], betaS.en[,2], betaS.rid[,2]))

colnames(betaS) =     c("method", "lag.index", "beta")
 
beta.plot       =   ggplot(betaS, aes(x=lag.index, y=beta, fill= method)) 
beta.plot       =   beta.plot + geom_bar(position = "dodge", colour = "black", stat="identity") +
  ggtitle("Comparison of Betas Among the Models")
beta.plot


#run time to train each model
mean(time.las)
mean(time.en)
mean(time.rid)
mean(time.rf)
