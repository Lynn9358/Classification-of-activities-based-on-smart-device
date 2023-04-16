library(dplyr)
library(caret)
library(glmnet)
library(randomForest)
library(randomForestSRC)
library(doParallel)
library(caret)
library(gbm)
library(xgboost)
library(e1071)
library(SwarmSVM)
library(class)
library(MASS)

#The object of this task is to classify activities base on the data from smart device, the training data are the datasets for subject 1 ,3 ,5 ,6 ,7, 8, 11, 14, 15, 16, 17, 19, 21, 22, 23, 25, 26, 27, 28, 29, 30, and test data are the datasets for subject 2, 4, 9, 10, 12, 13, 18, 20, 24. However, for different individuals, there are differences in the data levels under the same activity, therefore, the training data are classified by subject index as training set and test set and the evaluation of model can be more accurate.

#Task 1
#data cleaning for training data
set.seed(224)
trd = training_data
tsd = test_data
trd$status1 <- ifelse(trd$activity > 3,0,1)
trd$status2 <- ifelse(trd$activity <= 6, trd$activity, 7)
trd = trd[,!colnames(trd) %in% "activity"]


#Create training set and test set
index = as.vector(as.numeric(rownames(table(trd$subject))))
testindex <- list()
testindex[[1]] = sample(index,7,replace = FALSE)
testindex[[2]] = sample(index[!(index %in% testindex[[1]])],7,replace = FALSE)
testindex[[3]] = setdiff(index,union(testindex[[1]],testindex[[2]]))


#Lasso regression with 3-fold cross validatio
#Lasso model
acc1_lasso <- c()
for(i in 1:3){
  testset = trd[trd$subject %in% testindex[[i]],]
  trainset = trd[!(trd$subject %in% testindex[[i]]),]
  model1_lasso = glmnet(x = as.matrix(trainset[2:562]), y = trainset$status1, family = "binomial", alpha = 1,lambda = 1)
  pred1_lasso <- predict(model1_lasso, newx = as.matrix(testset[2:562]),s = 1, type = "response")
  pred1_lasso <-  ifelse(pred_lasso > 0.5, 1 ,0)
  matrix1_lasso  <- table(as.matrix(pred_lasso), as.matrix(testset$status1))
  acc1_lasso[i] = sum(diag(matrix_lasso))/nrow(testset)
}
acc1_lasso = sum(acc1_lasso)/3


#Optimized Lasso model
acc1_lasso2 <- c()

cv_fit <- cv.glmnet(x = as.matrix(trainset[2:562]), y = trainset$status1, family = "binomial", nfolds = 3, alpha = 1)
lambda_opt <- cv_fit$lambda.min

for(i in 1:3){
  testset = trd[trd$subject %in% testindex[[i]],]
  trainset = trd[!(trd$subject %in% testindex[[i]]),]
  model1_lasso2 <- glmnet(x = as.matrix(trainset[2:562]), y = trainset$status1, family = "binomial", alpha = 1, lambda = lambda_opt)
  pred1_lasso2 <- predict(model1_lasso2, newx = as.matrix(testset[2:562]), s = lambda_opt, type = "response")
  pred1_lasso2 <-  ifelse(pred1_lasso2 > 0.5, 1 ,0)
  matrix1_lasso2  <- table(as.matrix(pred1_lasso2), as.matrix(testset$status1))
  acc1_lasso2[i] = sum(diag(matrix1_lasso2))/nrow(testset)
}
  acc1_lasso2 = sum(acc1_lasso2)/3

#BaselineGLM model with
acc1_glm <- c()
  for(i in 1:3){
    testset = trd[trd$subject %in% testindex[[i]],]
    trainset = trd[!(trd$subject %in% testindex[[i]]),]
    model1_glm = glm(status1~. , data = trainset[2:563], family = binomial)
    pred1_glm = predict(model1_glm, newdata = testset[2:563])
    pred1_glm = ifelse(pred1_glm<0, 0,1)
    matrix1_glm =  table(as.matrix(pred1_glm), as.matrix(testset$status1))
    acc1_glm[i] = sum(diag(matrix1_glm))/nrow(testset)
    print(matrix1_glm)
    print(acc1_glm[i])
  }
 
mean(acc1_glm)
  
  
  
#GLM model with changed threshold
#3-fold cv
acc1_glm <- c()
  for(i in 1:3){
    testset = trd[trd$subject %in% testindex[[i]],]
    trainset = trd[!(trd$subject %in% testindex[[i]]),]
    model1_glm = glm(status1~. , data = trainset[2:563], family = binomial)
    pred1_glm = predict(model1_glm, newdata = testset[2:563])
    pred1_glm = ifelse(pred1_glm<-1, 0,1)
    matrix1_glm = table(as.matrix(pred1_glm), as.matrix(testset$status1))
    acc1_glm[i] = sum(diag(matrix1_glm))/nrow(testset)
  }
  acc1_glm = sum(acc1_glm)/3

  
#Task 2
  
#lda model(remove status1 because its strong multicollinearity prevent lda from running)
acc_lda <- c()
for(i in 1:3){
    testset = trd[trd$subject %in% testindex[[i]],]
    trainset = trd[!(trd$subject %in% testindex[[i]]),]
    model_lda = lda(formula= status2~. -status1 , data= trainset[2:564])
    pred_lda =  predict(model_lda, newdata = testset[2:564])$class         
    matrix_lda = table(as.matrix(pred_lda), as.matrix(testset$status2))
    acc_lda[i] = mean(as.matrix(pred_lda) == testset$status2)
}
acc_lda
acc_lda = sum(acc_lda)/3
  
#knn
#training without classified the data(but add the binary classification result as new variable)
preds_knn =  knn(train = trainset[2:563], test=testset[,2:563] , trainset$status2, k= 8)
matrix_knn= table(preds_knn, as.matrix(testset[564]))
acc_knn = sum(diag(matrix_knn))/nrow(testset)

#classified  the data into still and dynamic
trainset_st = trainset[trainset$status1 == 0,] 
trainset_dy = trainset[trainset$status1 == 1,] 
testset_st = testset[testset$status1== 0,]
testset_dy = testset[testset$status1== 1,] 

#training with classified the data 
preds_knn_dy =  knn(train = trainset_dy[2:563], test=testset_dy[,2:563] , trainset_dy$status2, k= 5)
preds_knn_st =  knn(train = trainset_st[2:563], test=testset_st[,2:563], trainset_st$status2, k= 5)
  
matrix_knn_dy = table(preds_knn_dy, as.matrix(testset_dy[564]))
matrix_knn_st = table(preds_knn_st, as.matrix(testset_st[564]))
  
acc_knn_dy = sum(diag(matrix_knn_dy)) /nrow(testset_dy)
acc_knn_st = sum(diag(matrix_knn_st)) /nrow(testset_st)
acc_knn_2 = (sum(diag(matrix_knn_st))+ sum( diag(matrix_knn_dy))) /nrow(testset)

#Result: The accuracy of adding the binary classification result as new variable and accuracy of directly split the data into still and dynamic do not show much difference in knn

#SVM
#the accuracy of SYM model
acc2_svm <- c()
for(i in 1:3){
  testset = trd[trd$subject %in% testindex[[i]],]
  trainset = trd[!(trd$subject %in% testindex[[i]]),]
  model_svm = svm(factor(status2)~.,data=trainset[2:564],kernel = "linear",family = multinomial,tolerance = 0.01)
  preds_svm = predict(model_svm, newdata = testset[,2:564])
  matrix_svm = table(preds_svm, as.matrix(testset[564]))
  acc2_svm[i] = sum(diag(matrix_svm)/nrow(testset))
  print(matrix_svm)
  print(acc2_svm[i])
}

mean(acc2_svm)

#Lagrange multiplier SVM
#Add binary classification as new variable
acc2_svm_alpha <- c()
for(i in 1:3){
  testset = trd[trd$subject %in% testindex[[i]],]
  trainset = trd[!(trd$subject %in% testindex[[i]]),]
  model_svm_alpha = alphasvm(factor(status2)~.,data=trainset[2:564],kernel = "linear", degree = 3,cost =1, tolerance= 0.02)
  preds_svm_alpha = predict(model_svm_alpha, newdata = testset[,2:563])
  matrix_svm_alpha = table(preds_svm_alpha, as.matrix(testset[564]))
  acc2_svm_alpha[i] = sum(diag(matrix_svm_alpha)/nrow(testset))
  print(matrix_svm_alpha)
  print(acc2_svm_alpha[i])
}
#svm training with classified data 
#svm_alpha_dynamic only
model_svm_alpha_dy = alphasvm(factor(status2)~.,data=trainset_dy[2:564],kernel = "linear",   degree = 3,cost =1, tolerance= 0.02)
preds_svm_alpha_dy = predict(model_svm_alpha_dy, newdata = testset_dy[,2:563])
matrix_svm_alpha_dy = table(preds_svm_alpha_dy, as.matrix(testset_dy[564]))
acc_svm_alpha_dy = sum(diag(matrix_svm_alpha_dy)/nrow(testset_dy))
matrix_svm_alpha_dy
acc_svm_alpha_dy

#svm_alpha_still only
model_svm_alpha_st = alphasvm(factor(status2)~.,data=trainset_st[2:564],kernel = "linear",   degree = 3,cost =1, tolerance= 0.02)
preds_svm_alpha_st = predict(model_svm_alpha_st, newdata = testset_st[,2:563])
matrix_svm_alpha_st = table(preds_svm_alpha_st, as.matrix(testset_st[564]))
acc_svm_alpha_st = sum(diag(matrix_svm_alpha_st)/nrow(testset_st))
matrix_svm_alpha_st
acc_svm_alpha_st
#the rusult does not show any difference
  
  
  
