---
title: "Index"
author: "AK"
date: "26 06 2018"
output: 
  html_document: 
    keep_md: yes
---

#Peer-graded Assignment: Prediction Assignment Writeup
One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, our goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants.

The objective of our project is to predict the manner in which they did the exercise. This is the "classe" variable in the training set. Any other variables can be used to predict with. We should create a report describing how we built our model, how we used cross validation, what we think the expected out of sample error is, and why we made the choices we did. We will also use our prediction model to predict 20 different test cases.


```r
library(caret)
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```


```r
train <- read.csv("~/Desktop/Data science lessons/PML/pml-training.csv")
test <- read.csv("~/Desktop/Data science lessons/PML/pml-testing.csv")
```

Before working with the full set of data, we can split the training data (train) into a smaller training set (train1) and a validation set (train2) to address out-of-sample error:


```r
set.seed(10)
inTrain <- createDataPartition(y=train$classe, p=0.7, list=F)
train1 <- train[inTrain, ]
train2 <- train[-inTrain, ]
```

We need to clean the data by removing variables with nearly zero variance, variables that are almost always NA, and variables that don’t make intuitive sense for prediction in train1 and train2:


```r
nzv <- nearZeroVar(train1)
train1 <- train1[, -nzv]
train2 <- train2[, -nzv]


mostlyNA <- sapply(train1, function(x) mean(is.na(x))) > 0.95
train1 <- train1[, mostlyNA==F]
train2 <- train2[, mostlyNA==F]

#to remove X, user_name, raw_timestamp_part_1, raw_timestamp_part_2, cvtd_timestamp
train1 <- train1[, -(1:5)]
train2 <- train2[, -(1:5)]
```

#Random Forest model
To select tuning parameters in train 1 for the model using 3-fold cross-validation.

```r
fitControl <- trainControl(method="cv", number=3, verboseIter=F)
fit <- train(classe ~ ., data=train1, method="rf", trControl=fitControl)
```

The model displays the following parameters:

```r
fit$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 27
## 
##         OOB estimate of  error rate: 0.22%
## Confusion matrix:
##      A    B    C    D    E  class.error
## A 3904    1    0    0    1 0.0005120328
## B    6 2649    3    0    0 0.0033860045
## C    0    7 2389    0    0 0.0029215359
## D    0    0    6 2245    1 0.0031083481
## E    0    0    0    5 2520 0.0019801980
```

It applied 500 trees and 27 variables at each split were tried.

#Model Selection
To predict the variable “classe” in train2:

```r
preds <- predict(fit, newdata=train2)

# display confusion matrix and out-of-sample error
confusionMatrix(train2$classe, preds)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1674    0    0    0    0
##          B    3 1132    3    1    0
##          C    0    3 1023    0    0
##          D    0    0    2  962    0
##          E    0    0    0    1 1081
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9978          
##                  95% CI : (0.9962, 0.9988)
##     No Information Rate : 0.285           
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9972          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9982   0.9974   0.9951   0.9979   1.0000
## Specificity            1.0000   0.9985   0.9994   0.9996   0.9998
## Pos Pred Value         1.0000   0.9939   0.9971   0.9979   0.9991
## Neg Pred Value         0.9993   0.9994   0.9990   0.9996   1.0000
## Prevalence             0.2850   0.1929   0.1747   0.1638   0.1837
## Detection Rate         0.2845   0.1924   0.1738   0.1635   0.1837
## Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
## Balanced Accuracy      0.9991   0.9979   0.9973   0.9988   0.9999
```

As it can be seen, it is 99.8% accurate and out-of-sample error is 0.2%.
With such as a high accuracy we can proceed with Random Forests for further prediction.

#Apply Selected Model

To repeat processing actions on the train and test:

```r
nzv <- nearZeroVar(train)
train <- train[, -nzv]
test <- test[, -nzv]

mostlyNA <- sapply(train, function(x) mean(is.na(x))) > 0.95
train <- train[, mostlyNA==F]
test <- test[, mostlyNA==F]

train <- train[, -(1:5)]
test <- test[, -(1:5)]

fitControl <- trainControl(method="cv", number=3, verboseIter=F)
fit <- train(classe ~ ., data=train, method="rf", trControl=fitControl)
```

#Making Test Set Predictions
To predict the values of classe for the observations in the test data set:

```r
preds <- predict(fit, newdata=test)
preds <- as.character(preds)
```

Random Forest model predicts the 20 quiz results (testing dataset) as shown below.


```r
preds
```

```
##  [1] "B" "A" "B" "A" "A" "E" "D" "B" "A" "A" "B" "C" "B" "A" "E" "E" "A"
## [18] "B" "B" "B"
```
