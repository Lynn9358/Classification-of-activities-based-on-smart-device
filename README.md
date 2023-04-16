# BIOSTAT626-Midterm1

## Task Description 

The object of this task is to classify activities base on the data from smart device. Two tab-delimited text files data ```training_data.txt``` and ```test_data.txt``` are provided. 

Movement are labeled as (1)WALKING,   (2) WALKING_UPSTAIRS,  (3) WALKING_DOWNSTAIRS, (4) SITTING,  (5) STANDING, (6) LYING,            (7) STAND_TO_SIT,      (8) SIT_TO_STAND,      (9) SIT_TO_LIE,        (10)LIE_TO_SIT        (11) STAND_TO_LIE,     (12) LIE_TO_STAND in ```training_data.txt``` 


The first task is to classify the activity of each time window into static (0) and dynamic (1). For this task, consider postural transitions as static (0).   
The second task is to a build refined multi-class classifier to classify walking (1), walking_upstairs (2), walking_downstairs (3), sitting (4), standing (5), lying (6), and static postural transition (7)

the training data are the datasets for subjects: 1 ,3 ,5 ,6 ,7, 8, 11, 14, 15, 16, 17, 19, 21, 22, 23, 25, 26, 27, 28, 29, 30, and test data are the datasets for subjects: 2, 4, 9, 10, 12, 13, 18, 20, 24. 
However, for different individuals, there are differences in the data levels under the same activity. Since this is no intersection of subject between training data and test data, there is no need to use subject as a variable. Also, in model training process, the training data are classified by subject index as training set and test set and then the evaluation of model can be more accurate.

## Task Procedure
### Task 1 
**Model:**  Fro task 1 logistic regression model and Lasso regression model are used.

**Evaluation:**  Three-fold cv. The reason for using 3-fold cross-validation is that the ratio of subjects between the training and testing sets is 2:1, and using 3-fold cross-validation can make the model's evaluation closest to the real results. Also, this test and training set is splited based on subject index, so it is not evenly 3 portioned, however, it is a more accurate evaluation of model comparing to randomly and equally 3-portioned..

**Lasso regression model:** For lasso regression, the baseline algorithm show a high accuracy with out optimized the lambda. And the accuracy can improve by optimized the lambda.  

**Logistic regression model:** All the 561 features are used as variables and the subject index are removed. For the baseline model, some static cases are misclassfied as dynamic cases, therefore the threshold is changed from 0 to -5 to improve accuracy. Based on different dataset, the best threshold can be different.
**Final result generating:** The final result are generated using logistic regression model with threshold of -5.

### Task 2  

**Model:**  For task2, LDA, knn, and svm are used.  

**Evaluation:**  Three-fold cv as in task2. Although some of the model have build-in cv. For some models, training data are classified by the result of first task and evaluation are performed within each group.

**LDA:** This model has high accuracy, however, is can not binary classification result as its  multicollinearity prevent lda from running.  

**KNN:** based on its performance on baseline model, the cases with same behaviour may cluster together or may have relative clear linear borderline  which makes KNN a porper model. The classification result from task 1 are experimentally added in as a new variable and the result are compared with another situation where task 1 classification are used as a index to distinguish differennt training group. Under the second circumstanc, in Group 0(static) we have 4 sub classification:sitting (4), standing (5), lying (6), and static postural transition (7), and group 1(dynamic) has 3:walking (1), walking_upstairs (2), walking_downstairs (3). Under both cases, 15 is the optimized k range from(1 to 50), other number of neighbours can be tested by changing the value of k.

**SVM & Lagrange multiplier SVM:**  Two different ways of adding the previous classification results in training process are tried. The resistance are test from 0.01(defaulted) to 1.00, among which 0.02 showed the best performance in both models. In the Lagrange multiplier SVM, the value of alpha can be optimized with build in function.


## Instructions on files
```code.R```:All models used
```training_data.txt```: The dataset for training with all the movement labeled.  
```test_data.txt```: The dataset for testing with all no movement labeled.  
```binary_9358.txt```: Result of task 1, Accuracy:1.000
```multiclass_9358.txt```: Result of task 2, Accuracy:0.965
```626problemset1.Rmd```: Problemset for the taskes
```626 muilt.Rmd```: Models in task2





