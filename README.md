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

**Evaluation:**  three-fold cv. The reason for using 3-fold cross-validation is that the ratio of subjects between the training and testing sets is 2:1, and using 3-fold cross-validation can make the model's evaluation closest to the real results.  

**Lasso regression model:** For lasso regression, the baseline algorithm show a high accuracy with out optimized the lambda. And the accuracy can improve by optimized the lambda.  

**Logistic regression model:** All the 561 features are used as variables and the 
**Final result generating:** The final result are generated using logistic regression model with threshold of zero



