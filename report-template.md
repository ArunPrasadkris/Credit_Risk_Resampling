# Credit_Risk_Resampling

## Overview of the Analysis

Credit risk poses a classification problem for machine learning.The problem of credit risk presents a classification challenge for machine learning, largely due to the unequal distribution of healthy loans compared to risky loans. In this analysis, we aim to address this issue by training and evaluating models with imbalanced classes. Specifically, we use a logistic regression model to resample the data, using the RandomOverSampler module from the imbalanced-learn library, and compare two versions of the dataset.

To compare the models, we calculate the count of the target classes, train logistic regression classifiers, calculate balanced accuracy scores, generate confusion matrices, and create classification reports for both versions of the dataset. We use a peer-to-peer lending services dataset of historical lending activity to build our models, with the goal of identifying the creditworthiness of borrowers based on the loan_status column. A value of 0 indicates that the loan is healthy, while a value of 1 signifies a high risk of default.

After splitting the dataset into X and y dataframes, we use the value_counts function to determine the number of healthy and risky loans in the original dataset. We then split the data into training and testing sets using the train_test_split function, and fit our logistic regression model using the training data. We evaluate the model's performance by calculating its accuracy score, generating a confusion matrix, and printing the classification report.

As the number of healthy loans greatly outweighs the number of risky loans in the dataset, we predict a new logistic regression model with resampled training data by oversampling the high-risk loans using the RandomOverSampler module. We confirm that the labels now have an equal number of data points using the value_counts function, and once again train our logistic regression classifier, this time using the oversampled data. We then generate the same reports as before to compare the performance of our oversampled model with the original model.

In the next section, we will examine and discuss the results of these models.

## Results

* Machine Learning Model 1:
  * The balanced accuracy score of Model 1 is 0.952
  * The precision score of the 0 class is 1.0 and 1 class is 0.85.



* Machine Learning Model 2:
  * The balanced accuracy score of Model 1 is 0.993
  * The precision score of the 0 class is 1.0 and 1 class is 0.84.

## Summary

Based on the balanced accuracy score, the second model appears to be the better option. However, we need to prioritize predicting high-risk loans, represented by a value of 1, over overall performance. To address the imbalance of the dataset, we implemented random oversampling, which involves adding instances of the minority class to the training set. This approach improves recall for the 1 class significantly (0.99 vs 0.91), but at the cost of precision (0.84 vs 0.85). Given that the focus is on predicting high-risk loans, the lower precision is acceptable. In conclusion, when dealing with potential loan defaults, it is recommended to use the oversampled model to improve the accuracy of predicting high-risk loans

Assignment completed by Arun Prasad