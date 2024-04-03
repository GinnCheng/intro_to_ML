#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np
### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

# print(len(features_train[0]))
# # print(np.shape(labels_train))

#########################################################
### your code goes here ###
clf = DecisionTreeClassifier(min_samples_split=40)
t_train = time()
clf.fit(features_train,labels_train)
print(f'Time for training is {round(time()-t_train,3)}s')
#########################################################
t_pred = time()
pred = clf.predict(features_test)
print(f'Time for predicting is {round(time()-t_pred,3)}s')
acc = accuracy_score(labels_test,pred)
print(f'The accuracy score is {acc}')


