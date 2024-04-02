#!/usr/bin/python3

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()
# simplifiy the data
# features_train = features_train[:int(len(features_train)/100)]
# labels_train = labels_train[:int(len(labels_train)/100)]
#########################################################
### your code goes here ###
clf = SVC(kernel='rbf', C=10000.0, gamma='auto')
t0 = time()
clf.fit(features_train,labels_train)
print(f'Time for training is {round(time()-t0,3)}s')
#########################################################
t1 = time()
pred = clf.predict(features_test)
print(f'Time for predicting is {round(time()-t1,3)}s')
#########################################################
accuracy = accuracy_score(labels_test,pred)
print(f'The accuracy is {accuracy}')
'''
You'll be Provided similar code in the Quiz
But the Code provided in Quiz has an Indexing issue
The Code Below solves that issue, So use this one
'''

# features_train = features_train[:int(len(features_train)/100)]
# labels_train = labels_train[:int(len(labels_train)/100)]

#########################################################
print(f'The answer for prediction[10,26,50] are {pred[10],pred[26],pred[50]}')
print(f'The number of prediction "Chris" is {pred.sum()}')