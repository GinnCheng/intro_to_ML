#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""
import os
import joblib
import sys
sys.path.append(os.path.abspath("../tools/"))
from feature_format import featureFormat, targetFeatureSplit
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier

data_dict = joblib.load(open("../final_project/final_project_dataset.pkl", "rb") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list, sort_keys = '../tools/python2_lesson14_keys.pkl')
labels, features = targetFeatureSplit(data)

# def tree_con_matrix(x_train, x_test, y_train, y_test):
#     clf = DecisionTreeClassifier()
#     clf.fit(x_train, y_train)
#     pred = clf.predict(x_test)
#     print(f'The accuracy is {confusion_matrix(y_test, pred)}')

### your code goes here
x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)
clf = DecisionTreeClassifier()
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
print(sum(y_pred))
print(sum(y_test))
### confusion matrix
import numpy as np
true_pos_indx = np.where((y_pred*y_test==1))
print(true_pos_indx)
print(precision_score(y_test,y_pred))
print(recall_score(y_test,y_pred))

## made up test
predictions = np.array([0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1])
true_labels = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0])
print(len(np.where((predictions==0) * (true_labels==1))[0]))
print(precision_score(true_labels,predictions))
print(recall_score(true_labels,predictions))