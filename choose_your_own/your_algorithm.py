#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
# plt.show()
################################################################################
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from time import time
from sklearn.metrics import accuracy_score
### your code here!  name your classifier object clf if you want the
### visualization code (prettyPicture) to show you the decision boundary
clf_knn = KNeighborsClassifier(n_neighbors=10,weights='distance',leaf_size=5)
clf_adb = AdaBoostClassifier(n_estimators=10,learning_rate=0.5)
clf_rfc = RandomForestClassifier(min_samples_leaf=8)

### start to train
t_tr_knn = time()
clf_knn.fit(features_train, labels_train)
print(f'The knn training takes {round(time() - t_tr_knn,3)}s')
t_tr_adb = time()
clf_adb.fit(features_train, labels_train)
print(f'The adaboost training takes {round(time() - t_tr_adb,3)}s')
t_tr_rfc = time()
clf_rfc.fit(features_train, labels_train)
print(f'The randomforest training takes {round(time() - t_tr_rfc,3)}s')

### start to predict
t_pr_knn = time()
pred_knn = clf_knn.predict(features_test)
# del clf_knn
print(f'The knn prediction takes {round(time() - t_pr_knn,3)}s')
t_pr_adb = time()
pred_adb = clf_adb.predict(features_test)
# del clf_adb
print(f'The adaboost prediction takes {round(time() - t_pr_adb,3)}s')
t_pr_rfc = time()
pred_rfc = clf_rfc.predict(features_test)
# del clf_rfc
print(f'The random forest prediction takes {round(time() - t_pr_adb,3)}s')

### calculate the accuracy
acc_knn = accuracy_score(labels_test,pred_knn)
acc_adb = accuracy_score(labels_test,pred_adb)
acc_rfc = accuracy_score(labels_test,pred_rfc)
print(f'The accuracy of knn is {acc_knn}')
print(f'The accuracy of adb is {acc_adb}')
print(f'The accuracy of rfc is {acc_rfc}')





clf = clf_knn

try:
    prettyPicture(clf, features_test, labels_test)
    plt.show()
except NameError:
    pass
