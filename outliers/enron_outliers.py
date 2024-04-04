#!/usr/bin/python3
import os
import joblib
import sys
import matplotlib.pyplot
sys.path.append(os.path.abspath("../tools/"))
from feature_format import featureFormat, targetFeatureSplit
import matplotlib.pyplot as plt
import numpy as np
### read in data dictionary, convert to numpy array
data_dict = joblib.load( open("../final_project/final_project_dataset.pkl", "rb") )
## pop the outlier that we found
data_dict.pop('TOTAL')
features = ["salary", "bonus"]
data = featureFormat(data_dict, features)

### your code below
salary, bonus = data[:,0], data[:,1]
plt.scatter(salary,bonus,color='blue')
# plt.show()
# max_ind = np.argsort(data[:,0])[::-1]
the_name = []
for name in data_dict.keys():
    tmp_salary, tmp_bonus = data_dict[name]['salary'], data_dict[name]['bonus']
    if (type(tmp_salary) in [int,float]) and \
            (type(tmp_bonus) in [int,float]):
        if (tmp_salary > 1e6) and (tmp_bonus > 5e6):
            the_name.append(name)
print(the_name)
