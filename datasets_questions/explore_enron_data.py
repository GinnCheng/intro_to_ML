#!/usr/bin/python3

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import joblib

enron_data = joblib.load(open("../final_project/final_project_dataset.pkl", "rb"))

## looking for number of pois
# num_poi = 0
# for key in enron_data.keys():
#     if enron_data[key]['poi'] == 1:
#         num_poi += 1
# print(num_poi)

## import the text file
# txtfile = '../final_project/poi_names.txt'
# with open(txtfile,'r') as f:
#     lines = f.readlines()
# line_array = []
# for i,line in enumerate(lines):
#     if i <= 1:
#         pass
#     else:
#         line_array.append(line.strip())
# print(len(line_array))

## check number of NaNs
import numpy as np
# ## analyse the dataset
# num_salary = 0
# names = []
# num_email = 0
# emails = []
# print(enron_data['METTS MARK'].keys())
# for name in enron_data.keys():
#     if (enron_data[name]['salary'] is not np.nan) and \
#             (name not in names):
#         names.append(name)
#         num_salary += 1
#     if (enron_data[name]['email_address'] is not np.nan) and \
#         (enron_data[name]['email_address'] not in emails):
#         emails.append(enron_data[name]['email_address'])
#         num_email += 1
# print(num_salary)
# print(num_email)
# print(names, emails)



## create an array data for training
# from intro_to_ML.tools.feature_format import featureFormat,targetFeatureSplit
# features_list = ['poi','salary','total_payments','bonus','total_stock_value',
#                  'expenses','from_poi_to_this_person','from_this_person_to_poi']
# features_data = featureFormat(enron_data,features_list)
# features,labels = targetFeatureSplit(features_data)
# print(features,labels)

## looking for NaN payment
num_NA_tot_pay = 0
for name in enron_data.keys():
    if (enron_data[name]['total_payments'] == 'NaN'):
        num_NA_tot_pay += 1

print(f'There are {num_NA_tot_pay} NaN total payments which '
      f'is about {num_NA_tot_pay/len(enron_data)*100}% of the total list')

## looking for NaN payments in POIs
# num_NA_tot_pay_in_poi = 0
# num_pois = 0
# for name in enron_data.keys():
#     if enron_data[name]['poi'] == 1:
#         num_pois += 1
#         if (enron_data[name]['total_payments'] == 'NaN'):
#             num_NA_tot_pay_in_poi += 1
#
# print(f'There are {num_NA_tot_pay_in_poi} NaN total payments which '
#       f'is about {num_NA_tot_pay_in_poi/num_pois*100}% of the total list')