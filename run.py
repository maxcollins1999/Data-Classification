### Preamble ###################################################################
#
# Author            : Max Collins
#
# Github            : https://github.com/maxcollins1999
#
# Title             : run.py
#
# Date Last Modified: 26/9/2020
#
# Notes             : The python script to be run when generating the 
#                     predictions for comp3009 assignment.
#
################################################################################

### Imports ####################################################################

#Local
from operator import index
from format_tools import raw_formatter

#Global
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import pickle as pkl

################################################################################

cat_cols = ['C2','C3','C4','C5','C7','C9','C10','C11','C12','C14','C16','C18','C19','C23','C24','C25','C27','C31','C32']
num_cols = ['C6', 'C8', 'C13', 'C20', 'C26']

#Preparing training and test data
data = pd.read_csv('data2020.student.csv')
data = data.loc[0:999,:]
data['Class'] = data['Class'].astype('category')

#Formatting training and test data
rformat = raw_formatter()
data = rformat.fit_transform(data, 'Class', num_cols, cat_cols)

#Splitting training and test
train_data, test_data = train_test_split(data,test_size = 0.15, stratify = data['Class'], random_state = 79)

#Formatting data for model fitting
x_train = train_data.drop('Class',axis=1)
y_train = train_data['Class']
x_test = test_data.drop('Class',axis=1)
y_test = test_data['Class']


#Unpickling classifiers
with open('classifiers.pkl', 'rb') as fstrm:
    clst = pkl.load(fstrm)

cvt = clst[0]
cnn = clst[1]

#Calculating test errors
vt_err = sum(cvt.predict(x_test) == y_test)/len(y_test)
nn_err = sum(cnn.predict(x_test) == y_test)/len(y_test)

#Checking models have built correctly
if vt_err - .76 < 1e-2 and nn_err - .79 < 1e-2:
    print('SUCCESS: MODELS CORRECTLY BUILT')
else:
    print('WARNING: MODELS MAY NOT BE CORRECTLY BUILT')

#Predicting data
to_pred = pd.read_csv('data2020.student.csv')
to_pred = to_pred.loc[1000:,:]
to_pred['Class'] = to_pred['Class'].astype('category')

to_pred = rformat.transform(to_pred)
to_pred = to_pred.drop('Class',axis=1)

pred1 = cnn.predict(to_pred)
pred2 = cvt.predict(to_pred)

check1 = [1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0,
 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0,
 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1,
 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0,
 1, 1, 0, 0]

check2 = [0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0,
 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0,
 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1,
 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0,
 1, 1, 0, 0]

if (pred1 == check1).all():
    print('SUCCESS: PREDICTED VALUES 1 ARE THE SAME AS NOTEBOOK')
else:
    print('WARNING: PREDICTED VALUES 1 ARE NOT THE SAME AS NOTEBOOK')

if (pred2 == check2).all():
    print('SUCCESS: PREDICTED VALUES 2 ARE THE SAME AS NOTEBOOK')
else:
    print('WARNING: PREDICTED VALUES 2 ARE NOT THE SAME AS NOTEBOOK')

preds = pd.DataFrame({'ID' : [i for i in range(1001,1101)]})
preds['Predict 1'] = pred1
preds['Predict 2'] = pred2

print('\nPREDICTED 1 INFO')
print(preds['Predict 1'].value_counts())
print('\nPREDICTED 2 INFO')
print(preds['Predict 2'].value_counts())

preds.to_csv('predict.sample.csv', index=False)

################################################################################