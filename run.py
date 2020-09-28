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
from format_tools import raw_formatter

#Global
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

################################################################################

cat_cols = ['C2','C3','C4','C5','C7','C9','C10','C11','C12','C14','C16','C18','C19','C23','C24','C25','C27','C29','C31','C32']
num_cols = ['C6', 'C8', 'C13', 'C20', 'C26']

#Preparing training and test data
data = pd.read_csv('data2020.student.csv')
data = data.loc[0:999,:]
data['Class'] = data['Class'].astype('category')

#Formatting training and test data
rformat = raw_formatter()
data = rformat.fit_transform(data, 'Class', num_cols, cat_cols)

#Splitting training and test
train_data, test_data = train_test_split(data,test_size = 0.15, stratify = data['Class'], random_state = 21)
train_data = data.loc[data['Class']==0,:].append(resample(train_data.loc[data['Class']==1,:], replace = True, n_samples = 553, random_state = 76))
print(train_data['Class'].value_counts())

#Removing outliers from training data
#cLOF = LocalOutlierFactor(n_neighbors=6,metric='manhattan').fit_predict(train_data)
#train_data = train_data.loc[cLOF==1,:]

#Formatting data for model fitting
x_train = train_data.drop('Class',axis=1)
y_train = train_data['Class']
#x_test = test_data.drop('Class',axis=1)
#y_test = test_data['Class']

#Setting optimal random forest and neural network paramaters
rf_params = {'RF__criterion': 'gini', 'RF__min_samples_split': 3, 'RF__n_estimators': 100}
nn_params = {'NN__hidden_layer_sizes': (90, 90, 90), 'NN__solver': 'adam'}

#Preparing data pipelines
numerical_transform = Pipeline(steps=[
    ('Standardisation', StandardScaler()),
    ('Principal Component Analysis', PCA())
])

preprocessor = ColumnTransformer(transformers=[
    ('Numeric Transformations', numerical_transform, num_cols)
], remainder='passthrough', n_jobs=-1)

#Building neural network
cnn = Pipeline(steps=[
    ('Numeric Transformations', preprocessor),
    ('NN', MLPClassifier(random_state=29))
])

cnn.set_params(**nn_params)
cnn.fit(x_train,y_train)
#nn_err = sum(cnn.predict(x_test) == y_test)/len(y_test)

#Building random forest
crf = Pipeline(steps=[
    ('Numeric Transformations', preprocessor),
    ('RF', RandomForestClassifier(random_state=32))
])

crf.set_params(**rf_params)
crf.fit(x_train,y_train)
#rf_err = sum(crf.predict(x_test) == y_test)/len(y_test)

#Checking models have built correctly
#if nn_err - 0.8974358974358975 < 1e-7 and rf_err - 0.9128205128205128 < 1e-7:
#    print('SUCCESS: MODELS CORRECTLY BUILT')
#else:
#    print('WARNING: MODELS MAY NOT BE CORRECTLY BUILT')

#Predicting data
to_pred = pd.read_csv('data2020.student.csv')
to_pred = to_pred.loc[1000:,:]
to_pred['Class'] = to_pred['Class'].astype('category')

to_pred = rformat.transform(to_pred)
to_pred = to_pred.drop('Class',axis=1)

print(sum(cnn.predict(to_pred)))

print(sum(crf.predict(to_pred)))

print(len(to_pred))
################################################################################