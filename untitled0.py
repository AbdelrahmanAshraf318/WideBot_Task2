# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 14:19:05 2020

@author: Abdelrahman Ashraf
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


train_data = pd.read_csv('training.csv')
train_data.head()

test_data = pd.read_csv('validation.csv')

print(train_data.head())

y = train_data.classLabel
binary_predictors = train_data.drop(['classLabel'], axis=1)
X = binary_predictors.select_dtypes(exclude=['object'])

binary_predictors2 = train_data.drop(['classLabel'], axis=1)
X2 = binary_predictors2.select_dtypes(exclude=['object'])

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
imputer.fit(X)

Xtrans = imputer.transform(X)

Xtrans2 = imputer.transform(X2)

#to know the rows with missing values
print('Missing: %d' % sum(np.isnan(Xtrans).flatten()))


#to know the columns with categorical data
s = (X.dtypes == 'object')
object_cols = list(s[s].index)

print("Categorical variables:",object_cols)


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(Xtrans , y)
predictions = model.predict(Xtrans2)
print(predictions)



















