#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 18:34:05 2023

@author: [@fabioacl], [@PFranciscoValente], [@DiogoMPessoa], [@Julio-CMedeiros].

"""

# Libraries
import numpy as np
import pandas as pd
import Utils
from datetime import datetime
import matplotlib.pyplot as plt
import lightgbm as lgb

# Load data
data_dir = 'Data for participants/Data Files/'
# train_data = pd.read_parquet(data_dir + 'train_data.parquet', engine='pyarrow')

#data_dir = '/Users/fabiolopes/Desktop/6th-NOVARTIS-DATATHON---Data-Crocodiles/Data for participants/Data Files'

train_data = pd.read_parquet(f'{data_dir}/train_data.parquet', engine = 'pyarrow')
train_data['country'] = pd.Categorical(train_data.country)
train_data['brand'] = pd.Categorical(train_data.brand)

train_data['day'] = train_data['date'].dt.day
train_data['month'] = train_data['date'].dt.month
train_data['year'] = train_data['date'].dt.year

relative_frequencies = train_data['dayweek'].value_counts(normalize=True) * 100

columns_categories = ['brand', 'country', 'ther_area', 'main_channel']

# Summary of unique values for each column
unique_summary = [list(train_data[col].unique()) for col in train_data.columns if col in columns_categories]

# Summary statistics for numeric columns
numeric_summary = train_data.describe()

X_train = train_data[train_data['year'].between(2013,2020)]
Y_train = X_train['phase']
X_train = X_train.drop(['date', 'monthly', 'phase'], axis = 1)

X_val = train_data[train_data['year']==2021]
Y_val = X_val['phase']
X_val = X_val.drop(['date', 'monthly', 'phase'], axis = 1)

model = lgb.LGBMRegressor()

model.fit(X_train,
          Y_train,
          eval_names=['train', 'valid'],
          eval_set=[(X_train, Y_train), (X_val, Y_val)],
          eval_metric='rmse')

submission_data = pd.read_parquet(f'{data_dir}/submission_data.parquet', engine = 'pyarrow')
submission_data['day'] = submission_data['date'].dt.day
submission_data['month'] = submission_data['date'].dt.month
submission_data['year'] = submission_data['date'].dt.year

submission_data = submission_data.drop(['date'], axis = 1)
submission_data['country'] = pd.Categorical(submission_data.country)
submission_data['brand'] = pd.Categorical(submission_data.brand)

submission_labels = model.predict(submission_data)
