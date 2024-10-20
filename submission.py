import pandas as pd
import numpy as np
import Utils
from datetime import datetime
import lightgbm as lgb
from catboost import CatBoostRegressor
# from xgboost import XGBRegressor
from sklearn import linear_model
import pickle

def prepare_submission_data(submission_template, test_data, predictions):

    test_data['prediction'] = predictions
    
    # make prediction sum 1 - calibration / normalization
    test_data.loc[test_data['prediction'] < 0, 'prediction'] = 0
    test_data['prediction'] = test_data.groupby(
        ['country', 'brand', 'month'])['prediction'].transform(lambda x: x / x.sum())
    
    # Drop the intermediate column used for normalization
    submission_template['date'] = submission_template['date'].astype('datetime64[ns]')
    submission_template = submission_template.drop('prediction', axis = 1)
    submission = pd.merge(submission_template, 
                          test_data[['country', 'brand', 'date', 'prediction']], 
                          on=['country', 'brand', 'date'], how='left')
    
    # create file
    csv_path = DATA_DIR + 'Submission/submission.csv'
    submission.reset_index(drop=True).to_csv(csv_path, index=False)



DATA_DIR = 'Data for participants/Data files/'

# Load and get data

# train_data = pd.read_parquet(DATA_DIR + 'train_data.parquet', engine='pyarrow')
# train_data = Utils.dataProcessing(train_data)

# test_data = pd.read_parquet(DATA_DIR + 'submission_data.parquet', engine='pyarrow')
# test_data = Utils.dataProcessing(test_data)

# ####combined train and validation for feature extraction
train_data = pd.read_parquet(DATA_DIR + 'train_data.parquet', engine='pyarrow')
train_data['Group'] = np.zeros((train_data.shape[0],1))

test_data = pd.read_parquet(DATA_DIR + 'submission_data.parquet', engine='pyarrow')
test_data['Group'] = np.ones((test_data.shape[0],1))
test_data['phase'] = np.nan
test_data['monthly'] = np.nan

data_all = pd.concat([train_data, test_data]).reset_index(drop = True)
data_all = Utils.dataProcessing(data_all)
data_all = Utils.compute_features(data_all)

train_data = data_all.loc[data_all['Group'] == 0]
test_data = data_all.loc[data_all['Group'] == 1]

train_data = train_data.drop('Group', axis = 1)
test_data = test_data.drop('Group', axis = 1)
test_data = test_data.drop('phase', axis = 1)
test_data = test_data.drop('monthly', axis = 1)

# -----------------------------------------------------------------------------
submission_template = pd.read_csv(DATA_DIR + 'submission_template.csv')

# Select data
target_train = train_data['phase']
features_train = train_data.drop(['phase','monthly','date'], axis=1)
features_test = test_data[features_train.columns]

# Load and train final model
selected_model_file = 'lgbmr_optimization_lastOne_maxInteractions.pkl'#lgbmr_optimization.pkl'
optuna_selection = pickle.load(open(selected_model_file, 'rb'))
selected_params = optuna_selection.best_params
model = lgb.LGBMRegressor(**selected_params)
model.fit(features_train, target_train)

# Get predictions and create submission file
predictions = model.predict(features_test)
prepare_submission_data(submission_template, test_data, predictions)


