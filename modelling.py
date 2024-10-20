import pandas as pd
import numpy as np
import Utils
from datetime import datetime
# import lightgbm as lgb
from catboost import CatBoostRegressor
# from xgboost import XGBRegressor
from sklearn import linear_model


DROP_NA = False
FEATURES_TO_REMOVE = ['phase', 'brand', 'country', 'date', 'monthly']
DATA_DIR = 'Data for participants/Data Files/'

def select_baseline_model(model):
    
    if model == "lr":
        return linear_model.LinearRegression()
    elif model == "cat":
        return CatBoostRegressor(iterations=50,
                                 learning_rate=0.5,
                                 depth=3)


def pre_process_features(train_data, test_data):
    
    # date separation
    train_data['date_day'] = train_data['date'].dt.day
    train_data['date_month'] = train_data['date'].dt.month
    train_data['date_year'] = train_data['date'].dt.year
    test_data['date_day'] = test_data['date'].dt.day
    test_data['date_month'] = test_data['date'].dt.month
    test_data['date_year'] = test_data['date'].dt.year
    
    # one hot encode categorical variables
    categorical = ['ther_area', 'main_channel']
    train_data = pd.get_dummies(train_data, columns=categorical)
    test_data = pd.get_dummies(test_data, columns=categorical)
    
    # remove variables        
    features_train = train_data.drop(FEATURES_TO_REMOVE, axis=1)
    features_test = test_data[features_train.columns]
    
    return features_train, features_test

def post_processing(submission):
    
    submission['month'] = submission['date'].dt.month    

    combinations = submission.groupby(['month','country','brand']).size().reset_index().rename(columns={0:'count'})
    
    for index, row in combinations.iterrows():
        idx_month_brand = submission.index[(submission['country'] == row['country']) &
                                           (submission['brand'] == row['brand']) &
                                           (submission['month'] == row['month'])].tolist()

        submission.loc[idx_month_brand,['prediction']] = submission.loc[idx_month_brand,['prediction']] * (1/np.sum(submission.loc[idx_month_brand,['prediction']], axis=0))
           
    return submission

# def calibrate(test_data, predictions):
    
#     grouped = df.groupby(['A', 'B'])['C'].sum().reset_index()




def prepare_submission_data(test_data, predictions):
    
    submission = test_data[['country','brand','date', 'date_month']]
    submission['pre_prediction'] = predictions
    
<<<<<<< HEAD
    submission = post_processing(submission)
    
=======
    # calibration
    # grouped = submission.groupby(
    #     ['country', 'brand', 'date_month'])['prediction'].sum().reset_index()
    # grouped['prediction_normalized'] = grouped.groupby(
    #     ['country', 'brand', 'date_month'])['prediction'].transform(lambda x: x / x.sum())
    
    submission['prediction'] = submission.groupby(
        ['country', 'brand', 'date_month'])['pre_prediction'].transform(lambda x: x / x.sum())
    
    # Drop the intermediate column used for normalization
    submission = submission.drop(columns=['pre_prediction','date_month' ])
    
    # create file
>>>>>>> c6ecc5847984e7b9073b0771c764aa701a9fb841
    csv_path = DATA_DIR + 'submission_file.csv'
    submission.reset_index(drop=True).to_csv(csv_path, index=False)


# Load data
train_data = pd.read_parquet(DATA_DIR + 'train_data.parquet', engine='pyarrow')
test_data = pd.read_parquet(DATA_DIR + 'submission_data.parquet', engine='pyarrow')
target_train = train_data['phase']

if DROP_NA:
    train_data = train_data.dropna()
    test_data = test_data.dropna()
    
features_train, features_test = pre_process_features(train_data, test_data)
    
model = select_baseline_model("cat")
model.fit(features_train, target_train)
predictions = model.predict(features_test)

# calibrate_predictions()

prepare_submission_data(test_data, predictions)


template = pd.read_csv(DATA_DIR + 'submission_template.csv')
proposed = pd.read_csv(DATA_DIR + 'submission_file.csv')
# template = template.drop(columns=["prediction"])
# proposed =proposed.drop(columns=["prediction"])

