import pandas as pd
import Utils
from datetime import datetime
import lightgbm as lgb
from catboost import CatBoostRegressor
import xgboost as xgb
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, f1_score
import optuna
import warnings
import pickle
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')


DROP_NA = False
FEATURES_TO_REMOVE = ['phase', 'brand', 'country', 'date', 'monthly']
DATA_DIR = 'Data for participants/Data Files/'
USE_OPTUNA = True
TEST_SIZE_OPTUNA = 0.2
GLOBAL_SEED = 42
NUM_TRIALS = 100


# Load data
train_data = pd.read_parquet(DATA_DIR + 'train_data.parquet', engine='pyarrow')
# clean data
train_data = Utils.dataProcessing(train_data)
# extract features
train_data = Utils.compute_features(train_data)

train_data = train_data.dropna()


##### Data split
X_train = train_data[train_data['year'].between(2013,2020)]
Y_train = X_train['phase']
X_train = X_train.drop(['phase','monthly','date'], axis=1)

X_val = train_data[train_data['year']==2021]
Y_val = X_val['phase']
monthly_weights_val = X_val['monthly']
X_val = X_val.drop(['phase','monthly','date'], axis=1)

X = (X_train, X_val)
Y = (Y_train, Y_val)

def objective(trial, data = X, target = Y):
    
    X_train, X_val = data
    Y_train, Y_val = target
    
    param = {
        'metric': 'rmse',
        'verbose': -1,
        'random_state': GLOBAL_SEED,
        'n_estimators': trial.suggest_int('n_estimarors', 100, 5000),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-3, 10.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-3, 10.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1),
        'subsample': trial.suggest_float('subsample', 0.5, 1),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.1, log=True),
        'num_leaves' : trial.suggest_int('num_leaves', 8, 64),
        'min_child_samples': trial.suggest_int('min_child_samples', 1, 100),
    }
    
    model = lgb.LGBMRegressor(**param)
    
    # param = {
    #     'max_depth': trial.suggest_int('max_depth', 1, 10),
    #     'learning_rate': trial.suggest_float('learning_rate', 0.01, 1.0),
    #     'n_estimators': trial.suggest_int('n_estimators', 50, 5000),
    #     'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
    #     'gamma': trial.suggest_float('gamma', 0.01, 1.0),
    #     'subsample': trial.suggest_float('subsample', 0.01, 1.0),
    #     'colsample_bytree': trial.suggest_float('colsample_bytree', 0.01, 1.0),
    #     'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 1.0),
    #     'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 1.0),
    #     'random_state': trial.suggest_int('random_state', 1, 1000)
    # }

    # model = xgb.XGBRegressor(enable_categorical = True,
    #                           **param)  
    
    # param = {
    #     "iterations": 1000,
    #     "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
    #     "depth": trial.suggest_int("depth", 1, 10),
    #     "subsample": trial.suggest_float("subsample", 0.05, 1.0),
    #     "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.05, 1.0),
    #     "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
    # }
    
    # model = CatBoostRegressor(**param)
    
    model.fit(X_train,Y_train,eval_set=[(X_val,Y_val)])#,early_stopping_rounds=100,verbose=False)
    
    preds = model.predict(X_val)
    
    X_val['phase'] = Y_val
    X_val['predicted'] = preds
    X_val['prediction'] = X_val.groupby(
        ['country', 'brand', 'month'])['predicted'].transform(lambda x: x / x.sum())
    X_val['monthly'] = monthly_weights_val
    
    metric_value = Utils.metric(X_val)
    
    X_val = X_val.drop(['phase','predicted','prediction','monthly'], axis=1, inplace = True)
    
    return metric_value

if USE_OPTUNA:    
    study = optuna.create_study(direction='minimize', study_name='Optimize boosting hyperparameters')
    study.optimize(objective, n_trials=NUM_TRIALS)
    
    print('----------------------')
    print('Best trial:', study.best_trial.params)
    print('Best trial score:', study.best_trial._values)


with open('lgbmr_optimization_lastOne_maxInteractions.pkl', 'wb') as f:
    pickle.dump(study, f)
    
    
# # Select data
# target_train = Y_train
# features_train = X_train
# features_test = X_val[features_train.columns]

# # Load and train final model
# selected_model_file = 'lgbmr_optimization.pkl'
# optuna_selection = pickle.load(open(selected_model_file, 'rb'))
# selected_params = optuna_selection.best_params
# model = lgb.LGBMRegressor(**selected_params)
# model.fit(features_train, target_train)

# # Get predictions and create submission file
# predictions = model.predict(features_test)

# #### plot
# plt.figure()
# plt.scatter(predictions, Y_val, marker = '.')

# plt.figure()
# plt.hist(Y_val, bins=50)
# plt.hist(predictions, bins=50)
    
