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
warnings.filterwarnings('ignore')


DROP_NA = False
FEATURES_TO_REMOVE = ['phase', 'brand', 'country', 'date', 'monthly']
DATA_DIR = 'Data for participants/Data Files/'
USE_OPTUNA = True
TEST_SIZE_OPTUNA = 0.2
GLOBAL_SEED = 42
NUM_TRIALS = 20


# Load data
train_data = pd.read_parquet(DATA_DIR + 'train_data.parquet', engine='pyarrow')

countries = train_data['country'].unique().tolist()
dictionary_best_parameters_country = dict.fromkeys(countries)

# clean data
train_data = Utils.dataProcessing(train_data)
## extract features
#train_data = Utils.compute_features(train_data)


for country in countries: 
    train_data_country = train_data.loc[train_data['country'] == country]
    
    ##### Data split
    X_train = train_data_country[train_data_country['year'].between(2013,2020)]
    Y_train = X_train['phase']
    X_train = X_train.drop(['phase','monthly','date'], axis=1)
    
    X_val = train_data_country[train_data_country['year']==2021]
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
        
        print('---------------------- > ', country)
        print('Best trial:', study.best_trial.params)
        print('Best trial score:', study.best_trial._values)
        
        dictionary_best_parameters_country[country] = study.best_trial
    
    
    
    
#### Test on validation set
##### Data split
X_train = train_data[train_data['year'].between(2013,2020)]
#Y_train = X_train['phase']
X_train = X_train.drop(['monthly','date'], axis=1)

X_val = train_data[train_data['year']==2021]
X_val_aux = X_val
Y_val = X_val['phase']
monthly_weights_val = X_val['monthly']
X_val = X_val.drop(['phase','monthly','date'], axis=1)


countries = X_train['country'].unique().tolist()[0:1]
for country in countries: 
    X_train_country = X_train.loc[X_train['country'] == country]
    Y_train_country = X_train_country['phase']
    X_train_country = X_train_country.drop(['phase'], axis = 1)
    X_val_country = X_val.loc[X_val['country'] == country]
    
    # Load and train final model
    optuna_selection = dictionary_best_parameters_country[country]
    selected_params = optuna_selection.params
    model = lgb.LGBMRegressor(**selected_params)
    model.fit(X_train_country, Y_train_country)

    # Get predictions and create submission file
    predictions = model.predict(X_val_country)
    X_val_country['predicted'] = predictions
    
    X_val_aux = pd.merge(X_val_aux, X_val_country, 
                         on=['country', 'brand', 'year', 'month', 'dayweek'], how='left')

        
X_val_aux['prediction'] = X_val_aux.groupby(
    ['country', 'brand', 'month', 'year'])['predicted'].transform(lambda x: x / x.sum())
metric_value = Utils.metric(X_val_aux)
        
