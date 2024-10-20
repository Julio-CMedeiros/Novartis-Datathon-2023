import numpy as np
import pandas as pd
import optuna
import Utils
import lightgbm as lgb

def objective(trial):
    data_dir = 'Data for participants/Data Files/'
    train_data = pd.read_parquet(data_dir + 'train_data.parquet', engine='pyarrow')
    train_data['country'] = pd.Categorical(train_data.country)
    train_data['brand'] = pd.Categorical(train_data.brand)

    train_data['day'] = train_data['date'].dt.day
    train_data['month'] = train_data['date'].dt.month
    train_data['year'] = train_data['date'].dt.year

    X_train = train_data[train_data['year'].between(2013,2020)]
    Y_train = X_train['phase']
    X_train = X_train.drop(['date', 'monthly', 'phase'], axis = 1)
    
    X_val = train_data[train_data['year']==2021]
    Y_val = X_val['phase']
    X_val_monthly = X_val['monthly']
    X_val_date = X_val['date']
    X_val = X_val.drop(['date', 'monthly', 'phase'], axis = 1)
    
    dtrain = lgb.Dataset(X_train, label=Y_train)

    param = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 2, 256),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
    }

    gbm = lgb.train(param, dtrain)
    preds = gbm.predict(X_val)
    
    X_val['phase'] = Y_val
    X_val['predicted'] = preds
    X_val['monthly'] = X_val_monthly 
    X_val['date'] = X_val_date
    X_val['prediction'] = X_val.groupby(
        ['country', 'brand', 'month'])['predicted'].transform(lambda x: x / x.sum())
    
    eval_metric = Utils.metric(X_val)
    return  eval_metric


if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=5)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
