import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime


def metric(df: pd.DataFrame) -> float:
    """Compute performance metric.

    :param df: Dataframe with target ('phase') and 'prediction', and identifiers.
    :return: Performance metric
    """
    df = df.copy()
    assert 'monthly' in df.columns, "Missing 'monthly' column, only available in the train set"
    assert 'phase' in df.columns, "Missing 'phase' column, only available in the train set"
    assert 'prediction' in df.columns, "Missing 'prediction' column with predictions"

    # Sum of phasing country-brand-month = 1
    df['sum_pred'] = df.groupby(['year', 'month', 'brand', 'country'])['prediction'].transform(sum)
    assert np.isclose(df['sum_pred'], 1.0, rtol=1e-04).all(), "Condition phasing year-month-brand-country must sum 1 is not fulfilled"
    
    # define quarter weights 
    df['quarter_w'] = np.where(df['quarter'] == 1, 1, 
                    np.where(df['quarter'] == 2, 0.75,
                    np.where(df['quarter'] == 3, 0.66, 0.5)))
                    
    # compute and return metric
    return round(np.sqrt((1 / len(df)) * sum(((df['phase'] - df['prediction'])**2) * df['quarter_w'] * df['monthly'])), 8)


def dataProcessing(data: pd.DataFrame):
    """ 
    Clean data
    """
    
    data['day'] = data['date'].dt.day
    data['month'] = data['date'].dt.month
    data['year'] = data['date'].dt.year
    data['quarter'] = data['date'].dt.quarter
    
    country_means = data.groupby('country')['hospital_rate'].mean()
    brand_means = data.groupby('brand')['hospital_rate'].mean()
    
    def fill_na(row):
        if pd.isnull(row['hospital_rate']):
            if row['country'] in country_means.index:
                country_mean = country_means[row['country']]
            else:
                country_mean = data['hospital_rate'].mean()

            if row['brand'] in brand_means.index:
                brand_mean = brand_means[row['brand']]
            else:
                brand_mean = data['hospital_rate'].mean()

            return (country_mean + brand_mean) / 2

        return row['hospital_rate']
    
    # replace nans
    data['main_channel'] = data['main_channel'].cat.add_categories("UNKNOWN").fillna('UNKNOWN')
    data['ther_area'] = data['ther_area'].cat.add_categories("UNKNOWN").fillna('UNKNOWN')
    
    data['Sintetic_hospital_rate'] = data['hospital_rate'].isna().astype(int)
    data['hospital_rate'] = data.apply(fill_na, axis=1)

    
    # remove / correct wrong values and outliers
    data['n_weekday_0'] = data.groupby(['country', 'brand', 'month', 'year'])['dayweek'].transform(lambda x: (x == 0).sum())
    data['n_weekday_1'] = data.groupby(['country', 'brand', 'month', 'year'])['dayweek'].transform(lambda x: (x == 1).sum())
    data['n_weekday_2'] = data.groupby(['country', 'brand', 'month', 'year'])['dayweek'].transform(lambda x: (x == 2).sum())
    data['n_weekday_3'] = data.groupby(['country', 'brand', 'month', 'year'])['dayweek'].transform(lambda x: (x == 3).sum())
    data['n_weekday_4'] = data.groupby(['country', 'brand', 'month', 'year'])['dayweek'].transform(lambda x: (x == 4).sum())
    data['n_weekday_5'] = data.groupby(['country', 'brand', 'month', 'year'])['dayweek'].transform(lambda x: (x == 5).sum())
    data['n_weekday_6'] = data.groupby(['country', 'brand', 'month', 'year'])['dayweek'].transform(lambda x: (x == 6).sum())
    
    data.loc[data['wd'] > 23, 'wd'] = (data['n_weekday_0'] + data['n_weekday_1'] + data['n_weekday_2'] +
                                       data['n_weekday_3'] + data['n_weekday_4'] + data['n_weekday_5'] +
                                       data['n_weekday_6'])
        
    data.loc[data['wd_left'] > 22, 'wd_left'] = (data['n_weekday_0'] + data['n_weekday_1'] + data['n_weekday_2'] +
                                                 data['n_weekday_3'] + data['n_weekday_4'] + data['n_weekday_5'] +
                                                 data['n_weekday_6']) - 1
    data.loc[data['n_nwd_bef'] > 4, 'n_nwd_bef'] = 4
    data.loc[data['n_nwd_aft'] > 4, 'n_nwd_aft'] = 4

    # encoding
    data['country'] = pd.Categorical(data.country)
    data['brand'] = pd.Categorical(data.brand)
    
    return data

def compute_features(df):
    
    #df['hospital_rate_bin'] = pd.cut(df['hospital_rate'], 4)
    df['hospital_rate_bin'] = pd.cut(x=df['hospital_rate'],bins=4, labels=[f'bin_{i}' for i in range(4)])

    # interactions
    df['SumNNwd'] = df['n_nwd_bef'] + df['n_nwd_aft']
   
    df['day_of_year_sin'] = np.sin((2 * np.pi * df['date'].dt.dayofyear) / (365 + df['date'].dt.is_leap_year.astype(int)))
    df['day_of_year_cos'] = np.cos((2 * np.pi * df['date'].dt.dayofyear) / (365 + df['date'].dt.is_leap_year.astype(int)))
    
    df['country_brand_interaction'] = df['country'].astype(str) + '-' + df['brand'].astype(str)
    df['hospital_rate_brand_interaction'] = df['hospital_rate_bin'].astype(str) + '-' + df['brand'].astype(str)
    df['weekday_channel_interaction'] = df['dayweek'].astype(str) + '-' + df['main_channel'].astype(str)
    df['month_channel_interaction'] = df['month'].astype(str) + '-' + df['main_channel'].astype(str)
    df['country_therapeutic_interaction'] = df['country'].astype(str) + '-' + df['ther_area'].astype(str)
    df['dayweek_brand_interaction'] = df['dayweek'].astype(str) + '-' + df['brand'].astype(str)
    df['month_brand_interaction'] = df['month'].astype(str) + '-' + df['brand'].astype(str)
    df['country_channel_interaction'] = df['country'].astype(str) + '-' + df['main_channel'].astype(str)
    df['year_channel_interaction'] = df['year'].astype(str) + '-' + df['main_channel'].astype(str)
    
    df['brand_country_weekday_interaction'] = df['brand'].astype(str) + '-' + df['country'].astype(str) + '' + df['dayweek'].astype(str)
    df['brand_country_month_interaction'] = df['brand'].astype(str) + '-' + df['country'].astype(str) + '' + df['month'].astype(str)

    
    #### lag features
    #df['year_lag'] = df.groupby(['brand', 'country', 'month', 'day'])['phase'].shift()
    #df['day_lag_year_lag'] = df.groupby(['brand', 'country', 'month', 'year'])['year_lag'].shift()
    #df['year_lag'] = df.groupby(['brand', 'country', 'month', 'wd', 'dayweek'])['phase'].shift()
    #df['year_lag'] = df.groupby(['brand', 'country', 'month', 'day'])['phase'].shift()
    #df['2_years_before'] = df.groupby(['brand', 'country', 'month', 'wd'])['phase'].shift(2)
    df['3year_lag'] = df.groupby(['brand', 'country', 'month', 'wd'])['phase'].shift(3)
    df['2year_lag'] = df.groupby(['brand', 'country', 'month', 'wd'])['phase'].shift(2)
    df['year_lag'] = df.groupby(['brand', 'country', 'month', 'wd'])['phase'].shift()
    df['avg_year_lag'] = df[['year_lag', '2year_lag', '3year_lag']].mean(axis=1)
    df['std_year_lag'] = df[['year_lag', '2year_lag', '3year_lag']].std(axis=1)
    df['min_year_lag'] = df[['year_lag', '2year_lag', '3year_lag']].min(axis=1)
    df['max_year_lag'] = df[['year_lag', '2year_lag', '3year_lag']].max(axis=1)
    df['median_year_lag'] = df[['year_lag', '2year_lag', '3year_lag']].median(axis=1)
    df['diff_year_2year_lag'] = (df['year_lag'] - df['2year_lag'])
    df['diff_year_3year_lag'] = (df['year_lag'] - df['3year_lag'])
    df['diff_2year_3year_lag'] = (df['2year_lag'] - df['3year_lag'])
    df['diff_rel_year_2year_lag'] = df['diff_year_2year_lag']/df['year_lag']
    df['diff_rel_year_3year_lag'] = df['diff_year_3year_lag']/df['year_lag']
    df['diff_rel_2year_3year_lag'] = df['diff_2year_3year_lag']/df['2year_lag']

    # encoding
    df['SumNNwd'] = pd.Categorical(df.SumNNwd)
    
    df['country_brand_interaction'] = pd.Categorical(df.country_brand_interaction)
    df['hospital_rate_brand_interaction'] = pd.Categorical(df.hospital_rate_brand_interaction)
    df['weekday_channel_interaction'] = pd.Categorical(df.weekday_channel_interaction)
    df['month_channel_interaction'] = pd.Categorical(df.month_channel_interaction)
    df['country_therapeutic_interaction'] = pd.Categorical(df.country_therapeutic_interaction)
    df['dayweek_brand_interaction'] = pd.Categorical(df.dayweek_brand_interaction)
    df['month_brand_interaction'] = pd.Categorical(df.month_brand_interaction)
    df['country_channel_interaction'] = pd.Categorical(df.country_channel_interaction)
    df['year_channel_interaction'] = pd.Categorical(df.year_channel_interaction)
    
    df['brand_country_weekday_interaction'] = pd.Categorical(df.brand_country_weekday_interaction)
    df['brand_country_month_interaction'] = pd.Categorical(df.brand_country_month_interaction)

    return df





# idx_sintetic = data.index[data['Sintetic'] == 1].tolist()
# mean_hospital_rate_by_country = data.groupby('country')['hospital_rate'].mean().fillna(0)
# for i in idx_sintetic:
#     data.loc[i,'hospital_rate'] = (mean_hospital_rate_by_country[data.loc[i,'country']] + data['hospital_rate'].mean(skipna=True)) / 2
    

#data['hospital_rate'] = data['hospital_rate'].fillna(data['hospital_rate'].transform(lambda x: (mean_hospital_rate_by_country[x['country']] + data['hospital_rate'].mean()) / 2 ))



