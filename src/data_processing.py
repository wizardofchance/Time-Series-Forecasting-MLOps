
# from pathlib import Path
# from datetime import datetime, timedelta
# from typing import Optional, List, Tuple
# from pdb import set_trace as stop

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm



#################################################################################
#### Download taxi data and consolidate into a pandas dataframe
#################################################################################

def taxi_download_month(year: int, month: int, location: str) -> pd.DataFrame:

    URL = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet'

    try:
        df = pd.read_parquet(URL)
        df = df[['tpep_pickup_datetime', 'PULocationID']]
        df.columns = 'date_time location'.split()

        c = df.location == location
        df_month = df[c]
        df_month.set_index('date_time', inplace=True)

        c1 = df_month.index >= f'{year}-{month}-1'
        
        if month == 12:
            c2 = df_month.index < f'{year+1}-1-1'
        else:
            c2 = df_month.index < f'{year}-{month+1}-1' 

        df_month = df_month[c1 & c2]
        df_month = df_month.sort_values(by = 'date_time')
        return df_month
    
    except:
        print(f'not available: {URL}')


        
def taxi_download_year(year: int, location: str)-> pd.DataFrame:

    collect, collect_exceptions = [], []

    for month in tqdm(list(range(1, 13))):
        try:
            df = taxi_download_month(year, month, location)
            collect.append(df)
        except:
            collect_exceptions.append((year, month))

    print(f'exceptions encountered: {collect_exceptions}')

    if len(collect)>0:            
        df_year = pd.concat(collect)
        return df_year   
    

    
def taxi_download_years(year_s, location): 

    def taxi_download_year(year: int, location: str)-> pd.DataFrame:

        collect, collect_exceptions = [], []
        for month in range(1, 13):
            try:
                df = taxi_download_month(year, month, location)
                collect.append(df)
            except:
                collect_exceptions.append((year, month))

        print(f'exceptions encountered: {collect_exceptions}')
        if len(collect)>0:            
            df_year = pd.concat(collect)
            return df_year     

    collect = []
    for year in tqdm(year_s):
        df_year = taxi_download_year(year, location)
        collect.append(df_year)

    df_years = pd.concat(collect)
    return df_years




#################################################################################
#### Create DEMAND dataset (datetime - taxi demand) for a specified location:
#################################################################################

def taxi_resample_timeseries(df_raw, resample_rate, start_date, end_date):

    df_raw = df_raw.assign(demand = np.ones(len(df_raw)))
    df_raw.drop('location', axis = 1, inplace = True)


    df_demand = df_raw.resample(resample_rate).count()

    # Fill 0 demand for any missing time periods:
    full_range = pd.date_range(start=start_date,end=end_date, freq=resample_rate, inclusive='left')
    df_demand = df_demand.reindex(full_range, fill_value=0)
    df_demand['date_time'] = df_demand.index
    df_demand.index = range(len(df_demand))
    #
    df_demand = df_demand[['date_time', 'demand']]

    return df_demand



#################################################################################
#### Compare & visualise y & y_pred:
#################################################################################


def f_compare_ys(y, y_pred, figsize = (10, 2.5)):
    plt.figure(figsize = figsize)
    plt.plot(y, label = 'actual', alpha = 0.7)
    plt.plot(y_pred, label = 'pred', linestyle = '--', alpha = 0.9)
    plt.xticks(rotation = 45)
    plt.legend()




