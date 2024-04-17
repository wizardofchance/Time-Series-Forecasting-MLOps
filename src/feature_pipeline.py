import os
from dotenv import load_dotenv, find_dotenv
import numpy as np
import pandas as pd
import hopsworks
from tqdm import tqdm


#######################################################################################
### DOWNLOAD & PREPROCESS:
#######################################################################################


def taxi_download_month(year, month, location, start_date, end_date):

    url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet'

    try:
        df = pd.read_parquet(url)
        df = df[['tpep_pickup_datetime', 'PULocationID']]
        df.columns = 'date_time location'.split()

        c = df.location == location
        df_month = df[c]
        df_month.set_index('date_time', inplace=True)

        c1 = df_month.index >= start_date
        c2 = df_month.index < end_date 
        df_month = df_month[c1 & c2]

        df_month = df_month.sort_values(by = 'date_time')
        return df_month
    
    except:
        print(f'not available: {url}')

        
def taxi_resample_timeseries(df_raw, resample_rate, start_date, end_date):

    df_raw = df_raw.assign(demand = np.ones(len(df_raw)))
    df_raw.drop('location', axis = 1, inplace = True)


    df_demand = df_raw.resample(resample_rate).count()

    # Fill 0 demand for any missing time periods:
    full_range = pd.date_range(start=start_date,end=end_date, freq=resample_rate, inclusive='left')
    df_demand = df_demand.reindex(full_range, fill_value=0)
    
    df_demand['date_time'] = df_demand.index
    df_demand.index = range(len(df_demand))    
    df_demand = df_demand[['date_time', 'demand']]
    return df_demand


def taxi_start_and_end_dates(year, month):
    
    assert (month > 0) and (month <= 12), 'month should be > 0 and < 13'
    assert (isinstance(year, int)) and (isinstance(month, int)), 'month/year should be int'

    start_date = f'{year}-{month}-1'
    end_date = f'{year}-{month+1}-1' if month < 12 else f'{year+1}-{1}-1'
    return start_date, end_date
    


############## !! 2 MAIN FUNCTIONS !! #############################


def taxi_download_timeseries_month(year, month, location, resample_rate = 'h'):
    """
    1. Downloads monthly data from nyc website.
    2. Transforms raw data into required format (index, date_time, demand).
    """

    start_date, end_date = taxi_start_and_end_dates(year, month)
    df_month = taxi_download_month(year, month, location, start_date, end_date)
    df_demand = taxi_resample_timeseries(df_month, resample_rate, start_date, end_date)
    return df_demand.astype({'date_time': 'datetime64[ns]'})



def taxi_download_timeseries_year(year, location, resample_rate = 'h'):

    collect, collect_exceptions = [], []
    for month in tqdm(list(range(1, 13))):
        try:
            df = taxi_download_timeseries_month(year, month, location, resample_rate = resample_rate)
            collect.append(df)
        except:
            collect_exceptions.append((year, month))

    print(f'exceptions encountered: {collect_exceptions}')
    if len(collect)>0:            
        df_year = pd.concat(collect)
        return df_year         




##############################################################################
# FINAL FUNCTION FOR MONTHLY DOWNLOAD & PUSHING TO FEATURE STORE:  
##############################################################################
      

def taxi_start_feature_pipeline(project, year, month, feat_group_name, feat_group_ver, 
                                location = 90, resample_rate = 'h'):
    """
    1. Downloads data from nyc website.
    2. Transforms raw data into required format (index, date_time, demand).
    3. Inserts transformed data into hopsworks feature store.
    """

    print('Downloading data from nyc taxi website...')

    df_demand = taxi_download_timeseries_month(year, month, location, resample_rate)
    print('Download complete, inserting taxi data into hopsworks feature store..')

    taxi_insert_data_into_feature_store(project, df_demand, feat_group_name, feat_group_ver)
    print('Data successfully pushed to feature store.')

    return df_demand.astype({'date_time': 'datetime64[ns]'})
