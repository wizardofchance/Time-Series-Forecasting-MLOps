import os
import numpy as np
import pandas as pd
import hopsworks

from src.utility import taxi_connect_to_project, taxi_insert_data_into_feature_store
from src.feature_pipeline import taxi_download_timeseries_month
from src.inference_pipeline import load_model_from_registry, taxi_make_forecast


# 1. CONNECT TO HOPSWORKS PROJECT:

project = taxi_connect_to_project()



# 2. DOWNLOAD & PROCESS PREV MONTH'S DATA:

year, month = 2023, 2
location = 90

df_demand_prev_month = taxi_download_timeseries_month(year, month, location)

# 2.1 SAVE PROCESS PREV MONTH'S DATA TO FEATURE STORE:

data = df_demand_prev_month
feat_group_name = f'nyc_taxi_demand_{year}_{month}'
feat_group_ver = 1

taxi_insert_data_into_feature_store(project, data, feat_group_name, feat_group_ver)



# 3. DOWNLOAD PREDICTOR MODEL FROM MODEL REGISTRY:

model_name, version = 'model_2020_jan23', 1

trained_model = load_model_from_registry(project, model_name, version)



# 4. MAKE PREDICTION FOR CURRENT MONTH:

start_ = f'{year}-{month+1}-1'
end_ = f'{year}-{month+2}-1' 
if month == 11:
    start_ = f'{year}-{month+1}-1'
    end_ = f'{year+1}-{1}-1' 
if month == 12:
    start_ = f'{year+1}-1-1'
    end_ = f'{year+1}-2-1' 


date_time_ = pd.date_range(start = start_, end = end_, freq= 'h', inclusive = 'left')
past_28_days_demand = df_demand_prev_month.demand.values[-24*28:]

y_pred =  taxi_make_forecast(past_28_days_demand, trained_model, forecast_horizon = len(date_time_))
y_pred = pd.DataFrame().assign(date_time = date_time_, demand = y_pred)



# 4. SAVE CURRENT PREDICTIONS TO FEATURE STORE:

data = y_pred
year_, month_ = pd.to_datetime(start_).year, pd.to_datetime(start_).month
feat_group_name = f'nyc_taxi_demand_prediction_{year_}_{month_}'
feat_group_ver = 1

taxi_insert_data_into_feature_store(project, data, feat_group_name, feat_group_ver)








