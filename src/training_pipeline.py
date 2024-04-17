import numpy as np
import pandas as pd


import hopsworks
import xgboost as xgb

import os
import joblib
from hsml.schema import Schema
from hsml.model_schema import ModelSchema

import matplotlib.pyplot as plt
import seaborn as sns
from seaborn import displot
sns.set_style('whitegrid')

from src.paths import MODELS_DIR





##################################################################################
##### TRAINING:
##################################################################################
    
def taxi_create_ml_dataset(df_demand, n_lags = 24*28):

    timeseries = df_demand.set_index('date_time')
    collect = []   
        
    for i in range(0, n_lags+1):
        collect.append(timeseries.shift(i))

    df_ml = pd.concat(collect[::-1], axis = 1).dropna()
            
    columns = [f't{i}' for i in range(df_ml.shape[1])][::-1]
    df_ml.columns = columns

    return df_ml.astype('int32')


def taxi_tr_ts_split(df_ml):

    thresh = int(len(df_ml)*0.85)
    thresh = df_ml.index[thresh]

    df_tr, df_ts = df_ml[df_ml.index <= thresh], df_ml[df_ml.index > thresh]
    return df_tr, df_ts



def taxi_train_ml_model(untrained_model, df_ml_tr):    
    X, y = df_ml_tr.iloc[:, :-1], df_ml_tr.iloc[:, -1]
    trained_model = untrained_model.fit(X, y)
    return trained_model



def taxi_make_ml_prediction(trained_model, df_ml):
    X, y = df_ml.iloc[:, :-1], df_ml.iloc[:, -1]
    y_pred = trained_model.predict(X)
    return y, y_pred




#################################################################################
#### REGRESSION metrics:
#################################################################################

def taxi_compare_timeseries(y, y_pred, figsize = (10, 2.5)):    
    plt.figure(figsize = figsize)
    plt.plot(y, label = 'actual', alpha = 0.7)
    plt.plot(y_pred, label = 'pred', linestyle = '--', alpha = 0.9)
    plt.xticks(rotation = 45)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))



    
def taxi_prediction_residuals(y, y_pred):
    residuals = y-y_pred
    plt.hist(residuals, bins = 30, alpha = 0.5)
    plt.xlabel('RESIDUALS')
    plt.show()



def taxi_reg_performance(y, y_pred):

    rmse = (((y-y_pred)**2).mean())**(1/2)
    mae = abs(y-y_pred).mean()
    mape = (abs(y-y_pred)/abs(y + 1e-15)).mean()
    r2 = 1 - (y - y_pred).std()**2/(y + 1e-15).std()**2 

    return rmse, mae, mape, r2



def taxi_performance_metrics(y, y_pred, tr_alpha = 0.2, val_alpha = 0.45, fig_aspect = 0.4):

    metrics = np.array(taxi_reg_performance(y, y_pred)).round(3)
    rmse, mae, mape, r2  = list(metrics)
    absolute_error, percent_error = abs(y - y_pred), abs(y - y_pred)/abs(y+1e-15)

    print()
    print('#############################')
    print(f'REGRESSION METRICS: \nmae = {mae}, r2 = {r2}')
    print('#############################')

    fig = plt.figure(figsize=plt.figaspect(fig_aspect))

    ax1 = plt.subplot(111)
    sns.kdeplot(absolute_error, fill=True, cumulative=True, linewidth = 2,
                alpha = tr_alpha,  ax=ax1)
    
    plt.vlines(mae, 0, 1, label = f'Mean Absolute Error: {mae}', 
               linestyle = '--', linewidth = 2, color = 'black')
    plt.title('CDF ABSOLUTE ERROR')
    plt.legend(loc='lower right', prop={'weight':'bold'})
    plt.ylabel('% dataset', fontsize = 15)

    plt.tight_layout()
    plt.show()

    return {'mae': mae.item(), 'r2': r2.item()}





####################### 2 MAIN FUNCSTIONS #######################################

def taxi_display_mlmodel_performance(y, y_pred):
    
    taxi_prediction_residuals(y, y_pred)
    metrics = taxi_performance_metrics(y, y_pred, tr_alpha = 0.2, val_alpha = 0.45, fig_aspect = 0.4)
    return metrics



def taxi_display_forecaster_performance(y, y_pred):   

    y, y_pred = pd.Series(y), pd.Series(y_pred) 
    
    taxi_compare_timeseries(y, y_pred, figsize = (10, 2.5))
    metrics = taxi_performance_metrics(y, y_pred, tr_alpha = 0.2, val_alpha = 0.45, fig_aspect = 0.4)
    return metrics






#################################################################################
##### HOPSWORKS:
################################################################################

def taxi_save_model_to_registry(project, trained_model, model_name, description, df_ml_tr, 
                                performance_dict):
    
    def taxi_model_schema(X_tr, y_tr):
        input_schema = Schema(X_tr)
        output_schema = Schema(y_tr)
        model_schema = ModelSchema(input_schema=input_schema, 
                                output_schema=output_schema)
        return model_schema

    model_path = str(MODELS_DIR / f'{model_name}.pkl')
    joblib.dump(trained_model, model_path) # Save model at some location first

    X_tr, y_tr = df_ml_tr.values[:, :-1], df_ml_tr.values[:, -1]
    model_schema = taxi_model_schema(X_tr, y_tr)


    model_registry = project.get_model_registry()

    model_registry = model_registry.sklearn.create_model(
        name = model_name,
        metrics = performance_dict,
        description = description,
        input_example = X_tr[0],
        model_schema = model_schema
        )    

    model_registry.save(model_path) # Push model stored here to hopsworks registry




        