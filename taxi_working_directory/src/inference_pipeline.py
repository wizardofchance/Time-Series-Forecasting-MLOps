import numpy as np
import pandas as pd

import hopsworks

import os
import joblib
from pathlib import Path



def load_model_from_registry(project, model_name, version):    

    mr = project.get_model_registry()
    model = mr.get_model(model_name, version = version)
    model_dir = model.download()
    trained_model = joblib.load(model_dir +  f'/{model_name}.pkl')
    
    return trained_model




def taxi_make_forecast(past_28_days_demand, trained_model, forecast_horizon = 24*30):

    past_28_days_demand = [*past_28_days_demand]

    y_pred = []
    for day in range(forecast_horizon):

        y = np.array(past_28_days_demand).reshape(1, -1)
        pred = trained_model.predict(y).item()
        
        y_pred.append(pred)
        past_28_days_demand.pop(0)
        past_28_days_demand.append(pred)

    return np.array(y_pred)