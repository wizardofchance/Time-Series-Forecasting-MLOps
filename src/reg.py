#################################################################################
################################################################################
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from seaborn import displot
sns.set_style('whitegrid')

#################################################################################
#################################################################################

def f_compare_ys(y, y_pred, figsize = (10, 2.5)):
    plt.figure(figsize = figsize)
    plt.plot(y, label = 'actual', alpha = 0.7)
    plt.plot(y_pred, label = 'pred', linestyle = '--', alpha = 0.9)
    plt.xticks(rotation = 45)
    plt.legend()

    
def f_residuals(residuals):
    plt.hist(residuals, bins = 30, alpha = 0.5)
    plt.xlabel('RESIDUALS')
    plt.show()
#################################################################################
#################################################################################


def fn_reg_performance(y, y_pred):

    rmse = (((y-y_pred)**2).mean())**(1/2)
    mae = abs(y-y_pred).mean()
    mape = (abs(y-y_pred)/abs(y + 1e-15)).mean()
    r2 = 1 - (y - y_pred).std()**2/(y + 1e-15).std()**2 

    return rmse, mae, mape, r2


def fn_reg_metrics(y, y_pred, tr_alpha = 0.2, val_alpha = 0.45, fig_aspect = 0.4):

    metrics = np.array(fn_reg_performance(y, y_pred)).round(3)
    rmse, mae, mape, r2  = list(metrics)
    absolute_error, percent_error = abs(y - y_pred), abs(y - y_pred)/abs(y+1e-15)

    print()
    print('#############################')
    print(f'REGRESSION METRICS: \nmae = {mae}, mape = {mape}, r2 = {r2}')
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

    return {'mape': mape.item(), 'mae': mae.item(), 'r2': r2.item()}

#################################################################################
#################################################################################



