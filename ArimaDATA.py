from __future__ import division
import numpy as np
from matplotlib import pyplot as plt
import array
import scipy.io
import scipy.io as io

import re
import os
import signal
import subprocess

import rasterio
import rasterio.plot
from rasterio.plot import show
import matplotlib.pyplot as plt
from rasterio.plot import show_hist
from osgeo import gdal

import geopandas as gpd
import pandas as pd
import csv
from collections import defaultdict
from datetime import datetime, timedelta
import os
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from pylab import rcParams
rcParams['figure.figsize'] = 10, 6
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
#from pmdarima.arima import auto_arima

from statsmodels.tsa.arima.model import ARIMA
#from pmdarima.arima import auto_arima
#from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import numpy as np
from statsmodels.tsa.stattools import kpss
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf


def kpss_test(series, **kw):
    statistic, p_value, n_lags, critical_values = kpss(series, **kw)
    # Format Output
    print(f"KPSS Statistic: {statistic}")
    print(f"p-value: {p_value}")
    print(f"num lags: {n_lags}")
    print("Critial Values:")
    for key, value in critical_values.items():
        print(f"   {key} : {value}")
    print(f'Result: The series is {"not " if p_value < 0.05 else ""}stationary')




def merge(dicts):
    result = defaultdict(list)
    for d in dicts:
        for key, value in d.items():
            result[key].append(value)
    return result


def time2sec(T):
   h=int(T[0])*3600; min=int(T[1])*60; seg=float(T[2])
   res = h+min+seg
   
   return res

def timeLine(time):
    count=0; timeSec=[]
    for i in range(len(time)):
        timeSec.append(time[i]-time[0])
    return timeSec

def OrderData(time, tput):
    Time=[]; Tput=[]
    for i in range(len(time)):
        Time.append(np.mean(time[i]))
        Tput.append(np.mean(tput[i]))
    Order_time=sorted(Time)
    Indexes = sorted(range(len(Time)),key=Time.__getitem__)
    Order_tput=[]
    for i in Indexes:
        Order_tput.append(Tput[i])
    return Order_time, Order_tput


def Dataset_Test(N, test_type):
    
    Training_time = 2

    
    


    return Training_time

# def hour_rounder(t):
#     # Rounds to nearest hour by adding a timedelta hour if minute >= 30
#     return (t.replace(second=0, microsecond=0, minute=0, hour=t.hour)+timedelta(hours=t.minute//30))
def rounder(t):
    if t.minute >= 30:
        return t.replace(second=0, microsecond=0, minute=0, hour=t.hour+1)
    else:
        return t.replace(second=0, microsecond=0, minute=0)

# Function to normalize the grid values
def normalize(band):
    # Calculate min and max of band
    band_max, band_min = band.max(), band.min()

    # Normalizes numpy arrays into scale 0.0 - 1.0
    return ((band - band_min)/(band_max - band_min))



def get_data(file, temp, wind): 
    path = 'data_sum/'
    
    raster = rasterio.open(path+file)
    type(raster)
    array = raster.read()
    
    #stats = []
    for band in array:
        avg = band.mean()
    
    info = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", file)
    ttime = info[0]+' '+info[1]
    d = datetime.strptime(ttime, "%Y%m%d %H%M%S")
    date = d.strftime("%Y-%m-%d %H:%M:%S")
    date_rounded = rounder(d)
    
        
    data_temp = pd.read_csv(temp)
    df = data_temp[['id', 'geom','timestamp','city', 'temperature', 'st_astext']]
    df.tail()
    
   
    
    for i in range(len(df.temperature)):
        d = datetime.strptime(df.timestamp[i], "%Y-%m-%d %H:%M:%S")
        if d==date_rounded:
            temp_analysis = df.temperature[i]
            
    data_wind = pd.read_csv(wind)
    df = data_wind[['id', 'geom','timestamp','city', 'angle', 'speed', 'st_astext']]
    df.tail()
    
    for i in range(len(df.speed)):
        d = datetime.strptime(df.timestamp[i], "%Y-%m-%d %H:%M:%S")
        if d==date_rounded:
            wind_analysis = df.speed[i]
            
            
    return date, file, avg, temp_analysis, wind_analysis




if __name__ == "__main__":
    
    t_global = pd.read_json(
    "http://userpage.fu-berlin.de/soga/soga-py/300/307000_time_series/t_global.json")
    t_global["Date"] = pd.to_datetime(t_global["Date"], format="%Y-%m-%d", errors="coerce")
    
    t_global = t_global.set_index("Date")["Monthly Anomaly_global"]
    print(t_global)
    
    

    temp_global_year = t_global.groupby(t_global.index.to_period("Y")).agg("mean")
    print(temp_global_year)
    
    temp_global_training = temp_global_year["1850-01-01":"2000-01-01"]
    temp_global_test = temp_global_year["2000-01-01":]
    
    # plt.figure(figsize=(18, 6))
    # plt.title("Earth Surface Temperature Anomalies", fontsize=14)
    # temp_global_training.plot(label="training set 1850-2000", fontsize=14)
    # temp_global_test.plot(label="test set 2001-2016", fontsize=14)

    # plt.legend()
    # plt.show()
    

    from scipy.stats import boxcox

    boxcox_transformed_data, boxcox_lamba = boxcox(temp_global_training + 10)
    boxcox_transformed_data = pd.Series(
        boxcox_transformed_data, index=temp_global_training.index
    )
    
    # fig, ax = plt.subplots(2, 1, figsize=(16, 8))
    # temp_global_training.plot(ax=ax[0], color="black", fontsize=14)
    # ax[0].set_title("Original time series", fontsize=14)


    # boxcox_transformed_data.plot(
    #     ax=ax[1],
    #     color="grey",
    # )
    # ax[1].set_title("Box-Cox transformed time series", fontsize=14)

    # ax[0].grid()
    # ax[1].grid()

    # plt.tight_layout()
    # plt.show()
    
    tes = kpss_test(temp_global_training)
    
    #print(tes)
    
    temp_global_training_diff1 = temp_global_training.diff()
    tes1=kpss_test(temp_global_training_diff1.dropna())  ## ignore NaN for kpss
    #print(tes1)
    
    # plt.figure(figsize=(18, 6))
    # plt.title("Differenced data set: temp_global_training_diff1`")
    # temp_global_training_diff1.plot(color="black")

    # plt.grid()
    # plt.show()
    #plt.figure(figsize=(18, 6))


    

    # fig, ax = plt.subplots(1, 2, figsize=(13, 5))

    # plot_acf(temp_global_training_diff1.dropna(), ax=ax[0])
    # ax[0].set_title("ACF")


    # plot_pacf(
    #     temp_global_training_diff1.dropna(), method="ywm", ax=ax[1]
    # )  ## add the calculation method running in the background ("ywm")

    # ax[1].set_title("PACF")
    # plt.show()
    
    # fit model
    model = ARIMA(temp_global_training, order=(3, 1, 0))
    model_fit = model.fit()
    print(model_fit.summary())
    
    print(
    "AR 1 =",
    round(model_fit.params["ar.L1"], 4),
    "AR 2 =",
    round(model_fit.params["ar.L2"], 4),
    "AR 3 =",
    round(model_fit.params["ar.L3"], 4),
    "sigma =",
    round(model_fit.params["sigma2"], 4),
    ) 
    
    print(model_fit.aicc)
    
    



    
    

    model = ARIMA(temp_global_training, order=(3, 1, 0))
    model_fit = model.fit()
    print(f"ARIMA(3,1,0) - AICc: {round(model_fit.aicc,2)}")

    model = ARIMA(temp_global_training, order=(3, 1, 1))
    model_fit = model.fit()
    print(f"ARIMA(3,1,1) - AICc: {round(model_fit.aicc,2)}")

    model = ARIMA(temp_global_training, order=(3, 1, 2))
    model_fit = model.fit()
    print(f"ARIMA(3,1,2) - AICc: {round(model_fit.aicc,2)}")

    model = ARIMA(temp_global_training, order=(2, 1, 2))
    model_fit = model.fit()
    print(f"ARIMA(2,1,2) - AICc: {round(model_fit.aicc,2)}")
    
    # auto_model = auto_arima(temp_global_training)
    # print(auto_model.summary())
    
    model = ARIMA(temp_global_training, order=(2, 1, 3))
    fitted = model.fit()
    
    

    plt.figure(figsize=(13, 4))
    plt.title("Earth Surface Temperature Anomalies")
    temp_global_training.plot(color="black", label="training set 1850-2000")
    plt.plot(fitted.fittedvalues, color="blue", label="fitted values")

    plt.legend()
    plt.show()










        
