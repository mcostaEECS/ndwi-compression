from __future__ import division
import numpy as np
from matplotlib import pyplot as plt
import array
import scipy.io
import scipy.io as io
from xml.dom import minidom

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
#plt.style.use('fivethirtyeight')
from pylab import rcParams
rcParams['figure.figsize'] = 10, 6
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
#from statsmodels.tsa.arima_model import ARIMA
#from pmdarima.arima import auto_arima
#from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
import numpy as np
from matplotlib.patches import Rectangle



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
    
    data = ['blue', 'green', 'red', 'nir', 'ndwi']
    season = ['summer', 'winter', 'sea']
    #season = season[2]
    
    ndwi = []
    for i in range(len(season)):
        
        if season[i] == 'summer':
            window = rasterio.windows.Window(1000,1000,3000,1000)
        elif season[i] == 'winter':
            window = rasterio.windows.Window(800,1400,3000,1000)
        elif season[i] == 'sea':
            window = rasterio.windows.Window(1000,1400,3000,1000) # sea check in the winter (Västerskaten)
            
        with rasterio.open('data/%s/%s.tif'%(season[i],data[4])) as src:
            ndwi.append(src.read(1, window=window))
            
            
      

    
    # # Create subplot and set figure size
    fig, axhist = plt.subplots(1, 1, figsize=(8, 4))
    
    cmap = plt.get_cmap('jet')
    low = cmap(0.5)
    medium =cmap(0.25)
    high = cmap(0.8)

    # Red
    show_hist(ndwi[0], bins=200, histtype='step',
            lw=1, edgecolor= low, alpha=0.8, facecolor='r', ax=axhist)

    # Green
    show_hist(ndwi[1], bins=200, histtype='step',
            lw=1, edgecolor= medium, alpha=0.8, facecolor='r', ax=axhist)

    # Blue
    show_hist(ndwi[2], bins=200, histtype='step',
            lw=1, edgecolor= high, alpha=0.8, facecolor='r', ax=axhist)
    
    #create legend

    handles = [Rectangle((0,0),1,1,color=c,ec="k") for c in [low,medium, high]]
    labels= ["summer","winter", "sea"]
    axhist.set_xlabel('Normalized Amplitude')
    plt.legend(handles, labels)
    

    
    plt.show()
    
    # Save Image
    # create the mask ARMA - mean Rayleigh

    
    
    
    
    
    
    #print('min:', ndwi.min(), 'mean:', ndwi.mean(), 'median:', np.median(ndwi), 'max:', ndwi.max())
    
    


    
    
    
#     raster_fp = path1+df1.filename[0]
    
#     raster = rasterio.open(raster_fp)
#     type(raster)
    
#     print(raster.crs)
    
#     print(raster.count)
    

    
#     print(raster.transform)
    
#     print(raster.meta)
    
#     array = raster.read()
    
#     print(type(array))
#     print(array.shape)
    
#     # # Read band 1, 2, and 3
#     band1 = raster.read(1)
#     # band2 = raster.read(2)
#     # band3 = raster.read(3)
#     # band4 = raster.read(4)
    
#     # assert band1.all() == band1.all(), "The bands are not the same!"
    
#     # print(band1.dtype)
    
#     # # print(band1)
    
#     # stats = []
#     # for band in array:
#     #     stats.append({
#     #         'min': band.min(),
#     #         'mean': band.mean(),
#     #         'median': np.median(band),
#     #         'max': band.max()})
        
#     # print(stats)
    
#     #show((raster,4))
#     print(band1.dtype)
    
#     # # Create subplots
#     # fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, nrows=1, figsize=(12, 5), sharey=True)

#     # # Plot Red, Green and Blue
#     # show((raster, 1), cmap='Reds', ax=ax1)
#     # show((raster, 2), cmap='Greens', ax=ax2)
#     # show((raster, 3), cmap='Blues', ax=ax3)

#     # # Set titles
#     # ax1.set_title("Red")
#     # ax2.set_title("Green")
#     # ax3.set_title("Blue")

#     # plt.show()

#     blue = raster.read(1)
#     green = raster.read(2)
#     red = raster.read(3)
#     nir = raster.read(4)

#     # # # Normalize all bands for natural  color composite
#     nblue = normalize(blue)
#     ngreen = normalize(green)
#     nred = normalize(red)
#     nnir = normalize(nir)
    
# #     False_composite = np.dstack((nnir, ngreen, nblue))

# # Let's see how our color composite looks like
#     plt.imshow(False_composite)


    
#     #plt.imshow(nnir)
#     plt.show()

    # Create RGB natural color composite
    #NDWI_composite = np.dstack((nnir, ngreen))
    #print(NDWI_composite.shape)

    

    # # # Let's see how our color composite looks like
    # plt.imshow(nnir)
    # plt.show()
    
    



 
    

    
    
  
                
                
                
                
                



                








      
    