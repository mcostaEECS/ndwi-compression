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
import geopandas as gpd
# import earthpy as et
# import earthpy.spatial as es
# import earthpy.plot as ep



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

def brighten(band):
    alpha=0.13
    beta=0
    return np.clip(alpha*band+beta, 0,255)


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
    
    
    
   
    
    
    season = ['summer', 'winter', 'sea']
    season = season[2]
    
 
    
    
    
    if season == 'summer':
         window = rasterio.windows.Window(1000,1000,3000,1000)
         path = 'data/%s/'%season
    elif season == 'winter':
         window = rasterio.windows.Window(800,1400,3000,1000)
         path = 'data/%s/'%season
    elif season == 'sea':
         window = rasterio.windows.Window(1000,1400,3000,1000) # sea check in the winter (Västerskaten)
         path = 'data/%s/'%season
         
    files = os.listdir(path)
    data = [i for i in files if i.endswith('.tif')]
    meta=  [i for i in files if i.endswith('.xml')]
    
    image_file = path+data[0]
    metadata_file = path+meta[0]

    
    with rasterio.open(image_file) as src:
        band_blue = src.read(1, window=window)
    
    with rasterio.open(image_file) as src:
        band_green = src.read(2, window=window)
        
    with rasterio.open(image_file) as src:
        band_red = src.read(3, window=window)
        
    # with rasterio.open(image_file) as src:
    #     band_nir = src.read(4)
        
    xmldoc = minidom.parse(metadata_file)
    nodes = xmldoc.getElementsByTagName("ps:bandSpecificMetadata")

    
    
    
    # XML parser refers to bands by numbers 1-4
    coeffs = {}
    for node in nodes:
        bn = node.getElementsByTagName("ps:bandNumber")[0].firstChild.data
        if bn in ['1', '2', '3', '4']:
            i = int(bn)
            value = node.getElementsByTagName("ps:reflectanceCoefficient")[0].firstChild.data
            coeffs[i] = float(value)
        
    # Multiply by corresponding coefficients
    band_blue = band_blue * coeffs[1]
    band_green = band_green * coeffs[2]
    band_red = band_red * coeffs[3]
    #band_nir = band_nir * coeffs[4]
    
    
    band_red=brighten(band_red)
    band_blue=brighten(band_blue)
    band_green=brighten(band_green)

    # red_bn = normalize(red_b)
    # green_bn = normalize(green_b)
    # blue_bn = normalize(blue_b)
        
    
    nblue = normalize(band_blue)
    band_green = normalize(band_green)
    nred = normalize(band_red)
    
    
    
    
    
    #band_nir = normalize(band_nir)
    
    # Allow division by zero
    # with np.errstate(divide='ignore', invalid='ignore'):
    #     ndwi = (band_green.astype(float) - band_nir.astype(float)) / (band_green + band_nir)
        
    # #Geotob's solution
    # ndwi[np.isnan(ndwi)] = 0
    
    
    rgb_composite_n= np.dstack((nred, band_green, nblue))
    plt.imshow(rgb_composite_n)


    # Calculate NDVI
    #ndvi = (band_nir.astype(float) - band_red.astype(float)) / (band_nir + band_red)
    
    #ndwi = (band_green - band_nir) / (band_green + band_nir)
    
  
    # k = band_green - band_nir
    # m = band_nir + band_red
    
    # print(ndwi.mean())
    
    # plt.imshow(ndwi)
    plt.show()
    
    
    # data = src.read([4,3,2])
    # norm = (data * (255 / np.max(data))).astype(np.uint8)
    # plt.imshow(norm)
    
    # show(rgb.read([4,3,2])/4000,transform=src.transform,title='Image-bands 4,3,2 IN 0-255 Values')
    
    # Set spatial characteristics of the output object to mirror the input
    # kwargs = src.meta
    # kwargs.update(
    #     dtype=rasterio.float32,
    #     count = 1)

    # # Create the file
    # with rasterio.open(path+'ndwi.tif', 'w', **kwargs) as dst:
    #         dst.write_band(1, ndwi.astype(rasterio.float32))

    
    # Create subplot and set figure size
    # fig, axhist = plt.subplots(1, 1, figsize=(8, 4))

    # # Red
    # show_hist(band_green, bins=200, histtype='step',
    #         lw=1, edgecolor= 'g', alpha=0.8, facecolor='r', ax=axhist)

    # # Green
    # show_hist(band_nir, bins=200, histtype='step',
    #         lw=1, edgecolor= 'r', alpha=0.8, facecolor='r', ax=axhist)

    # # Blue
    # show_hist(ndwi, bins=200, histtype='step',
    #         lw=1, edgecolor= 'b', alpha=0.8, facecolor='r', ax=axhist)
    
    # # #nir
    # # show_hist(nnir, bins=200, histtype='step',
    # #         lw=1, edgecolor= 'k', alpha=0.8, facecolor='r', ax=axhist)
    
    # plt.show()
    
    # Save Image
    # create the mask ARMA - mean Rayleigh

    
    
    
    
    
    
    #print('min:', ndwi.min(), 'mean:', ndwi.mean(), 'median:', np.median(ndwi), 'max:', ndwi.max())
    
    


    
    
    
#     raster_fp = path1+df1.filename[0]
    
#     raster = rasterio.open(raster_fp)
#     type(raster)
    
#     print(raster.crs)
    
#     print(raster.count)
    
#     # Dimensions
#     print(raster.width)
#     print(raster.height)
    
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
    
    



 
    

    
    
  
                
                
                
                
                



                








      
    