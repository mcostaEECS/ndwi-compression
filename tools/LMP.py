from __future__ import division
import numpy as np
import array
import cv2
import scipy.io
import time
from skimage import data
from skimage.morphology import disk
from numpy import zeros,sqrt, mean,linspace,concatenate, cumsum
from matplotlib import pyplot as plt
import scipy.io as sio

import sys
from PIL import Image
from load_data import load_data
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
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats import norm

# For image analysis
from skimage.filters.rank import minimum, maximum, mean, median
from skimage.filters import gaussian
import cv2
from scipy.spatial import distance
from pprint import pp
from scipy.io import savemat
from scipy import stats
import os
import signal
import subprocess
import datetime
import pyprind
import time


#------------------------- Multiplexer Function ------------------------#
def moving_average(im):
    kernel1 = np.ones((7,7),np.float32)/49
    Filtered_data = cv2.filter2D(src=im, ddepth=-1, kernel=kernel1) 
    return Filtered_data


#------------------------- Split Image ------------------------#
def subarrays(arr, nrows, ncols):
    h, w = arr.shape
    assert h % nrows == 0, f"{h} rows is not evenly divisible by {nrows}"
    assert w % ncols == 0, f"{w} cols is not evenly divisible by {ncols}"
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))
    
#------------------- Correlation Detection (LMP) ----------------------#
def corrTest(t,r):
    
    # work here for rho local in the recursion
    pab = [a*b for a,b in zip(t,r)]
    paa = [a*b for a,b in zip(t,t)]
    pbb = [a*b for a,b in zip(r,r)]
    den = [a+b for a,b in zip(paa,pbb)]
    return [a/b for a,b in zip(2*pab,den)]  


#------------------------- Change Detection Tests ------------------------#
def test(a,b, temp, wind, N, K):
    
    th = 0.95
    

    kernel = np.ones([K, K]) 
    temparray = np.copy(a)
    refarray = np.copy(b)
    
    kernel = kernel
    Arraylist = []; arraylist = []
   
    

    for y in range(len(kernel)):
        temparray = np.roll(temparray, y - 1, axis=0)
        refarray = np.roll(refarray, y - 1, axis=0)
        
        
        for x in range(len(kernel)):
            temparray_X = np.copy(temparray)
            refarray_X = np.copy(refarray)
            
            temparray_X = np.roll(temparray_X, x - 1, axis=1)
            refarray_X = np.roll(refarray_X, x - 1, axis=1)
                   
            # LMP
            rho_local = corrTest(temparray_X*kernel[y,x], refarray_X*kernel[y,x])
            avg_rho = 1 - np.mean(rho_local)
            T = temparray_X*kernel[y,x]
            
            ## joga na recursao na condicao local T*rho_local ao inves do loop
            
            # UMP
            cnt = 0
            res = []
    
            for i in range(len(T)):
              for j in range(len(T[0])):
                if 1 - rho_local[i][j] >= th*avg_rho:
                  novaC = 1 #T[i][j]*1
                  cnt += 1
                else:
                  novaC = 0
                res.append(novaC)
            arraylist.append(np.reshape(res, (N, N)))
            
    Arraylist = np.array(arraylist)
    
    return Arraylist[0]




#------------------------- Data Input/Output ------------------------#
 
if __name__ == "__main__":
    
     band = ['blue', 'green', 'red', 'nir', 'ndwi']
     season = ['summer', 'winter', 'sea']
     season = season[1]
    
     if season == 'summer':
         window = rasterio.windows.Window(1000,1000,3000,1000)
     elif season == 'winter':
         window = rasterio.windows.Window(800,1400,3000,1000)
     elif season == 'sea':
         window = rasterio.windows.Window(1000,1400,3000,1000) # sea check in the winter (Västerskaten)
        

     with rasterio.open('data/%s/%s.tif'%(season,band[1])) as src:
         band_green = src.read(1, window=window)
        
     with rasterio.open('data/%s/%s.tif'%(season,band[3])) as src:
         band_nir = src.read(1, window=window)
        
     with rasterio.open('data/%s/%s.tif'%(season,band[4])) as src:
         band_ndwi = src.read(1, window=window)
    
     temp = 230; wind=1.5; N=10; K=10
     
     band_ndwi = band_ndwi[0:500,0:500]
     
     H,W = band_ndwi.shape
     dimH=H; dimW=W
     
     Itest = subarrays(band_ndwi, N, N)
    
     CD=[]; lmp=[]; ump=[]; count = []
     i = 0
     while i < len(Itest):
        
         if i <= len(Itest)-2:
             OUT0=test(Itest[i+1], Itest[i], temp, wind, N, K)
             CD.append(OUT0)
             #lmp.append(lmp_time); ump.append(ump_time); count.append(cnt)
        
         else:
             OUT0=test(Itest[len(Itest)-1], Itest[len(Itest)-2], temp, wind, N, K)
             CD.append(OUT0)
             #lmp.append(lmp_time); ump.append(ump_time); count.append(cnt)
         i = i + 1
        
     ICD=[]
     im = []
     for j in range(0,len(CD),int(W/N)):
         a=[]
         for l in range(int(W/N)):
             a.append(CD[j+l])
         im.append(np.hstack((a)))
     ICD = np.vstack((im))
     
     
     im = [band_ndwi, ICD]
         
     fig = plt.figure(figsize=(8, 8))
     columns = 1
     rows = 2
     for i in range(1, columns*rows+1):
       
         img = im[i-1]
         fig.add_subplot(rows, columns, i)
         plt.imshow(img)
     plt.show()
     
     
    #  plt.imshow(ICD, cmap = plt.cm.gray)
    #  plt.axis("off")
    #  plt.show()
    
         
    
    
    



     
    #  plt.imshow(band_ndwi)
    #  plt.show()
    
         
    #  dat = LMP(band_ndwi, temp, wind,N)
         
    #  im = [band_ndwi, dat]
         
    #  fig = plt.figure(figsize=(8, 8))
    #  columns = 1
    #  rows = 2
    #  for i in range(1, columns*rows+1):
       
    #      img = im[i-1]
    #      fig.add_subplot(rows, columns, i)
    #      plt.imshow(img)
    #  plt.show()
     
    