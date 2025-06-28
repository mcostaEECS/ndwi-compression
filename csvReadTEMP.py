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
import pandas as pd
import csv
from collections import defaultdict

import rasterio
import rasterio.plot
from rasterio.plot import show
import matplotlib.pyplot as plt
from rasterio.plot import show_hist
import datetime
from osgeo import gdal

import geopandas as gpd
from datetime import datetime


def Dataset_Test(N, test_type):
    
    Training_time = 2

    
    


    return Training_time

def merge(dicts):
    result = defaultdict(list)
    for d in dicts:
        for key, value in d.items():
            result[key].append(value)
    return result

# Function to normalize the grid values
def normalize(band):
    # Calculate min and max of band
    band_max, band_min = band.max(), band.min()

    # Normalizes numpy arrays into scale 0.0 - 1.0
    return ((band - band_min)/(band_max - band_min))




if __name__ == "__main__":
    
    # file path
    path = 'temperature/'
    
    Localization= 'Vaxholm'
    Month = 6
    
   
    
    
    files = os.listdir(path)
    files_logs = [i for i in files if i.endswith('.csv')]
    
    print(files_logs[0])
    
    dataset = pd.read_csv(path+files_logs[0])
    #id	geom	timestamp	city	temperature	st_astext
    #id	geom	timestamp	city	angle	speed	st_astext
    # id	geom	timestamp	city	angle	speed	st_astext


    df = dataset[['id', 'geom','timestamp','city', 'temperature', 'st_astext']]
    
    # print(df.timestamp[0])
    
    # RES2 = int(re.findall(r'[+]?(?:\d*\.\d+|\d+)', df.timestamp[0])[1])
    # print(RES2)
    
    
    df.tail()
    
    res = []
    for i in range(len(df.temperature)):
        #print(df.temperature[i])
        if df.city[i]==Localization and int(re.findall(r'[+]?(?:\d*\.\d+|\d+)', df.timestamp[i])[1])==Month:
            a = df.timestamp[i]
            b = df.id[i]; c = df.geom[i]; d = df.city[i]
            e = df.temperature[i]; f = df.st_astext[i]
            res.append(str(a)+';'+str(b)+';'+str(c)+';'+str(d)+';'+str(e)+';'+str(f))
            
  
    fmt = "%Y-%m-%d %H:%M:%S"
    
    dataset = sorted(res, key=lambda s: datetime.strptime(s, fmt+';'+s[20:250]))
    
    # make the csv dict again
    
          
   
    #print(dataset)
    
    pathOut = 'Output/'
    nameFile=pathOut+'temperature_%s.csv'%Localization
    
    dicList = []
    energy = 0
    keys =  ['id', 'geom','timestamp','city', 'temperature', 'st_astext']
    for i in range(len(dataset)):
        Time = dataset[i].split(';')[0]
        City = dataset[i].split(';')[3]
        Temperature = dataset[i].split(';')[4]
        Id = dataset[i].split(';')[1]
        St_astext = dataset[i].split(';')[5]
        Geom = dataset[i].split(';')[2]
      
        
        
        
        dicts = {}
        keys = ['id', 'geom','timestamp','city', 'temperature', 'st_astext']
        values = [Id, Geom, Time, City, Temperature, St_astext]
        for k in range(len(keys)):
            dicts[keys[k]] = values[k]
        dicList.append(dicts)
    dic = merge(dicList)
    pd.DataFrame(dic).to_csv(nameFile)
  
    
    
    
    
    
    
    
    
    

    
    
    # 02/06
    # data1 = '20200602_084908_98_1065_3B_AnalyticMS_SR.tif' # Stockolm Soth Flemingsberg
    # data2 = '20200602_084911_04_1065_3B_AnalyticMS_SR.tif' # Taby
    # data3 = '20200602_084911_04_1065_3B_AnalyticMS_SR.tif' # Taby - Bogesund EAST
    # data4 = '20200602_091121_40_2212_3B_AnalyticMS_SR.tif' # Stockolm North - Danderyd - Solna
    # data5 = '20200602_095241_01_105c_3B_AnalyticMS_SR.tif' # Vaxholm/Bogesund
    
    
    # 01/06
    #data1 = '20200601_093723_1008_3B_AnalyticMS_SR.tif'  # Åkersberga
    #data2 = '20200601_093724_1008_3B_AnalyticMS_SR.tif'  # Vaxholm/Bogesund
    #data3 = '20200601_093725_1008_3B_AnalyticMS_SR.tif'  # NACKA
    
    
    # 03/06
    # data1 = '20200603_085131_21_1067_3B_AnalyticMS_SR.tif' # Jakobsberg/Stocholm/Lindigo
    # data2 = '20200603_092608_0e20_3B_AnalyticMS_SR.tif' # Taby - Bogesund (short)
    # data3 = '20200603_092609_0e20_3B_AnalyticMS_SR.tif' # Danderyd - Lidingö  (short)
    # data4 = '20200603_094045_0f4e_3B_AnalyticMS_SR.tif' # Vaxholm/Bogesund (short)
    # data5 = '20200603_094046_0f4e_3B_AnalyticMS_SR.tif' # Nacka-Gustavsberg (short)
    
    # 05/06
    # data1 = '20200605_100725_77_1064_3B_AnalyticMS_SR.tif' # Upssala Vasby - Taby - lidingö (long)
    # data2= '20200605_100727_27_1064_3B_AnalyticMS_SR.tif'  # Stocholm / solentuna / Jakobsberg / Farsta (long)
    
    # 07/06
    # data1 = '20200607_091654_18_2257_3B_AnalyticMS_SR.tif' # Vaxholm-Åkersberga-Torsby  (long)
    # data2= '20200607_091656_57_2257_3B_AnalyticMS_SR.tif'  # Vaxholm-Nacka-Gustavsberg (long)
    
    # 09/06
    # data1 = '20200609_100701_55_105e_3B_AnalyticMS_SR.tif' # Vaxholm-Åkersberga  (long)
    # data2= '20200609_100703_07_105e_3B_AnalyticMS_SR.tif'  # Lidingö-Bogesund-Nacka-stockholm east (long)
    # data3= '20200609_100704_59_105e_3B_AnalyticMS_SR.tif'  # Socholm soyth - tyresö (long)
    
    #winter data
   
    
    
    
    
    #data[12] = Löka (20190206_093035_1010_3B_AnalyticMS_SR.tif)
    #data[15] = stromma-torsby (20190213_093111_0f43_3B_AnalyticMS_SR.tif)
    #data[19] = Stromma / Gustavsvi
    # 18 Rinö
    
    #data[24] = lidingo-stockolm east - taby
    #data[25] = upllands vasby
    #data 26 = Skärmaräng
    # data 29 Rindö
    # 30-32 Nacka - Boo - Gustavsberg
    # 34 Hägernäs
    # 36 Åkersberga
    # 0 = Fågelbrolandet
    # 1 = Värmdö-Evlinge
    # 2 = Finnhamn löka
    # see 4
    # 5 = Torsby Vaxholm
    # 7 = vaxholm Bogesund
    # 39 Ljusterö and Åkersberga
    
    # data[18] see
    
    #02/06
    #raster_fp = 'demo/20200602_084908_98_1065_3B_AnalyticMS_SR.tif'   # Stockolm lan mot Flemingsberg
    #raster_fp = 'demo/20200602_084911_04_1065_3B_AnalyticMS_SR.tif'   # NA
    #raster_fp = path+data[24] 
    
    
    
    
     
    
    
    
    
    
    
    #raster_fp = 'demo/20200601_093725_1008_3B_AnalyticMS_SR.tif' # 4 bands and uint16 NACKA
    
    
    #raster_fp = 'demo/20200811_084329_83_1067_3B_udm2.tif' # 8 bands and uint8
    #raster_fp = 'demo/20200811_084329_83_1067_3B_AnalyticMS_DN_udm.tif' # 1 band1 and uint8
    
    #raster_fp = 'demo2/20190206_092500_0e0f_3B_AnalyticMS_SR.tif'
    
    # open file with rasterio
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
#     # band1 = raster.read(1)
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
    
#     show((raster,4))
    
    
# # # Create subplots
# #     fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, nrows=1, figsize=(12, 5), sharey=True)

# #     # Plot Red, Green and Blue
# #     show((raster, 1), cmap='Reds', ax=ax1)
# #     show((raster, 2), cmap='Greens', ax=ax2)
# #     show((raster, 3), cmap='Blues', ax=ax3)

# #     # Set titles
# #     ax1.set_title("Red")
# #     ax2.set_title("Green")
# #     ax3.set_title("Blue")

# #     plt.show()

#     # blue = raster.read(1)
#     # green = raster.read(2)
#     # red = raster.read(3)
#     # nir = raster.read(4)

#     # # # Normalize all bands for natural  color composite
#     # nblue = normalize(blue)
#     # ngreen = normalize(green)
#     # nred = normalize(red)
#     # nnir = normalize(nir)
    
# #     False_composite = np.dstack((nnir, ngreen, nblue))

# # # Let's see how our color composite looks like
# #     plt.imshow(False_composite)


    
# #     #plt.imshow(nnir)
# #     plt.show()

#     # Create RGB natural color composite
#     # RGB_composite = np.dstack((nred, ngreen, nblue))

#     # print(RGB_composite.shape)

    

#     # # Let's see how our color composite looks like
#     # plt.imshow(RGB_composite)
    
#     # Create subplot and set figure size
#     # fig, axhist = plt.subplots(1, 1, figsize=(8, 4))

#     # # Red
#     # show_hist(nred, bins=200, histtype='step',
#     #         lw=1, edgecolor= 'r', alpha=0.8, facecolor='r', ax=axhist)

#     # # Green
#     # show_hist(ngreen, bins=200, histtype='step',
#     #         lw=1, edgecolor= 'g', alpha=0.8, facecolor='r', ax=axhist)

#     # # Blue
#     # show_hist(nblue, bins=200, histtype='step',
#     #         lw=1, edgecolor= 'b', alpha=0.8, facecolor='r', ax=axhist)
    
#     # #nir
    # show_hist(nnir, bins=200, histtype='step',
    #         lw=1, edgecolor= 'k', alpha=0.8, facecolor='r', ax=axhist)
    
    # plt.show()




 
    

    
    
  
                
                
                
                
                



                








      