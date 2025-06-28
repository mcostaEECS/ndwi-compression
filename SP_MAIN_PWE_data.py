from __future__ import division
import numpy as np
import time
from matplotlib import pyplot as plt
import array
import scipy.io
import scipy.io as io

import pyprind
import time
import psutil
import datetime
import pyprind
import time
import psutil
import re
import os
import signal
import subprocess
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



if __name__ == "__main__":
    
    test_type = ['GLRT', 'LMP']
    test_type  = test_type[1]
    K = 9
    N = 40
    
    campaign = 'Runtime_%s_N_%d_K_%d'%(test_type, N, K)
    path = '/home/marcello-costa/workspace/LMP/Output/logs/'
    
    nroPairs = 1
    dimH = 3000; dimW = 2000
           
    TEST_time=[]; CFAR_time = []; Nop=[]
    bar = pyprind.ProgBar(nroPairs, monitor=True, title=campaign)
    for s in range(nroPairs):
        s = s + 17
        par=load_data(test_type)[s]
        Itest=par[20][0:dimH,0:dimW]
        
        Iref=[par[0][0:dimH,0:dimW],  par[1][0:dimH,0:dimW],  par[2][0:dimH,0:dimW],  par[3][0:dimH,0:dimW],
              par[4][0:dimH,0:dimW],  par[5][0:dimH,0:dimW],  par[6][0:dimH,0:dimW],  par[7][0:dimH,0:dimW], 
              par[8][0:dimH,0:dimW],  par[9][0:dimH,0:dimW],  par[10][0:dimH,0:dimW], par[11][0:dimH,0:dimW],
              par[12][0:dimH,0:dimW], par[13][0:dimH,0:dimW], par[14][0:dimH,0:dimW], par[15][0:dimH,0:dimW],
              par[16][0:dimH,0:dimW], par[17][0:dimH,0:dimW]]
        
        tp=par[18]; tp =  np.fliplr(tp); TP =par[19]
        pair = par[-1] - 1 
        
        if pair==0 or pair==4 or pair==8 or pair==12 or pair==16 or pair==20:
            start=400; end = 800
        elif pair==1 or pair==5 or pair==9 or pair==13 or pair==17 or pair==21:
            start=200; end = 600 
            
        elif pair==2 or pair==6 or pair==10 or pair==14 or pair==18 or pair==22:
            start=1600; end = 2000
        elif pair==3 or pair==7 or pair==11 or pair==15 or pair==19 or pair==23:
            start=1800; end = 2200     
            
        Itest = Itest[start:end]  #36 / 18 limite de memoria
        Iref = Iref[0][start:end]
        #Iref = Iref[0]
        
        name_file = path+campaign+'.txt'
        with open(name_file, 'a') as f:
            if test_type == 'LMP':                
                [test_time, cfar_time, Nt] = LMP(Itest, Iref, TP, N, K, s)
                f.write(str(test_type)+';'+'N_'+str(N)+';'+'K_'+str(K)+';'+'pair_'+str(s)+';'+'HT_runtime_'+str(test_time)+';'+'cfar_runtime_'+str(cfar_time)+';'+'Threshold_op_'+str(Nt)+'\n')
                bar.update()
            elif test_type == 'GLRT':
                [test_time, cfar_time, Nt] = GLRT_CFAR(Itest, Iref, TP, N, K, s)
                f.write(str(test_type)+';'+'N_'+str(N)+';'+'K_'+str(K)+';'+'pair_'+str(s)+';'+'HT_runtime_'+str(test_time)+';'+'cfar_runtime_'+str(cfar_time)+';'+'Threshold_op_'+str(Nt)+'\n')
                bar.update()
            TEST_time.append(test_time)
            CFAR_time.append(cfar_time)
            Nop.append(Nt)
        
        
       

    