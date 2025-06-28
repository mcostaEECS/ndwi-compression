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
import scipy.io
import psutil
import datetime
import os
import re
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats import norm

# For image analysis
from skimage.filters.rank import minimum, maximum, mean, median
from skimage.filters import gaussian
import cv2
from scipy.spatial import distance
from pprint import pp
from scipy.io import savemat
from numpy.random import normal
from numpy import hstack
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats import norm
from torch.multiprocessing import Pool, set_start_method, freeze_support
from concurrent.futures import ProcessPoolExecutor
from operator import mul
import os



#-------------------------------- CLI R script -------------------------------#
def Rscript(cmd):
    
    df = cmd[0].split(";")
    
    dataTest = df[3]+df[0]; dataRef = df[3]+df[1]; dataOut = df[4]+'RES%s.mat'%df[2]
    ImageRead1=sio.loadmat(dataTest); ImageRead2=sio.loadmat(dataRef)
    Itest=ImageRead1['yt']; Iref=ImageRead2['y0'] 
    
   
    CD = []; ths = 4
    for i in range(len(Itest)):
            
        sample = hstack((Itest[i], Iref[i]))
        
        ecdf = ECDF(sample)   # cdf computation
        ecdf.y[ecdf.y==1]=0.99999  # sigficance level (assuming Gaussian distribution)
        ecdf.y[ecdf.y==0]=0.00001
        Q = norm.ppf(ecdf.y)
        
        RES=[]; cnt=0
        for j in range(len(Itest[i])):
            if (Q[j] <= -ths) or (Q[j] >= ths):
                res=0
            else:
                if Itest[i][j] > 2*Iref[i][j]:
                    res=Itest[i][j]         # if is out of gaaussian iid = change
                    cnt += 1
                else:
                    res = 0
            RES.append(res)
        CD.append(RES)
        
    ICD = np.array(CD)
    
    
    
    scipy.io.savemat(dataOut, {'ht':ICD})
    
    
    return cnt
        
        
    # print(Iref.shape)
    # print(Itest.shape)
    # print(ICD.shape)
    
    # fig = plt.figure('test')
    # plt.suptitle('Binarizada')

    # ax = fig.add_subplot(1, 3, 1)
    # plt.imshow(Itest, cmap = plt.cm.gray)
    # plt.axis("off")
     
    # ax = fig.add_subplot(1, 3, 2)
    # plt.imshow(Iref, cmap = plt.cm.gray)
    # plt.axis("off")
    
    # ax = fig.add_subplot(1, 3, 3)
    # plt.imshow(ICD, cmap = plt.cm.gray)
    # plt.axis("off")
     
    # plt.show() 
    
    
    
  


def AR1(test_type, N, pair):
     
    folder ='%sn%dpair%d/'%(test_type,N, pair)
    path = '/home/marcello-costa/workspace/2DAR1/Input/DataAR1/' 
    path1 = '/home/marcello-costa/workspace/2DAR1/Output/DataAR1/' 
    
    input = path+folder
    output = path1+folder
    os.mkdir(output)
       
    files = os.listdir(input)
    files_images = [i for i in files if i.endswith('.mat')]
   
    dataset1 = []; dataset2 = []
    for i in range(len(files_images)):
        if files_images[i].find('T') != -1:
            dataset1.append(files_images[i])
        elif files_images[i].find('A') != -1:
            dataset2.append(files_images[i])
            
            
            
    #---------------------------- Reduced Image Test -----------------------------#
    
    # if pair==0 or pair==4 or pair==8 or pair==12 or pair==16 or pair==20:
    #     #start=500; end = 600
    #     start=400; end = 800
    # elif pair==1 or pair==5 or pair==9 or pair==13 or pair==17 or pair==21:
    #     start=200; end = 600 
    # elif pair==2 or pair==6 or pair==10 or pair==14 or pair==18 or pair==22:
    #     start=1600; end = 2000
    # elif pair==3 or pair==7 or pair==11 or pair==15 or pair==19 or pair==23:
    #     start=1800; end = 2200        
  
    dataset1 = sorted(dataset1, key=lambda s: int(re.search(r'\d+', s).group()))
    dataset2 = sorted(dataset2, key=lambda s: int(re.search(r'\d+', s).group()))
    
    # dataset1 = dataset1[start:end]  #36 / 18 limite de memoria
    # dataset2 = dataset2[start:end]
     
    
    # dataset1 = sorted(dataset1, key=lambda s: int(re.search(r'\d+', s).group()))
    # dataset2 = sorted(dataset2, key=lambda s: int(re.search(r'\d+', s).group()))
    
    
    step = 1
    com = 't:'  # script code in R
    im1 = []; im2 = []
    for j in range(0,len(dataset1),step):
        a=[]; b=[]; c=[]; d=[]
        for i in range(step):
            a.append(dataset1[i+j]+';'+dataset2[i+j]+';'+ re.findall(r"[-+]?(?:\d*\.\d+|\d+)", dataset2[i+j])[-1]+';'+input+';'+output)
        im1.append(a)
    
    # print(im1[0])
    CNT=[]
    START = time.time()
    for i in range(len(im1)):
        Cnt = Rscript(im1[i])
        CNT.append(Cnt)
    END = time.time()

    time_AR1 = END - START
    
    cnt = np.sum(CNT)
    
    
    return time_AR1, cnt
    
  
    
    
# if __name__ == "__main__":
    
#     test_type = ['GLRT', 'AR']
#     test_type  = test_type[1]
    
#     nroPairs = 1
#     s = 0
#     dimH = 1500
#     dimW = 1000
#     K = 10
#     N = 100
    
#     AR1(test_type, N, s)
    
    


  
     
 
 
 
 
 