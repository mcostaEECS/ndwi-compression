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
    

    
    df = cmd.split(";")
    
    dataTest = df[20]+df[0]; 
    dataRef1 = df[20]+df[1]; dataRef6 = df[20]+df[6]; dataRef11 = df[20]+df[11]; dataRef15 = df[20]+df[15]
    dataRef2 = df[20]+df[2]; dataRef7 = df[20]+df[7]; dataRef12 = df[20]+df[12]; dataRef16 = df[20]+df[16]
    dataRef3 = df[20]+df[3]; dataRef8 = df[20]+df[8]; dataRef13 = df[20]+df[13]; dataRef17 = df[20]+df[17]
    dataRef4 = df[20]+df[4]; dataRef9 = df[20]+df[9]; dataRef14 = df[20]+df[14]; dataRef18 = df[20]+df[18]
    dataRef5 = df[20]+df[5]; dataRef10 = df[20]+df[10]
    
  
    dataOut = df[21]+'RES%s.mat'%df[19]
    
    ImageRead0=sio.loadmat(dataTest); Itest=ImageRead0['yt']
    
    ImageRead1=sio.loadmat(dataRef1); Iref1=ImageRead1['y0'] 
    ImageRead2=sio.loadmat(dataRef2); Iref2=ImageRead2['y0']
    ImageRead3=sio.loadmat(dataRef3); Iref3=ImageRead3['y0']
    ImageRead4=sio.loadmat(dataRef4); Iref4=ImageRead4['y0']
    ImageRead5=sio.loadmat(dataRef5); Iref5=ImageRead5['y0']
    ImageRead6=sio.loadmat(dataRef6); Iref6=ImageRead6['y0']
    ImageRead7=sio.loadmat(dataRef7); Iref7=ImageRead7['y0']
    ImageRead8=sio.loadmat(dataRef8); Iref8=ImageRead8['y0']
    ImageRead9=sio.loadmat(dataRef9); Iref9=ImageRead9['y0']
    ImageRead10=sio.loadmat(dataRef10); Iref10=ImageRead10['y0']
    ImageRead11=sio.loadmat(dataRef11); Iref11=ImageRead11['y0']
    ImageRead12=sio.loadmat(dataRef12); Iref12=ImageRead12['y0']
    ImageRead13=sio.loadmat(dataRef13); Iref13=ImageRead13['y0']
    ImageRead14=sio.loadmat(dataRef14); Iref14=ImageRead14['y0']
    ImageRead15=sio.loadmat(dataRef15); Iref15=ImageRead15['y0']
    ImageRead16=sio.loadmat(dataRef16); Iref16=ImageRead16['y0']
    ImageRead17=sio.loadmat(dataRef17); Iref17=ImageRead17['y0']
    ImageRead18=sio.loadmat(dataRef18); Iref18=ImageRead18['y0']
    
    
    
    Iref = [Iref1,Iref2,Iref3,Iref4,Iref5,Iref6,Iref7,Iref8,Iref9, Iref10,Iref11,Iref12,Iref13,Iref14,Iref15,Iref16,Iref17,Iref18]
    
   
   
    
    CD = []
    for k in range(len(Iref)):
        IM=[]
        ths = 4
        for i in range(len(Itest)):
            sample = hstack((Itest[i], Iref[k][i]))
            
      


            ecdf = ECDF(sample)   # cdf computation
    
            ecdf.y[ecdf.y==1]=0.99999  # sigficance level (assuming Gaussian distribution)
            ecdf.y[ecdf.y==0]=0.00001
            Q = norm.ppf(ecdf.y)
            RES=[]; cnt=0
            for j in range(len(Itest[0])):
                if (Q[j] <= -ths) or (Q[j] >= ths):
                    res=0
                else:
                    if Itest[i][j] > 2*Iref[k][i][j]:
                        res=Itest[i][j]         # if is out of gaaussian iid = change
                        cnt += 1
                    else:
                        res = 0
                RES.append(res)
            IM.append(RES)
        CD.append(IM)
        
    
    Corr = [abs(np.corrcoef(np.array(Itest).flatten(),np.array(CD[0]).flatten())[0][1]), abs(np.corrcoef(np.array(Itest).flatten(),np.array(CD[1]).flatten())[0][1]), 
                abs(np.corrcoef(np.array(Itest).flatten(),np.array(CD[2]).flatten())[0][1]), abs(np.corrcoef(np.array(Itest).flatten(),np.array(CD[3]).flatten())[0][1]),
                abs(np.corrcoef(np.array(Itest).flatten(),np.array(CD[4]).flatten())[0][1]), abs(np.corrcoef(np.array(Itest).flatten(),np.array(CD[5]).flatten())[0][1]),
                abs(np.corrcoef(np.array(Itest).flatten(),np.array(CD[6]).flatten())[0][1]), abs(np.corrcoef(np.array(Itest).flatten(),np.array(CD[7]).flatten())[0][1]),
                abs(np.corrcoef(np.array(Itest).flatten(),np.array(CD[8]).flatten())[0][1]), abs(np.corrcoef(np.array(Itest).flatten(),np.array(CD[9]).flatten())[0][1]),
                abs(np.corrcoef(np.array(Itest).flatten(),np.array(CD[10]).flatten())[0][1]), abs(np.corrcoef(np.array(Itest).flatten(),np.array(CD[11]).flatten())[0][1]),
                abs(np.corrcoef(np.array(Itest).flatten(),np.array(CD[12]).flatten())[0][1]), abs(np.corrcoef(np.array(Itest).flatten(),np.array(CD[13]).flatten())[0][1]),
                abs(np.corrcoef(np.array(Itest).flatten(),np.array(CD[14]).flatten())[0][1]), abs(np.corrcoef(np.array(Itest).flatten(),np.array(CD[15]).flatten())[0][1]),
                abs(np.corrcoef(np.array(Itest).flatten(),np.array(CD[16]).flatten())[0][1]), abs(np.corrcoef(np.array(Itest).flatten(),np.array(CD[17]).flatten())[0][1])]
        
        
        #Idx = sorted(range(len(Corr)), key=lambda sub: Corr[sub])[:12]
        #Idx2 = sorted(range(len(Corr)), key=lambda sub: Corr[sub])[-3:]
    Idx2 = sorted(range(len(Corr)), key=lambda sub: Corr[sub])[:4]
    
    
    CDmc = []#; CDmc2 = [] 
    for i in range(len(Itest)):
        IM=[]
        for j in range(len(Itest[0])):
            #c = CD[Idx2[0]][i][j]+CD[Idx2[1]][i][j]+CD[Idx2[2]][i][j]+CD[Idx2[3]][i][j]+CD[Idx2[4]][i][j]+CD[Idx2[5]][i][j]+CD[Idx2[6]][i][j]+CD[Idx2[7]][i][j]
            c = CD[Idx2[0]][i][j]+CD[Idx2[1]][i][j]+CD[Idx2[2]][i][j]+CD[Idx2[3]][i][j]
            
            res2 = c / 4
            IM.append(res2)
        CDmc.append(IM)

           
    ICD = np.array(CDmc)
    
    # print(Iref[0].shape)
    # print(Itest.shape)
    
    # print(ICD.shape)
    
  
    
    
    
    
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
    
    
    
  


def MC(test_type, N, pair):
     
    folder ='%sn%dpair%d/'%(test_type,N, pair)
    path = '/home/marcello-costa/workspace/2DAR1/Input/DataMC/' 
    path1 = '/home/marcello-costa/workspace/2DAR1/Output/DataMC/' 
    
    input = path+folder
    output = path1+folder
    os.mkdir(output)
       
    files = os.listdir(input)
    files_images = [i for i in files if i.endswith('.mat')]
   
    dataset0 = []
    dataset1 = []; dataset2 = []; dataset3 = []; dataset4 = []; dataset5 = []; dataset6 = []
    dataset7 = []; dataset8 = []; dataset9 = []; dataset10 = []; dataset11 = []; dataset12 = []
    dataset13 = []; dataset14 = []; dataset15 = []; dataset16 = []; dataset17 = []; dataset18 = []
    
    for i in range(len(files_images)):
        if files_images[i].find('T') != -1:
            dataset0.append(files_images[i])
            
        elif files_images[i].find('A') != -1:
            dataset1.append(files_images[i])
        elif files_images[i].find('B') != -1:
            dataset2.append(files_images[i])
        elif files_images[i].find('C') != -1:
            dataset3.append(files_images[i])
        elif files_images[i].find('D') != -1:
            dataset4.append(files_images[i])
        elif files_images[i].find('E') != -1:
            dataset5.append(files_images[i])
        elif files_images[i].find('F') != -1:
            dataset6.append(files_images[i])
        elif files_images[i].find('G') != -1:
            dataset7.append(files_images[i])
        elif files_images[i].find('H') != -1:
            dataset8.append(files_images[i])
        elif files_images[i].find('I') != -1:
            dataset9.append(files_images[i])
        elif files_images[i].find('J') != -1:
            dataset10.append(files_images[i])
        elif files_images[i].find('K') != -1:
            dataset11.append(files_images[i])
        elif files_images[i].find('L') != -1:
            dataset12.append(files_images[i])
        elif files_images[i].find('M') != -1:
            dataset13.append(files_images[i])
        elif files_images[i].find('N') != -1:
            dataset14.append(files_images[i])
        elif files_images[i].find('O') != -1:
            dataset15.append(files_images[i])
        elif files_images[i].find('P') != -1:
            dataset16.append(files_images[i])
        elif files_images[i].find('Q') != -1:
            dataset17.append(files_images[i])
        elif files_images[i].find('R') != -1:
            dataset18.append(files_images[i])
            
    dataset0 = sorted(dataset0, key=lambda s: int(re.search(r'\d+', s).group()))
    
    dataset1 = sorted(dataset1, key=lambda s: int(re.search(r'\d+', s).group()))
    dataset2 = sorted(dataset2, key=lambda s: int(re.search(r'\d+', s).group()))
    dataset3 = sorted(dataset3, key=lambda s: int(re.search(r'\d+', s).group()))
    dataset4 = sorted(dataset4, key=lambda s: int(re.search(r'\d+', s).group()))
    dataset5 = sorted(dataset5, key=lambda s: int(re.search(r'\d+', s).group()))
    dataset6 = sorted(dataset6, key=lambda s: int(re.search(r'\d+', s).group()))
    dataset7 = sorted(dataset7, key=lambda s: int(re.search(r'\d+', s).group()))
    dataset8 = sorted(dataset8, key=lambda s: int(re.search(r'\d+', s).group()))
    dataset9 = sorted(dataset9, key=lambda s: int(re.search(r'\d+', s).group()))
    dataset10 = sorted(dataset10, key=lambda s: int(re.search(r'\d+', s).group()))
    dataset11 = sorted(dataset11, key=lambda s: int(re.search(r'\d+', s).group()))
    dataset12 = sorted(dataset12, key=lambda s: int(re.search(r'\d+', s).group()))
    dataset13 = sorted(dataset13, key=lambda s: int(re.search(r'\d+', s).group()))
    dataset14 = sorted(dataset14, key=lambda s: int(re.search(r'\d+', s).group()))
    dataset15 = sorted(dataset15, key=lambda s: int(re.search(r'\d+', s).group()))
    dataset16 = sorted(dataset16, key=lambda s: int(re.search(r'\d+', s).group()))
    dataset17 = sorted(dataset17, key=lambda s: int(re.search(r'\d+', s).group()))
    dataset18 = sorted(dataset18, key=lambda s: int(re.search(r'\d+', s).group()))
    
    
    
    
    step = 1
    com = 't:'  # script code in R
    im1 = []; im2 = []
    for j in range(0,len(dataset1),step):
        a=[]; b=[]; c=[]; d=[]
        for i in range(step):
            a.append(dataset0[i+j]+';'+dataset1[i+j]+';'+dataset2[i+j]+';'+dataset3[i+j]+';'+dataset4[i+j]+';'+dataset5[i+j]+';'+dataset6[i+j]
                     +';'+dataset7[i+j]+';'+dataset8[i+j]+';'+dataset9[i+j]+';'+dataset10[i+j]+';'+dataset11[i+j]+';'+dataset12[i+j]+';'+dataset13[i+j]
                     +';'+dataset14[i+j]+';'+dataset15[i+j]+';'+dataset16[i+j]+';'+dataset17[i+j]+';'+dataset18[i+j]
                     +';'+ re.findall(r"[-+]?(?:\d*\.\d+|\d+)", dataset1[i+j])[-1]+';'+input+';'+output)
        im1.append(a)
    
    # print(im1[0])
    # CNT=[]
    
    # for i in range(len(im1)):
    #     Cnt = Rscript(im1[i])
    #     CNT.append(Cnt)
    # END = time.time()

    # time_AR1 = END - START
    
    # cnt = np.sum    ### you have to apply agauin the trial test here!!!
    
    
    
    
    
    #------------------------ Parallel Processing MASTER -------------------------#
    START = time.time()
    cnt = 0
    for i in range(len(im1)):
        
        set_start_method('fork', force=True)
        try:
            pool = Pool(12) # the number of cores has been greatar than 8
            nova = pool.starmap(Rscript, zip(im1[i])) 
            cnt =+ cnt+1

        except Exception as e:
            print('Main Pool Error: ', e)
        except KeyboardInterrupt:
            exit()
        finally:
            pool.terminate()
            pool.close() 
            pool.join() 

    END_IM = time.time()
    time_AR1 = (END_IM - START)
    
    
    
    
    
    return time_AR1, cnt
    
  
    
    
# if __name__ == "__main__":
    
#     test_type = ['GLRT', 'AR', 'MC']
#     test_type  = test_type[2]
    
#     nroPairs = 1
#     s = 0
#     dimH = 3000
#     dimW = 2000
#     K = 10
#     N = 100
    
#     MC(test_type, N, s)
    
    


  
     
 
 
 
 
 
