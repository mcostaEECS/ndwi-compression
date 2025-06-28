from __future__ import division
import numpy as np
from matplotlib import pyplot as plt
import cv2
import os
import scipy.io as sio
from skimage.morphology import disk
from skimage.filters.rank import median
from scipy.spatial.distance import cdist
from scipy.special import erfcinv
import re
from matplotlib import pyplot as plt
import array
import scipy.io
import scipy.io as io
import math
from ast import literal_eval



def convert(img, target_type_min, target_type_max, target_type):
    imin = img.min()
    imax = img.max()

    a = (target_type_max - target_type_min) / (imax - imin)
    b = target_type_max - a * imax
    new_img = (a * img + b).astype(target_type)
    return new_img


def subarrays(arr, nrows, ncols):
    h, w = arr.shape
    assert h % nrows == 0, f"{h} rows is not evenly divisible by {nrows}"
    assert w % ncols == 0, f"{w} cols is not evenly divisible by {ncols}"
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))
    
    
def compress(d,m,N,th):
  

  
  vec= np.array(m).flatten()
  vec=vec.tolist()
  
  pct=(vec.count(1)/len(vec))
  
 
  
  cnt=0
  if pct >= th:
    im = convert(d, 0,255, np.uint8)
    cnt += 1
  else:
    im = d
    
    
  
  return np.array(im), cnt
    
  
  



if __name__ == "__main__":
  
    path = '/home/marcello-costa/workspace/MimirDATA/COMPRESS/'

    data=sio.loadmat(path+'nir-w'+'.mat')  
    data=data['NIR']
    
    mask=sio.loadmat(path+'mask-w'+'.mat')  
    mask=mask['MASK']
    
    print(mask.shape)
    print(data.shape)
    
    N=10
    W=500
    
    m = subarrays(mask, N, N)
    d = subarrays(data, N, N)
    TH = 0.75
    
    CD=[]; 
    for i in range(len(d)):
        [OUT0, cnt]=compress(d[i], m[i], N, TH)
        CD.append(OUT0)
        
    #print(CD)
        
    
    ICD=[]
    im = []
    for j in range(0,len(CD),int(W/N)):
        a=[]
        for l in range(int(W/N)):
            a.append(CD[j+l])
        im.append(np.hstack((a)))
    ICD = np.vstack((im))
     
     
    plt.imshow(data, cmap = plt.cm.gray)
    plt.axis("off")
    plt.show()
    
    # name_file = path+'nirComp'+'.mat'  
    # scipy.io.savemat(name_file, {'COMP':ICD})
        
         
    #gdal_translate -of GTiff -co COMPRESS=LZW -co BIGTIFF=YES input.tif output.tif.
    
# 

    
    
    
    # print(ICD1.shape)
      
    
    # fig = plt.figure('test')
    # ax = fig.add_subplot()
    # plt.imshow(ICD1, cmap = plt.cm.gray)
    # plt.axis("off")

    
    
   
     


    # plt.show() 
      
        
 
