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
import sys
from numpy import float16 as NPfloat16
from numpy import asarray as NPasarray
import rasterio
from math import log10
from numpy import *
import math
import array
from PIL import Image
import scipy.io
import os
import numpy
from numpy.random import randn, seed, random
from numpy import unpackbits, transpose, packbits, mean, newaxis, append, sqrt

# Set the compression environment to LZ77
#arcpy.env.compression = "LZ77"

# Set the compression environment to JPEG with a quality of 80
#arcpy.env.compression = "JPEG 80"

def psnr2(dataA,dataB):
  dataA, dataB = 1.*ravel(dataA), 1.*ravel(dataB)
  mse = sum((dataA-dataB)**2)/product(dataA.shape[:2])
  res= 10*log10((255**2)/mse)
  
  return res


def psnr1(firstImage, secondImage):
   # Compute the difference between corresponding pixels
   diff = np.subtract(firstImage, secondImage)
   # Get the square of the difference
   squared_diff = np.square(diff)

   # Compute the mean squared error
   mse = np.mean(squared_diff)

   # Compute the PSNR
   max_pixel = 1
   psnr = 20 * np.log10(max_pixel) - 10 * np.log10(mse)
    
   return psnr
    
    
    
    # enc = open('out/encoded.bpe','rb')
    # SizeEnc=os.stat('out/encoded.bpe').st_size
    # databits3= packbits(databits, axis=0)
    # data4 = sc.reshape(databits3,(SizeEnc))
    # rec = open('out/rec.bpe','wb')
    # rec.write(data4)
    # rec.close()
    
       
    # #CCSDS decoder
    # cmdstring2 = 'wine CCSDS_C.exe out/rec.bpe out/decoded.raw 1 %s %s 256 0 0 3' % (W, H)
    # os.system(cmdstring2)
    # img2 = open('out/decoded.raw','rb')
    # bin = array.array('B')
    # bin.read(img2,W*H)
    # data3 = sc.array(bin,dtype=sc.uint8)
    # data2 = sc.reshape(data3,(W,H))
    # img2.close()
    # imRec = Image.fromarray(data2[:,])
    # #imRec.show()
    
    # # PSNR computation    
    # dataA, data3 = 1.*ravel(dataA), 1.*ravel(data3)
    # mse = sum((dataA-data3)**2)/product(dataA.shape[:2])
    # psnr= 10*log10((255**2)/mse)
    
    # return psnr



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
  #print(pct, th)
  
  B=[]; C=[]
  if pct >= th:
    #im = convert(d, 0,255, np.uint8)
    im = d #NPasarray(d, dtype=NPfloat16) #np.float16(d)
    B=im
    cnt=0
  else:
    
    #im = convert(d, 0,255, np.uint8)
    im = NPasarray(d, dtype=NPfloat16) #np.float16(d)
    C=im
    
    cnt=1
   
   

  
  return np.array(im), np.array(B), np.array(C), cnt
    
  



if __name__ == "__main__":
  
    path = '/home/marcello-costa/workspace/MimirDATA/COMPRESS/'
    

    

    data=sio.loadmat(path+'nir-w'+'.mat')  
    data=data['NIR']
    
    mask=sio.loadmat(path+'mask-w'+'.mat')  
    mask=mask['MASK']
    
    #print(np.max(data))
    
    # print(mask.shape)
    # print(data.shape)
    
    N=10
    W=500
    
    m = subarrays(mask, N, N)
    d = subarrays(data, N, N)
    
    blocks=len(d)
    print('blocks:', len(d))
    
    TH = 0.75  #> more compress
    
    CD=[];  Idx=[]; b=[]; c=[]
    for i in range(len(d)):
        [OUT0, B, C, cnt]=compress(d[i], m[i], N, TH)
        CD.append(OUT0); b.append(B); c.append(C); Idx.append(cnt)
        
    nb=Idx.count(1)
    print('nb:',nb)
    
    cr=nb/blocks
    print('cr:', cr)
    
    
    
    # approach for testing analysis of
    
    # print(len(b))
    # print(len(data))
    
    # uncomp = sys.getsizeof(b)
    # print('sizeDataUncomp:',uncomp)
    
    # comp = sys.getsizeof(c)
    # print('sizeDataComp:',comp/2)
    
    #cr = (sys.getsizeof(data)-sys.getsizeof(b))/sys.getsizeof(data)
        
   
    
    
    
    
    

        
    # Write to TIFF
    # kwargs = data1[0].meta
    # kwargs.update(
    #     dtype=rasterio.float32,
    #     count=1,
    #     compress='lzw')

    # with rasterio.open(os.path.join(path, 'ndvi2.tif'), 'w', **kwargs) as dst:
    #     dst.write_band(4, b.astype(rasterio.float32))

        
        
    
    ICD=[]
    im = []
    for j in range(0,len(CD),int(W/N)):
        a=[]
        for l in range(int(W/N)):
            a.append(CD[j+l])
        im.append(np.hstack((a)))
    ICD = np.vstack((im))
    
    # a = sys.getsizeof(ICD)
    # print('sizeICD:',a)
    # print(ICD.nbytes)
   
    # b = sys.getsizeof(data)
    # print('sizeData:',b)
    # print(data.nbytes)
    
    
    
    # print(data.shape)
    # print(ICD.shape)
#     print(b)


    res = psnr1(data,ICD)
    print('psnr:',res)
    
    # res2 = calculate_psnr(data,ICD)
    # print('psnr2:',res2)
    
    
    
     
     
    plt.imshow(ICD, cmap = plt.cm.gray)
    plt.axis("off")
    plt.show()
    
#     # name_file = path+'nirComp'+'.mat'  
#     # scipy.io.savemat(name_file, {'COMP':ICD})
        
         

    
# 

    
    
    
    # print(ICD1.shape)
      
    
    # fig = plt.figure('test')
    # ax = fig.add_subplot()
    # plt.imshow(ICD1, cmap = plt.cm.gray)
    # plt.axis("off")

    
    
   
     


    # plt.show() 
      
        
 
