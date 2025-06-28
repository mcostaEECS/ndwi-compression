from __future__ import division
import numpy as np
import time
from matplotlib import pyplot as plt
import array
import scipy.io
import scipy.io as io
from GLRT_CFAR import GLRT
from LMP import LMP
from load_data import load_data
import pyprind
import time
import psutil
import datetime
import re


def Dataset_Test(N,K, test_type):



    nroPairs = 1
    dimH = 3000
    dimW = 2000
    
    GLRT_time=[]
    CFAR_time = []
    NT=[]
    campaign = 'Campaign_%s_N_%d_K_%d'%(test_type, N, K)

    path = '/home/marcello-costa/workspace/2DAR1/runtime/'
    nowTime = datetime.datetime.now()
    date_time = nowTime.strftime("%d_%m_%Y_%H_%M_%S.%f")[:-10]

    nameFile = 'Campaign_%s_N_%d_K_%d_date_%s'%(test_type, N, K, date_time) 
    id = path+nameFile+'.txt'

    

    bar = pyprind.ProgBar(nroPairs, monitor=True, title=campaign)
    for s in range(nroPairs):  
            par=load_data()[s]
            Itest=par[0][0:dimH,0:dimW]
            Iref=par[1][0:dimH,0:dimW]
            tp=par[2]
            tp =  np.fliplr(tp)
            TP =par[3]

            with open(id, 'a') as f:
                if test_type == 'LMP':                
                    [glrt_time, cfar_time, Nt] = LMP(Itest, Iref, TP, N, K, s)
                    f.write(str(test_type)+';'+'N_'+str(N)+';'+'K_'+str(K)+';'+'pair_'+str(s)+';'+'HT_runtime_'+str(glrt_time)+';'+'cfar_runtime_'+str(cfar_time)+';'+'Threshold_op_'+str(Nt)+'\n')
                elif test_type == 'GLRT':
                    [glrt_time, cfar_time, Nt] = GLRT(Itest, Iref, TP, N, K, s)
                    f.write(str(test_type)+';'+'N_'+str(N)+';'+'K_'+str(K)+';'+';'+'pair_'+str(s)+';'+'HT_runtime_'+str(glrt_time)+';'+'cfar_runtime_'+str(cfar_time)+';'+'Threshold_op_'+str(Nt)+'\n')

                GLRT_time.append(glrt_time)
                CFAR_time.append(cfar_time)
                NT.append(Nt)
                

            bar.update()
    
    GLRT_runtime= np.sum(GLRT_time)
    CFAR_runtime = np.sum(CFAR_time)
    NT_total = np.sum(NT)

    return GLRT_runtime, CFAR_runtime, NT_total
    


if __name__ == "__main__":

    test_type = ['GLRT', 'LMP']
    test_type  = test_type[0]

    k_range = [10]
    N_range = [100]

    for w in range(len(N_range)):
                [GLRT_runtime, CFAR_runtime, NT_total] = Dataset_Test(N_range[w],k_range[w], test_type)
      
