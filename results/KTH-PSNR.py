from cProfile import label
import numpy as np
import pandas as pd
import csv
import re
import os
import scipy.stats as sps
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.io import savemat, loadmat
import datetime
from collections import defaultdict
import scienceplots
from ast import literal_eval
# import plottools



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

def findMiddle(list):
  l = len(list)
  if l/2:
    return (list[l/2-1]+list[l/2])/2.0
  else:
    return list[(l/2-1)/2]
            

if __name__ == "__main__":


    mainDir='/home/marcello-costa/workspace/MimirDATA/RESS/'

    files = os.listdir(mainDir)
    files_logs = [i for i in files if i.endswith('.csv')]
    
    dataset = pd.read_csv(mainDir+files_logs[0])
    
     
            
    # # #------------------------ Graphic Analysis -------------------------#
 
    DPI = 250
    plt.rc('text', usetex=True)
    DPI = 250


    SMALL_SIZE = 10
    MEDIUM_SIZE = 16
    BIGGER_SIZE = 12

    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    

    
 
    with plt.style.context(['science', 'ieee', 'std-colors']):
            
            fig, ax1 = plt.subplots(figsize=(4,2), dpi=DPI) #, constrained_layout=True)
            ax1.set_prop_cycle(color=['darkred', 'indianred', 'indigo'],marker = ['o','*', 'o'], alpha=[0.95, 0.4, 0.2]) #linewidth=2
            for i in range(0, len(dataset)):
                
                df = dataset[['test_type', 'psnr', 'cr', 'nb']]
                
                
                Psnr = literal_eval(df.psnr[i])
                Cr = literal_eval(df.cr[i])
                test= literal_eval(df.test_type[i])
                
          
                df.tail()
                ax1.plot(Cr,Psnr, linewidth=2, label=list(set(test))[0])

            #ax1.set(**pparam2)
            ax1.legend(prop={'size': 12})
            ax1.set_ylabel(r'PSNR [dB]', fontsize=12)  
            ax1.set_xlabel(r'CR', fontsize=12)  
            ax1.xaxis.set_label_coords(.5, -.5)
            ax1.yaxis.set_label_coords(-0.09, 0.5)
                
       
            



            ax1.legend(loc='lower center', bbox_to_anchor=(0.5, 1), fancybox=True, shadow=True, ncol=3)
            
          
    #fig.subplots_adjust(bottom=0.2)
    path='/home/marcello-costa/workspace/'
 
    fig.savefig(path + 'kthPSNR.png', dpi=DPI)
    fig.tight_layout()
    plt.show()      
