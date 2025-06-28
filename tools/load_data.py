import scipy.io as sio


def load_data(test_type):

    path = '/home/marcello-costa/workspace/2DAR1/data/'
    path2 = '/home/marcello-costa/workspace/2DAR1/data/alvos/'
    
    # images of mission by pair
    ImageRead1=sio.loadmat(path+'ImageRead1.mat')
    Im1=ImageRead1['ImageRead1']
    ImageRead2=sio.loadmat(path+'ImageRead2.mat')
    Im2=ImageRead2['ImageRead2']
    ImageRead3=sio.loadmat(path+'ImageRead3.mat')
    Im3=ImageRead3['ImageRead3']
    ImageRead4=sio.loadmat(path+'ImageRead4.mat')
    Im4=ImageRead4['ImageRead4']
    ImageRead5=sio.loadmat(path+'ImageRead5.mat')
    Im5=ImageRead5['ImageRead5']
    ImageRead6=sio.loadmat(path+'ImageRead6.mat')
    Im6=ImageRead6['ImageRead6']
    ImageRead7=sio.loadmat(path+'ImageRead7.mat')
    Im7=ImageRead7['ImageRead7']
    ImageRead8=sio.loadmat(path+'ImageRead8.mat')
    Im8=ImageRead8['ImageRead8']
    ImageRead9=sio.loadmat(path+'ImageRead9.mat')
    Im9=ImageRead9['ImageRead9']
    ImageRead10=sio.loadmat(path+'ImageRead10.mat')
    Im10=ImageRead10['ImageRead10']
    ImageRead11=sio.loadmat(path+'ImageRead11.mat')
    Im11=ImageRead11['ImageRead11']
    ImageRead12=sio.loadmat(path+'ImageRead12.mat')
    Im12=ImageRead12['ImageRead12']
    ImageRead13=sio.loadmat(path+'ImageRead13.mat')
    Im13=ImageRead13['ImageRead13']
    ImageRead14=sio.loadmat(path+'ImageRead14.mat')
    Im14=ImageRead14['ImageRead14']
    ImageRead15=sio.loadmat(path+'ImageRead15.mat')
    Im15=ImageRead15['ImageRead15']
    ImageRead16=sio.loadmat(path+'ImageRead16.mat')
    Im16=ImageRead16['ImageRead16']
    ImageRead17=sio.loadmat(path+'ImageRead17.mat')
    Im17=ImageRead17['ImageRead17']
    ImageRead18=sio.loadmat(path+'ImageRead18.mat')
    Im18=ImageRead18['ImageRead18']
    ImageRead19=sio.loadmat(path+'ImageRead19.mat')
    Im19=ImageRead19['ImageRead19']
    ImageRead20=sio.loadmat(path+'ImageRead20.mat')
    Im20=ImageRead20['ImageRead20']
    ImageRead21=sio.loadmat(path+'ImageRead21.mat')
    Im21=ImageRead21['ImageRead21']
    ImageRead22=sio.loadmat(path+'ImageRead22.mat')
    Im22=ImageRead22['ImageRead22']
    ImageRead23=sio.loadmat(path+'ImageRead23.mat')
    Im23=ImageRead23['ImageRead23']
    ImageRead24=sio.loadmat(path+'ImageRead24.mat')
    Im24=ImageRead24['ImageRead24']
    
    
    # tragets position by mission
    tp1=sio.loadmat(path2+'S1.mat')
    tp1=tp1['S1']
    tp2=sio.loadmat(path2+'K1.mat')
    tp2=tp2['K1']
    tp3=sio.loadmat(path2+'F1.mat')
    tp3=tp3['F1']
    tp4=sio.loadmat(path2+'AF1.mat')
    tp4=tp4['AF1']
    
    if test_type == 'AR' or  test_type == 'MC' or  test_type == 'GLRT' or test_type == 'weiARMA' or test_type == 'GLRTa':
        
          par=[[Im7,Im8,Im9,Im10, Im11, Im12, Im13, Im14, Im15, Im16, Im17, Im18, Im19, Im20, Im21, Im22, Im23, Im24, tp1,'S1', Im1, 1],
             [Im13, Im1,Im2,Im3,Im4, Im5, Im6, Im14, Im15, Im16, Im17, Im18, Im19, Im20, Im21, Im22, Im23, Im24, tp2,'K1', Im7, 2],
             [Im19, Im1,Im2,Im3,Im4, Im5, Im6, Im7, Im8, Im9, Im10, Im11, Im12, Im20, Im21, Im22, Im23, Im24, tp3,'F1', Im13, 3],
             [Im1,Im2,Im3,Im4, Im5, Im6, Im7, Im8, Im9, Im10, Im11, Im12, Im13, Im14, Im15, Im16, Im17, Im18, tp4,'AF1', Im19, 4],   
           
            [Im14, Im7,Im8,Im9,Im10, Im11, Im12, Im13,  Im15, Im16, Im17, Im18, Im19, Im20, Im21, Im22, Im23, Im24, tp1,'S1', Im2, 5],
            [Im20, Im1,Im2,Im3,Im4, Im5, Im6, Im13, Im14, Im15, Im16, Im17, Im18, Im19, Im21, Im22, Im23, Im24, tp2,'K1', Im8, 6],
            [Im2,Im1, Im3,Im4, Im5, Im6, Im7, Im8, Im9, Im10, Im11, Im12, Im19, Im20, Im21, Im22, Im23, Im24, tp3,'F1', Im14, 7],
            [Im8, Im1,Im2,Im3,Im4, Im5, Im6, Im7,  Im9, Im10, Im11, Im12, Im13, Im14, Im15, Im16, Im17, Im18, tp4,'AF1', Im20,8],
            
            
            [Im21, Im7,Im8,Im9,Im10, Im11, Im12, Im13, Im14, Im15, Im16, Im17, Im18, Im19, Im20,  Im22, Im23, Im24, tp1,'S1', Im3, 9],
            [Im3, Im1,Im2,Im4, Im5, Im6, Im13, Im14, Im15, Im16, Im17, Im18, Im19, Im20, Im21, Im22, Im23, Im24, tp2,'K1', Im9, 10],
            [Im9, Im1,Im2,Im3,Im4, Im5, Im6, Im7, Im8,  Im10, Im11, Im12, Im19, Im20, Im21, Im22, Im23, Im24, tp3,'F1', Im15,11],
            [Im15, Im1,Im2,Im3,Im4, Im5, Im6, Im7, Im8, Im9, Im10, Im11, Im12, Im13, Im14,  Im16, Im17, Im18, tp4,'AF1', Im21,12],
            
            
            [Im10, Im7,Im8,Im9, Im11, Im12, Im13, Im14, Im15, Im16, Im17, Im18, Im19, Im20, Im21, Im22, Im23, Im24, tp1,'S1', Im4, 13],
            [Im16, Im1,Im2,Im3,Im4, Im5, Im6, Im13, Im14, Im15,  Im17, Im18, Im19, Im20, Im21, Im22, Im23, Im24, tp2,'K1', Im10, 14],
            [Im22, Im1,Im2,Im3,Im4, Im5, Im6, Im7, Im8, Im9, Im10, Im11, Im12, Im19, Im20, Im21,  Im23, Im24, tp3,'F1', Im16, 15],
            [Im4, Im1,Im2,Im3, Im5, Im6, Im7, Im8, Im9, Im10, Im11, Im12, Im13, Im14, Im15, Im16, Im17, Im18, tp4,'AF1', Im22, 16],
            
                      
            [Im17, Im7,Im8,Im9,Im10, Im11, Im12, Im13, Im14, Im15, Im16,  Im18, Im19, Im20, Im21, Im22, Im23, Im24, tp1,'S1', Im5, 17],
            [Im23, Im1,Im2,Im3,Im4, Im5, Im6, Im13, Im14, Im15, Im16, Im17, Im18, Im19, Im20, Im21, Im22,  Im24, tp2,'K1', Im11, 18],
            [Im5, Im1,Im2,Im3,Im4,  Im6, Im7, Im8, Im9, Im10, Im11, Im12, Im19, Im20, Im21, Im22, Im23, Im24, tp3,'F1', Im17, 19],
            [Im11, Im1,Im2,Im3,Im4, Im5, Im6, Im7, Im8, Im9, Im10,  Im12, Im13, Im14, Im15, Im16, Im17, Im18, tp4,'AF1', Im23, 20],
            
            
            
            [Im24, Im7,Im8,Im9,Im10, Im11, Im12, Im13, Im14, Im15, Im16, Im17, Im18, Im19, Im20, Im21, Im22, Im23, tp1,'S1', Im6, 21],
            [Im6, Im1,Im2,Im3,Im4, Im5,  Im13, Im14, Im15, Im16, Im17, Im18, Im19, Im20, Im21, Im22, Im23, Im24, tp2,'K1', Im12, 22],
            [Im12, Im1,Im2,Im3,Im4, Im5, Im6, Im7, Im8, Im9, Im10, Im11,  Im19, Im20, Im21, Im22, Im23, Im24, tp3,'F1', Im18, 23],
            [Im18, Im1,Im2,Im3,Im4, Im5, Im6, Im7, Im8, Im9, Im10, Im11, Im12, Im13, Im14, Im15, Im16, Im17,  tp4,'AF1', Im24, 24]]
    

        # par=[[Im1,Im7,tp1,'S1'],[Im7,Im13,tp2,'K1'],[Im13,Im19,tp3,'F1'],[Im19,Im1,tp4,'AF1'],
        #     [Im2,Im14,tp1,'S1'],[Im8,Im20,tp2,'K1'],[Im14,Im2,tp3,'F1'],[Im20,Im8,tp4,'AF1'],
        #     [Im3,Im21,tp1,'S1'],[Im9,Im3,tp2,'K1'],[Im15,Im9,tp3,'F1'],[Im21,Im15,tp4,'AF1'],
        #     [Im4,Im10,tp1,'S1'],[Im10,Im16,tp2,'K1'],[Im16,Im22,tp3,'F1'],[Im22,Im4,tp4,'AF1'],
        #     [Im5,Im17,tp1,'S1'],[Im11,Im23,tp2,'K1'],[Im17,Im5,tp3,'F1'],[Im23,Im11,tp4,'AF1'],
        #     [Im6,Im24,tp1,'S1'],[Im12,Im6,tp2,'K1'],[Im18,Im12,tp3,'F1'],[Im24,Im18,tp4,'AF1']]
        
    else:
        
        print('Invalid test')
    
        
        
            
    
        
    return par
    

    
    
