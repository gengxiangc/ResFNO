# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 11:11:25 2021

@author: cgx@nuaa.edu.cn

Implement of 

'[1] Residual Fourier Neural Operators for Composite Curing Modelling '

"""

import torch 
import numpy as np
import scipy.io as sio
from utilsResFNO import ResFNO, FNO1d, RangeNormalizer,Predict
import matplotlib.pyplot as plt

if __name__ == '__main__':
    
    '''
    load data of case1 [1]:
    dataT: temperature history(K) from FEM, 200*51*223
    dataA: Cure of degree of composites, 200*51*223
    
    200: number of samples
    51: x=0 ~ x=50mm
    223: t=0min to t= 222min
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data  = sio.loadmat('data/Designed_2hold_200.mat') 
    dataA = data['dataA']  # x=0 ~ x=20 is Inval tool, cure of degree = 0
    dataT = data['dataT'] 
    dataTair = data['dataTair']
    
    seed = 1
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    SAVE_MODELS = 1 # if save trained model
    SAVE_LOSS = 1   # is save training loss and test loss
    task = 'T'      # task: temperature or degree of cure
    dic = 'logs/' 
    ntrain = 50     # size of training data
    TRAIN = 1

    ''' 
    Note:  
    ResFNO can be generalized to 2D field, more quick, but more complex.
    Here is the simple iteration of 1d case. 
    x=0mm to x=51mm, 51 sub-models
    
    '''
    if not TRAIN:
        T_index = 9
        x_index = 35
        model = FNO1d(16, 64, task).to(device) 
        model.load_state_dict(torch.load(dic+'ResFNO_T_X35_net_params.pkl'))
        model.eval()
        x_input, T_Pre, T_Real = Predict(model, dataTair, dataT, x_index, T_index)

        ERROR = np.max(np.abs(T_Real- T_Pre))
        nx = dataT.shape[1]
        nt = dataT.shape[2]
        tn = 223
        T_list = np.linspace(0, tn, nt)
        plt.figure(figsize=(6,4))   
        plt.text(160, 490, r'${\Delta}T_{max}$ ='+str(np.round(ERROR, 2)) + ' K', 
                 fontsize=12 ) 
        plt.plot(T_list, x_input, 'k-', label='T of Air')        
        plt.plot(T_list, T_Real, 'r-', label='T simulated')
        plt.plot(T_list, T_Pre, 'b-', label='T ResFNO')
        plt.title('(b) Temperature history at $x=35mm$', fontsize=12)
        plt.ylabel('Temperature (K)', fontsize=12)
        plt.ylim([270, 520])
        plt.legend(fontsize=12, loc='upper left')
        # plt.grid()
        plt.show()
        
        
    if TRAIN: # Training
    # for i in range(51):
        for i in [35]:

            x_index = i 
    
            model_output, loss_dict, out_data_dict  = ResFNO(dataT, dataA, dataTair,
                                                             task    = task, 
                                                             x_index = x_index, 
                                                             ntrain  = ntrain)
            
            Title = str(task)+ '_X' + str(x_index)  

            if SAVE_MODELS:
                torch.save(model_output, dic+'ResFNO_' + str(Title) + '_net_params.pkl')
            
            if SAVE_LOSS:
                sio.savemat(dic+'ResFNO_' + str(Title) + '_loss_' + str(ntrain) + '.mat', mdict=loss_dict)  
                                                                 
            sio.savemat(dic+'ResFNO_' + str(Title) + '_pre_' + str(ntrain) + '.mat', mdict=out_data_dict)

            
            
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
