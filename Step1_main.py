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
from utilsResFNO import ResFNO

if __name__ == '__main__':
    
    '''
    load data of case1 [1]:
    dataT: temperature history(K) from FEM, 200*51*223
    dataA: Cure of degree of composites, 200*51*223
    
    200: number of samples
    51: x=0 ~ x=50mm
    223: t=0min to t= 222min
    '''
    data  = sio.loadmat('data/Case1.mat') 
    dataA = data['dataA']  # x=0 ~ x=20 is Inval tool, cure of degree = 0
    dataT = data['dataT'] 
    dataTair = data['dataTair']
    
    seed = 1
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    SAVE_MODELS = 0 # if save trained model
    SAVE_LOSS = 1   # is save training loss and test loss
    task = 'T'      # task: temperature or degree of cure
    dic = 'logs/' 
    ntrain = 50     # size of training data

    ''' 
    Note:  
    ResFNO can be generalized to 2D field, more quick, but more complex.
    Here is the simple iteration of 1d case. 
    x=0mm to x=51mm, 51 sub-models
    '''
    
    for i in range(51):

    
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
                                                             
        sio.savemat(dic+'ResFNO_' + str(Title) + '_pre_' + str(ntrain) + '.mat',mdict=out_data_dict)
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
