# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 11:11:25 2021

@author: cgx@nuaa.edu.cn

Implement of 

'Residual Fourier Neural Operators for Composite Curing Modelling '

"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import os
import matplotlib 
matplotlib.rcParams['backend'] = 'SVG'
matplotlib.rcParams['mathtext.fontset'] = 'cm' 
plt.rc('font', family='arial', size=12)
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# From 0 ~150

index = 23

data  = sio.loadmat('data/Case1.mat') 
dataT = data['dataT']
nx = dataT.shape[1]
nt = dataT.shape[2]
L = 0.05
discrete = 1
tn = 223
T_Pre  = np.ones((nx,nt))
T_Real = np.ones((nx,nt))
T_list = np.linspace(0, tn, nt)
dic = 'logs/'

for i in range(51):
    
    data = sio.loadmat(dic + 'ResFNO_T_X' + str(i) + '_pre_50.mat')
    pre_test  = data['pre_test'][:,:,0]
    x_test    = data['x_test'][:,:,0]
    x_train   = data['x_train'][:,:,0]
    y_test    = data['y_test']
    Title = '2d_T_Test_ID_' + str(index)
    T_Pre[i]  =  pre_test[index]
    T_Real[i] =  y_test[index]


###### Fig (a)  Temperature fileds simulated by FEM ############
x_position = 35
plt.figure(figsize=(12,5))   
plt.suptitle('Temperature history of training dataset ')
grid = plt.GridSpec(2, 5, wspace=0.35, hspace=0.35, left=0.07, right=0.95)
       
plt.subplot(grid[0,0:3])
T_grid, X_grid = np.meshgrid(np.linspace(0, 224, nt), np.linspace(0, L*1000, nx))
cs = plt.contourf(T_grid[:,::discrete], 
                  X_grid[:,::discrete], 
                  T_Real[:,::discrete], levels=200, origin='lower',cmap='rainbow')
cb = plt.colorbar(cs)
cb.set_ticks([300, 350, 400, 450, 500])
plt.title('(a) Temperature simulated by FEM', fontsize=12)
plt.plot([0,230],[x_position,x_position],ls="-",lw=3,c="white")
plt.xlim([0, 222])
plt.ylim([0, 50])
plt.yticks([0, 10, 20, 30, 40, 50])
plt.ylabel('$x$(mm)', fontsize=12)
plt.show()
################################################################


###### Fig (b) Temperature history at $x=35mm$ ############
plt.subplot(grid[0,3:5])
ERROR = np.max(np.abs(T_Real[x_position] - T_Pre[x_position]))
plt.text(160, 490, r'${\Delta}T_{max}$ ='+str(np.round(ERROR, 2)) + ' K', 
         fontsize=12 ) 
plt.plot(T_list, x_test[index], 'k-', label='T of Air')
    
plt.plot(T_list, T_Real[x_position], 'r-', label='T simulated')
plt.plot(T_list, T_Pre[x_position], 'b-', label='T ResFNO')
plt.title('(b) Temperature history at $x=35mm$', fontsize=12)
plt.ylabel('Temperature (K)', fontsize=12)
plt.ylim([270, 520])
plt.legend(fontsize=12, loc='upper left')
# plt.grid()
plt.show()
################################################################

###### Fig (c) Temperature predicted by ResFNO   ############ 
x_position = 21
plt.subplot(grid[1,0:3])
T_grid, X_grid = np.meshgrid(np.linspace(0, 224, nt), np.linspace(0, L*1000, nx))
cs = plt.contourf(T_grid[:,::discrete], 
                  X_grid[:,::discrete], 
                  T_Pre[:,::discrete], levels=200, origin='lower',cmap='rainbow')
cb = plt.colorbar(cs)
cb.set_ticks([300, 350, 400, 450, 500])
plt.title('(c) Temperature predicted by ResFNO  ', fontsize=12)
plt.plot([0,230],[x_position,x_position],ls="-",lw=3,c="white")
plt.xlabel('Time (min)', fontsize=12)
plt.ylabel('$x$(mm)', fontsize=12)
plt.xlim([0, 222])
plt.yticks([0, 10, 20, 30, 40, 50])
plt.show()
################################################################

###### Fig (d) Temperature history at  $x=21mm$  ############ 
plt.subplot(grid[1,3:5])
ERROR = np.max(np.abs(T_Real[x_position] - T_Pre[x_position]))
plt.text(160, 490, r'${\Delta}T_{max}$ ='+str(np.round(ERROR, 2)) + ' K', 
         fontsize=12 )
plt.plot(T_list, x_test[index], 'k-', label='T of Air')
plt.plot(T_list, T_Real[x_position], 'r-', label='T simulated')
plt.plot(T_list, T_Pre[x_position], 'b-', label='T ResFNO')
plt.title('(d) Temperature history at  $x=21mm$', fontsize=12)
plt.xlabel('Time (min)', fontsize=12)
plt.ylabel('Temperature (K)', fontsize=12)
plt.ylim([270, 520])
plt.legend(fontsize=12, loc='upper left')
plt.show()
################################################################


plt.savefig('Fig/Fig-'+str(Title)+'.svg',format='svg')
plt.savefig('Fig/Fig-'+str(Title)+'.pdf',format='pdf')