# -*- coding: utf-8 -*-
"""
Created on Wed May 18 12:48:48 2022

@author: GengxiangCHEN
"""

import mph
import numpy as np
import matplotlib.pyplot as plt
import time
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'
T1 = time.time()

# Print model
def Comsol_FEM(T_list, Ta, client):

    model = client.load('better2.mph')
    cure_cycle  = np.vstack((T_list,  Ta)).T
    np.savetxt('cure_cycle.csv', cure_cycle, delimiter=',')
    if 1:

        T2 = time.time()
    
        model.build()
        model.mesh()
    
        print( '--- Solving Model ---')
        model.solve('研究 1')
        # model.clear()
        T3 = time.time()
        
        client.remove(model)
        client.clear()
        
    
        print('加载时间:%.2f秒' % ((T2 - T1)))
        print('仿真时间:%.2f秒' % ((T3 - T2)))
    
    # Read Results
    fin = open('output.txt')
    b = fin.readlines()   
    a = b[5:]
            
    t_list = []
    T_air  = []
    T_FEM  = []
    
    for j in range(len(a)): 
    
        temp = a[j].split()     
        t_list.append(float(temp[0]))  
        T_FEM.append(float(temp[1]))
    
    t_list = np.array(t_list)
    T_FEM  = np.array(T_FEM)
    
    return T_FEM
# # thermal_lag = np.max(Ta[0:int(dt1*1)]-T[0:int(dt1*1)])
# # exotherm = np.max(T[int(dt1*1):int((dt1+dt2)*1)]-Ta[int(dt1*1):int((dt1+dt2)*1)])

# # Output[j] = np.array([thermal_lag,exotherm])
# # print('thermal_lag:',thermal_lag)
# # print('exotherm:',exotherm)


    # ## 画一条工艺曲线
    # if 0:
    #     plt.figure(figsize=(6,4))     
    #     plt.subplots_adjust(hspace=0.3)
    #     plt.plot(t_list, T_FEM, 'r-', label='T_FEM')
    #     plt.plot(t_list, T_air, 'g-', label='T_air')
    #     plt.title('Temperature', fontsize=12)
    #     # plt.ylim([0.4*L*1000,L*1000])
    #     plt.xlabel('Time (min)', fontsize=12)
    #     plt.ylabel('Temperature (K)', fontsize=12)
    #     plt.ylim([270, 530])
    #     plt.legend()
    #     plt.grid()
    #     plt.show()
