# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 11:11:25 2021

@author: cgx@nuaa.edu.cn

Implement of 

'[1] Residual Fourier Neural Operators for Composite Curing Modelling '

"""

from datetime import datetime
import scipy.io as sio
import streamlit as st
from vega_datasets import data
import pandas as pd
import torch 
import numpy as np
import scipy.io as sio
from utilsResFNO import ResFNO, FNO1d, RangeNormalizer,Predict,T_random
import matplotlib.pyplot as plt
from utils import chart, db
import numpy as np
import random

COMMENT_TEMPLATE_MD = """{} - {}
> {}"""


def space(num_lines=1):
    """Adds empty lines to the Streamlit app."""
    for _ in range(num_lines):
        st.write("")


st.set_page_config(layout="centered", page_icon="💬", page_title="Commenting app")

# Data visualisation part

st.title("💬 Try to compare FNO with FEM")

# Load data
data1  = sio.loadmat('data/Case1.mat') 
dataA = data1['dataA']  
dataT = data1['dataT'] 
dataTair = data1['dataTair']
nx = dataT.shape[1]
nt = dataT.shape[2]
tn = 223
T_list = np.linspace(0, tn, nt)

# Load model
dic = 'logs/' 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FNO1d(16, 64, 'T').to(device) 
model.load_state_dict(torch.load(dic+'ResFNO_T_X35_net_params.pkl'))
model.eval()

# Start 
col1, col2, col3 = st.columns(3)

with col1:
    t1 = st.slider(
        "Time of heat 1 (min)", min_value=20, max_value=80, step=1, value=50
    )
with col2:
    t2 = st.slider(
        "Time of hold 1 (min)", min_value=t1, max_value=110, step=1, value=100
    )
with col3:
    t3 = st.slider(
        "Time of heat 2 (min)", min_value=t2, max_value=150, step=1, value=120
    )   

col4, col5, col6 = st.columns(3)
with col1:
    t4 = st.slider(
        "Time of hold 2 (min)", min_value=t3, max_value=190, step=1, value=180
    )
with col2:
    dh1 = st.slider(
        "Teperature of hold 1 (K)", min_value=50, max_value=120, step=1, value=80
    )
with col3:
    dh2 = st.slider(
        "Teperature of hold 2 (K)", min_value=50, max_value=120, step=1, value=80
    )   
    
# dh1 = random.randrange(50, 120)
# dh2 = random.randrange(50, 120) 
# t1 = random.randrange(20, 80)
# t2 = random.randrange(t1+20, 110)
# t3 = random.randrange(t2+20, 150)
# t4 = random.randrange(t3+20, 190)
t5 = tn 
h0 = 20
Ta = [T_random(t, t1, t2-t1, t3-t2, t4-t3, t5-t4, h0, dh1, dh2) + 273
  for t in np.linspace(0, tn, nt) ] 

space(1)

#################
import time
T1 = time.time()

## 网络求解
x_index = 35
T_index = 10
# st.write("ResFNO predicting  ...")
T_input, T_Pre, _ = Predict(model, dataTair, dataT, x_index, T_index, Ta)
T2 = time.time()
# st.write('ResFNO finished in %.2f second' % ((T2 - T1)))

FEM = 0

if FEM:
## 有限元求解
    st.write("FEM  calculating  ...")
    from mph_2_hold import Comsol_FEM
    T_Real = Comsol_FEM(T_list, Ta)
    T3 = time.time()
    st.write('FEM  finished in %.2f second' % ((T3 - T2)))


    list_Air   = ['T Air' for i in range(nt)]
    list_Real  = ['T Real'  for i in range(nt)]
    list_Pre   = ['T Pre'  for i in range(nt)]
    
    list_Temperature  = np.hstack((T_input, T_Real, T_Pre))
    list_Method       = np.hstack((list_Air, list_Real, list_Pre))
    list_Time         = np.hstack((T_list,  T_list, T_list))

    cure_cycle = pd.DataFrame({
        'Time'       : list_Time, 
        'method'     : list_Method,
        'Temperature': list_Temperature})
    
if not FEM:

    list_Air   = ['T Air' for i in range(nt)]
    list_Pre   = ['T Pre'  for i in range(nt)]
    
    list_Temperature  = np.hstack((T_input, T_Pre))
    list_Method       = np.hstack((list_Air, list_Pre))
    list_Time         = np.hstack((T_list, T_list))

    cure_cycle = pd.DataFrame({
        'Time'       : list_Time, 
        'method'     : list_Method,
        'Temperature': list_Temperature})
################

# source = source[source.symbol.isin(symbols)]

chart = chart.get_chart(cure_cycle)
st.altair_chart(chart, use_container_width=True)

space(2)



    
    
    
    
    
    
    
    
    
    
    
    
    
    
