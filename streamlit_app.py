# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 11:11:25 2021

@author: cgx@nuaa.edu.cn

Implement of 

'[1] Residual Fourier Neural Operators for Composite Curing Modelling '

"""

import scipy.io as sio
import streamlit as st
import pandas as pd
import torch 
import numpy as np
from utilsResFNO import ResFNO, FNO1d, RangeNormalizer,Predict,T_random
from utils import chart
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
data1  = sio.loadmat('data/double_hold_200_HL.mat') 
dataA = data1['dataA']  
dataT = data1['dataT'] 
dataTair = data1['dataTair']
nx = dataT.shape[1]
nt = dataT.shape[2]
tn = 501
FEM = 0

  
@st.cache(suppress_st_warning=True)
def load_client():
    import mph
    client = mph.Client(cores=1)  
    return client

T_list = np.linspace(0, tn, nt)

# Load model
dic = 'logs/' 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FNO1d(16, 64, 'T').to(device) 
model.load_state_dict(torch.load(dic+'ResFNO_T_X11_net_params100.pkl'))
model.eval()

# Start 
col1, col2, col3 = st.columns(3)

with col1:
    r1 = st.slider(
        "Rate of T 1 (K/min)", min_value=1., max_value=3., step=0.1, value=1.5
    )
with col2:
    r2 = st.slider(
        "Rate of T 2 (K/min)", min_value=1., max_value=3., step=0.1, value=1.5
    )
with col3:
    dt2 = st.slider(
        "Time of hold 1 (min)", min_value=30, max_value=80, step=1, value=70
    )   

col4, col5, col6 = st.columns(3)
with col1:
    dt4 = st.slider(
        "Time of hold 2 (min)", min_value=30, max_value=120, step=1, value=90
    )
with col2:
    T1 = st.slider(
        "Teperature of hold 1 (K)", min_value=100, max_value=150, step=1, value=120
    )
with col3:
    T2 = st.slider(
        "Teperature of hold 2 (K)", min_value=155, max_value=220, step=1, value=180
    )   
    
rate3 = float(2.5)
dh1 = T1-20
dh2 = T2-T1
dt1 = dh1/r1
dt3 = dh2/r2
dt5 = (T2-20)/rate3
dt6 = tn-dt1-dt2-dt3-dt4-dt5

t5 = tn 
h0 = 20
Ta = [T_random(t, dt1, dt2, dt3, dt4, dt5, h0, dh1, dh2) + 273
  for t in np.linspace(0, tn, nt) ] 

space(1)

#################
import time
T1 = time.time()

## 网络求解
x_index = 11
T_index = 10
# st.write("ResFNO predicting  ...")
T_input, T_Pre, _ = Predict(model, dataTair, dataT, x_index, T_index, Ta)
T2 = time.time()
st.write('ResFNO finished in %.4f second' % ((T2 - T1)))

if FEM:
## 有限元求解
    client = load_client()
    st.write("FEM  calculating  ...")
    from mph_2_hold_HL import Comsol_FEM
    T_Real = Comsol_FEM(T_list, Ta, client)
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

chart = chart.get_chart_local(cure_cycle)
st.altair_chart(chart, use_container_width=True)

space(2)



    
    
    
    
    
    
    
    
    
    
    
    
    
    
