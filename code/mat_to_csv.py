#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 13 22:37:24 2021

@author: ubuntu
"""
import numpy as np
import pandas as pd

def save_csv(rows, file_name):
    np.savetxt(file_name,
		rows,
		delimiter =", ",
		fmt ='% s')
    
def read_csv(file_name):
    data_frame = pd.read_csv(file_name)   
    return(data_frame.to_numpy())
    
'''    
#import dataset
import scipy.io# load mat files

file_name = '/home/ubuntu/Documents/project/Btech_Project/Project documentation/datasets/set2/s1_2kg.mat'
mat = scipy.io.loadmat(file_name)
mat = {k:v for k, v in mat.items() if k[0] != '_'}

#assign values
emg1 = mat['data'][:,0]#[30000:,0]
emg2 = mat['data'][:,1]

rows = [emg1, emg2]

save_csv(rows, 'raw-dataset.csv')
arr = read_csv('raw-dataset.csv')
'''
    
    
