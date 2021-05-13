#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 20:15:37 2021

@author: ubuntu
"""

import main as pr
import scipy.io


sampling_frequency = 10000
frame = 2500
step = int(2500 / 2)
rms = []

for i in range(4):
    for j in range(7):
        if(j != 5):
            #print('s' + str(i+1) + '_' + str(j) + 'kg.mat')
            subject = 's' + str(i+1) + '_' + str(j) + 'kg.mat'
            file_name = '/home/ubuntu/Documents/project/Btech_Project/Project documentation/datasets/set2/' + subject
            mat = scipy.io.loadmat(file_name)
            mat = {k:v for k, v in mat.items() if k[0] != '_'}
            emg1 = mat['data'][:,0]
            #emg2 = mat['data'][:,1]
            #sampling_frequency = 1e3 / mat['isi'][0][0]
            #pr. plot_signal(emg1, sampling_frequency, subject + '- biceps')
            #pr.plot_signal(emg2, sampling_frequency, subject + '- triceps')
            rms.append(pr.smooth(pr.rootmeansquare(emg1, frame, step, sampling_frequency, 'biceps', show=False)))

import matplotlib.pyplot as plt # plotting

for i in range(6):
    plt.plot(rms[i])
            

            