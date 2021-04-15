# -*- coding: utf-8 -*-
import scipy.io# load mat files
import matplotlib.pyplot as plt # plotting
#import numpy as np # linear algebra
#from scipy import signal
import scipy.fftpack
#import math
#import pywt
import main as pp

#import dataset
file_name = '/home/ubuntu/Documents/project/Btech_Project/Project documentation/datasets/set2/s1_1kg.mat'
mat = scipy.io.loadmat(file_name)
mat = {k:v for k, v in mat.items() if k[0] != '_'}

#assign values
emg1 = mat['data'][30000:,0]
emg2 = mat['data'][:,1]
sampling_frequency = 1e3 / mat['isi'][0][0]
frame = 10000
step = 5000


#plot data
#pp.plot_signal(emg1, sampling_frequency, 'biceps')
#pp.plot_signal(emg2, sampling_frequency, 'triceps')


#apply filters
filtered_emg1 = pp.notch_filter(emg1, sampling_frequency, False)
filtered_emg1 = pp.bp_filter(filtered_emg1, 10, 500, sampling_frequency, False)

filtered_emg2 = pp.notch_filter(emg2, sampling_frequency, False)
filtered_emg2 = pp.bp_filter(filtered_emg2, 10, 500, sampling_frequency, False)

rms1 = pp.rootmeansquare(filtered_emg1, frame, step, sampling_frequency, 'biceps', show=True)
rms2 = pp.rootmeansquare(filtered_emg2, frame, step, sampling_frequency, 'biceps', show=True)
import numpy as np
filtered_emg2 = np.multiply(filtered_emg2, 10)
plt.autoscale(tight=True)
plt.plot(filtered_emg1)
plt.plot(filtered_emg2)
plt.close()


pos = []
for i in rms1:
    if i <= 0.06 : 
        pos.append(0)
    elif i > 0.06 and i <= 0.10 :
        pos.append(1)
    elif i > 0.12:
        pos.append(2)

plt.plot(pos)    








