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
file_name = '/home/ubuntu/Documents/project/Btech_Project/Project documentation/datasets/set2/s1_2kg.mat'
mat = scipy.io.loadmat(file_name)
mat = {k:v for k, v in mat.items() if k[0] != '_'}

#assign values
emg1 = mat['data'][:,0]#[30000:,0]
emg2 = mat['data'][:,1]
sampling_frequency = 1e3 / mat['isi'][0][0]
frame = 2500
step = int(2500 / 2)


#plot data
#pp.plot_signal(emg1, sampling_frequency, 'biceps')
#pp.plot_signal(emg2, sampling_frequency, 'triceps')


#apply filters
filtered_emg1 = pp.notch_filter(emg1, sampling_frequency, False)
filtered_emg1 = pp.bp_filter(filtered_emg1, 10, 500, sampling_frequency, False)

# filtered_emg2 = pp.notch_filter(emg2, sampling_frequency, False)
# filtered_emg2 = pp.bp_filter(filtered_emg2, 10, 500, sampling_frequency, False)

rms1 = pp.rootmeansquare(filtered_emg1, frame, step, sampling_frequency, 'biceps', show=True)
'''mav1 = pp.meanabsolutevalue(filtered_emg1, frame, step, sampling_frequency, channel_name='biceps', show=False)
var1 = pp.variance(filtered_emg1, frame, step, sampling_frequency, 'biceps', show=False)

rms2 = pp.rootmeansquare(filtered_emg2, frame, step, sampling_frequency, 'triceps', show=False)

emg3 = abs(filtered_emg1)
rms3 = pp.rootmeansquare(emg3, frame, step, sampling_frequency, 'biceps', show=False)
mav3 = pp.meanabsolutevalue(emg3, frame, step, sampling_frequency, 'biceps', show=False)
var3 = pp.variance(emg3, frame, step, sampling_frequency, 'biceps', show=False)
'''

# import numpy as np
# filtered_emg2 = np.multiply(filtered_emg2, 10)
# plt.autoscale(tight=True)
# plt.plot(rms1)
#plt.plot(filtered_emg2)



pos = []
z = []
for i in z:
    if i <= 0.15 : 
        pos.append(0)
    elif i > 0.21 and i <= 0.30 :
        pos.append(1)
    elif i > 0.3:
        pos.append(2)
plt.autoscale(tight=True)
plt.plot(pos)    

import numpy as np

plt.plot(z)    

angle = pos[:-7]
plt.plot(angle)
y = smooth(pos, window_len=7,window='hanning')
plt.plot(y)    

plt.plot(rms1)    
plt.plot(pos)
pos = []

for i in pos:
    if i <= 0.15 : 
        pos.append(0)
    elif i > 0.26 and i <= 1.20 :
        pos.append(1)
    elif i > 1.3:
        pos.append(2)
plt.autoscale(tight=True)
plt.plot(pos)    



import mat_to_csv as m2c

i = 0
angle_data = []
for i in angle:
    for j in np.linspace(i,i, int((len(emg1)/len(angle)))):
        angle_data.append(j)

plt.plot(angle_data)


rows = angle_data
m2c.save_csv(angle_data, "angle-dataset.csv")



    