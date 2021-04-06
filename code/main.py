# -*- coding: utf-8 -*-

import scipy.io# load mat files
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
from scipy import signal
import scipy.fftpack


def plot_signal(x, samplerate, chname):
    t = np.arange(0, len(x) / samplerate, 1 / samplerate)
    plt.plot(t, x)
    plt.autoscale(tight=True)
    plt.xlabel('Time')
    plt.ylabel('Amplitude (mV)')
    plt.title(chname)
    plt.show()
    
def notch_filter(x, samplerate, plot=False):
    #x = x - np.mean(x)

    notch_freq = 50 # Frequency to be removed from signal (Hz)
    Q = 30.0  # Quality factor
    w0 = notch_freq / (samplerate/2)  # Normalized Frequency
    # Design notch filter
    b, a = signal.iirnotch(w0, Q)
    x_filt = signal.filtfilt(b, a, x.T)

    if plot:
        t = np.arange(0, len(x) / samplerate, 1 / samplerate)
        plt.plot(t, x)
        plt.plot(t, x_filt.T, 'k')
        plt.autoscale(tight=True)
        plt.xlabel('Time')
        plt.ylabel('Amplitude (mV)')
        plt.show()

    return x_filt


def bp_filter(x, low_f, high_f, samplerate, plot=False):
    # x = x - np.mean(x)

    low_cutoff_bp = low_f / (samplerate / 2)
    high_cutoff_bp = high_f / (samplerate / 2)

    [b, a] = signal.butter(5, [low_cutoff_bp, high_cutoff_bp], btype='bandpass')

    x_filt = signal.filtfilt(b, a, x)

    if plot:
        t = np.arange(0, len(x) / samplerate, 1 / samplerate)
        plt.plot(t, x)
        plt.plot(t, x_filt, 'k')
        plt.autoscale(tight=True)
        plt.xlabel('Time')
        plt.ylabel('Amplitude (mV)')
        plt.show()

    return x_filt

def powerspectrum(signal, fs):    
    fourier_transform = np.fft.rfft(signal)
    abs_fourier_transform = np.abs(fourier_transform)
    power_spectrum = np.square(abs_fourier_transform)
    frequency = np.linspace(0, fs/2, len(power_spectrum))
    plt.plot(frequency, power_spectrum)

''' Feature extraction'''
def plotfeature(signal, channel_name, fs, feature, feature_name, step):
    ts = np.arange(0, len(signal) / fs, 1 / fs)
    
    # for idx, f in enumerate(tfeatures.T):
    tf = step * (np.arange(0, len(feature) / fs, 1 / fs))
    fig = plt.figure()
    ax = fig.add_subplot(111, label="1")
    ax2 = fig.add_subplot(111, label="2", frame_on=False)
    ax.plot(ts, signal, color="C0")
    
    ax.autoscale(tight=True)
    plt.title(channel_name + ": " + feature_name)
    ax.set_xlabel("Time")
    ax.set_ylabel("mV")
    ax2.plot(tf, feature, color="red")
    ax2.yaxis.tick_right()
    ax2.autoscale(tight=True)
    ax2.set_xticks([])
    ax2.set_yticks([])
    #mng = plt.get_current_fig_manager()
    #mng.window.state('zoomed')
    plt.show()
    
    
def variance(signal, frame, step, channel_name, show=False):
    var = []
    for i in range(frame, signal.size, step):
        x = signal[i - frame:i]
        var.append(np.var(x))
    
    if show:
        plotfeature(signal, channel_name, fs, var, "variance", step)
    return(var)

def rootmeansquare(signal, frame, step, channel_name, show=False):
    rms = []
    for i in range(frame, signal.size, step):
        x = signal[i - frame:i]
        rms.append(np.sqrt(np.mean(x ** 2)))
    
    if show:
        plotfeature(signal, channel_name, fs, rms, "Root Mean Square", step)
    return(rms)

def integralemg(signal, frame, step, channel_name, show=False):
    iemg = []
    for i in range(frame, signal.size, step):
        x = signal[i - frame:i]
        iemg.append(np.sum(abs(x)))  # Integral
    
    if show:
        plotfeature(signal, channel_name, fs, iemg, "Integral", step)
    return(iemg)

def meanabsolutevalue(signal, frame, step, channel_name, show=False):
    mav = []
    for i in range(frame, signal.size, step):
        x = signal[i - frame:i]
        mav.append(np.sum(np.absolute(x)) / frame)  # Mean Absolute Value
    
    if show:
        plotfeature(signal, channel_name, fs, mav, "Mean  Absolute Value", step)
    return(mav)




#import dataset
file_name = '/home/ubuntu/Documents/project/Btech_Project/Project documentation/datasets/set2/s1_2kg.mat'
mat = scipy.io.loadmat(file_name)
mat = {k:v for k, v in mat.items() if k[0] != '_'}

#assign values
emg1 = mat['data'][:,0]
emg2 = mat['data'][:,1]
sampling_frequency = 1e3 / mat['isi'][0][0]

#plot data
plot_signal(emg1, sampling_frequency, 'biceps')
plot_signal(emg2, sampling_frequency, 'triceps')

#apply filters
filtered_emg1 = notch_filter(emg1, sampling_frequency, False)
filtered_emg1 = bp_filter(filtered_emg1, 10, 500, sampling_frequency, False)

filtered_emg2 = notch_filter(emg2, sampling_frequency, False)
filtered_emg2 = bp_filter(filtered_emg2, 10, 500, sampling_frequency, False)

# EMG Feature Extraction
frame = 10000
step = 5000
channel_name = 'biceps'
fs = 10000

var = variance(filtered_emg1, frame, step, 'biceps', show=True)
rms1 = rootmeansquare(filtered_emg1, frame, step, 'biceps', show=True)
rms2 = rootmeansquare(filtered_emg2, frame, step, 'triceps', show=True)
iemg = integralemg(filtered_emg1, frame, step, 'biceps', show=True)
mav = meanabsolutevalue(filtered_emg2, frame, step, 'biceps', show=True)

def wilson_amplitude(signal, th):
    
    """x = abs(np.diff(signal))
    umbral = x >= th
    return np.sum(umbral)"""
    
    


def wilsonamplitude(signal, frame, step, channel_name, show=False):
    wamp = []
    #th = np.mean(signal) + 1 * np.std(signal)
    th = 0.4
    print(th)
    for i in range(frame, signal.size, step):
        x = signal[i - frame:i]
        wamp.append(wilson_amplitude(x, th))  # Willison amplitude
    
    if show:
        plotfeature(signal, channel_name, fs, wamp, "wilson Amplitude", step)
    return(wamp)
    
wamp = wilsonamplitude(filtered_emg2, frame, step, 'biceps', show=True)


wamp = []
#th = np.mean(signal) + 1 * np.std(signal)
th = 0.4
for i in range(frame, filtered_emg1.size, step):
    x = filtered_emg1[i - frame:i]
    y = abs(np.diff(x))
    umbral = y >= th
    wamp.append(np.sum(umbral))  # Willison amplitude



    