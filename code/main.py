# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
from scipy import signal
import math
import pywt
import scipy.io# load mat files


def plot_signal(x, samplerate, chname):
    if type(x) != 'list':
        x = [x]

    fig = plt.figure(figsize=(10,5))    
    for i in range(len(x)):
        t = np.arange(0, len(x[i]) / samplerate, 1 / samplerate)
        plt.subplot(1,len(x), i+1)
        plt.plot(t, x[i])
        plt.autoscale(tight=True)
        plt.xlabel('Time')
        plt.ylabel('Amplitude (mV)')
        plt.title(chname)
        #fig.set_size_inches(w=15,h=10)

    
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
    
    
def variance(signal, frame, step, fs, channel_name, show=False):
    var = []
    for i in range(frame, signal.size, step):
        x = signal[i - frame:i]
        var.append(np.var(x))
    
    if show:
        plotfeature(signal, channel_name, fs, var, "variance", step)
    return(var)

def rootmeansquare(signal, frame, step, fs, channel_name, show=False):
    rms = []
    for i in range(frame, signal.size, step):
        x = signal[i - frame:i]
        rms.append(np.sqrt(np.mean(x ** 2)))
    
    if show:
        plotfeature(signal, channel_name, fs, rms, "Root Mean Square", step)
    return(rms)

def integralemg(signal, frame, step, fs, channel_name, show=False):
    iemg = []
    for i in range(frame, signal.size, step):
        x = signal[i - frame:i]
        iemg.append(np.sum(abs(x)))  # Integral
    
    if show:
        plotfeature(signal, channel_name, fs, iemg, "Integral", step)
    return(iemg)

def meanabsolutevalue(signal, frame, step, fs, channel_name, show=False):
    mav = []
    for i in range(frame, signal.size, step):
        x = signal[i - frame:i]
        mav.append(np.sum(np.absolute(x)) / frame)  # Mean Absolute Value
    
    if show:
        plotfeature(signal, channel_name, fs, mav, "Mean  Absolute Value", step)
    return(mav)

def log_detector(signal, frame, step, fs, channel_name, show=False):
    log_det = []
    for i in range(frame, signal.size, step):
        x = signal[i - frame:i]
        log_det.append(np.exp(np.sum(np.log10(np.absolute(x))) / frame))
    
    if show:
        plotfeature(signal, channel_name, fs, log_det, "Log Detector", step)
    return(log_det)

def wave_length(signal, frame, step, fs, channel_name, show=False):
    wl = []
    for i in range(frame, signal.size, step):
        x = signal[i - frame:i]
        wl.append(np.sum(abs(np.diff(x))))  # Wavelength
    
    if show:
        plotfeature(signal, channel_name, fs, wl, "Wavelength", step)
    return(wl)

def avg_amplitude_change(signal, frame, step, fs, channel_name, show=False):
    aac = []
    for i in range(frame, signal.size, step):
        x = signal[i - frame:i]
        aac.append(np.sum(abs(np.diff(x))) / frame)  # Average Amplitude Change
    
    if show:
        plotfeature(signal, channel_name, fs, aac, "Average Amplitude Change", step)
    return(aac)

def difference_absolute_standard_deviation(signal, frame, step, fs, channel_name, show=False):
    dasdv = []
    for i in range(frame, signal.size, step):
        x = signal[i - frame:i]
        dasdv.append(math.sqrt((1 / (frame - 1)) * np.sum((np.diff(x)) ** 2)))  # Difference absolute standard deviation value
    
    if show:
        plotfeature(signal, channel_name, fs, dasdv, "Difference absolute standard deviation value", step)
    return(dasdv)

def zcruce(X, th):
    th = 0
    cruce = 0
    for cont in range(len(X) - 1):
        can = X[cont] * X[cont + 1]
        can2 = abs(X[cont] - X[cont + 1])
        if can < 0 and can2 > th:
            cruce = cruce + 1
    return cruce

def zero_crossing(signal, frame, step, fs, channel_name, show=False):
    zc = []
    th = np.mean(signal) + 3 * np.std(signal)
    for i in range(frame, signal.size, step):
        x = signal[i - frame:i]
        zc.append(zcruce(x, th))  # Zero-Crossing

    if show:
        plotfeature(signal, channel_name, fs, zc, "Zero-Crossing", step)
    return(zc)

def myopulse(signal, th):
    umbral = signal >= th
    return np.sum(umbral) / len(signal)

def myopulse_percentage_rate(signal, frame, step, channel_name, show=False):
    myop = []
    th = np.mean(signal) + 3 * np.std(signal)
    for i in range(frame, signal.size, step):
        x = signal[i - frame:i]
        myop.append(myopulse(x, th))  # Myopulse percentage rate
    fs = 10
    if show:
        plotfeature(signal, channel_name, fs, myop, "Myopulse percentage rate", step)
    return(myop)


# freq domain feature extraction methods

def next_power_of_2(x):
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


def spectrum(signal, fs):
    m = len(signal)
    n = next_power_of_2(m)
    y = np.fft.fft(signal, n)
    yh = y[0:int(n / 2 - 1)]
    fh = (fs / n) * np.arange(0, n / 2 - 1, 1)
    power = np.real(yh * np.conj(yh) / n)

    return fh, power

def freq_ratio(frequency, power):
    power_low = power[(frequency >= 30) & (frequency <= 250)]
    power_high = power[(frequency > 250) & (frequency <= 500)]
    ULC = np.sum(power_low)
    UHC = np.sum(power_high)

    return ULC / UHC

def frequency_ratio(signal, frame, step, fs, channel_name, show=False):
    fr = []
    for i in range(frame, signal.size, step):
        x = signal[i - frame:i]
        frequency, power = spectrum(x, fs)

        fr.append(freq_ratio(frequency, power))  # Frequency ratio
   
    if show:
        plotfeature(signal, channel_name, fs, fr, "Frequency ratio", step)
    return(fr)

def mean_power(signal, frame, step, fs, channel_name, show=False):
    mnp = []
    for i in range(frame, signal.size, step):
        x = signal[i - frame:i]
        frequency, power = spectrum(x, fs)

        mnp.append(np.sum(power) / len(power))  # Mean power
   
    if show:
        plotfeature(signal, channel_name, fs, mnp, "Mean power", step)
    return(mnp)

def total_power(signal, frame, step, fs, channel_name, show=False):
    tot = []
    for i in range(frame, signal.size, step):
        x = signal[i - frame:i]
        frequency, power = spectrum(x, fs)

        tot.append(np.sum(power))  # Total power
   
    if show:
        plotfeature(signal, channel_name, fs, tot, "Total power", step)
    return(tot)


def mean_freq(frequency, power):
    num = 0
    den = 0
    for i in range(int(len(power) / 2)):
        num += frequency[i] * power[i]
        den += power[i]

    return num / den

def median_freq(frequency, power):
    power_total = np.sum(power) / 2
    temp = 0
    tol = 0.01
    errel = 1
    i = 0

    while abs(errel) > tol:
        temp += power[i]
        errel = (power_total - temp) / power_total
        i += 1
        if errel < 0:
            errel = 0
            i -= 1

    return frequency[i]

def mean_frequency(signal, frame, step, fs, channel_name, show=False):
    mnf = []
    mdf = []
    pkf = []
    for i in range(frame, signal.size, step):
        x = signal[i - frame:i]
        frequency, power = spectrum(x, fs)

        mnf.append(mean_freq(frequency, power))  # Mean frequency
        mdf.append(median_freq(frequency, power))  # Median frequency
        pkf.append(frequency[power.argmax()])  # Peak frequency
   
    if show:
        plotfeature(signal, channel_name, fs, mnf, "Mean frequency", step)
        plotfeature(signal, channel_name, fs, mdf, "Median frequency", step)
        plotfeature(signal, channel_name, fs, pkf, "Peak frequency", step)
    return(np.column_stack((mnf, mdf, pkf)))


#time freq feature extraction method
def wavelet_energy(x, mother, nivel):
    coeffs = pywt.wavedecn(x, wavelet=mother, level=nivel)
    arr, _ = pywt.coeffs_to_array(coeffs)
    Et = np.sum(arr ** 2)
    cA = coeffs[0]
    Ea = 100 * np.sum(cA ** 2) / Et
    Ed = []

    for k in range(1, len(coeffs)):
        cD = list(coeffs[k].values())
        cD = np.asarray(cD)
        Ed.append(100 * np.sum(cD ** 2) / Et)

    return Ea, Ed

def time_frequency_features_estimation(signal, frame, step, channel_name):

    h_wavelet = []

    for i in range(frame, signal.size, step):
        x = signal[i - frame:i]

        E_a, E = wavelet_energy(x, 'db2', 4)
        E.insert(0, E_a)
        E = np.asarray(E) / 100

        h_wavelet.append(-np.sum(E * np.log2(E)))
        
    plotfeature(signal, channel_name, 10000, h_wavelet, "time frequency feature", step)
    return h_wavelet
    
'''
#import dataset
file_name = '/home/ubuntu/Documents/project/Btech_Project/Project documentation/datasets/set2/s1_0kg.mat'
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

'''//////////////////////////////'''
plot_signal([emg1, filtered_emg1], sampling_frequency, 'biceps')

x = [filtered_emg1, rms1]
samplerate = sampling_frequency
chname = 'biceps brachii'

fig = plt.figure(figsize=(10,5))    
for i in range(len(x)):
    t = np.arange(0, len(x[i]) / samplerate, 1 / samplerate)
    plt.subplot(1,len(x), i+1)
    plt.plot(t, x[i])
    plt.autoscale(tight=True)
    plt.xlabel('Time')
    plt.ylabel('Amplitude (mV)')
    plt.title(chname)
    #fig.set_size_inches(w=15,h=10)
'''//////////////////////////////'''
# EMG Feature Extraction
frame = 2500
step = 1250
channel_name = 'biceps'
fs = 10000

"""
  Time features
    Compute time features from signal using sliding window method.
    :param signal: numpy array signal.
    :param frame: sliding window size.
    :param step: sliding window step size.
"""
var = variance(filtered_emg1, frame, step, 'biceps', show=True)
rms1 = rootmeansquare(filtered_emg1, frame, step, fs, 'biceps', show=False)
rms2 = rootmeansquare(filtered_emg1, frame, step, fs, 'biceps', show=True)
iemg = integralemg(filtered_emg1, frame, step, 'biceps', show=True)
mav = meanabsolutevalue(filtered_emg1, frame, step, 'biceps', show=True)
log_det = log_detector(filtered_emg1, frame, step, 'biceps', show=True)
wl = wave_length(filtered_emg1, frame, step, 'biceps', show=True)
aac = avg_amplitude_change(filtered_emg1, frame, step, 'biceps', show=True)
dasdv = difference_absolute_standard_deviation(filtered_emg1, frame, step, 'biceps', show=True)
zc = zero_crossing(filtered_emg1, frame, step, 'biceps', show=True)
myop = myopulse_percentage_rate(filtered_emg1, frame, step, 'biceps', show=True)


"""
  Frequency features
    Compute frequency features from signal using sliding window method.
    :param signal: numpy array signal.
    :param fs: sampling frequency of the signal.
    :param frame: sliding window size
    :param step: sliding window step size
"""
fr = frequency_ratio(filtered_emg1, frame, step, sampling_frequency, 'biceps', show=True)
mnp = mean_power(filtered_emg1, frame, step, sampling_frequency, 'biceps', show=True)
tot = total_power(filtered_emg1, frame, step, sampling_frequency, 'biceps', show=True)
mnf = mean_frequency(filtered_emg1, frame, step, sampling_frequency, 'biceps', show=True)


"""
  Time-frequency features 
    Compute time-frequency features from signal using sliding window method.
    :param signal: numpy array signal.
    :param frame: sliding window size
    :param step: sliding window step size
"""
time_frequency_matrix = time_frequency_features_estimation(filtered_emg2, frame, step)
'''

























