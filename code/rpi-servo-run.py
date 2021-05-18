import pandas as pd
import numpy as np 
from scipy import signal
import RPi.GPIO as GPIO # Import Raspberry Pi GPIO library
from time import sleep # Import the sleep function from the time module
import matplotlib.pyplot as plt # plotting
import scipy.io# load mat files

# import matplotlib.pyplot as plt # plotting
# import main as pp

def read_csv(file_name):
    data_frame = pd.read_csv(file_name)   
    return(data_frame.to_numpy())


def fplot(x, x_filt):
    samplerate = 10000
    t = np.arange(0, len(x) / samplerate, 1 / samplerate)
    plt.plot(t, x)
    plt.plot(t, x_filt, 'k')
    plt.autoscale(tight=True)
    plt.xlabel('Time')
    plt.ylabel('Amplitude (mV)')
    plt.show()

def notch_filter(x, samplerate=10000):
    notch_freq = 50 # Frequency to be removed from signal (Hz)
    Q = 30.0  # Quality factor
    w0 = notch_freq / (samplerate/2)  # Normalized Frequency
    # Design notch filter
    b, a = signal.iirnotch(w0, Q)
    x_filt = signal.filtfilt(b, a, x.T)

    # fplot(x, x_filt.T)
    return x_filt



def bp_filter(x, low_f, high_f, samplerate=10000):
    low_cutoff_bp = low_f / (samplerate / 2)
    high_cutoff_bp = high_f / (samplerate / 2)
    [b, a] = signal.butter(5, [low_cutoff_bp, high_cutoff_bp], btype='bandpass')
    x_filt = signal.filtfilt(b, a, x)
    # fplot(t, x_filt)
    return x_filt

def rootmeansquare(signal, frame, step, fs=10000):
    rms = []
    for i in range(frame, signal.size, step):
        x = signal[i - frame:i]
        rms.append(np.sqrt(np.mean(x ** 2)))
    return(rms)

def smooth(x,window_len=11,window='hanning'):
    s=np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')
    y=np.convolve(w/w.sum(),s,mode='valid')
    return y

'''def angle_classify(emg_rms):
    arm_positions = []
    for i in emg_rms:
        if i <= 0.15 : 
            arm_positions.append(0)
        elif i > 0.21 and i <= 0.30 :
            arm_positions.append(1)
        elif i > 0.3:
            arm_positions.append(2)
    return(arm_positions)'''

def angle_classify(emg_rms):
    arm_positions = []
    new_value=0
    
    for i in emg_rms:
        new_value=((i-min(emg_rms))*150)/(max(emg_rms)-min(emg_rms))
        #new_value= new_value*2
        arm_positions.append(new_value)
        new_value=0
            
    return(arm_positions)


def angle_map(positions):
    angles = []
    for i in positions:
        if int(i) == 0:
            angles.append(0)
        elif int(i) == 1:
            angles.append(90)
        elif int(i) == 2:
            angles.append(150)
        else:
            angles.append(angles[-1])
    return(angles)

def SetAngle(angle):
    duty = angle / 18 + 2
    GPIO.output(8, True)
    pwm.ChangeDutyCycle(duty)
    sleep(1)
    GPIO.output(8, False)
    pwm.ChangeDutyCycle(0)

def run_servo(arm_angles):
    GPIO.setwarnings(False) # Ignore warning for now
    GPIO.setmode(GPIO.BOARD) # Use physical pin numbering
    GPIO.setup(8, GPIO.OUT, initial=GPIO.LOW) # Set pin 8 to be an output pin and s>
    pwm=GPIO.PWM(8, 50)
    pwm.start(0)
    
    for angle in arm_angles:
        SetAngle(angle)
        sleep(100)

    pwm.stop()
    GPIO.cleanup()

#import dataset
file_name = '/home/ubuntu/Documents/project/Btech_Project/Project documentation/datasets/set2/s1_2kg.mat'
mat = scipy.io.loadmat(file_name)
mat = {k:v for k, v in mat.items() if k[0] != '_'}
emg_biceps = mat['data'][:,0]#[30000:,0]

#emg_biceps = read_csv("subj1.csv")
sampling_frequency = 10000
frame = 2500
step = int(2500 / 2)
# femg_biceps = notch_filter(emg1)
femg_biceps = bp_filter(notch_filter(emg_biceps), 10, 500)

remg_biceps = rootmeansquare(femg_biceps, frame, step)
# plt.plot(remg_biceps)
remg_biceps = smooth(remg_biceps, window_len=7,window='flat')
remg_biceps = smooth(remg_biceps, window_len=7,window='flat')
# plt.plot(remg_biceps)
arm_positions = angle_classify(remg_biceps)
plt.plot(arm_positions)
# plt.plot(arm_positions)
arm_angles = angle_map(arm_positions)
run_servo(arm_angles)

plt.subplot(1,2,1)
plt.plot(remg_biceps)
plt.subplot(1,2,2)
plt.plot(arm_positions)


