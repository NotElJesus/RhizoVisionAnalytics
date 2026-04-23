import AttenuationLibrary as AL
import soundfile as sl
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate
from scipy.signal import medfilt
import os


def get_max_from_boxes(arr, box_size):
    # Ensure the array can be evenly divided; if not, truncate the end
    n_boxes = len(arr) // box_size
    arr = np.array(arr[:n_boxes * box_size])
    
    # Reshape to (number of boxes, box size) and find max along axis 1
    return np.max(arr.reshape(-1, box_size), axis=1)
def get_freq_for_boxes(freqs, box_size):
    n_boxes = len(freqs) // box_size
    freqs = np.array(freqs[:n_boxes * box_size])
    return np.mean(freqs.reshape(-1, box_size), axis=1)

os.chdir("jeffatten") #Change to the folder that holds the files, should be jeffatten

#Testing whether its worth it to use a window or not
onion = AL.Soundfile("SecondScan/Outputs/A_09_J.wav") #Get the baseline object, making a new one each time because the correlate function changes the object
nothing = AL.Soundfile("SecondScan/unwanted/A_45_J.wav") #Get the measurement file, should be A_00_X then A_01_X etc.
touching = AL.Soundfile("SecondScan/Outputs/Touchingagain_12_J.wav")
onion.kernelSize = 11
nothing.kernelSize = 11
touching.kernelSize = 11
boxwidth = 512
xlim = [2200,4500]
plt.rcParams.update({'font.size': 16})
plt.figure(figsize=(6, 6))
plt.semilogx(get_freq_for_boxes(onion.freqs, boxwidth), 20*np.log10(get_max_from_boxes(np.abs(onion.windowFFT), boxwidth)), label="Onion + Soil")
plt.semilogx(get_freq_for_boxes(nothing.freqs, boxwidth), 20*np.log10(get_max_from_boxes(np.abs(nothing.windowFFT), boxwidth)), label="Only Soil")
plt.semilogx(get_freq_for_boxes(touching.freqs, boxwidth), 20*np.log10(get_max_from_boxes(np.abs(touching.windowFFT), boxwidth)), label="Sent Signal")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")

plt.xlim(xlim)
plt.ylim([-125,-58])
plt.axvline(x=2500,label="Speaker Signal (2.5 kHz)",color="red",linestyle="--")
plt.grid(True, which="both")
plt.legend()
plt.show()
