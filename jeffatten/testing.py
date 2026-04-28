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

print(os.getcwd())
speaker = AL.Soundfile("2026-03-05/speaker_2point5.wav") #Get the baseline object, making a new one each time because the correlate function changes the object
piezo = AL.Soundfile("SecondScan/unwanted/A_45_J.wav") #Get the measurement file, should be A_00_X then A_01_X etc.
speaker.kernelSize = 11
piezo.kernelSize = 11
boxwidth = 64
xlim = [2000,3500]
plt.rcParams.update({'font.size': 16})
plt.figure(figsize=(8, 6),dpi=300)
plt.axvline(x=2500,label="Electrical Signal (2.5 kHz)",color="red",linestyle="--")
plt.semilogx(get_freq_for_boxes(speaker.freqs, boxwidth), 20*np.log10(get_max_from_boxes(np.abs(speaker.windowFFT), boxwidth)), label="Speaker Signal")
plt.semilogx(get_freq_for_boxes(piezo.freqs, boxwidth), 20*np.log10(get_max_from_boxes(np.abs(piezo.windowFFT), boxwidth)), label="Piezo Signal")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")

plt.xlim(xlim)
plt.ylim([-135,-58])

plt.grid(True, which="both")
plt.legend()
#plt.show()
plt.savefig("piezovspeaker.png")
