import AttenuationLibrary as AL
import soundfile as sl
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate
from scipy.signal import medfilt

baseline = AL.Soundfile("SecondScan/Outputs/Touchingagain_12_K.wav") #Get the baseline object, making a new one each time because the correlate function changes the object
measurement = AL.Soundfile("SecondScan/Outputs/A_00_K.wav") #Get the measurement file, should be A_00_X then A_01_X etc.

#FFT Raw
inraw = np.fft.rfft(baseline.soundfile)
outraw = np.fft.rfft(measurement.soundfile)
freqsraw = np.fft.rfftfreq(baseline.length,1/baseline.samplerate)

#FFT with a window
innorm = baseline.soundfile / np.max(np.abs(baseline.soundfile))
outnorm = measurement.soundfile / np.max(np.abs(measurement.soundfile)) #Normalizing signals
N = len(outnorm)
windowfreq = np.fft.rfftfreq(N,1/baseline.samplerate)
window = np.hanning(N)
inwindow = np.fft.rfft(innorm*window)/N
outwindow = np.fft.rfft(outnorm*window)/N

inraw = AL.convert_to_decibels(inraw)
outraw = AL.convert_to_decibels(outraw)
inwindow = AL.convert_to_decibels(inwindow)
outwindow = AL.convert_to_decibels(outwindow)
plt.semilogx(freqsraw,inraw,label="in raw")
plt.semilogx(freqsraw,outraw,label="out raw")
plt.semilogx(windowfreq,inwindow,label="in window")
plt.semilogx(windowfreq,outwindow,label="out window")
plt.legend()
plt.show()
