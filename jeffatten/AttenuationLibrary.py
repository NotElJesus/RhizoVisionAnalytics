import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate
from scipy.signal import medfilt

class Soundfile:
    def __init__(self,soundfilepath:str):
        self.soundfile,self.samplerate = sf.read(soundfilepath)
    def save(self,path:str):
        sf.write(path,self.soundfile,self.samplerate)
    @property 
    def length(self):
        return len(self.soundfile)
        
def correlate_soundfiles(sound1:Soundfile,sound2:Soundfile): #This takes two soundfile objects and does its best to align them, trying to make the signal start at the same time more precisely than a human could
    correlation = correlate(in1 = sound2.soundfile,in2 = sound1.soundfile,mode="full",method="auto") #Chat describes it as basically sliding a signal against another one and comparing them 
    lags = np.arange(-sound1.length + 1, sound2.length) #Generates a list of all possible lags, including negative values
    lag = lags[np.argmax(np.abs(correlation))]#Which lag corresponds to the highest correlation
    if lag > 0:
        # sound2 starts later → remove its delay
        sound2.soundfile = sound2.soundfile[lag:]
        sound1.soundfile = sound1.soundfile[:sound2.length]
    else:
        sound1.soundfile = sound1.soundfile[-lag:]
        sound2.soundfile = sound2.soundfile[:sound1.length]
        
def get_mag(signal:Soundfile):
    fft = np.fft.rfft(signal.soundfile)
    return 20*np.log10(np.abs(fft))
def calculate_attenuation(InitialSignal:Soundfile,FinalSignal:Soundfile,precorrelated:bool = False,filterafter = False):
    if precorrelated == False:
        correlate_soundfiles(InitialSignal,FinalSignal)
    fft_in = np.fft.rfft(InitialSignal.soundfile) #Take fourier of both
    fft_final = np.fft.rfft(FinalSignal.soundfile)
    
    mag_in = 20*np.log10(np.abs(fft_in)) #Convert to decibels
    mag_final = 20*np.log10(np.abs(fft_final))
    attenuationsraw = mag_final - mag_in
    if filterafter:
        mag_in_filtered = medfilt(mag_in, kernel_size=101)
        mag_final_filtered = medfilt(mag_final, kernel_size=101)
        return mag_final_filtered-mag_in_filtered
    else:
        return attenuationsraw