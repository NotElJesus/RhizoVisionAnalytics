import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate
from scipy.signal import medfilt, butter, sosfiltfilt


def convert_to_decibels(signal:np.array):
    return 20*np.log10(np.abs(signal))

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    # Design the filter in SOS format
    sos = butter(order, [lowcut, highcut], btype='bandpass', fs=fs, output='sos')
    # Apply the filter forward and backward to ensure zero phase shift
    output = sosfiltfilt(sos, data)
    return output

class Soundfile:
    def __init__(self,soundfilepath:str):
        self.soundfile,self.samplerate = sf.read(soundfilepath)
        self.fft = None #Added later in the process
        self.freq = None
        self.isCorrelated = False
    def save(self,path:str):
        sf.write(path,self.soundfile,self.samplerate) 
    def getmag(self):
        if self.fft is None:
            self.fft = np.fft.rfft(self.soundfile)
        return 20*np.log10(np.abs(self.fft))
    def bandpassFilter(self,lowcut,highcut,order=5): #Runs the filter on itself
        self.soundfile = butter_bandpass_filter(self.soundfile, lowcut, highcut, self.samplerate, order)
    @property
    def correlated(self):
        return self.isCorrelated
    @property 
    def length(self):
        return len(self.soundfile)
    @property
    def rawFFT(self):
        return np.fft.rfft(self.soundfile)
    @property
    def windowFFT(self):
        N = self.length
        window = np.hanning(N)
        return np.fft.rfft(self.soundfile*window)/N
class Sinogram:
    def __init__(self,sourceFile:Soundfile,receiverFiles:list[Soundfile],silenceCapsLengths:float=1):
        self.sourceAudio:Soundfile = sourceFile #This is the audio file of the source
        self.receiverAudio:list[Soundfile] = receiverFiles #This is a list of the audio files recorded on the hydrophones
        self.silenceCapsLengths = silenceCapsLengths #This is how many seconds of silence there were at the start and end of the audio file
    
class ConstantSinogram(Sinogram):
    def __init__(self, sourceFile, receiverFiles, silenceCapsLengths = 1):
        super().__init__(sourceFile, receiverFiles, silenceCapsLengths)        

class StairSinogram(Sinogram):
    def __init__(self, sourceFile, receiverFiles, silenceCapsLengths = 1,stairFreqHeight:float=500,stairTimeLength:float=0.5,startingFreq:float=2500,numberofSteps:int=1):
        super().__init__(sourceFile, receiverFiles, silenceCapsLengths) #Was the signal made of steps or was it a constant frequency?
        self.stairFreq = stairFreqHeight #G H I increased in uniform steps, so this is for that, as in what frequency was each step, they went from 2500 to 3000 to 3500 etc
        self.stairLength = stairTimeLength #This is how long each step was, in the scans it was uniform so we should't need to worry about varying lengths
        self.startingFreq = startingFreq #For stairs, this is what the 

class AttenCalculators: #These calculate the attenuation over the entire frequency spectrum 
    def NaiveSubtraction(InitialSignal:Soundfile,FinalSignal:Soundfile,useWindow:bool=False): #Just subtracting the magnitude of the recieved signal from the input signal
        if not InitialSignal.correlated or not FinalSignal.correlated:
            correlate_soundfiles(InitialSignal,FinalSignal)
        if useWindow:#Take fourier of both
            fft_in = InitialSignal.windowFFT 
            fft_final = FinalSignal.windowFFT
        else:
            fft_in = InitialSignal.rawFFT
            fft_final = FinalSignal.rawFFT
        mag_in = convert_to_decibels(fft_in)#Convert to decibels
        mag_final = convert_to_decibels(fft_final)
        return mag_final - mag_in
    def NaiveDivision(InitialSignal:Soundfile,FinalSignal:Soundfile,useWindow:bool=False): #Just dividing the final signal by the initial signal
        if not InitialSignal.correlated or not FinalSignal.correlated:
            correlate_soundfiles(InitialSignal,FinalSignal)
        if useWindow:
            fft_in = InitialSignal.windowFFT #Take fourier of both
            fft_final = FinalSignal.windowFFT
        else:
            fft_in = InitialSignal.rawFFT
            fft_final = FinalSignal.rawFFT
        mag_in = convert_to_decibels(fft_in)#Convert to decibels
        mag_final = convert_to_decibels(fft_final)
        return mag_final/mag_in
    def EstTransferFunction(InitialSignal:Soundfile,FinalSignal:Soundfile,useWindow:bool=False): #This estimates the transfer function of the system, so how much of the signal gets through at a certain frequency
        if not InitialSignal.correlated or not FinalSignal.correlated:
            correlate_soundfiles(InitialSignal,FinalSignal)
        if useWindow:
            fft_in = InitialSignal.windowFFT #Take fourier of both
            fft_final = FinalSignal.windowFFT
        else:
            fft_in = InitialSignal.rawFFT
            fft_final = FinalSignal.rawFFT
        eps = 1e-12 #To avoid division by zero
        atten = fft_final * np.conj(fft_in) / (np.abs(fft_in)**2 + eps)
        return convert_to_decibels(atten)
  
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
    max_len = max(sound1.length, sound2.length) #Find which file is longer
    sound1.soundfile = pad(sound1.soundfile, max_len) #Pad both to same length
    sound2.soundfile = pad(sound2.soundfile, max_len)
    sound1.isCorrelated = True
    sound2.isCorrelated = True

def pad(signal, target_len): #ChatGPT code to pad two signals :)
    pad_width = target_len - len(signal)
    if pad_width > 0:
        if signal.ndim == 1:
            return np.pad(signal, (0, pad_width))
        else:  # stereo or multi-channel
            return np.pad(signal, ((0, pad_width), (0, 0)))
    return signal

def calculate_attenuation(InitialSignal:Soundfile,FinalSignal:Soundfile,precorrelated:bool = False,filterafter:bool = False,kernelsize:float=101,minFreq:float=None,maxFreq:float=None):
    #Precorrelated is for if the correlate_soundfiles has already been run on the files, if not it runs it
    #Filterafter is for whether or not to filter the attenuation after to try to smooth things out
    #kernelsize controls the size of the kernel for filtering, I think its how many cells it averages over, must be odd
    #minFreq is the minimum frequency to be considered, everything below that frequency is discarded

    if precorrelated == False:
        correlate_soundfiles(InitialSignal,FinalSignal)
    fft_in = np.fft.rfft(InitialSignal.soundfile) #Take fourier of both
    fft_final = np.fft.rfft(FinalSignal.soundfile)
    freqs = np.fft.rfftfreq(FinalSignal.length, 1/FinalSignal.samplerate)

    mask = np.ones_like(freqs, dtype=bool)
    if minFreq is not None:
        mask &= freqs >= minFreq
    if maxFreq is not None:
        mask &= freqs <= maxFreq

    mag_in = 20*np.log10(np.abs(fft_in[mask])) #Convert to decibels
    mag_final = 20*np.log10(np.abs(fft_final[mask]))

    FinalSignal.freq = freqs[mask]
    FinalSignal.fft = mag_final

    attenuationsraw = mag_final - mag_in
    
    if filterafter:
        mag_in_filtered = medfilt(mag_in, kernel_size=kernelsize)
        mag_final_filtered = medfilt(mag_final, kernel_size=kernelsize)
        FinalSignal.fft = mag_final_filtered
        return (mag_final_filtered-mag_in_filtered)
    else:
        return attenuationsraw
    
# Doing this in a jank way, delete this upon review, adding a function to calculate attenuation at a specific frequency

def calculate_attenuation_at_freq(InitialSignal:Soundfile,
                                  FinalSignal:Soundfile,
                                  precorrelated:bool = False,
                                  kernelsize:float=101,
                                  desiredfreq:float = 1000):
    #Precorrelated is for if the correlate_soundfiles has already been run on the files, if not it runs it
    #kernelsize controls the size of the kernel for filtering, I think its how many cells it averages over, must be odd
    #minFreq is the minimum frequency to be considered, everything below that frequency is discarded
    freqsraw = np.fft.rfftfreq(InitialSignal.length,1/FinalSignal.samplerate)
    inraw = np.fft.rfft(InitialSignal.soundfile)
    outraw = np.fft.rfft(FinalSignal.soundfile)
    
    InitialSignal.soundfile = butter_bandpass_filter(InitialSignal.soundfile,1500,3500,InitialSignal.samplerate,order=2) #Butterworth 2000 to 3000
    FinalSignal.soundfile = butter_bandpass_filter(FinalSignal.soundfile,1500,3500,FinalSignal.samplerate,order=2)
    if precorrelated == False:
        correlate_soundfiles(InitialSignal,FinalSignal)
    Initial_Norm = InitialSignal.soundfile / np.max(np.abs(InitialSignal.soundfile)) #We should make this a function but whatever
    Final_Norm = FinalSignal.soundfile / np.max(np.abs(FinalSignal.soundfile)) #Normalizing signals
    N = len(Final_Norm)
    freqs = np.fft.rfftfreq(N, 1/FinalSignal.samplerate)
    window = np.hanning(N)
    fft_in = np.fft.rfft(Initial_Norm*window)/N #Take fourier of both
    fft_final = np.fft.rfft(Final_Norm*window)/N

    desiredFreqIndex = np.searchsorted(freqs,desiredfreq,side="right") #Find the first index where the freq is greater than the desired frequency
    eps = 1e-12
    atten = fft_final * np.conj(fft_in) / (np.abs(fft_in)**2 + eps)
    atten = 20*np.log10(medfilt(np.abs(atten),kernel_size=kernelsize))
    atten_raw = outraw * np.conj(inraw) / (np.abs(inraw)**2 + eps)
    atten_raw = 20*np.log10(medfilt(np.abs(atten_raw),kernel_size=kernelsize))
    fft_in = np.abs(fft_in)
    fft_final = np.abs(fft_final)
    #fft_in = medfilt(np.abs(fft_in),kernel_size=kernelsize) #Smooth out ffts
    #fft_final = medfilt(np.abs(fft_final),kernel_size=kernelsize)
    #plt.semilogx(freqs,20*np.log10(fft_in))
    plt.semilogx(freqsraw,inraw,label="in raw")
    plt.semilogx(freqsraw,outraw,label="out raw")
    plt.semilogx(freqsraw,atten_raw,label="atten raw")
    
    
    plt.semilogx(freqs,atten,label="atten")
    print(atten[desiredFreqIndex])
    plt.semilogx(freqs,20*np.log10(fft_in),label="in")
    plt.semilogx(freqs,20*np.log10(fft_final),label="out")
    

    
    plt.legend()
    plt.show()
    
    mag_in = 20*np.log10(np.abs(fft_in[desiredFreqIndex])) #Convert to decibels and take the loudness at certain value
    mag_final = 20*np.log10(np.abs(fft_final[desiredFreqIndex]))
    return atten[desiredFreqIndex]