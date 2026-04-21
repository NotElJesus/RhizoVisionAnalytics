import AttenuationLibrary as AL
import soundfile as sl
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate
from scipy.signal import medfilt
import os
import re
from PIL import Image


print(f"Current working dir:{os.getcwd()}, should be in jeffatten folder")
#The goal of this script is to take the files from the Second Scan, thus the audio files in 'SecondScan'
#And make the sinogram, for G,H,I,J,K,L
#What it needs to do is go through every file for a scanning method, compare it to the 'touching' file, which was recorded right next to the piezodisc
#Take the attenuation, a higher attenuation means more stuff in between so it should be whiter in the sinogram than the other ones

folder = "SecondScan/Outputs" #Name of folder that holds the files
sourceloudnessfile = "SecondScan/Outputs/Touchingagain_12_L.wav" #Name of the file that is recorded right ontop of the piezo, usually called touching


files = os.listdir(folder)
pattern = r'^[A-Za-z]_\d{2}_L.wav$' #Change the last letter in this to change which file is selected
print(files)

matches = [s for s in files if re.search(pattern, s)] #First get a list of all files that match the format
print(matches)


attenuations = np.zeros(len(matches))
print(len(matches))
ffts = []
freqs = []
for file in range(len(matches)):
    measurementfilename = matches[file]
    print(f"Currently processing {matches[file]} at {folder+"/"+measurementfilename}")
    baseline = AL.Soundfile(sourceloudnessfile) #Get the baseline object, making a new one each time because the correlate function changes the object
    measurement = AL.Soundfile(folder+"/"+measurementfilename) #Get the measurement file, should be A_00_X then A_01_X etc.
    measurementatten = AL.calculate_attenuation_at_freq(baseline,measurement,precorrelated=False,filterafter=True,kernelsize=1001,desiredfreq=2500) #Attenuation as function of freq
    attenuations[file] = measurementatten
print(attenuations)
maxatten = np.max(attenuations) #Find the largest atten
minatten = np.min(attenuations)
normalized_attenuations = 1 - (attenuations - minatten)/(maxatten-minatten) #The higher attenuation values actually mean less stuff between so we want those to be ba


p_low = np.percentile(normalized_attenuations, 5)
p_high = np.percentile(normalized_attenuations, 95)
# Percentile clipping + gamma scaling to try to get the center differences to be clearer
normalized_attenuations = np.clip(normalized_attenuations, p_low, p_high)
normalized_attenuations = (normalized_attenuations - p_low) / (p_high - p_low)
normalized_attenuations = normalized_attenuations ** 1.5   # tweak this
normalized_attenuations = normalized_attenuations * 255

image_attenuations = np.round(normalized_attenuations).astype(int) #Recale to 0-255 range
print(image_attenuations)
outputimage = Image.fromarray(np.abs(image_attenuations.astype(np.int8)), "L")
outputimage.save("Hi.bmp")

np.save("Attenuations_L3.npy",image_attenuations) #Saves for use in reconstruction