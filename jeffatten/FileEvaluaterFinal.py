import AttenuationLibrary as AL
import soundfile as sl
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate
from scipy.signal import medfilt
import os
import re
from PIL import Image
import rayVisualization as rv #TODO: make this actually refer to the one in Jesus' Work 

os.chdir("jeffatten") #Change to the folder that holds the files, should be jeffatten

print(f"Current working dir:{os.getcwd()}, should be in jeffatten folder")
#The goal of this script is to take the files from the Second Scan, thus the audio files in 'SecondScan'
#And make the sinogram, for G,H,I,J,K,L
#What it needs to do is go through every file for a scanning method, compare it to the 'touching' file, which was recorded right next to the piezodisc
#Take the attenuation, a higher attenuation means more stuff in between so it should be whiter in the sinogram than the other ones

folder = "SecondScan/Outputs" #Name of folder that holds the files
scanningletter = "K"
sourceloudnessfile = f"SecondScan/Outputs/Touchingagain_12_{scanningletter}.wav" #Name of the file that is recorded right ontop of the piezo, usually called touching


files = os.listdir(folder)

matches = []
#The matches list need to be reversed because the A matrix maker makes it from the bottom raw, while in the lab we started from the top so it needs to be in reverse numeric order
for i in ["A","B","C","D","E","F"]:
    print(i)
    pattern = r'^[' + i + r']_\d{2}_' + scanningletter + r'.wav'
    print(pattern)
    templist = [s for s in files if re.search(pattern, s)]
    templist.reverse()
    matches.append(templist)


attenuations = []
ScalebyRows = False #Whether to scale attenuations by row, instead of the full image at once
for filerange in matches:
    tempattenuations = []
    for file in filerange:
        measurementfilename = file
        print(f"Currently processing {measurementfilename} at {folder+"/"+measurementfilename}")
        baseline = AL.Soundfile(sourceloudnessfile) #Get the baseline object, making a new one each time because the correlate function changes the object
        measurement = AL.Soundfile(folder+"/"+measurementfilename) #Get the measurement file, should be A_00_X then A_01_X etc.
        baseline.kernelSize = 501
        measurement.kernelSize = 501
        #plt.semilogx(measurement.freqs, 20*np.log10(np.abs(measurement.rawFFT)), label=file)
        #measurementatten = AL.calculate_attenuation_at_freq(baseline,measurement,precorrelated=False,filterafter=True,kernelsize=11,desiredfreq=2500) #Attenuation as function of freq
        measurementatten = AL.AttenCalculators.ScalarizeTransferFunction(baseline,measurement,desiredFreq=2500,useWindow=True) #Scalar attenuation value
        tempattenuations.append(measurementatten)
    #plt.legend()
    #plt.show()
    print(attenuations)
    if ScalebyRows:
        tempattenuations = np.asarray(tempattenuations)
        tempattenuations = -tempattenuations
        p_low = np.percentile(tempattenuations, 25)
        p_high = np.percentile(tempattenuations, 75)
        scaled = 255 * (tempattenuations - p_low) / (p_high - p_low)
        scaled = np.clip(scaled, 0, 255).astype(np.uint8)
        attenuations.extend(scaled**0.7) #A little contrast boost
    else:
        attenuations.extend(tempattenuations)
attenuations = np.asarray(attenuations)
if not ScalebyRows:
    attenuations = -attenuations
    lo = np.percentile(attenuations, 10)
    hi = np.percentile(attenuations, 90)
    scaled = 255 * (attenuations - lo) / (hi - lo)
    scaled = np.clip(scaled, 0, 255).astype(np.uint8)
    attenuations = scaled**0.7 #Contrast boosting here 
savename = f"Attenuations_{scanningletter}4.npy"
rv.MakeSinogram(attenuations,f"OutputStorage/Sinograms/{savename}.bmp",6,13) #Makes and saves the sinogram as a bmp 
np.save(f"OutputStorage/NumpyArrays/{savename}", attenuations) #Saves for use in reconstruction
print(f"Done, saved as OutputStorage/NumpyArrays/{savename}")