import AttenuationLibrary as AL
import soundfile as sl
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate
from scipy.signal import medfilt
import os
import re
from PIL import Image

folder = "Second" #Name of folder that holds the files
sourceloudnessfile = "Touching_X.wav" #Name of the file that is recorded right ontop of the piezo, usually called touching

#What this program needs to do is go through each file, calculate the attenuation for each, and then compare
files = os.listdir("FirstScan")
print(files)
pattern = r'^[A-Za-z]_\d{2}_X.wav$' #Change the X in this to change which file is selected


matches = [s for s in files if re.search(pattern, s)] #First get a list of all files that match the format
print(matches)


attenuations = np.zeros(len(matches))
ffts = []
freqs = []
for file in range(len(matches)):
    print(f"Currently processing {matches[file]}")
    baseline = AL.Soundfile(folder+"/"+sourceloudnessfile) #Get the baseline object, making a new one each time because the correlate function changes the object
    measurementfilename = matches[file]
    measurement = AL.Soundfile(folder+"/"+measurementfilename) #Get the measurement file, should be A_00_X then A_01_X etc.
    measurementatten = AL.calculate_attenuation(baseline,measurement,precorrelated=False,filterafter=True,kernelsize=1001,minFreq=1000,maxFreq=7000,sampleFreq=3000) #Attenuation as function of freq
    #We want the most negative because I'm just calculating attenuation as mag_final-mag_in 
    lowestIndex = np.argmax(measurement.fft > 3000)
    attenuations[file] = measurementatten[lowestIndex]
    plt.semilogx(measurement.freq,measurement.fft,label = f"A_{file}_X") 
print(attenuations)
maxatten = np.min(attenuations) #Find the largest atten
attenuations = np.round(attenuations/maxatten*255) #Rescale to 0-255 range
print(attenuations)
plt.legend()
plt.show()

outputimage = Image.fromarray(np.abs(attenuations.astype(np.int8)), "L")
outputimage.save("Hi.bmp")