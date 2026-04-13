import AttenuationLibrary as AL
import soundfile as sl
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate
from scipy.signal import medfilt

kernelsize = 301
testinput_nothing = AL.Soundfile("2026-03-26_DuoPiezo_OpAmp_Hydrophone08_Sweep_A_TouchingFiltered.wav")
testoutput_nothing = AL.Soundfile("2026-03-26_DuoPiezo_OpAmp_Hydrophone08_Sweep_A_NothingBetween_Filterd.wav")

print(testinput_nothing.soundfile)

AL.correlate_soundfiles(testoutput_nothing,testinput_nothing)
print(f"testoutput length:{testoutput_nothing.length},test1input length:{testinput_nothing.length}")
freqs = np.fft.rfftfreq(testinput_nothing.length, 1/testinput_nothing.samplerate) #Should frequencies be put in to some other class or something??
atten_nothing_raw = AL.calculate_attenuation(testinput_nothing,testoutput_nothing)
atten_nothing_filtered = AL.calculate_attenuation(testinput_nothing,testoutput_nothing,filterafter=True)
indexforfreq = np.argmax(freqs > 2000)
print(f"--Nothing Analysis-- \nIndex: {indexforfreq}\nFreq: {freqs[indexforfreq]}\nBinsize: {freqs[indexforfreq+kernelsize]-freqs[indexforfreq]}\nAttenuation = f-i\nRaw Signal Attenuation: {atten_nothing_raw[indexforfreq]}\nFiltered Signal Attenuation: {atten_nothing_filtered[indexforfreq]}") #Get the first index

plt.semilogx(freqs,atten_nothing_raw)
plt.show()

#Plotting raw dB v. Freq 
plt.semilogx(freqs,AL.get_mag(testinput_nothing),label = "Input signal") 
plt.semilogx(freqs,AL.get_mag(testoutput_nothing),label = "Output signal")
plt.title("Raw dB v. Freq (Nothing)")
plt.legend()
plt.xlabel("Freq (Hz)")
plt.ylabel("dB")
plt.show()

#Plotting filtered dB v. Freq
plt.semilogx(freqs,medfilt(AL.get_mag(testinput_nothing),kernelsize),label = "Input signal")
plt.semilogx(freqs,medfilt(AL.get_mag(testoutput_nothing),kernelsize),label = "Output signal")
plt.title("Filtered dB v. Freq (Nothing)")
plt.legend()
plt.xlabel("Freq (Hz)")
plt.ylabel("dB")
plt.show()
# Repeating above but for onion
testinput_onion = AL.Soundfile("2026-03-26_DuoPiezo_OpAmp_Hydrophone08_Sweep_A_TouchingFiltered.wav")
testoutput_onion = AL.Soundfile("2026-03-26_DuoPiezo_OpAmp_Hydrophone08_Sweep_A_OnionBetweenFiltered.wav")

AL.correlate_soundfiles(testoutput_onion,testinput_onion)
freqs_onion = np.fft.rfftfreq(testinput_onion.length, 1/testinput_onion.samplerate) #Should frequencies be put in to some other class or something??
atten_onion_raw = AL.calculate_attenuation(testinput_onion,testoutput_onion)
atten_onion_filtered = AL.calculate_attenuation(testinput_onion,testoutput_onion,filterafter=True)
indexforfreq = np.argmax(freqs_onion > 2000)
print(f"--Onion Analysis-- \nIndex: {indexforfreq}\nFreq: {freqs_onion[indexforfreq]}\nAttenuation = f - i \nRaw Signal Attenuation: {atten_onion_raw[indexforfreq]}\nFiltered Signal Attenuation: {atten_nothing_filtered[indexforfreq]}")

#Plotting raw dB v. Freq 
plt.semilogx(freqs_onion,AL.get_mag(testinput_onion),label = "Input signal") 
plt.semilogx(freqs_onion,AL.get_mag(testoutput_onion),label = "Output signal")
plt.title("Raw dB v. Freq (Onion)")
plt.legend()
plt.xlabel("Freq (Hz)")
plt.ylabel("dB")
plt.show()

#Plotting filtered dB v. Freq
plt.semilogx(freqs_onion,medfilt(AL.get_mag(testinput_onion),kernelsize),label = "Input signal")
plt.semilogx(freqs_onion,medfilt(AL.get_mag(testoutput_onion),kernelsize),label = "Output signal")
plt.title("Filtered dB v. Freq (Onion)")
plt.legend()
plt.xlabel("Freq (Hz)")
plt.ylabel("dB")
plt.show()

# Overall plot
plt.semilogx(freqs,medfilt(AL.get_mag(testinput_nothing),kernelsize),label = "Input signal (Nothing)")
plt.semilogx(freqs,medfilt(AL.get_mag(testoutput_nothing),kernelsize),label = "Output signal (Nothing)")
plt.semilogx(freqs_onion,medfilt(AL.get_mag(testinput_onion),kernelsize),label = "Input signal (Onion)",linestyle="dashed")
plt.semilogx(freqs_onion,medfilt(AL.get_mag(testoutput_onion),kernelsize),label = "Output signal (Onion)",linestyle="dashed")
plt.legend()
plt.xlabel("Freq (Hz)")
plt.ylabel("dB")
plt.title("Filtered dB v. Freq")
plt.show()

#Overall Atten Plot
plt.semilogx(freqs,atten_nothing_raw,label = "Nothing Difference")
plt.semilogx(freqs_onion,atten_onion_raw,label = "Onion Difference")
plt.legend()
plt.xlabel("Freq (Hz)")
plt.ylabel("dB")
plt.title("Raw dB v. Freq delta signal")
plt.show()
#Filter it
plt.semilogx(freqs,medfilt(atten_nothing_raw,kernelsize),label = "Nothing Difference")
plt.semilogx(freqs_onion,medfilt(atten_onion_raw,kernelsize),label = "Onion Difference")
plt.legend()
plt.xlabel("Freq (Hz)")
plt.ylabel("dB")
plt.title("Filtered dB v. Freq delta signal")
plt.show()