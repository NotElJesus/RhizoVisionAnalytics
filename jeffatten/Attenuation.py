import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate
from scipy.signal import medfilt
import AttenuationLibrary as AL


input_signal,samplerate = sf.read("2026-03-26_DuoPiezo_OpAmp_Hydrophone08_Sweep_A_Touching.wav")
output_signal2,samplerate = sf.read("2026-03-26_DuoPiezo_OpAmp_Hydrophone08_Sweep_A_NothingBetween.wav")
output_signal,samplerate = sf.read("2026-03-26_DuoPiezo_OpAmp_Hydrophone08_Sweep_A_OnionBetween.wav")
fs = samplerate
plt.plot(input_signal)
plt.plot(output_signal)
plt.show()
# First align the signals
corr = correlate(output_signal,input_signal,mode="full",method="auto")
lags = np.arange(-len(input_signal) + 1, len(output_signal))
lag = lags[np.argmax(np.abs(corr))]#Lag between the input and output
if lag > 0:
    # y starts later → remove its delay
    output_aligned = output_signal[lag:]
    input_aligned = input_signal[:len(output_aligned)]
else:
    # x starts later
    input_aligned = input_signal[-lag:]
    output_aligned = output_signal[:len(input_aligned)]
sf.write("Input_Aligned1.wav",input_aligned,samplerate)
sf.write("Output_Aligned1.wav",output_aligned,samplerate)
fft_in = np.fft.rfft(input_aligned)
fft_out = np.fft.rfft(output_aligned)

mag_in = 20*np.log10(np.abs(fft_in))
mag_out = 20*np.log10(np.abs(fft_out))

attenuationsignal1 = mag_out - mag_in
freqs = np.fft.rfftfreq(len(input_aligned), 1/fs)
'''
plt.plot(input_aligned, label='x')
plt.plot(output_aligned, label='y')
plt.legend()
plt.show()
'''
mag_out_smooth = medfilt(mag_out, kernel_size=11)
mag_out_onion_smooth = mag_out_smooth
mag_out_onion = mag_out
mag_in_smooth = medfilt(mag_in, kernel_size=11)
mag_in_onion = mag_in
mag_in_onion_smooth = mag_in_smooth

plt.semilogx(freqs,mag_out_smooth,label = "Output")
plt.semilogx(freqs,mag_in_smooth,label = "Input")
plt.legend()
plt.title("Onion Between")
plt.show() 
plt.semilogx(freqs, attenuationsignal1,label = "Onion")

corr = correlate(output_signal2,input_signal,mode="full",method="fft")
lags = np.arange(-len(input_signal) + 1, len(output_signal2))
lag = lags[np.argmax(np.abs(corr))]#Lag between the input and output
if lag > 0:
    # y starts later → remove its delay
    output_aligned2 = output_signal2[lag:]
    input_aligned2 = input_signal[:len(output_aligned2)]
else:
    # x starts later
    input_aligned2 = input_signal[-lag:]
    output_aligned2 = output_signal2[:len(input_aligned2)]
sf.write("Input_Aligned2.wav",input_aligned2,samplerate)
sf.write("Output_Aligned2.wav",output_aligned2,samplerate)
fft_in2 = np.fft.rfft(input_aligned2)
fft_out2 = np.fft.rfft(output_aligned2)

mag_in_2 = 20*np.log10(np.abs(fft_in2))
mag_out_2 = 20*np.log10(np.abs(fft_out2))
attenuationsignal2 = mag_out_2 - mag_in_2
freqs = np.fft.rfftfreq(len(input_aligned), 1/fs)
freqs2 = np.fft.rfftfreq(len(input_aligned2), 1/fs)
mag_in_2_smooth = medfilt(mag_in_2, kernel_size=11)
mag_out_2_smooth = medfilt(mag_out_2, kernel_size=11)

# attendiff = attenuationsignal1 - attenuationsignal2
'''
plt.semilogx(freqs,mag_out_2,label = "Output (Nothing)")
plt.semilogx(freqs,mag_out_onion,label = "Output (Onion)")
plt.semilogx(freqs,mag_in_2,label = "Input (Nothing)")
plt.semilogx(freqs,mag_in_onion,label = "Input (Onion)")
plt.legend()
plt.title("Raw Data")
plt.show() 
plt.semilogx(freqs,mag_out_2_smooth,label = "Output (Nothing)")
plt.semilogx(freqs,mag_out_onion_smooth,label = "Output (Onion)")
plt.semilogx(freqs,mag_in_smooth,label = "Input (Nothing)")
plt.semilogx(freqs,mag_in_onion_smooth,label = "Input (Onion)")
plt.legend()
plt.title("Smoothed Data")
plt.show() 
'''
#plt.semilogx(freqs,attenuationsignal1,label = "Output (Nothing)")
#plt.semilogx(freqs2,attenuationsignal2,label = "Output (Onion)")
attenuationsignal1_smooth = medfilt(attenuationsignal1, kernel_size=99)
attenuationsignal2_smooth = medfilt(attenuationsignal2, kernel_size=99)
plt.semilogx(freqs,attenuationsignal1_smooth,label = "Attenuation (Nothing)")
plt.semilogx(freqs2,attenuationsignal2_smooth,label = "Attenuation (Onion)")
plt.legend()
plt.xlabel("Freq (Hz)")
plt.ylabel("dB")
plt.show()


