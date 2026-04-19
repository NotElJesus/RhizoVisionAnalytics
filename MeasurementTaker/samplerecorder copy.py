from WF_SDK import device, scope, wavegen, tools, error   # import instruments

import matplotlib.pyplot as plt   # needed for plotting
from time import sleep            # needed for delays
from time import time

from scipy.io.wavfile import write
import numpy as np
import sounddevice as sd #Used to make recordings

#print(sd.query_devices())# do this to look at ids of devices, you want the one that is (ZOOM AMS-22 AUDIO) (2 in, 0 out)
deviceid = 32
fs = 44100 #sample rate
duration = 5 #duration of recording in seconds
maxamplitude = (7+1/3)/2 #Because we are getting saturation at 22 Vs, and the amplification is 3 on the op amp, also this is mean to peak 
sd.default.samplerate = fs
sd.default.device = (None, deviceid)
sd.default.channels = 1
'---Defining Recording function---'
def record(samplenum:int = 9999,rotationletter:str = 'Z',seconds:float=5):
    steptimelength = 0.5
    Yfreqs = list(range(500,6500,500))
    
    '--- Setting Up Recording Device ---'
    recording1name = f"Outputs/{rotationletter}_{int(samplenum):02d}_G.wav"
    print(f"Recording {recording1name}...",end="")
    myrecording = sd.rec(int((2+steptimelength*len(Yfreqs))* fs))
    '--- G ---'
    sleep(1)
    for freq in Yfreqs:
        wavegen.generate(device_data, channel=1, function=wavegen.function.sine, offset=0, frequency=freq, amplitude=maxamplitude)
        sleep(steptimelength)
        wavegen.close(device_data)
    # reset the wavegen
    wavegen.close(device_data)
    sd.wait() #Wait until recording is over if it needs to be
    print(f"{recording1name} done, saving ...",end="")
    write(f"{recording1name}",fs,myrecording)
    print(f"{recording1name} saved!")
    
    '--- H ---' #Square wave stepping
    recording1name = f"Outputs/{rotationletter}_{int(samplenum):02d}_H.wav"
    print(f"Recording {recording1name}...",end="")
    myrecording = sd.rec(int((2+steptimelength*len(Yfreqs))* fs))
    
    sleep(1)
    for freq in Yfreqs:
        wavegen.generate(device_data, channel=1, function=wavegen.function.square, offset=0, frequency=freq, amplitude=maxamplitude)
        sleep(steptimelength)
    # reset the wavegen
    wavegen.close(device_data)
    sd.wait() #Wait until recording is over if it needs to be
    print(f"{recording1name} done, saving ...",end="")
    write(f"{recording1name}",fs,myrecording)
    print(f"{recording1name} saved!")
    
    '--- I ---' #Triangle wave stepping
    recording1name = f"Outputs/{rotationletter}_{int(samplenum):02d}_I.wav"
    print(f"Recording {recording1name}...",end="")
    myrecording = sd.rec(int((2+steptimelength*len(Yfreqs))* fs))
    
    sleep(1)
    for freq in Yfreqs:
        wavegen.generate(device_data, channel=1, function=wavegen.function.triangle, offset=0, frequency=freq, amplitude=maxamplitude)
        sleep(steptimelength)
    # reset the wavegen
    wavegen.close(device_data)
    sd.wait() #Wait until recording is over if it needs to be
    print(f"{recording1name} done, saving ...",end="")
    write(f"{recording1name}",fs,myrecording)
    print(f"{recording1name} saved!")
    '--- J ---' #Sinewave 2.5khz 3 seconds
    recording1name = f"Outputs/{rotationletter}_{int(samplenum):02d}_J.wav"
    print(f"Recording {recording1name}...",end="")
    myrecording = sd.rec(int(5*fs))
    
    sleep(1)
    wavegen.generate(device_data, channel=1, function=wavegen.function.sine, offset=0, frequency=2.5e03, amplitude=maxamplitude)
    # reset the wavegen
    sleep(3)
    wavegen.close(device_data)
    sd.wait() #Wait until recording is over if it needs to be
    print(f"{recording1name} done, saving ...",end="")
    write(f"{recording1name}",fs,myrecording)
    print(f"{recording1name} saved!")
    '--- K ---' #Squarewave 2.5khz 3 seconds
    recording1name = f"Outputs/{rotationletter}_{int(samplenum):02d}_K.wav"
    print(f"Recording {recording1name}...",end="")
    myrecording = sd.rec(int(5*fs))
    
    sleep(1)
    wavegen.generate(device_data, channel=1, function=wavegen.function.square, offset=0, frequency=2.5e03, amplitude=maxamplitude)
    # reset the wavegen
    sleep(3)
    wavegen.close(device_data)
    sd.wait() #Wait until recording is over if it needs to be
    print(f"{recording1name} done, saving ...",end="")
    write(f"{recording1name}",fs,myrecording)
    print(f"{recording1name} saved!")
    '--- L ---' #Trianglewave 2.5khz 3 seconds
    recording1name = f"Outputs/{rotationletter}_{int(samplenum):02d}_L.wav"
    print(f"Recording {recording1name}...",end="")
    myrecording = sd.rec(int(5*fs))
    
    sleep(1)
    wavegen.generate(device_data, channel=1, function=wavegen.function.triangle, offset=0, frequency=2.5e03, amplitude=maxamplitude)
    # reset the wavegen
    sleep(3)
    wavegen.close(device_data)
    sd.wait() #Wait until recording is over if it needs to be
    print(f"{recording1name} done, saving ...",end="")
    write(f"{recording1name}",fs,myrecording)
    print(f"{recording1name} saved!")
'--- Setting up Noise Maker ---'
try:
    # connect to the device
    device_data = device.open()
    # use instruments here
    sleep(1)
    
except error as e:
    print(e)
    # close the connection
    device.close(device.data)

'---Control---'

done = False
samplenum = 99
rotationlet = 'Z'
while done == False:
    print(f"Current config is {rotationlet}_{int(samplenum):02d}\n-Type in a letter to set what rotation we are in\n-Type a number to say which sample we're taking\n-Type \"Run\" to run the collection process\n-Type Baseline to collect a 30 second recording with no noise\n-Type Exit to leave ")
    choice = input("What would you like to do? \n")
    if choice.isnumeric():
        samplenum = int(choice)
    elif choice == "Run":
        print("Setting up wave generator...")
        scope.open(device_data)
        sleep(1) #Stabilizing
        record(samplenum,rotationlet,5)
    elif choice == "Exit":
        done = True
    elif choice == "Baseline":
        baselinerecording = sd.rec(int(30*fs))
        sd.wait()
        write(f"Outputs/{rotationlet}_Baseline.wav",fs,baselinerecording)
    else:
        rotationlet = choice
        print(f"Set rotation letter to \"{choice}\"")
print("Bye!")
# close the connection
device.close(device_data)