from WF_SDK import device, scope, wavegen, tools, error   # import instruments

import matplotlib.pyplot as plt   # needed for plotting
from time import sleep            # needed for delays
from time import time


from scipy.io.wavfile import write
import numpy as np
import sounddevice as sd #Used to make recordings

#print(sd.query_devices()) do this to look at ids of devices, you want the one that is (ZOOM AMS-22 AUDIO) (2 in, 0 out)
'--- Setting Up Recording Device ---'
deviceid = 32
fs = 44100 #sample rate
duration = 5 #duration of recording in seconds
sd.default.samplerate = fs
sd.default.device = deviceid
sd.default.channels = 1
'--- Setting up Noise Maker ---'
try:
    # connect to the device
    device_data = device.open()
    
    

    """-----------------------------------"""

    # initialize the scope with default settings
    scope.open(device_data)
    # use instruments here
    sleep(1)
    

except error as e:
    print(e)
    # close the connection
    device.close(device.data)



'--- Recording ---'
myrecording = sd.rec(int(duration * fs))


wavegen.generate(device_data, channel=1, function=wavegen.function.square, offset=0, frequency=4e03, amplitude=4)
sleep(1)
wavegen.generate(device_data, channel=1, function=wavegen.function.square, offset=0, frequency=3e03, amplitude=4)
sleep(1)
wavegen.generate(device_data, channel=1, function=wavegen.function.square, offset=0, frequency=4e03, amplitude=4)
sleep(1)
# reset the wavegen
wavegen.close(device_data)
sd.wait() #Wait until recording is over if it needs to be
write("test2.wav",fs,myrecording)
'--- shutdown ---'

# close the connection
device.close(device_data)