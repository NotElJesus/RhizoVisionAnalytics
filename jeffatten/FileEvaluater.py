import AttenuationLibrary as AL
import soundfile as sl
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate
from scipy.signal import medfilt
import os
import re

files = os.listdir("FirstScan")
print(files)
pattern = r'(A_)..(_X)*'

matches = [s for s in files if re.search(pattern, s)]
print(matches)