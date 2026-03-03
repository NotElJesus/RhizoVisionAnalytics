import numpy as np
from PIL import Image

def MakeSinogram(inputarray, filename, numberofrotations, detectors):
    max = np.max(inputarray)
    if max > 0:
        scale = 255 / max
    else:
        scale = 1
    output = np.abs(np.reshape((np.floor(inputarray * scale)).astype(np.int8), (numberofrotations, detectors)))
    outputimage = Image.fromarray(output, "L")
    outputimage.save(filename)