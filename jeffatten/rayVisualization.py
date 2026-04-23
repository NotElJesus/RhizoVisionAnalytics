import numpy as np
from PIL import Image
from geometry import Ray
from matplotlib.axes import Axes

def MakeSinogram(inputarray, filename, numberofrotations, detectors):
    max = np.max(inputarray)
    if max > 0:
        scale = 255 / max
    else:
        scale = 1
    output = np.abs(np.reshape((np.floor(inputarray * scale)).astype(np.int8), (numberofrotations, detectors)))
    outputimage = Image.fromarray(output, "L")
    outputimage.save(filename)

def PlotAMatrix(AMatrix, ax: Axes): 
    rays: list[Ray] = AMatrix.Rays
    for i in rays:
        ax.plot([i.point1.x, i.point2.x], [i.point1.y, i.point2.y])