import numpy as np

filename = "lysimeterTest_gray.png"
shortfile = "ABSLysimeterTest"

InputFolder = "Input/"
OutputFolder = "Output/"
WorkingFolder = "Workingdir/"

plotmatlab = True

ReconstructionWidth = 64

Detectors = 12
TotalIterations = 100

SourceLocation = 30
DetectorsDist = 65
FanAngleDegrees = 60

Rotations = np.arange(0, 181, 1)