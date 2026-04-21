import os
import numpy as np

filename = "basicfreddy_gray.png"
shortfile = "ABSTestingFred"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

InputFolder = os.path.join(BASE_DIR, "backend", "uploads")
OutputFolder = os.path.join(BASE_DIR, "backend", "Output")
WorkingFolder = os.path.join(BASE_DIR, "backend", "Workingdir")

os.makedirs(InputFolder, exist_ok=True)
os.makedirs(OutputFolder, exist_ok=True)
os.makedirs(WorkingFolder, exist_ok=True)

plotmatlab = True

ReconstructionWidth = 32
Detectors = 21
Rotations = np.arange(0, 181, 1)