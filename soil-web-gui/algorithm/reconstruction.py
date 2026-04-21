import os
import numpy as np
from PIL import Image
from algorithm.config import WorkingFolder, OutputFolder, shortfile, Rotations
from algorithm.visualization import MakeSinogram

class xMatrix:
    def __init__(self, Amatrix, ReconWidth: int = 3):
        self.ReconstructionWidth = ReconWidth
        self.xarray = np.zeros(self.ReconstructionWidth ** 2, dtype=float)

        self.Amatrix = Amatrix
        print("Using diagonal C and R vectors.")

        self.Pmatrix = np.empty([1, 1])
        self.iteration = 0
        self.maxiteration = 0
        self.detectors = 0
        self.frames = []

    def ImportPVector(self, p):
        self.Pmatrix = p
        print("P vector copied")

    def SetReconstruction(self, NewWidth=0):
        if NewWidth > 0:
            self.ReconstructionWidth = NewWidth
            self.APicture = Image.new("1", (self.ReconstructionWidth, self.ReconstructionWidth))
            self.xarray = np.zeros(self.ReconstructionWidth ** 2)
        else:
            print("ERROR: Reconstruction grid must be larger than 0!")

    def SetDetectors(self, detectors):
        self.detectors = detectors

    def SetIterations(self, iter):
        if iter >= 0:
            self.maxiteration = iter
        else:
            print("ERROR: Use a positive number")

    def Iterate(self):
        forwardProjection = self.Amatrix.AMatrix @ self.xarray
        result = self.Pmatrix - forwardProjection
        resultScaled = self.Amatrix.Rdiag * result
        resultATrans = self.Amatrix.AMatrix_Transpose @ resultScaled
        backProjection = self.Amatrix.Cdiag * resultATrans
        self.xarray = self.xarray + backProjection

    def DoAllIterations(self, computerSinograms=False):
        for i in range(0, self.maxiteration):
            print(f"Starting iteration {i+1} of {self.maxiteration}...", end='')
            self.Iterate()

            print(f"iteration {i} calculated, saving image...", end='')
            self.SaveImage(os.path.join(WorkingFolder, f"{shortfile}Iteration{i}.bmp"))
            print(f"image saved! Iteration {i} completed!")

            if computerSinograms:
                MakeSinogram(
                    np.matmul(self.Amatrix.AMatrix, self.xarray),
                    os.path.join(WorkingFolder, f"{shortfile}SinogramIteration{i}.bmp"),
                    np.size(Rotations),
                    self.detectors
                )

        print("Saving final output image")
        self.SaveImage(os.path.join(OutputFolder, f"{shortfile}.png"))
        print("Final image saved!")

    def SaveImage(self, filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        output = np.reshape(
            np.abs(self.xarray),
            (self.ReconstructionWidth, self.ReconstructionWidth)
        )

        max_val = np.max(output)
        if max_val > 0:
            scale = 255 / max_val
        else:
            scale = 1

        outputimage = Image.fromarray(
            np.clip(np.floor(output * scale), 0, 255).astype(np.uint8),
            "L"
        )

        self.frames.append(outputimage)
        outputimage.save(filename)

    def SaveGif(self, filename=None):
        if not self.frames:
            print("No frames to save!")
            return

        if filename is None:
            filename = os.path.join(OutputFolder, f"{shortfile}.gif")

        self.frames[0].save(
            filename,
            save_all=True,
            append_images=self.frames[1:],
            optimize=False,
            duration=50,
            loop=0
        )
        print("GIF saved!")
