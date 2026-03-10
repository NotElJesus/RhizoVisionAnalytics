import numpy as np
from PIL import Image
from config import WorkingFolder, OutputFolder, shortfile, Rotations
from visualization import MakeSinogram

# The x matrix should be done after the A matrix is constructed and this is the 
# class that manages everything related to the x matrix. The x matrix is the unknown 
# image being updated throughout the iterations and trying to get the original image. 
class xMatrix:
    def __init__(self, Amatrix, ReconWidth:int = 3):
        self.ReconstructionWidth = ReconWidth
        # Initializes the array for the xmatrix
        self.xarray = np.zeros(self.ReconstructionWidth ** 2, dtype=float)

        self.Amatrix = Amatrix
        print("Using diagonal C and R vectors.")

        # The P matrix starts off empty but should be imported right away.
        self.Pmatrix = np.empty([1,1])

        self.iteration = 0
        self.maxiteration = 0 
        self.detectors = 0

        # This is used to store frames, which will be used to create a GIF later.
        self.frames = []

    # The basic stuff that needs to be setup before hand as the previous programs have done.
    def ImportPVector(self, p):
        # Note From Jeff: "This is wasteful but its python so who cares"
        self.Pmatrix = p
        print("P vector copied")

    # Allows the Reconstruction width to be changed if it is necessary
    # Note from Jeff: "Don't know if this will be used"
    def SetReconstruction(self, NewWidth = 0):
        if NewWidth > 0: 
            self.ReconstructionWidth = NewWidth
            self.APicture = Image.new("1",(self.ReconstructionWidth, self.ReconstructionWidth))
            self.xarray = np.zeros(self.ReconstructionWidth**2)
        else:
            print("ERROR: Reconstruction grid must be larger than 0!")

    def SetDetectors(self, detectors):
        self.detectors = detectors

    # Set the total number of iterations to do during the program
    def SetIterations(self, iter):
        if iter >= 0:
            self.maxiteration = iter
        else:
            print("ERROR: Use a positive number")

    # Iterate Function
    def Iterate(self):
        # residual in projection space
        # Forward projection by multiplied AMatrix with the x matrix.
        forwardProjection = (self.Amatrix.AMatrix @ self.xarray)

        # Then the result of the forward projection is subtracted from the original projections
        result = self.Pmatrix - forwardProjection

        # Back projection
        # First apply R vector using element-wise multiplication, then the Amatrix transpose 
        # using matrix multiplication and then apply the C vector using element-wise again.
        resultScaled = self.Amatrix.Rdiag * result
        resultATrans = self.Amatrix.AMatrix_Transpose @ resultScaled
        backProjection = self.Amatrix.Cdiag * resultATrans

        # The back projection is now added to the x matrix. 
        self.xarray = self.xarray + backProjection

    # Does all the iterations. 
    def DoAllIterations(self, computerSinograms = False): 
    
        for i in range(0, self.maxiteration):
            print(f"Starting iteration {i+1} of {self.maxiteration}...", end='')
            self.Iterate()

            print(f"iteration {i} calculated, saving image...",end='')
            self.SaveImage(WorkingFolder+f"{shortfile}Iteration{i}.bmp")

            print(f"image saved! Iteration {i} completed!")
            
            #TODO: Disable for performance, just comment line out
            if computerSinograms:
                MakeSinogram(np.matmul(self.Amatrix.AMatrix,self.xarray),WorkingFolder+f"{shortfile}SinogramIteration{i}.bmp",np.size(Rotations),self.detectors)
        
        print(f"Saving final output image")
        self.SaveImage(OutputFolder+f"{shortfile}.bmp")
        
        print(f"Final image saved!")

    # Saving Stuff Section
    # Function to save the image and add it to frames to make a gif later
    def SaveImage(self,filename):
        output = np.reshape(np.abs(self.xarray).astype(np.int8),(self.ReconstructionWidth,self.ReconstructionWidth))
        max = np.max(output)
        if max > 0:
            scale = 255/max
        else:
            scale = 1
        outputimage = Image.fromarray(np.floor(output*scale).astype(np.int8),"L")
        self.frames.append(outputimage)
        outputimage.save(filename)

    def SaveGif(self,filename=OutputFolder+f"{shortfile}.gif"):
        self.frames[0].save(filename,
               save_all = True, append_images = self.frames[1:],
               optimize = False, duration = 50,loop = 0)
        print("GIF saved!")