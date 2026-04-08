import numpy as np
import math
from PIL import Image, ImageDraw

from config import Detectors, Rotations, WorkingFolder
from geometry import point, boundingBox, Ray

# The A matrix class of the program.
# Variables in the initialization definition:
#   ReconstructionWidth - The grid dimensions of how it will be built
#   Detectors - Number of Detectors used in the reconstruction
#   Rotations - Storing the number of rotations done in the program
#   APicture - Stores a picture that will be used to create a row of A
#   Rays - To store the ray objects that make up the detector
#   Center - This -0.5 is to align with the center of the pixel rather than the bottom left of a pixel
#   boundingBoxPoints - Creates the 
class AMatrix:

    def __init__(self):
        self.ReconstructionWidth = 3
        self.Detectors = Detectors 
        self.Rotations = list(Rotations)
        self.APicture = Image.new("1", (self.ReconstructionWidth, self.ReconstructionWidth))
        self.Rays: list[Ray] = []
        self.center:point = point(self.ReconstructionWidth/2- 0.5, self.ReconstructionWidth/2 - 0.5)
        self.boundingBoxPoints = boundingBox(self.ReconstructionWidth, self.ReconstructionWidth)

    # Setting definitions for the initialized variables
    # Pretty self explanatory definitions 
    # Note from Jeff: "Should probably use the property decorator but this also works the same"
    def SetReconstruction(self, NewWidth):
        if NewWidth > 0: 
            self.ReconstructionWidth = NewWidth
            self.APicture = Image.new("1",(self.ReconstructionWidth,self.ReconstructionWidth))
            self.center = point(self.ReconstructionWidth/2-.5,self.ReconstructionWidth/2-0.5)
        else:
            print("ERROR: Reconstruction grid must be larger than 0!")

    def SetDetectors(self, NewDetectors):
        if NewDetectors > 0:
            self.Detectors = NewDetectors
            print(f"Now using {self.Detectors} detectors")
        else:
            print("ERROR: There must be more than 0 detectors!")

    def UpdateBoundingBox(self,newWidth:float=2,newHeight:float=2):
        #updates the bounding box with the assumption it should be centered about the center of the image, and scaled by square root 2
        self.boundingBoxPoints.changeSize(newWidth,newHeight)
        centerofImage = point(newWidth/2,newHeight/2,0)
        self.boundingBoxPoints.scaleAboutPoint(scalex=math.sqrt(2),scaley=math.sqrt(2),point=centerofImage)
        self.boundingBoxPoints.translate(deltax=-0.5,deltay=-0.5) #To better align the rays with the pixels

    def SetOptimalDetectors(self):
        self.SetDetectors(math.ceil(self.ReconstructionWidth*math.sqrt(2)))

    def AppendRotation(self, AdditionalRotation):
        self.Rotations.append(AdditionalRotation)

    def AppendRotations(self, AdditionalRotations): # This is for adding multiple rotations
        self.Rotations.extend(AdditionalRotations)
    
    # Different Ray creation methods.
    # Parallel Rays - The rays are parallel to each other going all the way around.
    # Flat Fan Rays - One flat side of the array contains detectors all pointing to a source.
    # Curved Fan Rays - Same as the previous flat fan rays but with a curved fan 
    #                   design hopefully allowing for a better outcome for the 
    #                   same amount of work.
    def CreateParallelRays(self): 
        #TODO: Allow for the possibility of non-square bounding box
        self.UpdateBoundingBox(self.ReconstructionWidth, self.ReconstructionWidth)
        # The linspace forumla dont for the equal spacing of the rays
        raySpacing = (self.boundingBoxPoints.boundingBox_TL.y - self.boundingBoxPoints.boundingBox_BL.y) / (self.Detectors-1)

        for i in range(0, self.Detectors):
            #debug statement print(f"i: {i} | xCoodrinate: {xCoordinate} | yCoordinate: {xCoordinate + raySpacing*i}")
            ycoordinate = self.boundingBoxPoints.boundingBox_BL.y + raySpacing * i
            self.Rays.append(Ray(point1=point(self.boundingBoxPoints.boundingBox_BL.x,ycoordinate),
                                 point2=point(self.boundingBoxPoints.boundingBox_BR.x,ycoordinate),
                                 rotationOrigin=self.center))
    # Fan ray pattern
    def CreateFanRays(self):
        self.UpdateBoundingBox(self.ReconstructionWidth * 3,self.ReconstructionWidth)

        # Same linspace formula
        raySpacing = (self.boundingBoxPoints.boundingBox_TL.y-self.boundingBoxPoints.boundingBox_BL.y)/(self.Detectors-1)
        
        fanningPointY = self.center.y # Y coordinate of the points where the rays fan from
        fanningPointX = self.boundingBoxPoints.boundingBox_BR.x
        fanWallx = self.boundingBoxPoints.boundingBox_BL.x

        for i in range(0,self.Detectors):
            ycoord = self.boundingBoxPoints.boundingBox_BL.y + i * raySpacing
            self.Rays.append(Ray(point(fanningPointX,fanningPointY, 0),
                                 point(fanWallx, ycoord, 0), rotationOrigin=self.center))

    # Curved Pattern
    # sourceloc - This is the distance the source is from the center of the image/object.
    # detectors - This is the distance the detectors are from the center of the image/object.
    # fanAngleDegrees - This the total angular span of the detector arc meaning the angle from the start to the end.
    def CreateCurvedFanRays(self, sourceloc = None, detectors = None, fanAngleDegrees = 60):
        self.UpdateBoundingBox(self.ReconstructionWidth, self.ReconstructionWidth)

        boundingBoxWidth = self.boundingBoxPoints.boundingBox_TL.x - self.boundingBoxPoints.boundingBox_BL.x

        w = boundingBoxWidth
        c = self.center
    
        if sourceloc is None:
            sourceloc = 1.2 * w 
        if detectors is None:
            detectors = 1.5 * w
    
        source = point(c.x + sourceloc, c.y, 0)

        half = math.radians(fanAngleDegrees / 2.0)
        angles = np.linspace(-half, +half, self.Detectors)
    
        self.Rays = []

        for a in angles:
            # Point on an arc centered at the source, facing left
            det = point(source.x - detectors * math.cos(a), source.y + detectors * math.sin(a))
    
            self.Rays.append(Ray(point1 = point(source.x, source.y, 0), point2 = det, rotationOrigin = c))

    # Making and moving Rays
    def DrawRay(self, rayNum):
        drawer = ImageDraw.Draw(self.APicture)
        if rayNum < self.Detectors:
            coords = self.Rays[rayNum].returncoords()
            drawer.line(coords,fill="White",width=0)
        else:
            # Note from Jeff: Not sure what to do if condiiton fails.
            return
        
    def DrawRays(self):
        for i in range(0,len(self.Rays)):
            self.DrawRay(i)

    def RotateRaysTo(self,theta):
        for i in self.Rays:
            i.RotateTo(theta)

    # Image management definitions
    def ClearImage(self):
        TheDraw = ImageDraw.Draw(self.APicture)
        TheDraw.rectangle([0, 0, self.ReconstructionWidth, self.ReconstructionWidth], fill = "Black", outline = "Black")

    def SaveImage(self,filename = "DefaultA.bmp"):
        self.APicture.save(WorkingFolder + filename)

    # Finally making the important matrices for SIRT.
    # CreateAMatrix - This is the mathematical representation of the scanner with what the projections were taken. 
    # The following two are technically matrices in the original algorithm and it still works that way but they were
    # changed tp vectors because it helps lower the computational processing power needed for the program to work.
    # For example, originally a Reconstruction Width of 128 in the original matrices didn't even start the iterations
    # after 30 mins. Now with C/R being in the for of a vector and doing element-wise matrix multiplication, the program
    # can do it in about 30 - 40 seconds on my laptop.
    # CreateCDiagonal/CreateRDiagonal- The diagonal matrix storing only the diagonals in a vector 
    def CreateAMatrix(self):
        self.AMatrix = np.zeros([self.Detectors * len(self.Rotations), (self.ReconstructionWidth)**2])

        if len(self.Rays) == 0:
            print("WARNING: Rays should be added to the AMatrix before running this function")

        for angleNum in range(0,len(self.Rotations)): #This is basically having PIL draw the lines that each ray will make, and then adding it as a row vector to its spot in the A matrix
            self.RotateRaysTo(self.Rotations[angleNum])
            for ray in range(0,self.Detectors):
                self.ClearImage()
                self.DrawRay(ray)
                # Debug statement: self.APicture.save(f"Workingdir/AMatrixDrawings/Angle{angleNum:10d}Ray{ray:10d}.bmp")
                self.AMatrix[angleNum*self.Detectors+ray] = np.asarray(self.APicture).reshape(-1)

        print(f"A Matrix created, with a size of {np.shape(self.AMatrix)}")

        # Precomputing the transpose of A matrix. 
        self.AMatrix_Transpose = self.AMatrix.transpose()
        print("Transpose of A Matrix calculated")
    
    def CreateCMatrix(self):
        # C is diagonal: C[j] = 1/sum(A[:,j])
        denom = np.sum(self.AMatrix, axis = 0)
        self.Cdiag = np.zeros_like(denom, dtype=float)
        mask = denom > 0
        self.Cdiag[mask] = 1.0 / denom[mask]
        print("C diagonal matrix created")

    def CreateRMatrix(self):
        # R is diagonal: R[i] = 1/sum(A[i,:])
        denom = np.sum(self.AMatrix, axis = 1)
        self.Rdiag = np.zeros_like(denom, dtype=float)
        mask = denom > 0
        self.Rdiag[mask] = 1.0 / denom[mask]
        print("R diagonal created")

    # Debugging Tools below
    def DebugCreateX(self): 
        self.Rays.append(Ray(point1 = point(self.boundingBoxPoints.boundingBox_BL.x,
                                            self.boundingBoxPoints.boundingBox_BL.y, 0),
                             point2 = point(self.boundingBoxPoints.boundingBox_TR.x,
                                            self.boundingBoxPoints.boundingBox_TR.y, 0),
                             rotationOrigin = self.center))
        
        self.Rays.append(Ray(point1 = point(self.boundingBoxPoints.boundingBox_TL.x,
                                            self.boundingBoxPoints.boundingBox_TL.y, 0),
                             point2 = point(self.boundingBoxPoints.boundingBox_BR.x,
                                            self.boundingBoxPoints.boundingBox_BR.y, 0),
                             rotationOrigin = self.center))
        
    def PrintMatrixStats(self):
        A = self.AMatrix

        total_elements = A.size
        nonzero_elements = np.count_nonzero(A)
        zero_elements = total_elements - nonzero_elements

        sparsity = zero_elements / total_elements
        density = nonzero_elements / total_elements

        print("\n=== A Matrix Statistics ===")
        print(f"Shape: {A.shape}")
        print(f"Total elements: {total_elements}")
        print(f"Nonzero elements: {nonzero_elements}")
        print(f"Zero elements: {zero_elements}")
        print(f"Density (nonzero %): {density*100:.4f}%")
        print(f"Sparsity (zero %): {sparsity*100:.4f}%")

        # Average number of pixels each ray hits
        avg_nonzero_per_row = np.mean(np.count_nonzero(A, axis=1))
        print(f"Average nonzero elements per ray: {avg_nonzero_per_row:.2f}")
        
    def PrintMatrixSample(self, rows=5, cols=50):
        A = self.AMatrix

        print("\n=== Sample of A Matrix ===")
        for i in range(min(rows, A.shape[0])):
            print(f"Row {i}:")
            print(A[i, :cols])