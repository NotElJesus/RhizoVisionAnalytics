# The basic process of this program is Take image -> Do Scans -> Use Scans to see what x we can get
# This file is refactoring everything to use point classes instead of the 

from PIL import Image
from PIL import ImageDraw
import numpy as np
import math
import matplotlib.pyplot as plt

filename = "freddy.png"
shortfile = "ABSTestingFred"
InputFolder = "Input/"
OutputFolder = "Output/"
WorkingFolder = "Workingdir/"
plotmatlab = False #Set to True to create a gif of the scanning process 

ReconstructionWidth = 32 #Size of the reconstruction grid, just gonna do it as a square
Detectors = 21 #How many detectors are used
Rotations = np.arange(0,181,1) # [0,22.5,45,67.5,90,112.5,135,157.5] ## In Degrees, angles start from the positive x-axis

class point:
    def __init__(self,x:float=0,y:float=0,z:float=0):
        self._x:float = x
        self._y:float = y
        self._z:float = z #Probably not going to use this but it can't hurt to have
    # TODO: Add checks to make sure new values are actually numbers
    @property
    def x(self):
        return self._x
    @x.setter
    def x(self,newx:float):
        self._x = newx
    
    @property
    def y(self):
        return self._y
    @y.setter
    def y(self,newy:float):
        self._y = newy
        
    @property
    def z(self):
        return self._z
    @z.setter
    def z(self,newz:float):
        self._z = newz

    @property #Properties to make working with matrices easier
    def xy(self):
        return np.array([self._x,self._y])
    @xy.setter
    def xy(self,xy:list[float]):
        self._x = xy[0]
        self._y = xy[1]
    
    @property #3D version
    def xyz(self):
        return np.array([self._x,self._y,self._z])
    @xyz.setter
    def xyz(self,xyz:list[float]):
        self._x = xyz[0]
        self._y = xyz[1]
        self._z = xyz[2]
    def translate(self,deltax:float=0,deltay:float=0,deltaz:float=0):
        self._x += deltax
        self._y += deltay
        self._z += deltaz
    def scale(self,scalex:float=1,scaley:float=1,scalez:float=1):
        self._x *= scalex
        self._y *= scaley
        self._z *= scalez
        
class boundingBox:
    def __init__(self,width:float=1,height:float=1,plane:float=0): #The points that represent the four corners of the reconstruction grid, going clockwise from the bottom-left
        self.boundingBox_BL = point(0,0,plane)
        self.boundingBox_TL = point(0,height,plane)
        self.boundingBox_TR = point(width,height,plane)
        self.boundingBox_BR = point(width,0,plane)
    def translate(self,deltax:float = 0,deltay:float = 0,deltaz:float=0):
        self.boundingBox_BL.translate(deltax,deltay,deltaz)
        self.boundingBox_TL.translate(deltax,deltay,deltaz)
        self.boundingBox_TR.translate(deltax,deltay,deltaz)
        self.boundingBox_BR.translate(deltax,deltay,deltaz)
    def changeSize(self,newwidth:float=2,newheight:float=2,newplane:float=0): #Does this before moving it
        self.boundingBox_BL.xyz = [0,0,newplane]
        self.boundingBox_TL.xyz = [0,newheight,newplane]
        self.boundingBox_TR.xyz = [newwidth,newheight,newplane]
        self.boundingBox_BR.xyz = [newwidth,0,newplane]
    def scaleAboutOrigin(self,scalex:float=1,scaley:float=1,scalez:float=1):
        self.boundingBox_BL.scale(scalex=scalex,scaley=scaley,scalez=scalez)
        self.boundingBox_TL.scale(scalex=scalex,scaley=scaley,scalez=scalez)
        self.boundingBox_TR.scale(scalex=scalex,scaley=scaley,scalez=scalez)
        self.boundingBox_BR.scale(scalex=scalex,scaley=scaley,scalez=scalez)
    def scaleAboutPoint(self,scalex:float=1,scaley:float=1,scalez:float=1,point:point=point(0,0,0)):
        translation = point.xyz
        self.translate(deltax=-translation[0],
                       deltay=-translation[1],
                       deltaz=-translation[2])
        self.scaleAboutOrigin(scalex=scalex,scaley=scaley,scalez=scalez)
        self.translate(deltax=translation[0],
                       deltay=translation[1],
                       deltaz=translation[2])
class Ray:
    def __init__(self,point1:point = point(0,0,0),point2:point = point(1,0,0),rotationOrigin:point=point(0,0,0)): 
        self._coords = np.array([[0,0],[0,0]])
        self._point1:point = point1
        self._point2:point = point2
        self._coords = np.array([[self._point1.x,self._point2.x],[self._point1.y,self._point2.y]]) #Pay attention to this layout, did this to make rotation matrix work
        self._theta = 0 #How much the ray has been rotated from starting position
        self._rotationOrigin:point = rotationOrigin
            
        #Making the rotate command like this because it will make dealing with rotations easier later
    def RotateBy(self,theta:float=0): #Rotates By an angle in radians
        theta = math.radians(theta)
        rotation_array = np.array([[math.cos(theta),-math.sin(theta)],[math.sin(theta),math.cos(theta)]])
        translation = np.array([[self._rotationOrigin.x],[self._rotationOrigin.y]])
        self.coords = self.coords - translation #Translation, tried +=, didn't work, this is moving the ray to be centered around the origin so the rotation matrix rotates it as expected
        self.coords = np.matmul(rotation_array,self.coords) #Rotation
        self.coords = self.coords + translation #Translation
    def RotateTo(self,desiredtheta:float=0):
        deltatheta = desiredtheta - self._theta
        self.RotateBy(deltatheta)
        self._theta=desiredtheta
    def SetTheta(self,theta:float=0): #Don't know if this is a good idea
        self._theta = theta
        
    @property
    def point1(self):
        return self._point1
    @point1.setter
    def point1(self,newpoint:point):
        self._point1 = newpoint
    
    @property
    def point2(self):
        return self._point2
    @point2.setter
    def point2(self,newpoint:point):
        self._point2 = newpoint

    def returncoords(self):
        return [self.point1.x,self.point1.y,self.point2.x,self.point2.y]
        
    @property
    def rotationOrigin(self):
        return self._rotationOrigin
    @rotationOrigin.setter
    def rotationOrigin(self,newx:float=0,newy:float=0):
        self._rotationOrigin.x = newx
        self._rotationOrigin.y = newy
    @property
    def coords(self):
        return np.array([[self._point1.x,self._point2.x],[self._point1.y,self._point2.y]])
    @coords.setter
    def coords(self,npArray):
        self._point1.x = npArray[0][0]
        self._point1.y = npArray[1][0]
        self._point2.x = npArray[0][1]
        self._point2.y = npArray[1][1]
        self._coords = npArray

class AMatrix:
    def __init__(self):
        self.ReconstructionWidth:int = 3 #Reconstruction grid width
        self.Detectors:int = Detectors #How many detectors 
        self.Rotations:list[float] = Rotations # Storing the rotation angles
        self.APicture:Image = Image.new("1",(self.ReconstructionWidth,self.ReconstructionWidth)) #Stores the picture that is used to make a row of A
        self.Rays: list[Ray]= [] # To store the ray objects that make up the detector
        self.center:point = point(self.ReconstructionWidth/2-.5,self.ReconstructionWidth/2-0.5) #This -0.5 is to align with the center of the pixel rather than the bottom left of a pixel
        self.boundingBoxPoints = boundingBox(self.ReconstructionWidth,ReconstructionWidth)
    #Setting Methods, should probably use the property thing but this works    
    def SetReconstruction(self,NewWidth):
        if NewWidth > 0: 
            self.ReconstructionWidth = NewWidth
            self.APicture = Image.new("1",(self.ReconstructionWidth,self.ReconstructionWidth))
            self.center = point(self.ReconstructionWidth/2-.5,self.ReconstructionWidth/2-0.5)
        else:
            print("Reconstruction grid must be larger than 0")
    def SetDetectors(self,NewDetectors):
        if NewDetectors > 0:
            self.Detectors = NewDetectors
            print(f"Now using {self.Detectors} detectors")
        else:
            print("There must be more than 0 detectors")
    def UpdateBoundingBox(self,newWidth:float=2,newHeight:float=2):
        #updates the bounding box with the assumption it should be centered about the center of the image, and scaled by square root 2
        self.boundingBoxPoints.changeSize(newWidth,newHeight)
        centerofImage = point(newWidth/2,newHeight/2,0)
        self.boundingBoxPoints.scaleAboutPoint(scalex=math.sqrt(2),scaley=math.sqrt(2),point=centerofImage)
        self.boundingBoxPoints.translate(deltax=-0.5,deltay=-0.5) #To better align the rays with the pixels
    def SetOptimalDetectors(self):
        self.SetDetectors(math.ceil(self.ReconstructionWidth*math.sqrt(2)))
    def AppendRotation(self,AdditionalRotation):
        self.Rotations.append(AdditionalRotation)
    def AppendRotations(self,AdditionalRotations): #This is for adding multiple rotations
        self.Rotations.extend(AdditionalRotations)
    # Ray creation methods
    def CreateParallelRays(self): #Makes the rays parallel
        self.UpdateBoundingBox(self.ReconstructionWidth,self.ReconstructionWidth) #TODO: Allow for the possibility of non-square bounding box
        raySpacing = (self.boundingBoxPoints.boundingBox_TL.y-self.boundingBoxPoints.boundingBox_BL.y)/(self.Detectors-1)#Lin space forumala
        for i in range(0,self.Detectors):
            #debug statement print(f"i: {i} | xCoodrinate: {xCoordinate} | yCoordinate: {xCoordinate + raySpacing*i}")
            ycoordinate = self.boundingBoxPoints.boundingBox_BL.y+raySpacing*i
            self.Rays.append(Ray(point1=point(self.boundingBoxPoints.boundingBox_BL.x,ycoordinate),
                                 point2=point(self.boundingBoxPoints.boundingBox_BR.x,ycoordinate),
                                 rotationOrigin=self.center))
    def CreateFanRays(self): #Makes the rays into a fan pattern
        self.UpdateBoundingBox(self.ReconstructionWidth*3,self.ReconstructionWidth)
        raySpacing = (self.boundingBoxPoints.boundingBox_TL.y-self.boundingBoxPoints.boundingBox_BL.y)/(self.Detectors-1)
        fanningPointY = self.center.y # Y coordinate of the points where the rays fan from
        fanningPointX = self.boundingBoxPoints.boundingBox_BR.x
        fanWallx = self.boundingBoxPoints.boundingBox_BL.x
        for i in range(0,self.Detectors):
            ycoord = self.boundingBoxPoints.boundingBox_BL.y+i*raySpacing
            self.Rays.append(Ray(point(fanningPointX,fanningPointY,0),
                                 point(fanWallx,ycoord,0),
                                 rotationOrigin=self.center))
    # Making and moving Rays
    def DrawRay(self,rayNum):
        drawer = ImageDraw.Draw(self.APicture)
        if rayNum < self.Detectors:
            coords = self.Rays[rayNum].returncoords()
            drawer.line(coords,fill="White",width=0)
        else:
            return #don't know what to do if condition fails
    def DrawRays(self):
        for i in range(0,len(self.Rays)):
            self.DrawRay(i)
    def RotateRaysTo(self,theta):
        for i in self.Rays:
            i.RotateTo(theta)
    # Image management
    def ClearImage(self):
        TheDraw = ImageDraw.Draw(self.APicture)
        TheDraw.rectangle([0,0,self.ReconstructionWidth,self.ReconstructionWidth],fill="Black",outline="Black")
    def SaveImage(self,filename="DefaultA.bmp"):
        self.APicture.save(WorkingFolder+filename)
    # Make Matrices
    def CreateAMatrix(self): #Actually making the matrix :)
        self.AMatrix = np.zeros([self.Detectors*len(self.Rotations),(self.ReconstructionWidth)**2])
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
        self.AMatrix_Transpose = self.AMatrix.transpose() #Precomputing this
        print("Transpose of A Matrix calculated")
    def CreateCMatrix(self):
        columns = np.shape(self.AMatrix)[1]
        self.CMatrix = np.zeros([columns,columns]) #This needs to be a square
        for j in range(0,columns):
            denominator = np.sum(self.AMatrix[:,j])#Summing specific column
            if denominator > 0:
                self.CMatrix[j,j] = 1/denominator
            else:
                self.CMatrix[j,j] = 0
        print("C Matrix created")
    def CreateRMatrix(self):
        rows = np.shape(self.AMatrix)[0]
        self.RMatrix = np.zeros([rows,rows]) #This needs to be a square
        for i in range(0,rows):
            denominator = np.sum(self.AMatrix[i,:])
            if denominator > 0:
                self.RMatrix[i,i] = 1/denominator
            else:
                self.RMatrix[i,i] = 0
        print("R Matrix created")
    # Debugging
    def DebugCreateX(self): 
        self.Rays.append(Ray(point1 = point(self.boundingBoxPoints.boundingBox_BL.x,
                                            self.boundingBoxPoints.boundingBox_BL.y,0),
                             point2 = point(self.boundingBoxPoints.boundingBox_TR.x,
                                            self.boundingBoxPoints.boundingBox_TR.y,0),
                             rotationOrigin=self.center))
        self.Rays.append(Ray(point1 = point(self.boundingBoxPoints.boundingBox_TL.x,
                                            self.boundingBoxPoints.boundingBox_TL.y,0),
                             point2 = point(self.boundingBoxPoints.boundingBox_BR.x,
                                            self.boundingBoxPoints.boundingBox_BR.y,0),
                             rotationOrigin=self.center))
class xMatrix: #This should be done after the AMatrix is made, manages everything related to the constructed xMatrix
    def __init__(self,Amatrix:AMatrix,ReconWidth:float=3):
        self.ReconstructionWidth = ReconWidth
        self.xarray = np.zeros(self.ReconstructionWidth**2) #Initializes the array for the xmatrix
        self.Amatrix: AMatrix = Amatrix
        print("About to calculate C*At*R, stand by...")
        self.CAtransR = np.matmul(np.matmul(self.Amatrix.CMatrix,self.Amatrix.AMatrix_Transpose),self.Amatrix.RMatrix) #This saves alot of time 
        print("C*At*R calculated")
        self.Pmatrix = np.empty([1,1]) #Starts off empty, should be imported ASAP
        self.iteration = 0
        self.maxiteration = 0 
        self.frames = [] #Used to store frames to make a gif later
        self.detectors = 0
    #Setting Up Stuff Section
    def ImportPVector(self,p):
        self.Pmatrix = p #this is wasteful but its python so who cares
        print("P vector copied")
    def SetReconstruction(self,NewWidth=0): #Lets the reconstruction width be changed, don't know if this will be used
        if NewWidth > 0: 
            self.ReconstructionWidth = NewWidth
            self.APicture = Image.new("1",(self.ReconstructionWidth,self.ReconstructionWidth))
            self.xarray = np.zeros(self.ReconstructionWidth**2)
        else:
            print("Reconstruction grid must be larger than 0")
    def SetDetectors(self,detectors):
        self.detectors = detectors
    def SetIterations(self,iter): #Set the number of iterations to do
        if iter >= 0:
            self.maxiteration = iter
        else:
            print("Use a positive number")
    #Iteration Section
    def Iterate(self): #Do a single iteration of the xarray 
        #self.xarray = self.xarray + np.matmul((np.matmul(np.matmul(self.Amatrix.CMatrix,self.Amatrix.AMatrix_Transpose),self.Amatrix.RMatrix)),(p-np.matmul(self.Amatrix.AMatrix,self.xarray)))
        self.xarray = self.xarray + np.matmul(self.CAtransR,(self.Pmatrix-np.matmul(self.Amatrix.AMatrix,self.xarray)))
        # As a future improvement calculate the CAtRP and the CAtRA and then reuse it instead of recalculating it everytime
        #self.xarray = np.abs(self.xarray) #If you don't do this then you get random white dots because when going from signed to unsigned the low negative values turn to 255
        #TODO: Evaluate if this should be here or in the image processing
    def DoAllIterations(self): #Does all of iterations 
        for i in range(0,self.maxiteration):
            print(f"Starting iteration {i+1} of {self.maxiteration}...",end='')
            self.Iterate()
            print(f"iteration {i} calculated, saving image...",end='')
            self.SaveImage(WorkingFolder+f"{shortfile}Iteration{i}.bmp")
            print(f"image saved! Iteration {i} completed!")
            MakeSinogram(np.matmul(self.Amatrix.AMatrix,self.xarray),WorkingFolder+f"{shortfile}SinogramIteration{i}.bmp",np.size(Rotations),self.detectors) #TODO: Disable for performance, just comment line out
        print(f"Saving final output image")
        self.SaveImage(OutputFolder+f"{shortfile}.bmp")
        print(f"Final image saved!")
    #Saving Stuff Section
    def SaveImage(self,filename): #Function to save the image and add it to frames to make a gif later
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
        
def MakeSinogram(inputarray,filename,numberofrotations,detectors):
    max = np.max(inputarray)
    if max > 0:
        scale = 255/max
    else:
        scale = 1
    output = np.abs(np.reshape((np.floor(inputarray*scale)).astype(np.int8),(numberofrotations,detectors)))
    outputimage = Image.fromarray(output,"L")
    outputimage.save(filename)
    
def PlotAMatrix(AMatrix: AMatrix, ax: plt.axes):
    rays: list[Ray] = AMatrix.Rays
    for i in rays:
        ax.plot([i.point1.x,i.point2.x],[i.point1.y,i.point2.y])
def main():
    try:
        path = "Input/"+filename
        print(f"Loading {path}...")
        img = Image.open(path).resize([ReconstructionWidth,ReconstructionWidth])
        img.save(WorkingFolder+"Torture/inputimage.png")
        img.convert("L").save(WorkingFolder+"AsBMP.bmp")
        print(f"{path} loaded!")
    except IOError:
        print("File issues!!")
        pass
    xarray = np.asarray(img).reshape(-1) # Making x
    print(f"x array made with shape {np.shape(xarray)}")
    ScanningResult = AMatrix()
    ScanningResult.SetReconstruction(ReconstructionWidth)
    ScanningResult.SetOptimalDetectors()
    #ScanningResult.CreateParallelRays()
    ScanningResult.CreateFanRays()
    #ScanningResult.DebugCreateX()
    ScanningResult.DrawRays()
    fig, ax = plt.subplots()
    if plotmatlab: #Change at top to enable or disable
        frames = []
        img2 = Image.open(WorkingFolder+"Torture/inputimage.png")
        for i in range(0,181):#Debug block
            ax.imshow(img2)
            PlotAMatrix(ScanningResult,ax)
            ScanningResult.RotateRaysTo(i)
            if (ScanningResult.boundingBoxPoints.boundingBox_BR.x-ScanningResult.boundingBoxPoints.boundingBox_BL.x) > (ScanningResult.boundingBoxPoints.boundingBox_TL.y-ScanningResult.boundingBoxPoints.boundingBox_BL.y):
                ax.set_xlim([ScanningResult.boundingBoxPoints.boundingBox_BL.x-5,ScanningResult.boundingBoxPoints.boundingBox_BR.x+5])
                ax.set_ylim([ScanningResult.boundingBoxPoints.boundingBox_BL.x-5,ScanningResult.boundingBoxPoints.boundingBox_BR.x+5])  
            else:
                ax.set_xlim([ScanningResult.boundingBoxPoints.boundingBox_BL.y-5,ScanningResult.boundingBoxPoints.boundingBox_TL.y+5])
                ax.set_ylim([ScanningResult.boundingBoxPoints.boundingBox_BL.y-5,ScanningResult.boundingBoxPoints.boundingBox_TL.y+5])
            plt.savefig(f"Workingdir/Torture/{i}.png")
            gifFrame = Image.open(f"Workingdir/Torture/{i}.png")
            frames.append(gifFrame)
            ax.cla()   
        frames[0].save("Workingdir/Torture/Gif.gif",
               save_all = True, append_images = frames[1:],
               optimize = False, duration = 50,loop = 0)
        ScanningResult.RotateRaysTo(0)
    ScanningResult.CreateAMatrix() #These three functions should probably be linked into another function but whatever
    p = np.matmul(ScanningResult.AMatrix,xarray)
    MakeSinogram(p,f"{OutputFolder}Sinograms/{shortfile}.bmp",np.size(Rotations),ScanningResult.Detectors)
    ScanningResult.CreateCMatrix()
    ScanningResult.CreateRMatrix()
    #plt.bar(np.array(range(0,ScanningResult.Detectors)),p[0:ScanningResult.Detectors])
    #plt.show()
    Reconstuctor = xMatrix(ScanningResult,ScanningResult.ReconstructionWidth)
    Reconstuctor.ImportPVector(p)
    Reconstuctor.SetIterations(100)
    Reconstuctor.SetDetectors(ScanningResult.Detectors)
    Reconstuctor.DoAllIterations()
    Reconstuctor.SaveGif(OutputFolder+f"{shortfile}.gif")
    
main()
