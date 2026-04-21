import numpy as np
import math

# A container for a point in a 2D space.
# The z variable is a mostly useless variable but it is there just for future-proofing sake.
# Along with all the 3D space definitions related to the z variable.
class point:
    def __init__(self, x:float = 0, y:float = 0, z:float = 0):
        self._x = x
        self._y = y
        self._z = z
        
    # TODO: Add checks to make sure new values are actually numbers
    @property
    def x(self):
        return self._x
    
    @property
    def y(self):
        return self._y
    
    @property
    def z(self):
        return self._z
    
    @x.setter
    def x(self, newx:float):
        self._x = newx
    
    @y.setter
    def y(self, newy:float):
        self._y = newy
    
    @z.setter
    def z(self,newz:float):
        self._z = newz

    # These properties help working with matrices easier in the future.
    @property
    def xy(self):
        return np.array([self._x,self._y])
    
    @xy.setter
    def xy(self, xy:list[float]):
        self._x = xy[0]
        self._y = xy[1]
    
    # 3D version if it is necessary for the z variable.
    @property
    def xyz(self):
        return np.array([self._x,self._y,self._z])
    
    @xyz.setter
    def xyz(self, xyz:list[float]):
        self._x = xyz[0]
        self._y = xyz[1]
        self._z = xyz[2]
    
    def translate(self, deltax:float = 0, deltay:float = 0, deltaz:float = 0):
        self._x += deltax
        self._y += deltay
        self._z += deltaz

    def scale(self, scalex:float = 1, scaley:float = 1,scalez:float = 1):
        self._x *= scalex
        self._y *= scaley
        self._z *= scalez

class boundingBox:
    # Can be safely assumed what each one means but
    # BL - Back Left
    # TL - Top Left
    # TR - Top Right
    # BR - Back Right
    #The points that represent the four corners of the reconstruction grid, going clockwise from the bottom-left
    def __init__(self, width:float = 1, height:float = 1, plane:float = 0):
        self.boundingBox_BL = point(0, 0, plane)
        self.boundingBox_TL = point(0, height, plane)
        self.boundingBox_TR = point(width, height, plane)
        self.boundingBox_BR = point(width, 0, plane)

    def translate(self, deltax:float = 0, deltay:float = 0, deltaz:float=0):
        self.boundingBox_BL.translate(deltax, deltay, deltaz)
        self.boundingBox_TL.translate(deltax, deltay, deltaz)
        self.boundingBox_TR.translate(deltax, deltay, deltaz)
        self.boundingBox_BR.translate(deltax, deltay, deltaz)

    def changeSize(self, newwidth:float = 2, newheight:float = 2, newplane:float = 0): #Does this before moving it
        self.boundingBox_BL.xyz = [0, 0, newplane]
        self.boundingBox_TL.xyz = [0, newheight, newplane]
        self.boundingBox_TR.xyz = [newwidth,newheight,newplane]
        self.boundingBox_BR.xyz = [newwidth,0,newplane]

    def scaleAboutOrigin(self,scalex:float=1,scaley:float=1,scalez:float=1):
        self.boundingBox_BL.scale(scalex=scalex,scaley=scaley,scalez=scalez)
        self.boundingBox_TL.scale(scalex=scalex,scaley=scaley,scalez=scalez)
        self.boundingBox_TR.scale(scalex=scalex,scaley=scaley,scalez=scalez)
        self.boundingBox_BR.scale(scalex=scalex,scaley=scaley,scalez=scalez)
        
    def scaleAboutPoint(self, scalex:float=1, scaley:float=1, scalez:float=1, point:point = point(0,0,0)):
        translation = point.xyz
        self.translate(deltax=-translation[0],
                       deltay=-translation[1],
                       deltaz=-translation[2])
        self.scaleAboutOrigin(scalex = scalex, scaley = scaley, scalez = scalez)
        self.translate(deltax=translation[0],
                       deltay=translation[1],
                       deltaz=translation[2])
class Ray:
    def __init__(self, point1:point = point(0, 0, 0), point2:point = point(1, 0, 0), rotationOrigin:point = point(0, 0, 0)): 
        self._coords = np.array([[0,0],[0,0]])
        self._point1 = point1
        self._point2 = point2
        self._coords = np.array([[self._point1.x, self._point2.x], [self._point1.y, self._point2.y]]) #Pay attention to this layout, did this to make rotation matrix work
        self._theta = 0 #How much the ray has been rotated from starting position
        self._rotationOrigin = rotationOrigin
            
    # Making the rotate command like this because it will make dealing with rotations easier later
    # RotateBy an angle in radians.
    def RotateBy(self, theta:float = 0):
        theta = math.radians(theta)
        rotation_array = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
        translation = np.array([[self._rotationOrigin.x], [self._rotationOrigin.y]])
        self.coords = self.coords -translation #Translation, tried +=, didn't work, this is moving the ray to be centered around the origin so the rotation matrix rotates it as expected
        self.coords = np.matmul(rotation_array,self.coords) #Rotation
        self.coords = self.coords + translation #Translation
        
    def RotateTo(self,desiredtheta:float=0):
        deltatheta = desiredtheta - self._theta
        self.RotateBy(deltatheta)
        self._theta=desiredtheta

    # Jeff's note: Not sure if this is a good idea. 
    def SetTheta(self,theta:float=0):
        self._theta = theta
        
    @property
    def point1(self):
        return self._point1
    
    @point1.setter
    def point1(self, newpoint:point):
        self._point1 = newpoint
    
    @property
    def point2(self):
        return self._point2
    
    @point2.setter
    def point2(self, newpoint:point):
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
        return np.array([[self._point1.x, self._point2.x], [self._point1.y, self._point2.y]])
    
    @coords.setter
    def coords(self,npArray):
        self._point1.x = npArray[0][0]
        self._point1.y = npArray[1][0]
        self._point2.x = npArray[0][1]
        self._point2.y = npArray[1][1]
        self._coords = npArray