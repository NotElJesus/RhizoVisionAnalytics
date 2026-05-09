# Rebuilt SIRT program from Jeff

This is a completely separated version of the latest Try#.py and will in future use be moving forward with working with this. This readme will include a description of the purpose of each .py and in what order you should read it if you are trying to figure out the code/structure yourself. I will be doing my best to keep the readme as simple as possible.

# Directory format
```
.
├── config.py
├── geometry.py
├── rayVisualization.py
├── systemMatrix.py
├── reconstruction.py
├── barebonesMain.py
├── Input
│   └── DESCRIPTION: Input images are stored in here
├── Workingdir
│   ├── DESCRIPTION: Stores reconstruction iterations
│   ├── AMatrixDrawings
│   │   └── DESCRIPTION: Drawings for making the AMatrix are stored in here if visualization is enabled 
│   └── Torture
│       └── DESCRIPTION: Matlab plots get put in here if enabled
└── Output
    ├── DESCRIPTION: Reconstruction gets put in here
    └── Sinograms
        └── DESCRIPTION: Sinogram of input image gets put in here
```

## 1) [config.py](config.py)

This includes all the global variables used throughout the project that are important and necessary if ever needed to adjust. If you are wanting to use the simulation without adjusting the code, you will only need to change this.

## 2) [geometry.py](geometry.py)

The file includes the classes ```point```, ```boundingBox```, and ```Ray```; along with all its definitions that were part of the original. This is used later on with plotting and creating the A matrix.

## 3) [rayVisualization.py](rayVisualization.py)

This only contains the ```MakeSinogram``` and ```PlotAMatrix```definitions which does at it states for both; creates the sinogram that is used at reference point for the reconstruction and also creates the plotting for the fan geometry visual.

## 4) [systemMatrix.py](systemMatrix.py)

This contains the whole entire ```AMatrix``` class that is used in the program in order to create the system matrix and allow us to do the reconstruction which is the important part. I could try to explain the entire thing here but it is better if you read the documentation added in the python file itself to understand the entire idea.

The A matrix, also referred to as the *system matrix*, specifies the exact relation between the scanned object and the projection in a mathematical way. Each row expressing a single ray.

## 5) [reconstruction.py](reconstruction.py)

The *reconstruction.py* python file contains the whole x matrix class which is the unknown image that is getting updated through each iteration. This contains the iterations logic and as well the ability to save the image iterations and keep them as a gif.

## 6) [barebonesMain.py](barebonesMain.py)

The barebonesMain.py is where the magic happens for this whole program. It combines all the functions done in the previous programs and allows it work all at once, like it should in a normal algorithm. This allows it so that the only things that need to be adjusted would be *config.py* and this file. This one isn't as descriptive as the rest(or even at all to be honest) but I hope that with the rest of the program being descriptive, it allows this one to be pretty easy to follow along for the most part. 

# What do you need to run the program.

For the simulation using an 2D image, it is rather simple, you need an image and basically any image works but you need to make sure that it is greyscale in this case to work. Below is a simple few line program that can be run to convert that image into something that can be inputted without issues. YOu could just do this within main and verify it through there to make sure every inputted image fits this idea but I never did because it wasn't that much of a big deal to include. 

```cpp
from PIL import Image

img = Image.open("INPUT_IMAGE_NAME.png").convert("L")
img.save("OUTPUT_IMAGE_NAME.png")
```

If you are needing to input data, you would just skip this part and just import P vector through the main program. You would not need to do any of this because you already went through the more complicated thing of converting your measured data into that P vector.

The **Directory Format** is an important thing to note and necessary to do. Make sure that you have every folder that is stated above and their correctly directory level to make sure the program runs smoothly. That includes the surface level folders and their subdirectories.

# How to run the program.

If you are able to install the GitHub, go within the **Jesus' Work** folder. Adjust your parameters within config if necessary, whether that is detectors, rotations, input image name, etc. 

```bash
python barebonesMain.py
```

All you would to do is run the command above in your favorite command prompt/terminal and that would start the program and do the rest. 