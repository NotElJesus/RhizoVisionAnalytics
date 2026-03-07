# Rebuilt SIRT program from Jeff

This is a completely separated version of the latest Try#.py and will in future use be moving forward with working with this. This readme will include a description of the purpose of each .py and in what order you should read it if you are trying to figure out the code/structure yourself. I will be doing my best to keep the readme as simple as possible.

## Directory format
```
.
├── config.py
├── geometry.py
├── visualization.py
├── systemMatrix.py
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

## 3) [visualization.py](visualization.py)

This only contains the ```MakeSinogram``` and ```PlotAMatrix```definitions which does at it states for both; creates the sinogram that is used at reference point for the reconstruction and also creates the plotting for the fan geometry visual.

## 4) [systemMatrix.py](systemMatrix.py)

This contains the whole entire ```AMatrix``` class that is used in the program in order to create the system matrix and allow us to do the reconstruction which is the important part. I could try to explain the entire thing here but it is better if you read the documentation added in the python file itself to understand the entire idea.

The A matrix, also referred to as the *system matrix*, specifies the exact relation between the scanned object and the projection in a mathematical way. Each row expressing a single ray.

## 5) [reconstruction.py]()