# Rebuilt SIRT program from Jeff

This is a completely separated version of the latest Try#.py and will in future use be moving forward with working with this. This readme will include a description of the purpose of each .py and in what order you should read it if you are trying to figure out the code/structure yourself. I will be doing my best to keep the readme as simple as possible.
## Directory format
```
.
├── try7.py
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

## 1) config.py

This includes all the global variables used throughout the project that are important and necessary if ever needed to adjust. If you are wanting to use the simulation without adjusting the code, you will only need to change this.

## 2) geometry.py

The file includes the classes ```point```, ```boundingBox```, and ```Ray```; along with all its definitions that were part of the original. This is used later on with plotting and creating the A matrix.

## 3) imaging.py

