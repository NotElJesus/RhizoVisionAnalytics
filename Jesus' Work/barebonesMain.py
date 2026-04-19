import time
import numpy as np
from PIL import Image

from config import (
    filename,
    InputFolder,
    OutputFolder,
    WorkingFolder,
    ReconstructionWidth,
    Rotations,
    TotalIterations,
    SourceLocation,
    DetectorsDist,
    FanAngleDegrees,
)
from systemMatrix import AMatrix
from reconstruction import xMatrix
from rayVisualization import MakeSinogram, PlotAMatrix

import os
import io
import matplotlib.pyplot as plt

def plot_rays_animation(scanner, img_path, angles):
    os.makedirs("Workingdir/Torture", exist_ok=True)

    base_img = Image.open(img_path).convert("L")
    base_img = base_img.resize((scanner.ReconstructionWidth, scanner.ReconstructionWidth))

    fig, ax = plt.subplots()
    frames = []

    for angle in angles:
        ax.clear()

        ax.imshow(
            base_img,
            cmap="gray",
            extent=(0.0, float(scanner.ReconstructionWidth), float(scanner.ReconstructionWidth), 0.0)
        )

        scanner.RotateRaysTo(float(angle))
        PlotAMatrix(scanner, ax)

        margin = 10
        w = float(scanner.ReconstructionWidth)
        ax.set_xlim(-margin, w + margin)
        ax.set_ylim(w + margin, -margin)
        ax.set_title(f"Angle = {angle}°")

        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        frames.append(Image.open(buf).copy())
        buf.close()

    if frames:
        frames[0].save(
            "Workingdir/Torture/rays.gif",
            save_all=True,
            append_images=frames[1:],
            duration=50,
            loop=0
        )
    plt.close(fig)
    scanner.RotateRaysTo(0)

def main():
    
    start_time = time.perf_counter()

    path = InputFolder + filename
    print(f"Loading image: {path}")

    try:
        img = Image.open(path).convert("L")
        img = img.resize((ReconstructionWidth, ReconstructionWidth))
        img.save(f"{WorkingFolder}Torture/inputimage.png")
    except IOError:
        print("Could not load input image.")
        return

    xarray = np.asarray(img, dtype=float).reshape(-1)
    print(f"x array made with shape {xarray.shape}")

    scanner = AMatrix()
    scanner.SetReconstruction(ReconstructionWidth)

    scanner.CreateCurvedFanRays(
        sourceloc = SourceLocation,
        detectors = DetectorsDist,
        fanAngleDegrees = FanAngleDegrees
    )

    plot_rays_animation(scanner, InputFolder + filename, Rotations)

    print("Creating A matrix...")
    scanner.CreateAMatrix()

    if hasattr(scanner, "PrintMatrixStats"):
        scanner.PrintMatrixStats()

    print("Creating projection vector p...")
    p = np.matmul(scanner.AMatrix, xarray)

    MakeSinogram(
        p,
        f"{OutputFolder}Sinograms/output.bmp",
        len(Rotations),
        scanner.Detectors
    )

    print("Creating C and R matrices...")
    scanner.CreateCMatrix()
    scanner.CreateRMatrix()

    reconstructor = xMatrix(scanner, scanner.ReconstructionWidth)
    reconstructor.ImportPVector(p)
    reconstructor.SetIterations(TotalIterations)
    reconstructor.SetDetectors(scanner.Detectors)

    print("Running iterations...")
    reconstructor.DoAllIterations()

    print("Saving reconstruction GIF...")
    reconstructor.SaveGif(OutputFolder + "reconstruction.gif")

    end_time = time.perf_counter()
    print(f"Elapsed time: {end_time - start_time:.4f} seconds")
    print("Done.")


if __name__ == "__main__":
    main()