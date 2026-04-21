from pathlib import Path

import numpy as np
from PIL import Image

from algorithm.config import OutputFolder
from algorithm.reconstruction import xMatrix
from algorithm.systemMatrix import AMatrix


def run_simulation(
    image_path,
    reconstruction_width,
    iterations,
    sourceloc,
    detectors,
    fan_angle_degrees,
    output_basename=None,
):
    output_dir = Path(OutputFolder)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_basename = output_basename or Path(image_path).stem or "reconstruction"

    img = Image.open(image_path).convert("L").resize((reconstruction_width, reconstruction_width))
    xarray = np.asarray(img, dtype=float).reshape(-1)

    scanner = AMatrix()
    scanner.SetReconstruction(reconstruction_width)
    scanner.SetDetectors(detectors)
    scanner.CreateCurvedFanRays(
        sourceloc=sourceloc,
        detectors=detectors,
        fanAngleDegrees=fan_angle_degrees,
    )
    scanner.DrawRays()
    scanner.CreateAMatrix()

    p = np.matmul(scanner.AMatrix, xarray)

    scanner.CreateCMatrix()
    scanner.CreateRMatrix()

    reconstructor = xMatrix(scanner, reconstruction_width)
    reconstructor.ImportPVector(p)
    reconstructor.SetIterations(iterations)
    reconstructor.SetDetectors(detectors)
    reconstructor.DoAllIterations()

    default_output = output_dir / "ABSTestingFred.png"
    output_path = output_dir / f"{output_basename}_reconstruction.png"

    if default_output.exists():
        if output_path.exists():
            output_path.unlink()
        default_output.replace(output_path)

    return str(output_path)
