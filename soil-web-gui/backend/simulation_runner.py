from pathlib import Path

import numpy as np
from PIL import Image

from algorithm.config import OutputFolder
from algorithm.audio_attenuation import measurements_to_p_vector
from algorithm.reconstruction import xMatrix
from algorithm.systemMatrix import AMatrix
from algorithm.visualization import MakeSinogram


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


def run_audio_reconstruction(
    baseline_path,
    measurement_paths,
    reconstruction_width,
    iterations,
    sourceloc,
    detectors,
    fan_angle_degrees,
    rotations,
    desired_freq,
    kernel_size,
    use_window=True,
    scale_by_rows=False,
    output_basename=None,
):
    output_dir = Path(OutputFolder)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_basename = output_basename or "audio_scan"

    # Build the measured projection vector from WAV attenuation values instead of a synthetic image.
    p = measurements_to_p_vector(
        baseline_path=baseline_path,
        measurement_paths=measurement_paths,
        desired_freq=desired_freq,
        use_window=use_window,
        kernel_size=kernel_size,
        rotations=rotations,
        detectors=detectors,
        scale_by_rows=scale_by_rows,
    ).astype(float)

    scanner = AMatrix()
    scanner.SetReconstruction(reconstruction_width)
    scanner.SetDetectors(detectors)
    # Use the number of measured rotation groups supplied by the audio scan.
    scanner.Rotations = list(np.linspace(0, 180, rotations, endpoint=False))
    scanner.CreateCurvedFanRays(
        sourceloc=sourceloc,
        detectors=detectors,
        fanAngleDegrees=fan_angle_degrees,
    )
    scanner.DrawRays()
    scanner.CreateAMatrix()
    scanner.CreateCMatrix()
    scanner.CreateRMatrix()

    reconstructor = xMatrix(scanner, reconstruction_width)
    reconstructor.ImportPVector(p)
    reconstructor.SetIterations(iterations)
    reconstructor.SetDetectors(detectors)
    reconstructor.DoAllIterations()

    sinogram_path = output_dir / f"{output_basename}_sinogram.png"
    MakeSinogram(p, str(sinogram_path), rotations, detectors)

    default_output = output_dir / "ABSTestingFred.png"
    output_path = output_dir / f"{output_basename}_reconstruction.png"

    if default_output.exists():
        if output_path.exists():
            output_path.unlink()
        default_output.replace(output_path)

    return {
        "reconstruction_path": str(output_path),
        "sinogram_path": str(sinogram_path),
        "measurement_count": len(measurement_paths),
    }
