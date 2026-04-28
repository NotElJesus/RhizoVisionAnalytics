import shutil
import sys
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

try:
    from .simulation_runner import run_audio_reconstruction, run_simulation
except ImportError:
    from simulation_runner import run_audio_reconstruction, run_simulation

app = FastAPI(title="RhizoVisionAnalytics API")

OUTPUT_DIR = CURRENT_DIR / "Output"
WORKING_DIR = CURRENT_DIR / "Workingdir"
UPLOAD_DIR = CURRENT_DIR / "uploads"
AUDIO_UPLOAD_DIR = CURRENT_DIR / "audio_uploads"

for folder in (UPLOAD_DIR, AUDIO_UPLOAD_DIR, OUTPUT_DIR, WORKING_DIR):
    folder.mkdir(parents=True, exist_ok=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/output", StaticFiles(directory=str(OUTPUT_DIR)), name="output")


def safe_upload_path(upload: UploadFile, folder: Path = UPLOAD_DIR) -> Path:
    # Keep uploaded filenames stable while avoiding accidental overwrites.
    original = Path(upload.filename or "uploaded_image").name
    stem = Path(original).stem or "uploaded_image"
    suffix = Path(original).suffix or ".png"
    candidate = folder / f"{stem}{suffix}"
    counter = 1

    while candidate.exists():
        candidate = folder / f"{stem}_{counter}{suffix}"
        counter += 1

    return candidate


def save_upload(upload: UploadFile, folder: Path) -> Path:
    file_path = safe_upload_path(upload, folder)
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(upload.file, buffer)
    return file_path


@app.get("/")
def read_root():
    return {"message": "Backend is running"}


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/run")
async def run_reconstruction(
    request: Request,
    image: UploadFile = File(...),
    reconstruction_width: int = Form(...),
    iterations: int = Form(...),
    sourceloc: int = Form(...),
    detectors: int = Form(...),
    fan_angle_degrees: float = Form(...),
):
    if reconstruction_width <= 0:
        raise HTTPException(status_code=400, detail="reconstruction_width must be greater than 0")
    if iterations < 0:
        raise HTTPException(status_code=400, detail="iterations must be greater than or equal to 0")
    if sourceloc <= 0:
        raise HTTPException(status_code=400, detail="sourceloc must be greater than 0")
    if detectors <= 0:
        raise HTTPException(status_code=400, detail="detectors must be greater than 0")
    if fan_angle_degrees <= 0:
        raise HTTPException(status_code=400, detail="fan_angle_degrees must be greater than 0")

    file_path = save_upload(image, UPLOAD_DIR)

    try:
        result_path = Path(
            run_simulation(
                image_path=str(file_path),
                reconstruction_width=reconstruction_width,
                iterations=iterations,
                sourceloc=sourceloc,
                detectors=detectors,
                fan_angle_degrees=fan_angle_degrees,
                output_basename=file_path.stem,
            )
        )
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": f"Reconstruction failed: {exc}"},
        )

    result_filename = result_path.name
    output_url = str(request.url_for("output", path=result_filename))

    return {
        "success": True,
        "filename": image.filename,
        "saved_path": str(file_path),
        "output": str(result_path),
        "output_url": output_url,
        "params": {
            "reconstruction_width": reconstruction_width,
            "iterations": iterations,
            "sourceloc": sourceloc,
            "detectors": detectors,
            "fan_angle_degrees": fan_angle_degrees,
        },
        "message": "Reconstruction completed successfully",
    }


@app.post("/run-audio")
async def run_audio_scan(
    request: Request,
    baseline: UploadFile = File(...),
    measurements: list[UploadFile] = File(...),
    reconstruction_width: int = Form(...),
    iterations: int = Form(...),
    sourceloc: int = Form(...),
    detectors: int = Form(...),
    fan_angle_degrees: float = Form(...),
    rotations: int = Form(...),
    desired_freq: float = Form(2500),
    kernel_size: int = Form(11),
    use_window: bool = Form(True),
    scale_by_rows: bool = Form(False),
):
    # The measured projection vector must match the scanner geometry exactly.
    if reconstruction_width <= 0:
        raise HTTPException(status_code=400, detail="reconstruction_width must be greater than 0")
    if iterations < 0:
        raise HTTPException(status_code=400, detail="iterations must be greater than or equal to 0")
    if sourceloc <= 0:
        raise HTTPException(status_code=400, detail="sourceloc must be greater than 0")
    if detectors <= 0:
        raise HTTPException(status_code=400, detail="detectors must be greater than 0")
    if fan_angle_degrees <= 0:
        raise HTTPException(status_code=400, detail="fan_angle_degrees must be greater than 0")
    if rotations <= 0:
        raise HTTPException(status_code=400, detail="rotations must be greater than 0")
    if desired_freq <= 0:
        raise HTTPException(status_code=400, detail="desired_freq must be greater than 0")
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise HTTPException(status_code=400, detail="kernel_size must be a positive odd integer")
    if len(measurements) != rotations * detectors:
        raise HTTPException(
            status_code=400,
            detail=f"Expected {rotations * detectors} measurement files, got {len(measurements)}",
        )

    baseline_path = save_upload(baseline, AUDIO_UPLOAD_DIR)
    measurement_paths = [save_upload(file, AUDIO_UPLOAD_DIR) for file in measurements]

    try:
        result = run_audio_reconstruction(
            baseline_path=str(baseline_path),
            measurement_paths=[str(path) for path in measurement_paths],
            reconstruction_width=reconstruction_width,
            iterations=iterations,
            sourceloc=sourceloc,
            detectors=detectors,
            fan_angle_degrees=fan_angle_degrees,
            rotations=rotations,
            desired_freq=desired_freq,
            kernel_size=kernel_size,
            use_window=use_window,
            scale_by_rows=scale_by_rows,
            output_basename=Path(baseline_path).stem,
        )
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": f"Audio reconstruction failed: {exc}"},
        )

    reconstruction_path = Path(result["reconstruction_path"])
    sinogram_path = Path(result["sinogram_path"])

    return {
        "success": True,
        "filename": baseline.filename,
        "saved_path": str(baseline_path),
        "measurement_count": result["measurement_count"],
        "output": str(reconstruction_path),
        "output_url": str(request.url_for("output", path=reconstruction_path.name)),
        "sinogram": str(sinogram_path),
        "sinogram_url": str(request.url_for("output", path=sinogram_path.name)),
        "params": {
            "reconstruction_width": reconstruction_width,
            "iterations": iterations,
            "sourceloc": sourceloc,
            "detectors": detectors,
            "fan_angle_degrees": fan_angle_degrees,
            "rotations": rotations,
            "desired_freq": desired_freq,
            "kernel_size": kernel_size,
            "use_window": use_window,
            "scale_by_rows": scale_by_rows,
        },
        "message": "Audio reconstruction completed successfully",
    }
