from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/")
def read_root():
    return {"message": "Backend is running"}

@app.post("/run")
async def run_reconstruction(
    image: UploadFile = File(...),
    reconstruction_width: int = Form(...),
    iterations: int = Form(...),
    sourceloc: int = Form(...),
    detectors: int = Form(...),
    fan_angle_degrees: float = Form(...)
):
    file_path = os.path.join(UPLOAD_DIR, image.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    return {
        "success": True,
        "filename": image.filename,
        "saved_path": file_path,
        "params": {
            "reconstruction_width": reconstruction_width,
            "iterations": iterations,
            "sourceloc": sourceloc,
            "detectors": detectors,
            "fan_angle_degrees": fan_angle_degrees
        },
        "message": "Request received successfully"
    }