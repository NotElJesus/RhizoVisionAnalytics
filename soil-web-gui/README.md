# Soil Web GUI

This folder contains the FastAPI backend, Vue frontend, and reconstruction algorithm integration for RhizoVisionAnalytics.

The app supports two reconstruction modes:

- Image Simulation: upload one image and let the backend generate a simulated projection vector.
- Audio Scan: upload one baseline/touching WAV file plus `rotations * detectors` measurement WAV files. The backend computes a transfer-function attenuation value at the selected frequency, builds a sinogram, and reconstructs from that measured projection vector.

## Start the backend

Open a terminal from the repository root:

```powershell
cd D:\Github\RhizoVisionAnalytics\soil-web-gui
.\backend\venv\Scripts\python.exe -m pip install -r backend\requirements.txt
.\backend\venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

Backend health check:

```text
http://127.0.0.1:8000/health
```

## Start the frontend

Open a second terminal:

```powershell
cd D:\Github\RhizoVisionAnalytics\soil-web-gui\frontend\frontend
npm install
npm run dev -- --host 127.0.0.1 --port 5173
```

Frontend app:

```text
http://127.0.0.1:5173
```

The frontend sends reconstruction requests to the backend at `http://localhost:8000` by default.

For audio scans, measurement files are sorted by filename before processing. Name them so that filename order matches the expected scan order.
