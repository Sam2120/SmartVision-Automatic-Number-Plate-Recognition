# ALPR (YOLO11 + OpenOCR)

Local web app that:
- Uploads a video
- Runs YOLO (v11 via Ultralytics) to detect license-plate regions
- Runs OpenOCR (`openocr-python`) on crops
- Produces an annotated output video + a JSON file of detections

## 1) Setup (Windows)

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Notes:
- OCR uses `openocr-python` with the ONNX backend (`onnxruntime`).
- YOLO inference can run on GPU if you have a CUDA-enabled PyTorch installed.

## GPU (optional)

### YOLO on GPU

Ultralytics uses PyTorch. To run YOLO on your NVIDIA GPU you need a CUDA-enabled PyTorch build.

1) Install the correct PyTorch for your CUDA version using the official selector:
https://pytorch.org/get-started/locally/

2) In the web UI, keep **Use GPU (CUDA) if available** checked.

### OCR on GPU (optional)

OpenOCR ONNX can also use GPU if you install `onnxruntime-gpu` (and have CUDA installed).

1) Uninstall CPU ONNXRuntime:

```powershell
pip uninstall -y onnxruntime
```

2) Install GPU ONNXRuntime:

```powershell
pip install onnxruntime-gpu
```

3) Keep **Use GPU (CUDA) if available** checked.

## 2) Run

```powershell
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Open:
- http://127.0.0.1:8000

## 3) Usage

- Choose a video
- Set `model_path` to your license-plate YOLO11 `.pt` weight file
- Click **Upload & Process**

Outputs are stored in `jobs/<job_id>/`:
- `annotated.mp4`
- `result.json`
