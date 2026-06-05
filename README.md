# Stable Diffusion (Text-to-Image)

A small project that lets you generate images from a text prompt using Stable Diffusion.

## What’s included
- **FastAPI backend** (`app.py`) with:
  - `GET /health`
  - `POST /generate` → returns a PNG
- **Desktop UI** (`Stable_Diffusion.py`) using Tkinter + CustomTkinter

## Setup

```bash
git clone <REPO_URL>
cd Stable-diffusion
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configure the model (backend)
Uses environment variables:
- `MODEL_PATH` (default: `runwayml/stable-diffusion-v1-5`)
- `DEVICE` (default: `cpu`)

```bash
export MODEL_PATH=runwayml/stable-diffusion-v1-5
export DEVICE=cpu
```

## Run FastAPI

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Health check:
```bash
curl http://localhost:8000/health
```

Generate image:
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"a red sports car driving on a rainy street, cinematic lighting"}' \
  --output output.png
```

## Run the desktop UI

```bash
python3 Stable_Diffusion.py
```

Note: `Stable_Diffusion.py` contains a hardcoded `model_id` inside `__main__`. On macOS/Linux, change it to a valid Hugging Face repo id or a local model folder.

Example:
```python
model_id = "runwayml/stable-diffusion-v1-5"
device = "cpu"
```


