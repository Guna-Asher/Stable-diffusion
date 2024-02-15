from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import torch
from diffusers import StableDiffusionPipeline
import os
import uuid
import logging

app = FastAPI(title="Stable Diffusion Image Generator")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GenerateRequest(BaseModel):
    prompt: str

# Load model (assuming model path is set via env or default)
MODEL_PATH = os.getenv("MODEL_PATH", "runwayml/stable-diffusion-v1-5")
DEVICE = os.getenv("DEVICE", "cpu")

try:
    pipe = StableDiffusionPipeline.from_pretrained(MODEL_PATH, torch_dtype=torch.float32)
    pipe = pipe.to(DEVICE)
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise

@app.get("/health")
def health():
    logger.info("Health check requested")
    return {"status": "healthy"}

@app.post("/generate")
def generate_image(request: GenerateRequest):
    try:
        logger.info(f"Generating image for prompt: {request.prompt}")
        with torch.no_grad():
            result = pipe(request.prompt, guidance_scale=8.5)
            image = result["images"][0]

            # Save image temporarily
            temp_dir = "/tmp" if os.name != 'nt' else os.getenv("TEMP", "C:\\temp")
            os.makedirs(temp_dir, exist_ok=True)
            unique_filename = str(uuid.uuid4()) + ".png"
            image_path = os.path.join(temp_dir, unique_filename)
            image.save(image_path)

            logger.info("Image generated and saved successfully")
            return FileResponse(image_path, media_type="image/png", filename=unique_filename)
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail="Image generation failed")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
