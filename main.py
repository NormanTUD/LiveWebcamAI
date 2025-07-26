from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from threading import Lock
from io import BytesIO
from PIL import Image
import torch, time
from diffusers import StableDiffusionImg2ImgPipeline
import base64

app = FastAPI()

# CORS (für lokale Tests)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Load SD
pipe = StableDiffusionImg2ImgPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipe.to("cuda")
pipe.safety_checker = lambda images, **kwargs: (images, False)  # Disable safety filter for speed

lock = Lock()

@app.post("/process")
async def process_image(file: UploadFile, prompt: str = Form(...)):
    image = Image.open(BytesIO(await file.read())).convert("RGB")
    start = time.time()

    with lock:
        result = pipe(prompt=prompt, image=image, strength=0.75, guidance_scale=7.5).images[0]

    buffer = BytesIO()
    result.save(buffer, format="JPEG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    duration = time.time() - start

    return JSONResponse(content={"image": encoded, "duration": duration})
