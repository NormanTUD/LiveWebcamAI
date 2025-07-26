import base64
import os
import tempfile
import uuid
import shutil
import threading
from flask import Flask, request, jsonify, abort, Response
import uuid
import shutil
import tempfile
import os
import gc
import argparse
import logging
from PIL import Image
import torch
from diffusers import AutoPipelineForImage2Image, DEISMultistepScheduler
from flask import Flask

last_generated_image = None

app = Flask(__name__)

pipe = None

# Max 50 MB Upload limit (50 * 1024 * 1024 bytes)
MAX_UPLOAD_SIZE = 50 * 1024 * 1024

def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def clean_memory():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

def check_cuda():
    if not torch.cuda.is_available():
        logging.error("CUDA ist nicht verfügbar – überprüfe deine PyTorch/GPU-Installation!")
        exit(1)
    logging.info(f"CUDA verfügbar: {torch.cuda.get_device_name(0)}")
    return "cuda", torch.float16

def load_image(path: str, size=(512, 512)) -> Image.Image:
    if not os.path.exists(path):
        logging.error(f"Bilddatei '{path}' nicht gefunden!")
        exit(1)
    try:
        return Image.open(path).convert("RGB").resize(size)
    except Exception as e:
        logging.error(f"Fehler beim Laden oder Verarbeiten des Bildes: {e}")
        exit(1)

def load_pipeline(model_id: str, device: str, dtype=torch.float16):
    global pipe

    try:
        logging.info("Lade Pipeline...")
        pipe = AutoPipelineForImage2Image.from_pretrained(
            model_id,
            torch_dtype=dtype,
            variant="fp16",
            low_cpu_mem_usage=True,
        )

        pipe.scheduler = DEISMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.safety_checker = None

        pipe = pipe.to(device)
        logging.info(f"Pipeline erfolgreich geladen auf Gerät: {next(pipe.unet.parameters()).device}")
        return pipe
    except Exception as e:
        logging.error(f"Fehler beim Laden der Pipeline: {e}")
        exit(1)

def run_warmup(pipe, image: Image.Image):
    try:
        logging.info("Führe Warmup-Durchlauf durch...")
        _ = pipe(
            prompt="simple warmup",
            image=[image],
            num_inference_steps=5,
            guidance_scale=5.0,
        )
    except Exception as e:
        logging.warning(f"Warmup fehlgeschlagen (wird ignoriert): {e}")

def run_image2image_pipeline(
    prompt: str,
    negative_prompt: str,
    num_inference_steps: int,
    guidance_scale: float,
    strength: float,
    init_image: Image.Image = None,  # default None
    model_id: str = "lykon/dreamshaper-8",
    seed: int = 33,
    device: str = "cuda",
    dtype=torch.float16,
) -> Image.Image:
    global last_generated_image

    # Wenn kein init_image übergeben wurde, verwende das letzte
    if init_image is None:
        if last_generated_image is None:
            logging.error("Kein Startbild übergeben und auch kein vorheriges Bild vorhanden.")
            return None
        logging.info("Verwende vorheriges generiertes Bild als init_image.")
        init_image = last_generated_image

    run_warmup(pipe, init_image)

    generator = torch.Generator(device=device).manual_seed(seed)

    logging.info("Starte Bildgenerierung...")
    output = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=[init_image],
        generator=generator,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        strength=strength
    )

    if output and output.images and len(output.images) > 0:
        result = output.images[0]
        last_generated_image = result  # Merke dieses Bild für den nächsten Durchlauf
        return result
    else:
        logging.error("Kein Bild wurde generiert!")
        return None

def parse_args():
    parser = argparse.ArgumentParser(description="Image2Image mit Diffusers (dreamshaper-8)")
    parser.add_argument("--input", help="Pfad zum Eingabebild (z.B. 1.jpg)")
    parser.add_argument("--output", default="image2img_result.png", help="Pfad zur Ausgabedatei")
    parser.add_argument("--prompt", help="Textprompt für die Bildgenerierung")
    parser.add_argument("--negative_prompt", default="", help="Negative Prompt (optional)")
    parser.add_argument("--steps", type=int, default=25, help="Anzahl Inferenz-Schritte")
    parser.add_argument("--scale", type=float, default=7.5, help="Guidance-Scale")
    parser.add_argument("--seed", type=int, default=33, help="Zufalls-Seed")
    parser.add_argument("--model", default="lykon/dreamshaper-8", help="HuggingFace Modell-ID")
    parser.add_argument("--server", action="store_true", default=False, help="Starte den FastAPI-Server (default: False)")
    return parser.parse_args()

def main():
    setup_logging()
    clean_memory()
    device, dtype = check_cuda()

    args = parse_args()
    init_image = load_image(args.input)

    result = run_image2image_pipeline(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        num_inference_steps=args.steps,
        guidance_scale=args.scale,
        init_image=init_image,
        model_id=args.model,
        seed=args.seed,
        device=device,
        dtype=dtype,
    )

    if result:
        result.save(args.output)
        logging.info(f"Fertig! Ergebnis gespeichert als {args.output}")

@app.route("/generate", methods=["POST"])
def generate():
    # Datei auslesen
    input_file = request.files.get("input")
    if not input_file:
        return "No input file uploaded", 400

    contents = input_file.read()
    print(f"Upload received: {input_file.filename}, size: {len(contents)} bytes")

    if len(contents) > MAX_UPLOAD_SIZE:
        print(f"Upload rejected: Datei zu groß ({len(contents)} Bytes > {MAX_UPLOAD_SIZE})")
        abort(413, description="File too large (>50MB)")

    tmp_dir = tempfile.mkdtemp()
    print(f"Temporary directory created: {tmp_dir}")

    input_path = os.path.join(tmp_dir, input_file.filename)
    with open(input_path, "wb") as f:
        f.write(contents)
    print(f"Input file saved: {input_path}")

    # Setup Umgebung
    print("Setting up environment...")
    setup_logging()
    clean_memory()
    device, dtype = check_cuda()
    print(f"Environment setup done. Device: {device}, Dtype: {dtype}")

    # Bild laden
    print(f"Loading input image from {input_path}...")
    init_image = load_image(input_path)
    print("Image loaded successfully.")

    output_filename = f"{uuid.uuid4().hex}.png"
    output_path = os.path.join(tmp_dir, output_filename)
    print(f"Output filename generated: {output_filename}")

    # Pipeline ausführen
    print("Starting image2image pipeline...")
    result = run_image2image_pipeline(
        prompt=request.form.get("prompt", ""),
        negative_prompt=request.form.get("negative_prompt", ""),
        num_inference_steps=int(request.form.get("steps", 25)),
        guidance_scale=float(request.form.get("scale", 7.5)),
        init_image=init_image,
        model_id=request.form.get("model", "lykon/dreamshaper-8"),
        seed=int(request.form.get("seed", 33)),
        device=device,
        dtype=dtype,
        strength=float(request.form.get("strength", 0.4))
    )
    print("Pipeline finished.")

    if not result:
        print("Image generation failed, cleaning up.")
        shutil.rmtree(tmp_dir)
        abort(500, description="Image generation failed")

    # Ergebnis speichern
    result.save(output_path)
    print(f"Result saved at {output_path}")

    # Antwort senden (hier: Dateiname und Pfad im tmp, anpassen je nach Usecase)
    return Response(base64.b64encode(open(output_path, "rb").read()).decode(), mimetype="text/plain")

# Datei als Antwort (direkt anzeigen im Browser)
    print(f"Returning file response: {output_path}")
    return FileResponse(output_path, media_type="image/png")

if __name__ == "__main__":
    print("Parsing parameters")
    args = parse_args()

    print("Checking Cuda")
    device, dtype = check_cuda()

    print("Loading Pipeline")
    load_pipeline(args.model, device, dtype)

    if args.server:
        app.run(host="0.0.0.0", port=9932)
    else:
        print("Main")
        main()
