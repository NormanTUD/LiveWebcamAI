import sys
import base64
import os
import tempfile
import uuid
import shutil
import gc
import argparse
import logging
import time

from typing import Optional

from PIL import Image
import torch
from diffusers import AutoPipelineForImage2Image, DEISMultistepScheduler
from flask import Flask, request, abort, Response
from beartype import beartype

LAST_GENERATED_IMAGE = None

app = Flask(__name__)

PIPE = None

# Max 50 MB Upload limit (50 * 1024 * 1024 bytes)
MAX_UPLOAD_SIZE = 50 * 1024 * 1024

@beartype
def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

@beartype
def clean_memory() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

@beartype
def check_cuda() -> str:
    if not torch.cuda.is_available():
        logging.error("CUDA ist nicht verfügbar – überprüfe deine PyTorch/GPU-Installation!")
        return "cpu"
    logging.info(f"CUDA verfügbar: {torch.cuda.get_device_name(0)}")
    return "cuda"

@beartype
def load_image(path: str, size=(512, 512)) -> Image.Image:
    if not os.path.exists(path):
        logging.error(f"Bilddatei '{path}' nicht gefunden!")
        sys.exit(1)
    try:
        return Image.open(path).convert("RGB").resize(size)
    except Exception as e:
        logging.error(f"Fehler beim Laden oder Verarbeiten des Bildes: {e}")
        sys.exit(1)

@beartype
def load_pipeline(model_id: str):
    global PIPE

    try:
        logging.info("Lade Pipeline...")
        PIPE = AutoPipelineForImage2Image.from_pretrained(
            model_id,
            torch_dtype=dtype,
            variant="fp16",
            low_cpu_mem_usage=True,
        )

        PIPE.scheduler = DEISMultistepScheduler.from_config(PIPE.scheduler.config)
        PIPE.safety_checker = None
        PIPE.enable_xformers_memory_efficient_attention()
        PIPE.enable_attention_slicing()

        PIPE = PIPE.to(device)
        logging.info(f"Pipeline erfolgreich geladen auf Gerät: {next(PIPE.unet.parameters()).device}")
    except Exception as e:
        logging.error(f"Fehler beim Laden der Pipeline: {e}")
        sys.exit(1)

@beartype
def run_warmup(image: Image.Image, guidance_scale):
    try:
        logging.info("Führe Warmup-Durchlauf durch...")
        _ = PIPE(
            prompt="simple warmup",
            image=[image],
            num_inference_steps=5,
            guidance_scale=guidance_scale  # .0,
        )
    except Exception as e:
        logging.warning(f"Warmup fehlgeschlagen (wird ignoriert): {e}")

@beartype
def run_image2image_pipeline(
    prompt: str,
    negative_prompt: str,
    num_inference_steps: int,
    guidance_scale: float,
    strength: float = 0.3,
    init_image: Optional[Image.Image] = None,
    seed: int = 33
) -> Optional[Image.Image]:
    global LAST_GENERATED_IMAGE

    start_total = time.perf_counter()
    logging.info("🟢 Starte run_image2image_pipeline")

    # Schritt 1: Initialbild bestimmen
    start = time.perf_counter()
    if init_image is None:
        if LAST_GENERATED_IMAGE is None:
            logging.error("❌ Kein Startbild übergeben und auch kein vorheriges Bild vorhanden.")
            return None
        logging.info("ℹ️ Verwende vorheriges generiertes Bild als init_image.")
        init_image = LAST_GENERATED_IMAGE
    end = time.perf_counter()
    logging.info(f"✅ Schritt 1 (Init-Bild bestimmen) dauerte {end - start:.3f} Sekunden")

    # Schritt 2: Warmup
    start = time.perf_counter()
    run_warmup(init_image, guidance_scale)
    end = time.perf_counter()
    logging.info(f"✅ Schritt 2 (Warmup) dauerte {end - start:.3f} Sekunden")

    # Schritt 3: Generator vorbereiten
    start = time.perf_counter()
    generator = torch.Generator(device=device).manual_seed(seed)
    end = time.perf_counter()
    logging.info(f"✅ Schritt 3 (Seed setzen) dauerte {end - start:.3f} Sekunden")

    # Schritt 4: Bildgenerierung
    start = time.perf_counter()
    logging.info("🖼️ Starte Bildgenerierung mit Diffusion Pipeline...")
    output = PIPE(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=[init_image],
        generator=generator,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        strength=strength
    )
    end = time.perf_counter()
    logging.info(f"✅ Schritt 4 (Bildgenerierung) dauerte {end - start:.3f} Sekunden")

    # Schritt 5: Ergebnis prüfen und speichern
    start = time.perf_counter()
    if output and output.images and len(output.images) > 0:
        result = output.images[0]
        LAST_GENERATED_IMAGE = result
        end = time.perf_counter()
        logging.info(f"✅ Schritt 5 (Ergebnis speichern) dauerte {end - start:.3f} Sekunden")
        logging.info(f"🎉 Gesamtzeit: {time.perf_counter() - start_total:.3f} Sekunden")
        return result

    logging.error("❌ Kein Bild wurde generiert!")
    logging.info(f"❌ Gesamtzeit bis Fehler: {time.perf_counter() - start_total:.3f} Sekunden")
    return None

@beartype
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

@beartype
def main() -> None:
    setup_logging()
    clean_memory()

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

def setup_logging():
    # Hier eigenes Logging einrichten
    pass

def clean_memory():
    # GPU Speicher etc. freigeben
    pass

def load_image(path):
    from PIL import Image
    return Image.open(path).convert("RGB")

def run_image2image_pipeline(
    prompt,
    negative_prompt,
    num_inference_steps,
    guidance_scale,
    init_image,
    seed,
    strength
):
    # Hier die eigentliche Pipeline aufrufen
    # Muss exceptions werfen oder None zurückgeben, wenn Fehler
    # Beispiel Dummy-Implementierung:
    try:
        # --- Pipeline Code ---
        # z.B. output = PIPE(...)
        # return output_image
        pass
    except Exception as e:
        print(f"Pipeline error: {e}")
        return None

@beartype
@app.route("/generate", methods=["POST"])
def generate():
    tmp_dir = None
    try:
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

        safe_filename = os.path.basename(input_file.filename)
        if not safe_filename:
            safe_filename = "input_image"

        input_path = os.path.join(tmp_dir, safe_filename)
        with open(input_path, "wb") as f:
            f.write(contents)
        print(f"Input file saved: {input_path}")

        print("Setting up environment...")
        setup_logging()
        clean_memory()
        print(f"Environment setup done. Device: {device}, Dtype: {dtype}")

        print(f"Loading input image from {input_path}...")
        init_image = load_image(input_path)
        print("Image loaded successfully.")

        output_filename = f"{uuid.uuid4().hex}.png"
        output_path = os.path.join(tmp_dir, output_filename)
        print(f"Output filename generated: {output_filename}")

        # Parameter aus Formular holen & validieren
        def parse_int(name, default):
            try:
                return int(request.form.get(name, default))
            except Exception as e:
                print(f"Invalid int for {name}: {e}, fallback to {default}")
                return default

        def parse_float(name, default):
            try:
                return float(request.form.get(name, default))
            except Exception as e:
                print(f"Invalid float for {name}: {e}, fallback to {default}")
                return default

        params = {
            "prompt": request.form.get("prompt", "").strip(),
            "negative_prompt": request.form.get("negative_prompt", "").strip(),
            "num_inference_steps": parse_int("steps", 25),
            "guidance_scale": parse_float("scale", 7.5),
            "init_image": init_image,
            "seed": parse_int("seed", 33),
            "strength": parse_float("strength", 0.4)
        }

        print("Parameters for image2image pipeline:")
        for k, v in params.items():
            if k != "init_image":
                print(f"  {k}: {v}")
            else:
                print(f"  {k}: <PIL.Image.Image object>")

        print("Starting image2image pipeline...")
        result = run_image2image_pipeline(**params)
        print("Pipeline finished.")

        if result is None:
            print("Image generation failed, cleaning up.")
            abort(500, description="Image generation failed")

        result.save(output_path)
        print(f"Result saved at {output_path}")

        with open(output_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()

        # Cleanup temporäres Verzeichnis nach erfolgreichem Versand
        shutil.rmtree(tmp_dir)
        tmp_dir = None

        return Response(encoded, mimetype="text/plain")

    except Exception as e:
        print(f"Exception in /generate: {e}")
        # Bei Exception temp_dir löschen, falls existiert
        if tmp_dir and os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        abort(500, description=f"Internal Server Error: {e}")

dtype = torch.float16
device = check_cuda()
args = parse_args()

if __name__ == "__main__":
    print("Parsing parameters")

    print("Checking Cuda")

    print("Loading Pipeline")
    load_pipeline(args.model)

    if args.server:
        app.run(host="0.0.0.0", port=9932)
    else:
        print("Main")
        main()
