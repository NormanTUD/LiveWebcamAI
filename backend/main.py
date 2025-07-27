# Tested models:
# Work:
# - lykon/dreamshaper-8
# - runwayml/stable-diffusion-v1-5
# - dreamlike-art/dreamlike-photoreal-2.0
# - gsdf/Counterfeit-V2.5
# - dreamlike-art/dreamlike-photoreal-2.0
# Don't work:
# - stabilityai/stable-diffusion-2-inpainting
# - kandinsky-community/kandinsky-2-2
# - civitai/anything-v4.5
# - imagepipeline/Realistic-Stock-Photo

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
from rich.console import Console
from rich.table import Table
from rich import box
from rich.style import Style

from PIL import Image
import torch
from diffusers import AutoPipelineForImage2Image, DEISMultistepScheduler
from flask import Flask, request, abort, Response, send_file
from beartype import beartype

console = Console()

LAST_GENERATED_IMAGE = None
CURRENT_MODEL_ID = None

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
    global PIPE, CURRENT_MODEL_ID

    if model_id == CURRENT_MODEL_ID and PIPE is not None:
        logging.info(f"Modell '{model_id}' ist bereits geladen. Verwende bestehende Pipeline.")
        return PIPE

    try:
        logging.info(f"Lade Pipeline für Modell '{model_id}'...")
        PIPE = AutoPipelineForImage2Image.from_pretrained(
            model_id,
            torch_dtype=dtype,
            # variant="fp16",
            low_cpu_mem_usage=True,
        )

        PIPE.scheduler = DEISMultistepScheduler.from_config(PIPE.scheduler.config)
        PIPE.safety_checker = None
        PIPE.enable_xformers_memory_efficient_attention()
        PIPE.enable_attention_slicing()

        PIPE = PIPE.to(device)
        CURRENT_MODEL_ID = model_id
        logging.info(f"Pipeline erfolgreich geladen auf Gerät: {next(PIPE.unet.parameters()).device}")

        return PIPE

    except Exception as e:
        logging.error(f"Fehler beim Laden der Pipeline: {e}")
        sys.exit(1)

WARMUP_DONE = False

@beartype
def run_warmup(image: Image.Image, guidance_scale):
    global WARMUP_DONE
    if WARMUP_DONE:
        return
    try:
        logging.info("Führe Warmup-Durchlauf durch...")
        _ = PIPE(
            prompt="simple warmup",
            image=[image],
            num_inference_steps=2,
            guidance_scale=guidance_scale
        )
        WARMUP_DONE = True
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
    seed: int = 33,
    clamped_values: Optional[dict] = None
) -> Optional[Image.Image]:
    global LAST_GENERATED_IMAGE

    start_total = time.perf_counter()
    console.rule("[bold green]Start: run_image2image_pipeline")

    timings = {}

    # Schritt 1: Initialbild bestimmen
    start = time.perf_counter()
    if init_image is None:
        if LAST_GENERATED_IMAGE is None:
            console.print("[bold red]❌ Kein Startbild übergeben und auch kein vorheriges Bild vorhanden.")
            return None
        console.print("[yellow]ℹ️ Verwende vorheriges generiertes Bild als init_image.")
        init_image = LAST_GENERATED_IMAGE
    if init_image.mode != "RGB":
        console.print("[yellow]⚠️ Konvertiere init_image zu RGB")
        init_image = init_image.convert("RGB")
    if init_image.size[0] < 64 or init_image.size[1] < 64:
        console.print("[bold red]❌ Init-Image ist zu klein!")
    end = time.perf_counter()
    timings["Init-Bild bestimmen"] = end - start

    # Schritt 2: Warmup
    start = time.perf_counter()
    run_warmup(init_image, guidance_scale)
    end = time.perf_counter()
    timings["Warmup"] = end - start

    # Schritt 3: Generator vorbereiten
    start = time.perf_counter()
    generator = torch.Generator(device=device).manual_seed(seed)
    end = time.perf_counter()
    timings["Seed setzen"] = end - start

    # Schritt 4: Bildgenerierung
    start = time.perf_counter()
    console.print("🖼️ Starte Bildgenerierung mit Diffusion Pipeline...")
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
    timings["Bildgenerierung"] = end - start

    # Schritt 5: Ergebnis prüfen und speichern
    start = time.perf_counter()
    if output and output.images and len(output.images) > 0:
        result = output.images[0]
        LAST_GENERATED_IMAGE = result
        end = time.perf_counter()
        timings["Ergebnis speichern"] = end - start
        total_time = time.perf_counter() - start_total
        timings["Gesamtzeit"] = total_time

        console.rule("[bold cyan]⏱️ Zeiten der Schritte")
        table = Table(title="Pipeline-Schritte", box=box.SIMPLE)
        table.add_column("Schritt", style="bold")
        table.add_column("Dauer (Sekunden)", justify="right")
        for name, duration in timings.items():
            table.add_row(name, f"{duration:.3f}")
        console.print(table)

        if clamped_values:
            console.rule("[bold magenta]🔧 Clamped Parameter")
            clamp_table = Table(title="Parameter Clamping", box=box.SIMPLE_HEAVY)
            clamp_table.add_column("Parameter", style="bold")
            clamp_table.add_column("Original", justify="right")
            clamp_table.add_column("Clamped", justify="right")
            for key, (original, clamped) in clamped_values.items():
                highlight = Style(bgcolor="orange3") if original != clamped else None
                clamp_table.add_row(str(key), str(original), str(clamped), style=highlight)
            console.print(clamp_table)

        console.print(f"[bold green]🎉 Bildgenerierung abgeschlossen in {total_time:.3f} Sekunden")
        return result

    console.print("[bold red]❌ Kein Bild wurde generiert!")
    console.print(f"[red]❌ Gesamtzeit bis Fehler: {time.perf_counter() - start_total:.3f} Sekunden")
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

def clamp_params(params):
    # num_inference_steps: typischer Bereich 1-50, clamp auf 5-50 z.B.
    steps = params.get("num_inference_steps", 25)
    if steps < 5:
        steps = 5
    elif steps > 50:
        steps = 50
    params["num_inference_steps"] = steps

    # guidance_scale: üblicher Bereich ca. 1-20, clamp 1-20
    scale = params.get("guidance_scale", 7.5)
    if scale < 1:
        scale = 1.0
    elif scale > 20:
        scale = 20.0
    params["guidance_scale"] = scale

    # strength: meist 0..1, aber 1.0 kann problematisch sein, setze max 0.99
    strength = params.get("strength", 0.4)
    if strength <= 0:
        strength = 0.01
    elif strength >= 1.0:
        strength = 0.99
    params["strength"] = strength

    # seed >=0
    seed = params.get("seed", 33)
    if seed < 0:
        seed = 0
    params["seed"] = seed

    return params

@beartype
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

    clean_memory()

    # Bild laden
    print(f"Loading input image from {input_path}...")
    init_image = load_image(input_path)
    print("Image loaded successfully.")

    output_filename = f"{uuid.uuid4().hex}.png"
    output_path = os.path.join(tmp_dir, output_filename)
    print(f"Output filename generated: {output_filename}")

    # Pipeline ausführen

    prompt = request.form.get("prompt", "") + ", ultra detailed, 8k, realistic lighting, sharp focus"

    params = {
        "prompt": prompt,
        "negative_prompt": request.form.get("negative_prompt", ""),
        "num_inference_steps": int(request.form.get("steps", 25)),
        "guidance_scale": float(request.form.get("scale", 7.5)),
        "init_image": init_image,
        "seed": int(request.form.get("seed", 33)),
        "strength": float(request.form.get("strength", 0.4)),
        "model": request.form.get("model", "dreamlike-art/dreamlike-photoreal-2.0")
    }

    load_pipeline(params["model"])

    print("Original parameters:")
    for k, v in params.items():
        print(f"  {k}: {v}")

    # Clamp params vor Pipeline-Call
    params = clamp_params(params)

    print("Clamped parameters:")
    for k, v in params.items():
        print(f"  {k}: {v}")

    print("Starting image2image pipeline...")

    pipeline_start_time = time.time()

    try:
        result = run_image2image_pipeline(**params)
    except Exception as e:
        print(f"Pipeline error: {e}")
        # Fallback: versuche mit Standard-Params
        print("Retrying with fallback parameters...")
        fallback_params = {
            **params,
            "num_inference_steps": 25,
            "strength": 0.4,
            "guidance_scale": 7.5,
        }
        try:
            result = run_image2image_pipeline(**fallback_params)
        except Exception as e2:
            print(f"Fallback also failed: {e2}")
            shutil.rmtree(tmp_dir)
            abort(500, description="Image generation failed after fallback")

    pipeline_start_end = time.time()

    pipeline_time = pipeline_start_end - pipeline_start_time

    print(f"Pipeline finished, took {pipeline_time}s.")

    if not result:
        print("Image generation failed, cleaning up.")
        shutil.rmtree(tmp_dir)
        abort(500, description="Image generation failed")

    # Ergebnis speichern
    result.save(output_path)
    print(f"Result saved at {output_path}")

    # Antwort senden (hier: Dateiname und Pfad im tmp, anpassen je nach Usecase)
    return send_file(output_path, mimetype="image/png")

setup_logging()
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
