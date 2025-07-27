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
import inspect

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

CURRENTLY_LOADING_PIPELINE = False
LAST_GENERATED_IMAGE = None
CURRENT_MODEL_ID = None

app = Flask(__name__)

PIPES = []

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
def count_available_gpus() -> int:
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    else:
        return 0

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

def insert_or_replace(index, pipe):
    global PIPES
    if index < len(PIPES):
        PIPES[index] = pipe
    else:
        PIPES.append(pipe)

@beartype
def load_pipeline(model_id: str) -> None:
    global CURRENTLY_LOADING_PIPELINE

    while CURRENTLY_LOADING_PIPELINE:
        time.sleep(200)

    CURRENTLY_LOADING_PIPELINE = True

    global CURRENT_MODEL_ID

    if model_id == CURRENT_MODEL_ID and len(PIPES) == count_available_gpus():
        console.print(f"Modell '{model_id}' ist bereits geladen. Verwende bestehende Pipeline.")

        CURRENTLY_LOADING_PIPELINE = False
        return None

    nr_gpus = count_available_gpus()

    try:
        for i in range(0, nr_gpus):
            if i in PIPES:
                if PIPES[i]["name"] = model_id:
                    continue

            console.print("-> -> -> -> -> -> -> -> -> -> -> -> -> -> -> -> -> ->")
            console.print(f"Lade Pipeline für Modell '{model_id}' für GPU Nr. {i + 1}/{nr_gpus}...")
            console.print("-> -> -> -> -> -> -> -> -> -> -> -> -> -> -> -> -> ->")


            pipe = {}

            pipe["function"] = AutoPipelineForImage2Image.from_pretrained(
                model_id,
                torch_dtype=dtype,
                # variant="fp16",
                low_cpu_mem_usage=True,
            )

            pipe["is_blocked"] = False

            pipe["function"].scheduler = DEISMultistepScheduler.from_config(pipe["function"].scheduler.config)
            pipe["function"].safety_checker = None
            pipe["function"].enable_xformers_memory_efficient_attention()
            pipe["function"].enable_attention_slicing()

            pipe["function"] = pipe["function"].to(f"cuda:{i}")

            if i == 0:
                sig = inspect.signature(pipe["function"])

                table = Table(title="Parameters of the AutoPipelineForImage2Image")

                table.add_column("Name", style="bold")
                table.add_column("Default", style="dim")
                table.add_column("Annotation", style="cyan")

                for name, param in sig.parameters.items():
                    default = (
                        "–" if param.default is inspect.Parameter.empty else repr(param.default)
                    )
                    annotation = (
                        "–" if param.annotation is inspect.Parameter.empty else repr(param.annotation)
                    )
                    table.add_row(name, default, annotation)

                console.print(table)

            CURRENT_MODEL_ID = model_id
            LAST_GENERATED_IMAGE = None

            insert_or_replace(i, pipe)

            console.print(f"Pipeline erfolgreich geladen auf GPU Nr. {i + 1}/{nr_gpus}")

            PIPES[i]["name"] = model_id

    except Exception as e:
        CURRENTLY_LOADING_PIPELINE = False
        logging.error(f"Fehler beim Laden der Pipeline: {e}")
        sys.exit(1)

    CURRENTLY_LOADING_PIPELINE = False
    return None

WARMUP_DONE = False

@beartype
def run_warmup(image: Image.Image, guidance_scale: float, pipe_nr: int):
    global WARMUP_DONE
    if WARMUP_DONE:
        return
    try:
        console.print("Führe Warmup-Durchlauf durch...")
        _ = PIPES[pipe_nr]["function"](
            prompt="simple warmup",
            image=[image],
            num_inference_steps=2,
            guidance_scale=guidance_scale
        )
        WARMUP_DONE = True
    except Exception as e:
        logging.warning(f"Warmup fehlgeschlagen (wird ignoriert): {e}")

def merge_image_with_previous_one_if_available(img1):
    if LAST_GENERATED_IMAGE:
        return crossfade_images(img1, LAST_GENERATED_IMAGE, 0.1)

    return img1

@beartype
def get_pipe_nr():
    while True:
        for i in range(0, len(PIPES)):
            if not PIPES[i]["is_blocked"]:
                return i

        time.sleep(0.01)

@beartype
def block_pipe(pipe_nr) -> None:
    PIPES[pipe_nr]["is_blocked"] = True

@beartype
def release_pipe(pipe_nr) -> None:
    PIPES[pipe_nr]["is_blocked"] = False

@beartype
def run_image2image_pipeline(
    prompt: str,
    negative_prompt: str,
    num_inference_steps: int,
    guidance_scale: float,
    strength: float = 0.3,
    init_image: Optional[Image.Image] = None,
    seed: int = 33,
    clamped_values: Optional[dict] = None,
    model: str = "dreamlike-art/dreamlike-photoreal-2.0"
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

    pipe_nr = get_pipe_nr()

    block_pipe(pipe_nr)

    # Schritt 2: Warmup
    start = time.perf_counter()
    run_warmup(init_image, guidance_scale, pipe_nr)
    end = time.perf_counter()
    timings["Warmup"] = end - start

    # Schritt 3: Generator vorbereiten
    start = time.perf_counter()
    generator = torch.Generator(device=f"cuda:{pipe_nr}").manual_seed(seed)
    end = time.perf_counter()
    timings["Seed setzen"] = end - start

    # Schritt 4: Bildgenerierung
    start = time.perf_counter()
    console.print("🖼️ Starte Bildgenerierung mit Diffusion Pipeline...")
    output = PIPES[pipe_nr]["function"](
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=[merge_image_with_previous_one_if_available(init_image)],
        generator=generator,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        strength=strength
    )

    release_pipe(pipe_nr)

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
    parser.add_argument("--model", default="prompthero/openjourney", help="HuggingFace Modell-ID")
    parser.add_argument("--server", action="store_true", default=False, help="Starte den FastAPI-Server (default: False)")
    return parser.parse_args()

def crossfade_images(img1, img2, alpha):
    return Image.blend(img1, img2, alpha)

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
        console.print(f"Fertig! Ergebnis gespeichert als {args.output}")

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
