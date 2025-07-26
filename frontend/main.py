from flask import Flask, request, render_template, jsonify, Response
import requests
import uuid
import base64
import threading
import subprocess
import sys
import time
import os
import sys
import time
from beartype import beartype

app = Flask(__name__)

@beartype
@app.route("/")
def index():
    return render_template("index.html")

@beartype
@app.route("/generate", methods=["POST"])
def proxy_process():
    file = request.files.get("input")  # Name sollte "input" sein, wie im Backend erwartet
    if not file:
        print("[proxy_process] Kein Bild empfangen", file=sys.stderr)
        return jsonify({"error": "Kein Bild empfangen"}), 400

    # Lese Form-Daten aus
    prompt = request.form.get("prompt", "")
    negative_prompt = request.form.get("negative_prompt", "")
    steps = request.form.get("steps", "25")
    scale = request.form.get("scale", "7.5")
    seed = request.form.get("seed", "33")
    strength = request.form.get("strength", "0.4")
    model = request.form.get("model", "lykon/dreamshaper-8")

    try:
        # Bereite Files und Data für Backend-Request vor
        files = {"input": (file.filename, file.stream, file.mimetype)}
        data = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "steps": steps,
            "scale": scale,
            "seed": seed,
            "model": model,
            "strength": strength
        }

        print(f"[proxy_process] Sende Bild '{file.filename}' an Backend-Service...", file=sys.stderr)
        backend_res = requests.post("http://localhost:9932/generate", files=files, data=data, stream=True)
        backend_res.raise_for_status()
        print("[proxy_process] Antwort vom Backend erhalten, Status:", backend_res.status_code, file=sys.stderr)

        # Content-Type des Backend-Response weitergeben (z.B. image/png)
        content_type = backend_res.headers.get("Content-Type", "application/octet-stream")

        # Streaming Response an Client zurückgeben (Bild direkt)
        def generate():
            for chunk in backend_res.iter_content(chunk_size=8192):
                if chunk:
                    yield chunk

        return Response(generate(), content_type=content_type)

    except Exception as e:
        print(f"[proxy_process] Proxy-Fehler: {e}", file=sys.stderr)
        return jsonify({"error": f"Proxy-Fehler: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)
