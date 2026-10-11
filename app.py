"""Web front-end for Vercel.

Vercel can't run Stable Diffusion itself (no GPU, 250 MB size limit), so this
app sends the prompt to the Hugging Face Inference API and returns the image.
"""
import base64
import os

import requests
from flask import Flask, jsonify, render_template, request

app = Flask(__name__, static_folder="public", static_url_path="")

HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_MODEL = os.environ.get("HF_MODEL", "stable-diffusion-v1-5/stable-diffusion-v1-5")
API_URL = f"https://router.huggingface.co/hf-inference/models/{HF_MODEL}"


@app.route("/")
def index():
    return render_template("index.html", model=HF_MODEL)


@app.post("/generate")
def generate():
    if not HF_TOKEN:
        return jsonify(error="Server is missing HF_TOKEN."), 500
    data = request.get_json(silent=True) or {}
    prompt = str(data.get("prompt", "")).strip()
    if not prompt:
        return jsonify(error="Type a prompt first."), 400
    if len(prompt) > 300:
        return jsonify(error="Keep the prompt under 300 characters."), 400

    try:
        r = requests.post(
            API_URL,
            headers={"Authorization": f"Bearer {HF_TOKEN}", "Accept": "image/png"},
            json={"inputs": prompt,
                  "parameters": {"width": 512, "height": 512, "num_inference_steps": 25}},
            timeout=55,
        )
    except requests.RequestException:
        return jsonify(error="The image service didn't answer in time. Try again."), 504

    if r.status_code == 200 and r.headers.get("content-type", "").startswith("image/"):
        b64 = base64.b64encode(r.content).decode()
        return jsonify(image=f"data:{r.headers['content-type']};base64,{b64}")
    if r.status_code == 503:
        return jsonify(error="The model is warming up. Try again in about 30 seconds."), 503
    return jsonify(error=f"Image service returned {r.status_code}. Check HF_TOKEN and HF_MODEL."), 502


if __name__ == "__main__":
    app.run(debug=True)
