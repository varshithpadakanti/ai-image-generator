# AI-Powered Text-to-Image Generator
- generate.py: interactive CLI, runs Stable Diffusion locally with diffusers (GPU recommended).
- app.py: small web app for Vercel that calls the Hugging Face Inference API.

## Local CLI
    pip install -r requirements-local.txt
    python generate.py

## Web app locally
    pip install -r requirements.txt
    export HF_TOKEN=hf_xxx   # Windows PowerShell: $env:HF_TOKEN="hf_xxx"
    python app.py

## Vercel
Environment variables: HF_TOKEN (required), HF_MODEL (optional).
