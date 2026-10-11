"""Interactive CLI text-to-image generator (Stable Diffusion v1.5 via Hugging Face diffusers).

Run locally:  python generate.py            -> interactive prompt loop
              python generate.py -p "a lighthouse at dawn, oil painting"
Needs: pip install -r requirements-local.txt   (a GPU is strongly recommended)
"""
import argparse
import os
import re
from datetime import datetime

import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline

# The original runwayml/stable-diffusion-v1-5 repo was taken down; this is the community mirror.
# Override with:  SD_MODEL=runwayml/stable-diffusion-v1-5 python generate.py
MODEL_ID = os.environ.get("SD_MODEL", "stable-diffusion-v1-5/stable-diffusion-v1-5")
OUT_DIR = "outputs"


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_pipeline():
    device = pick_device()
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"Loading {MODEL_ID} on {device} (first run downloads ~4 GB)...")
    pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=dtype)
    # Inference optimisations: faster scheduler (fewer steps) + lower memory use.
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_attention_slicing()
    return pipe.to(device), device


def generate(pipe, device, prompt, negative="", steps=25, guidance=7.5, seed=None, size=512):
    gen = torch.Generator(device="cpu" if device == "mps" else device)
    if seed is not None:
        gen.manual_seed(seed)
    result = pipe(prompt, negative_prompt=negative or None, num_inference_steps=steps,
                  guidance_scale=guidance, width=size, height=size, generator=gen)
    os.makedirs(OUT_DIR, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", prompt.lower()).strip("-")[:40] or "image"
    path = os.path.join(OUT_DIR, f"{datetime.now():%Y%m%d-%H%M%S}-{slug}.png")
    result.images[0].save(path)
    return path


def main():
    ap = argparse.ArgumentParser(description="Text-to-image with Stable Diffusion")
    ap.add_argument("-p", "--prompt", help="Generate one image and exit")
    ap.add_argument("--negative", default="blurry, low quality, distorted", help="Negative prompt")
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--size", type=int, default=512, choices=[256, 384, 512], help="Square size in px")
    args = ap.parse_args()

    pipe, device = load_pipeline()
    opts = dict(negative=args.negative, steps=args.steps, guidance=args.guidance,
                seed=args.seed, size=args.size)

    if args.prompt:
        print("Saved:", generate(pipe, device, args.prompt, **opts))
        return

    print("Type a prompt and press Enter. Commands: /steps N, /seed N, /quit")
    while True:
        try:
            text = input("\nprompt> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            continue
        if text == "/quit":
            break
        if text.startswith("/steps "):
            opts["steps"] = max(5, min(100, int(text.split()[1]))); print("steps =", opts["steps"]); continue
        if text.startswith("/seed "):
            opts["seed"] = int(text.split()[1]); print("seed =", opts["seed"]); continue
        print("Saved:", generate(pipe, device, text, **opts))


if __name__ == "__main__":
    main()
