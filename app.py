import os
import tempfile
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from PIL import Image

app = FastAPI(title="ChatGPT Boost Cloud API")

@app.get("/")
def health():
    return {"status": "ok", "service": "chatgpt-boost"}

@app.post("/upscale")
async def upscale(file: UploadFile = File(...)):
    suffix = Path(file.filename or "input").suffix.lower() or ".png"
    with tempfile.TemporaryDirectory() as work:
        src = Path(work) / f"input{suffix}"
        dst = Path(work) / "output_4k.png"
        src.write_bytes(await file.read())
        try:
            image = Image.open(src).convert("RGB")
        except Exception as exc:
            raise HTTPException(400, "Invalid image") from exc

        # Placeholder integration point for the Real-ESRGAN inference pipeline.
        # The cloud container is intentionally stateless: files live only in this
        # request's temporary directory and are deleted when the request ends.
        target_h = 3840
        ratio = target_h / image.height
        target = (round(image.width * ratio), target_h)
        image.resize(target, Image.Resampling.LANCZOS).save(dst)

        return FileResponse(dst, media_type="image/png", filename="upscaled_4k.png")
