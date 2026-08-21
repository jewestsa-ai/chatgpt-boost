"""AI image upscaler targeting 4K output using Real-ESRGAN."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet


def upscale_to_4k(input_path: str, output_path: str) -> tuple[int, int]:
    """Enhance an image with Real-ESRGAN, then fit it to 4K without cropping."""
    image = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {input_path}")

    model = RRDBNet(
        num_in_ch=3, num_out_ch=3, num_feat=64,
        num_block=23, num_grow_ch=32, scale=4,
    )
    upsampler = RealESRGANer(
        scale=4,
        model_path="weights/RealESRGAN_x4plus.pth",
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=False,
    )

    enhanced, _ = upsampler.enhance(image, outscale=4)
    rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)

    target_w, target_h = 3840, 2160
    # Keep the original aspect ratio and fit inside a 4K canvas.
    scale = min(target_w / pil.width, target_h / pil.height)
    size = (round(pil.width * scale), round(pil.height * scale))
    pil = pil.resize(size, Image.Resampling.LANCZOS)

    canvas = Image.new("RGB", (target_w, target_h))
    canvas.paste(pil, ((target_w - pil.width) // 2, (target_h - pil.height) // 2))
    canvas.save(output_path, quality=98)
    return canvas.size


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    args = parser.parse_args()
    print(upscale_to_4k(args.input, args.output))


if __name__ == "__main__":
    main()
