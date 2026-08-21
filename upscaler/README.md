# AI 4K Upscaler

Uses Real-ESRGAN to reconstruct image detail before producing a 3840×2160 4K output.

## Setup

1. Install dependencies from `requirements.txt`.
2. Download `RealESRGAN_x4plus.pth` into `upscaler/weights/` from the official Real-ESRGAN release.
3. Run:

```bash
python upscaler/upscale.py input.jpg output.jpg
```

The script performs AI super-resolution first; it is not simple pixel resizing.
