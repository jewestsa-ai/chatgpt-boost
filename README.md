# ChatGPT Boost

Cloud-ready backend for running the project's tools and AI workloads away from the user's device.

## Architecture

`iPad/browser -> cloud API -> AI/model -> response`

Uploads are kept in a temporary directory during processing and removed when the request finishes.

## Run locally

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7860
```

## Container

```bash
docker build -t chatgpt-boost .
docker run -p 7860:7860 chatgpt-boost
```

The `/upscale` endpoint is the integration point for the Real-ESRGAN inference pipeline. The next deployment step is to connect this container to a cloud host with the required compute/GPU resources.
