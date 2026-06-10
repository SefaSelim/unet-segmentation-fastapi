# U-Net Segmentation FastAPI

This repository serves a previously trained PyTorch U-Net segmentation model with FastAPI.
It contains only inference code, Docker configuration, and the trained model weights required to run predictions.

## Repository Structure

```text
app/
  main.py
  model.py
  inference.py
  __init__.py
models/
  unet_segmentation_model.pth
requirements.txt
Dockerfile
docker-compose.yml
README.md
.gitignore
```

## Run With Docker Compose

From the repository root:

```bash
docker-compose up --build
```

The API will be available at:

```text
http://localhost:7001
```

Swagger documentation will be available at:

```text
http://localhost:7001/docs
```

## API Endpoints

### GET /

Returns a simple message confirming that the API is running.

Example:

```json
{
  "message": "U-Net Segmentation API is running"
}
```

### GET /health

Returns service health.

Example:

```json
{
  "status": "ok"
}
```

### POST /predict

Accepts an uploaded image using `multipart/form-data`, runs CPU inference with the U-Net model, and returns a JSON prediction.

Response fields:

- `filename`: uploaded file name
- `content_type`: uploaded file MIME type
- `prediction.label`: predicted class label
- `prediction.probability`: confidence score
- `prediction.mask_area_ratio`: ratio of pixels selected by the binary mask

## Test Prediction With Swagger

1. Start the service:

   ```bash
   docker-compose up --build
   ```

2. Open:

   ```text
   http://localhost:7001/docs
   ```

3. Open `POST /predict`.
4. Click `Try it out`.
5. Upload an image.
6. Click `Execute`.

## Test Prediction With curl

```bash
curl -X POST "http://localhost:7001/predict" \
  -H "accept: application/json" \
  -F "file=@sample.jpg;type=image/jpeg"
```

## Example JSON Response

```json
{
  "filename": "sample.jpg",
  "content_type": "image/jpeg",
  "prediction": {
    "label": "tumor",
    "probability": 0.6372,
    "mask_area_ratio": 0.1245
  }
}
```

## Configuration

The container uses CPU inference only. Runtime settings can be changed in `docker-compose.yml`:

- `MODEL_PATH`: model weights path
- `IMAGE_SIZE`: preprocessing resize size, default `256`
- `MASK_THRESHOLD`: sigmoid threshold for the binary mask, default `0.5`
- `AREA_RATIO_THRESHOLD`: minimum positive mask ratio for the positive label, default `0.01`
- `POSITIVE_LABEL`: default `tumor`
- `NEGATIVE_LABEL`: default `no_tumor`

## ARM macOS Notes

The Docker image uses the multi-architecture `python:3.11-slim` base image and CPU PyTorch packages from PyPI, so it is suitable for Apple Silicon machines such as M1, M2, and M3.
