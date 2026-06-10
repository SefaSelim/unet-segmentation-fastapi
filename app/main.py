from fastapi import FastAPI, File, HTTPException, UploadFile

from app.inference import predict


app = FastAPI(
    title="U-Net Segmentation API",
    description="FastAPI service for CPU inference with a trained PyTorch U-Net model.",
    version="1.0.0",
)


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "U-Net Segmentation API is running"}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)) -> dict[str, object]:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")

    try:
        prediction = predict(image_bytes)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "prediction": prediction,
    }
