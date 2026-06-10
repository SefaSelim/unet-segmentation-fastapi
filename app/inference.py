import os
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch

from app.model import load_model


MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/unet_segmentation_model.pth"))
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "256"))
MASK_THRESHOLD = float(os.getenv("MASK_THRESHOLD", "0.5"))
AREA_RATIO_THRESHOLD = float(os.getenv("AREA_RATIO_THRESHOLD", "0.01"))
POSITIVE_LABEL = os.getenv("POSITIVE_LABEL", "tumor")
NEGATIVE_LABEL = os.getenv("NEGATIVE_LABEL", "no_tumor")


@lru_cache(maxsize=1)
def get_model():
    return load_model(MODEL_PATH)


def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    image_array = np.frombuffer(image_bytes, np.uint8)
    bgr_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if bgr_image is None:
        raise ValueError("Uploaded file could not be decoded as an image.")

    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    resized_image = cv2.resize(rgb_image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    normalized_image = resized_image.astype(np.float32) / 255.0
    chw_image = np.transpose(normalized_image, (2, 0, 1))

    return torch.from_numpy(chw_image).unsqueeze(0)


def predict(image_bytes: bytes) -> dict[str, float | str]:
    tensor = preprocess_image(image_bytes)
    model = get_model()

    with torch.no_grad():
        logits = model(tensor)
        probability_map = torch.sigmoid(logits).squeeze().cpu().numpy()

    binary_mask = probability_map >= MASK_THRESHOLD
    mask_area_ratio = float(binary_mask.mean())
    has_positive_mask = mask_area_ratio >= AREA_RATIO_THRESHOLD
    label = POSITIVE_LABEL if has_positive_mask else NEGATIVE_LABEL

    if has_positive_mask and binary_mask.any():
        probability = float(probability_map[binary_mask].mean())
    else:
        probability = float(1.0 - probability_map.mean())

    return {
        "label": label,
        "probability": round(probability, 4),
        "mask_area_ratio": round(mask_area_ratio, 4),
    }
