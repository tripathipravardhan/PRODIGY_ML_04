"""Helpers shared by the training script and the Streamlit app."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_SIZE = (224, 224)


def gesture_name(folder_name: str) -> str:
    """Convert a dataset folder such as 01_palm into palm."""
    return folder_name.split("_", maxsplit=1)[-1].replace("_", " ")


def load_labels(path: str | Path) -> list[str]:
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def prepare_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize(IMAGE_SIZE)
    return np.expand_dims(np.asarray(image, dtype=np.float32), axis=0)