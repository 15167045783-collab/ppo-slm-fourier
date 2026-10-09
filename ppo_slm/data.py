from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError


def load_target(path: Path, image_size: int = 256) -> torch.Tensor:
    if image_size <= 0:
        raise ValueError("image_size must be positive")
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"target image does not exist: {path}")
    try:
        with Image.open(path) as image:
            grayscale = image.convert("L")
            if grayscale.size != (image_size, image_size):
                grayscale = grayscale.resize(
                    (image_size, image_size), Image.Resampling.BILINEAR
                )
            array = np.asarray(grayscale, dtype=np.float32) / 255.0
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError(f"could not decode target image: {path}") from exc

    if not np.isfinite(array).all():
        raise ValueError("target image contains non-finite values")
    maximum = float(array.max())
    if maximum <= 0.0:
        raise ValueError("target image is completely black")
    normalized = np.ascontiguousarray(array / maximum)
    return torch.from_numpy(normalized).unsqueeze(0)
