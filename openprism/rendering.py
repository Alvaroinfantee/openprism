"""Serialization helpers for the lightweight operator console."""

from __future__ import annotations

import base64
from io import BytesIO

import numpy as np
from PIL import Image


def image_data_url(
    array: np.ndarray,
    *,
    image_format: str = "JPEG",
    quality: int = 88,
) -> str:
    image = Image.fromarray(np.asarray(array, dtype=np.uint8), mode="RGB")
    buffer = BytesIO()
    if image_format.upper() == "JPEG":
        image.save(buffer, format="JPEG", quality=quality, optimize=True)
        mime = "image/jpeg"
    elif image_format.upper() == "PNG":
        image.save(buffer, format="PNG", optimize=True)
        mime = "image/png"
    else:
        raise ValueError(f"unsupported image format: {image_format}")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:{mime};base64,{encoded}"

