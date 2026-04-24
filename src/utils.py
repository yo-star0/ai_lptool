"""共通ユーティリティ."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def load_image(path: str | Path) -> Image.Image:
    img = Image.open(path).convert("RGBA")
    return img


def pil_to_np(img: Image.Image) -> np.ndarray:
    return np.array(img)


def np_to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(arr)


@dataclass
class BBox:
    """矩形領域 (左上原点, ピクセル)."""

    x: int
    y: int
    w: int
    h: int

    @property
    def x2(self) -> int:
        return self.x + self.w

    @property
    def y2(self) -> int:
        return self.y + self.h

    def expanded(self, pad: int, max_w: int, max_h: int) -> "BBox":
        x = max(0, self.x - pad)
        y = max(0, self.y - pad)
        x2 = min(max_w, self.x2 + pad)
        y2 = min(max_h, self.y2 + pad)
        return BBox(x, y, x2 - x, y2 - y)
