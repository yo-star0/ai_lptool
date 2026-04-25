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
    """画像をRGBで読み込む."""
    img = Image.open(path).convert("RGB")
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

    @classmethod
    def from_polygon(cls, points: np.ndarray) -> "BBox":
        """4点ポリゴンから軸並行bboxを作る."""
        xs = points[:, 0]
        ys = points[:, 1]
        x = int(np.floor(xs.min()))
        y = int(np.floor(ys.min()))
        x2 = int(np.ceil(xs.max()))
        y2 = int(np.ceil(ys.max()))
        return cls(x, y, max(1, x2 - x), max(1, y2 - y))


def dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    """2値マスクを膨張させる. radius<=0なら何もしない."""
    if radius <= 0:
        return mask
    import cv2

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.dilate(mask, kernel)
