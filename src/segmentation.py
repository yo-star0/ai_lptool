"""rembg ラッパ（人物/商品マスク抽出）.

TODO(次フェーズ): 実装を入れる。現状はインタフェース定義のみ。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .utils import BBox, get_logger

logger = get_logger(__name__)


@dataclass
class Subject:
    """切り抜かれた被写体."""

    rgba: Image.Image  # RGBA画像 (背景透過済み)
    mask: np.ndarray  # 2次元マスク (uint8, 0-255)
    bbox: BBox
    label: str = "subject"  # "person" / "product" / "subject"


class Segmenter:
    def __init__(self, model_name: str = "isnet-general-use"):
        self.model_name = model_name
        self._session = None

    def _ensure_loaded(self) -> None:
        if self._session is not None:
            return
        from rembg import new_session  # type: ignore

        logger.info("Loading rembg session (model=%s)...", self.model_name)
        self._session = new_session(self.model_name)

    def segment(self, image: Image.Image) -> list[Subject]:
        raise NotImplementedError("Segmenter.segment は次フェーズで実装します")
