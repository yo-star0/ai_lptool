"""simple-lama-inpainting ラッパ（背景復元）.

TODO(次フェーズ): 実装を入れる。現状はインタフェース定義のみ。
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from .utils import get_logger

logger = get_logger(__name__)


class Inpainter:
    def __init__(self, device: str = "cpu"):
        self.device = device
        self._lama = None

    def _ensure_loaded(self) -> None:
        if self._lama is not None:
            return
        from simple_lama_inpainting import SimpleLama  # type: ignore

        logger.info("Loading SimpleLama (device=%s)...", self.device)
        self._lama = SimpleLama()

    def inpaint(self, image: Image.Image, mask: np.ndarray) -> Image.Image:
        """`image` に対して `mask` で示す領域を背景復元する.

        mask: 2次元配列, 0-255 (255の領域が消去対象)
        """
        raise NotImplementedError("Inpainter.inpaint は次フェーズで実装します")
