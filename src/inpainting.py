"""simple-lama-inpainting ラッパ（背景復元）."""
from __future__ import annotations

import numpy as np
from PIL import Image

from .utils import get_logger

logger = get_logger(__name__)


class Inpainter:
    def __init__(self, device: str = "cpu"):
        """
        device: "cpu" / "cuda".
        simple-lama-inpainting は内部で torch を使う。CUDA 検出は torch 任せ。
        """
        self.device = device
        self._lama = None

    def _ensure_loaded(self) -> None:
        if self._lama is not None:
            return
        from simple_lama_inpainting import SimpleLama  # type: ignore

        logger.info("Loading SimpleLama (device=%s)...", self.device)
        # SimpleLama は引数で device を取らない実装が多い。torch.cuda.is_available() で自動判定。
        self._lama = SimpleLama()

    def inpaint(self, image: Image.Image, mask: np.ndarray) -> Image.Image:
        """`image` の `mask` 領域を背景復元する.

        mask: 2次元 uint8 (0-255). 255 (>127) の領域が消去対象。
        """
        self._ensure_loaded()
        assert self._lama is not None
        if mask.ndim != 2:
            raise ValueError(f"mask must be 2D, got shape {mask.shape}")
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)

        # マスクが完全に空ならインペイント不要
        if int(mask.max()) == 0:
            logger.info("Inpaint skipped: empty mask")
            return image.convert("RGB")

        rgb = image.convert("RGB")
        # SimpleLama は PIL もしくは np を受け取る。マスクは PIL 'L' を渡す。
        mask_pil = Image.fromarray((mask > 127).astype(np.uint8) * 255, mode="L")
        result = self._lama(rgb, mask_pil)
        if isinstance(result, np.ndarray):
            result = Image.fromarray(result)
        return result.convert("RGB")
