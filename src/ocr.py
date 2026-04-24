"""PaddleOCR ラッパ.

TODO(次フェーズ): 実装を入れる。現状はインタフェース定義のみ。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from .utils import BBox, get_logger

logger = get_logger(__name__)


@dataclass
class TextRegion:
    bbox: BBox
    text: str
    confidence: float
    is_vertical: bool = False


class OCREngine:
    def __init__(self, lang: str = "japan", use_gpu: bool = False, min_confidence: float = 0.5):
        self.lang = lang
        self.use_gpu = use_gpu
        self.min_confidence = min_confidence
        self._ocr = None

    def _ensure_loaded(self) -> None:
        if self._ocr is not None:
            return
        # 遅延ロード: モジュールimportコストと初回ダウンロードを避けるため
        from paddleocr import PaddleOCR  # type: ignore

        logger.info("Loading PaddleOCR (lang=%s, gpu=%s)...", self.lang, self.use_gpu)
        self._ocr = PaddleOCR(use_angle_cls=True, lang=self.lang, use_gpu=self.use_gpu, show_log=False)

    def detect(self, image_path: str | Path | Image.Image) -> list[TextRegion]:
        raise NotImplementedError("OCREngine.detect は次フェーズで実装します")
