"""PaddleOCR ラッパ.

PaddleOCRの出力（4点ポリゴン + テキスト + 信頼度）を、
本ツール内部で扱いやすい TextRegion に変換する。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from .utils import BBox, get_logger, pil_to_np

logger = get_logger(__name__)


@dataclass
class TextRegion:
    bbox: BBox
    text: str
    confidence: float
    is_vertical: bool = False
    polygon: np.ndarray = field(default_factory=lambda: np.zeros((4, 2), dtype=np.int32))


def _is_vertical(polygon: np.ndarray, text: str) -> bool:
    """縦書き判定. 矩形の縦横比 + 文字数の組み合わせで推定."""
    bbox = BBox.from_polygon(polygon)
    if bbox.w == 0:
        return True
    aspect = bbox.h / max(1, bbox.w)
    # 高さ/幅 > 1.5 かつ文字が複数 → 縦書き
    return aspect > 1.5 and len(text.strip()) >= 2


class OCREngine:
    def __init__(
        self,
        lang: str = "japan",
        use_gpu: bool = False,
        min_confidence: float = 0.5,
    ):
        self.lang = lang
        self.use_gpu = use_gpu
        self.min_confidence = min_confidence
        self._ocr = None

    def _ensure_loaded(self) -> None:
        if self._ocr is not None:
            return
        from paddleocr import PaddleOCR  # type: ignore

        logger.info("Loading PaddleOCR (lang=%s, gpu=%s)...", self.lang, self.use_gpu)
        # show_log=False で内部の冗長ログを抑制
        self._ocr = PaddleOCR(
            use_angle_cls=True,
            lang=self.lang,
            use_gpu=self.use_gpu,
            show_log=False,
        )

    def detect(self, image: str | Path | Image.Image) -> list[TextRegion]:
        self._ensure_loaded()
        assert self._ocr is not None

        if isinstance(image, (str, Path)):
            arr = pil_to_np(Image.open(image).convert("RGB"))
        else:
            arr = pil_to_np(image.convert("RGB"))

        # PaddleOCR は cls=True で角度補正、戻り値は [[poly, (text, conf)], ...] の入れ子
        raw = self._ocr.ocr(arr, cls=True)
        if not raw or raw[0] is None:
            return []
        # 新しい paddleocr は [page_results] の二重リストを返すことがある
        page = raw[0] if isinstance(raw[0], list) and len(raw[0]) and isinstance(raw[0][0], list) else raw

        regions: list[TextRegion] = []
        for entry in page:
            try:
                poly, (text, conf) = entry
            except (TypeError, ValueError):
                continue
            if conf is None or conf < self.min_confidence:
                continue
            polygon = np.array(poly, dtype=np.int32)
            if polygon.shape != (4, 2):
                continue
            bbox = BBox.from_polygon(polygon)
            regions.append(
                TextRegion(
                    bbox=bbox,
                    text=str(text),
                    confidence=float(conf),
                    is_vertical=_is_vertical(polygon, str(text)),
                    polygon=polygon,
                )
            )
        logger.info("OCR detected %d regions (>=%.2f confidence)", len(regions), self.min_confidence)
        return regions
