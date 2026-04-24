"""PSD 出力ロジック.

TODO(次フェーズ): 実装を入れる。現状はインタフェース定義のみ。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from .ocr import TextRegion
from .segmentation import Subject
from .utils import get_logger

logger = get_logger(__name__)


@dataclass
class PSDBuildSpec:
    """PSD 構築に必要な入力セット."""

    background: Image.Image
    subjects: list[Subject] = field(default_factory=list)
    text_regions: list[TextRegion] = field(default_factory=list)
    text_images: list[Image.Image] = field(default_factory=list)  # text_regions と対応


def write_psd(spec: PSDBuildSpec, output_path: str | Path) -> None:
    raise NotImplementedError("write_psd は次フェーズで実装します")
