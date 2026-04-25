"""LP画像 → 分割PSD 変換の統合パイプライン."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .inpainting import Inpainter
from .jsx_exporter import write_jsx
from .ocr import OCREngine, TextRegion
from .psd_writer import PSDBuildSpec, write_psd
from .segmentation import Segmenter, Subject
from .utils import dilate_mask, get_logger, load_image, pil_to_np

logger = get_logger(__name__)


@dataclass
class PipelineConfig:
    use_gpu: bool = False
    ocr_lang: str = "japan"
    ocr_min_confidence: float = 0.5
    rembg_model: str = "isnet-general-use"
    rembg_min_area_ratio: float = 0.005
    text_mask_padding: int = 4
    subject_mask_padding: int = 2
    emit_jsx: bool = True


def _build_text_mask(
    regions: list[TextRegion],
    size: tuple[int, int],
    padding: int,
) -> np.ndarray:
    """OCRで検出したポリゴン領域を塗りつぶしたマスクを作る."""
    W, H = size
    mask = np.zeros((H, W), dtype=np.uint8)
    for r in regions:
        if r.polygon.size == 0:
            x, y, w, h = r.bbox.x, r.bbox.y, r.bbox.w, r.bbox.h
            cv2.rectangle(mask, (x, y), (x + w, y + h), 255, thickness=-1)
        else:
            cv2.fillPoly(mask, [r.polygon.astype(np.int32)], 255)
    return dilate_mask(mask, padding)


def _build_subject_mask(
    subjects: list[Subject],
    size: tuple[int, int],
    padding: int,
) -> np.ndarray:
    W, H = size
    mask = np.zeros((H, W), dtype=np.uint8)
    for s in subjects:
        np.maximum(mask, s.mask, out=mask)
    return dilate_mask(mask, padding)


def _crop_text_image(image: Image.Image, region: TextRegion) -> Image.Image:
    """元画像からテキスト領域を矩形で切り出す (RGBA)."""
    bbox = region.bbox
    cropped = image.crop((bbox.x, bbox.y, bbox.x2, bbox.y2)).convert("RGBA")
    return cropped


class Pipeline:
    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()
        self.ocr = OCREngine(
            lang=self.config.ocr_lang,
            use_gpu=self.config.use_gpu,
            min_confidence=self.config.ocr_min_confidence,
        )
        self.segmenter = Segmenter(
            model_name=self.config.rembg_model,
            min_area_ratio=self.config.rembg_min_area_ratio,
        )
        self.inpainter = Inpainter(device="cuda" if self.config.use_gpu else "cpu")

    def run(self, input_path: str | Path, output_path: str | Path) -> Path:
        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("=== lp2psd start ===")
        logger.info("Input : %s", input_path)
        logger.info("Output: %s", output_path)

        image = load_image(input_path)
        W, H = image.size

        # 1. OCR
        regions = self.ocr.detect(image)

        # 2. Segmentation
        subjects = self.segmenter.segment(image)

        # 3. マスク合成
        text_mask = _build_text_mask(regions, (W, H), self.config.text_mask_padding)
        subject_mask = _build_subject_mask(subjects, (W, H), self.config.subject_mask_padding)
        combined = np.maximum(text_mask, subject_mask)

        # 4. テキスト画像の切り出し（インペイント前の元画像から取得）
        text_images = [_crop_text_image(image, r) for r in regions]

        # 5. インペイントで背景復元
        background = self.inpainter.inpaint(image, combined)

        # 6. PSD 出力
        spec = PSDBuildSpec(
            background=background,
            subjects=subjects,
            text_regions=regions,
            text_images=text_images,
        )
        psd_path = write_psd(spec, output_path)

        # 7. JSX 出力（同名・拡張子変更）
        if self.config.emit_jsx:
            jsx_path = output_path.with_suffix(".jsx")
            write_jsx(regions, jsx_path)

        logger.info("=== lp2psd done ===")
        return psd_path


def main() -> None:
    parser = argparse.ArgumentParser(description="LP画像 → レイヤー分割PSD 変換")
    parser.add_argument("--input", required=True, help="入力画像パス (PNG/JPG)")
    parser.add_argument("--output", required=True, help="出力PSDパス")
    parser.add_argument("--gpu", action="store_true", help="GPU を使う")
    parser.add_argument("--no-jsx", action="store_true", help="JSX スクリプトを生成しない")
    parser.add_argument("--min-conf", type=float, default=0.5, help="OCR信頼度しきい値")
    parser.add_argument("--text-pad", type=int, default=4, help="テキストマスク余白(px)")
    args = parser.parse_args()

    config = PipelineConfig(
        use_gpu=args.gpu,
        emit_jsx=not args.no_jsx,
        ocr_min_confidence=args.min_conf,
        text_mask_padding=args.text_pad,
    )
    pipeline = Pipeline(config)
    out = pipeline.run(args.input, args.output)
    logger.info("Done: %s", out)


if __name__ == "__main__":
    main()
