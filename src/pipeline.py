"""LP画像 → 分割PSD 変換の統合パイプライン.

現段階ではスケルトンのみ。各モジュールの実装は次フェーズで追加する。
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from .inpainting import Inpainter
from .jsx_exporter import write_jsx
from .ocr import OCREngine
from .psd_writer import PSDBuildSpec, write_psd
from .segmentation import Segmenter
from .utils import get_logger, load_image

logger = get_logger(__name__)


@dataclass
class PipelineConfig:
    use_gpu: bool = False
    ocr_lang: str = "japan"
    ocr_min_confidence: float = 0.5
    rembg_model: str = "isnet-general-use"
    text_mask_padding: int = 4
    emit_jsx: bool = True


class Pipeline:
    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()
        self.ocr = OCREngine(
            lang=self.config.ocr_lang,
            use_gpu=self.config.use_gpu,
            min_confidence=self.config.ocr_min_confidence,
        )
        self.segmenter = Segmenter(model_name=self.config.rembg_model)
        self.inpainter = Inpainter(device="cuda" if self.config.use_gpu else "cpu")

    def run(self, input_path: str | Path, output_path: str | Path) -> Path:
        """LP画像をPSDに変換し、出力パスを返す."""
        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Loading image: %s", input_path)
        _ = load_image(input_path)

        raise NotImplementedError(
            "Pipeline.run は次フェーズで実装します。"
            " 現状はスケルトンで、各モジュールのインタフェースのみ定義されています。"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="LP画像 → レイヤー分割PSD 変換")
    parser.add_argument("--input", required=True, help="入力画像パス (PNG/JPG)")
    parser.add_argument("--output", required=True, help="出力PSDパス")
    parser.add_argument("--gpu", action="store_true", help="GPU を使う")
    parser.add_argument("--no-jsx", action="store_true", help="JSX スクリプトを生成しない")
    args = parser.parse_args()

    config = PipelineConfig(use_gpu=args.gpu, emit_jsx=not args.no_jsx)
    pipeline = Pipeline(config)
    out = pipeline.run(args.input, args.output)
    logger.info("Done: %s", out)


if __name__ == "__main__":
    main()
