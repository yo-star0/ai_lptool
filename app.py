"""Gradio Web UI for lp2psd."""
from __future__ import annotations

import io
import logging
import tempfile
import traceback
from pathlib import Path

import gradio as gr

from src.pipeline import Pipeline, PipelineConfig


class _UILogHandler(logging.Handler):
    """Gradio に流すための簡易ハンドラ."""

    def __init__(self) -> None:
        super().__init__()
        self.buffer = io.StringIO()
        self.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        self.buffer.write(self.format(record) + "\n")

    def get(self) -> str:
        return self.buffer.getvalue()

    def clear(self) -> None:
        self.buffer.seek(0)
        self.buffer.truncate(0)


def _attach_ui_handler() -> _UILogHandler:
    handler = _UILogHandler()
    root = logging.getLogger("src")
    root.addHandler(handler)
    return handler


def _detach_ui_handler(handler: _UILogHandler) -> None:
    logging.getLogger("src").removeHandler(handler)


def convert(
    image_path: str | None,
    use_gpu: bool,
    min_conf: float,
    text_pad: int,
    subject_pad: int,
    min_area_ratio: float,
    emit_jsx: bool,
):
    if not image_path:
        return None, None, "画像をアップロードしてください"

    handler = _attach_ui_handler()
    try:
        config = PipelineConfig(
            use_gpu=use_gpu,
            ocr_min_confidence=min_conf,
            text_mask_padding=int(text_pad),
            subject_mask_padding=int(subject_pad),
            rembg_min_area_ratio=min_area_ratio,
            emit_jsx=emit_jsx,
        )

        out_dir = Path(tempfile.mkdtemp(prefix="lp2psd_"))
        stem = Path(image_path).stem or "lp"
        psd_path = out_dir / f"{stem}.psd"

        pipeline = Pipeline(config)
        pipeline.run(image_path, psd_path)

        jsx_path = psd_path.with_suffix(".jsx") if emit_jsx else None
        if jsx_path is not None and not jsx_path.exists():
            jsx_path = None

        return str(psd_path), (str(jsx_path) if jsx_path else None), handler.get()
    except NotImplementedError as exc:
        return None, None, handler.get() + f"\n[NotImplemented] {exc}"
    except Exception as exc:
        tb = traceback.format_exc()
        return None, None, handler.get() + f"\n[ERROR] {exc}\n{tb}"
    finally:
        _detach_ui_handler(handler)


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="lp2psd") as demo:
        gr.Markdown(
            "# lp2psd\n"
            "LP画像 (PNG/JPG) を、テキスト・人物・商品が別レイヤーに分かれた **PSD** ファイルに変換します。\n"
            "PSD と一緒に出力される **.jsx** スクリプトを Photoshop で実行すると、編集可能なテキストレイヤーが追加されます。"
        )
        with gr.Row():
            with gr.Column(scale=1):
                image_in = gr.Image(label="LP画像 (PNG/JPG)", type="filepath")
                use_gpu = gr.Checkbox(label="GPUを使う (要 CUDA)", value=False)
                min_conf = gr.Slider(
                    0.0, 1.0, value=0.5, step=0.05, label="OCR信頼度しきい値"
                )
                text_pad = gr.Slider(
                    0, 30, value=4, step=1, label="テキストマスク余白 (px)"
                )
                subject_pad = gr.Slider(
                    0, 30, value=2, step=1, label="被写体マスク余白 (px)"
                )
                min_area_ratio = gr.Slider(
                    0.0, 0.05, value=0.005, step=0.001,
                    label="被写体最小面積比 (画像面積比)",
                )
                emit_jsx = gr.Checkbox(label="JSXスクリプトも生成する", value=True)
                run_btn = gr.Button("変換", variant="primary")
            with gr.Column(scale=1):
                psd_out = gr.File(label="生成されたPSD")
                jsx_out = gr.File(label="生成されたJSX (Photoshop用)")
                log_out = gr.Textbox(label="ログ", lines=18, max_lines=40)

        run_btn.click(
            fn=convert,
            inputs=[image_in, use_gpu, min_conf, text_pad, subject_pad, min_area_ratio, emit_jsx],
            outputs=[psd_out, jsx_out, log_out],
        )
    return demo


def main() -> None:
    build_ui().launch()


if __name__ == "__main__":
    main()
