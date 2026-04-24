"""Gradio Web UI.

TODO(次フェーズ): パイプライン本体と接続する。現状は起動確認用のプレースホルダ。
"""
from __future__ import annotations

import gradio as gr


def _convert_placeholder(image, use_gpu: bool, min_conf: float, mask_padding: int):
    if image is None:
        return None, "画像をアップロードしてください"
    return None, (
        "現在はスケルトン段階のため、PSD生成は未実装です。\n"
        "次フェーズで pipeline.Pipeline.run をここに接続します。\n"
        f"設定: gpu={use_gpu}, min_conf={min_conf}, padding={mask_padding}"
    )


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="lp2psd") as demo:
        gr.Markdown("# lp2psd\nLP画像 → レイヤー分割PSD 変換ツール")

        with gr.Row():
            with gr.Column():
                image_in = gr.Image(label="LP画像 (PNG/JPG)", type="filepath")
                use_gpu = gr.Checkbox(label="GPUを使う", value=False)
                min_conf = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="OCR信頼度しきい値")
                mask_padding = gr.Slider(0, 20, value=4, step=1, label="テキストマスク余白(px)")
                run_btn = gr.Button("変換", variant="primary")
            with gr.Column():
                psd_out = gr.File(label="生成されたPSD")
                log_out = gr.Textbox(label="ログ", lines=10)

        run_btn.click(
            fn=_convert_placeholder,
            inputs=[image_in, use_gpu, min_conf, mask_padding],
            outputs=[psd_out, log_out],
        )
    return demo


def main() -> None:
    build_ui().launch()


if __name__ == "__main__":
    main()
