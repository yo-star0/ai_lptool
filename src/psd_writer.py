"""PSD 出力ロジック.

レイヤー構造:
    - background       (画像レイヤー: インペイント済み背景)
    - subject_1, ...   (画像レイヤー: 被写体ごとのRGBA)
    - text_*           (画像レイヤー: テキスト矩形を切り出した画像)
                        レイヤー名に元テキストを埋め込み、JSXで参照可能にする

レイヤー名の埋め込み形式:
    text|<text>|<v|h>|<conf>
    例: text|素肌が変わると、|h|0.98
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image
from psd_tools import PSDImage
from psd_tools.api.layers import Layer, PixelLayer
from psd_tools.constants import Tag

from .ocr import TextRegion
from .segmentation import Subject
from .utils import get_logger

logger = get_logger(__name__)


def _ascii_fallback_name(name: str, default: str) -> str:
    """PSDのレガシーPascal文字列(MacRoman)に書ける範囲だけ残す."""
    try:
        name.encode("mac_roman")
        return name
    except UnicodeEncodeError:
        # 非ASCIIを"_"に置換し、空ならdefaultを返す
        ascii_only = "".join(c if ord(c) < 128 else "_" for c in name).strip("_ ")
        return ascii_only or default


def _set_unicode_name(layer: Layer, full_name: str) -> None:
    """レイヤーに UNICODE_LAYER_NAME タグ付きブロックを設定して
    Photoshopで日本語が表示されるようにする."""
    layer._record.tagged_blocks.set_data(Tag.UNICODE_LAYER_NAME, full_name)


@dataclass
class PSDBuildSpec:
    """PSD 構築に必要な入力セット."""

    background: Image.Image
    subjects: list[Subject] = field(default_factory=list)
    text_regions: list[TextRegion] = field(default_factory=list)
    text_images: list[Image.Image] = field(default_factory=list)


def _safe_layer_name(text: str, max_len: int = 80) -> str:
    """レイヤー名に使えない文字を置換し、長さを制限."""
    sanitized = text.replace("\n", " ").replace("\r", " ").replace("|", "/").strip()
    if len(sanitized) > max_len:
        sanitized = sanitized[:max_len] + "…"
    return sanitized


def _encode_text_layer_name(region: TextRegion, idx: int) -> str:
    safe = _safe_layer_name(region.text)
    direction = "v" if region.is_vertical else "h"
    return f"text_{idx:02d}|{safe}|{direction}|{region.confidence:.2f}"


def write_psd(spec: PSDBuildSpec, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    bg = spec.background.convert("RGB")
    W, H = bg.size

    # キャンバスは白で初期化（合成後の見た目はレイヤーで決まる）
    psd = PSDImage.frompil(Image.new("RGB", (W, H), (255, 255, 255)))

    # 1. background
    bg_layer = PixelLayer.frompil(bg, parent=psd, name="background", top=0, left=0)
    _set_unicode_name(bg_layer, "background")

    # 2. subjects
    for subj in spec.subjects:
        rgba = subj.rgba.convert("RGBA")
        ascii_name = _ascii_fallback_name(subj.label, default="subject")
        layer = PixelLayer.frompil(
            rgba,
            parent=psd,
            name=ascii_name,
            top=int(subj.bbox.y),
            left=int(subj.bbox.x),
        )
        _set_unicode_name(layer, subj.label)

    # 3. text layers (画像レイヤー + 名前にテキスト埋め込み)
    if len(spec.text_images) != len(spec.text_regions):
        raise ValueError(
            f"text_images and text_regions length mismatch: "
            f"{len(spec.text_images)} vs {len(spec.text_regions)}"
        )
    for idx, (region, text_img) in enumerate(zip(spec.text_regions, spec.text_images), start=1):
        rgba = text_img.convert("RGBA")
        full_name = _encode_text_layer_name(region, idx)
        ascii_name = _ascii_fallback_name(full_name, default=f"text_{idx:02d}")
        layer = PixelLayer.frompil(
            rgba,
            parent=psd,
            name=ascii_name,
            top=int(region.bbox.y),
            left=int(region.bbox.x),
        )
        _set_unicode_name(layer, full_name)

    psd.save(str(output_path))
    logger.info(
        "PSD saved: %s (size=%dx%d, subjects=%d, texts=%d)",
        output_path, W, H, len(spec.subjects), len(spec.text_regions),
    )
    return output_path
