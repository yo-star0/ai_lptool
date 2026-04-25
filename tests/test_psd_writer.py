"""依存(rembg/paddleocr/lama)無しで動く範囲のスモークテスト.

実行:
    PYTHONPATH=. python -m pytest tests/ -v
あるいは:
    python tests/test_psd_writer.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# プロジェクトルートを sys.path に追加
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image, ImageDraw

from src.jsx_exporter import write_jsx
from src.ocr import TextRegion
from src.pipeline import _build_subject_mask, _build_text_mask, _crop_text_image
from src.psd_writer import PSDBuildSpec, write_psd
from src.segmentation import Subject
from src.utils import BBox


def _make_fixture():
    W, H = 300, 200
    img = Image.new("RGB", (W, H), (255, 240, 245))
    ImageDraw.Draw(img).rectangle((10, 10, 110, 40), fill=(20, 20, 20))
    ImageDraw.Draw(img).ellipse((150, 50, 230, 150), fill=(150, 100, 80))

    regions = [
        TextRegion(
            bbox=BBox(10, 10, 100, 30),
            text="素肌が変わると、",
            confidence=0.97,
            is_vertical=False,
            polygon=np.array([[10, 10], [110, 10], [110, 40], [10, 40]], dtype=np.int32),
        ),
        TextRegion(
            bbox=BBox(50, 100, 30, 80),
            text="縦書きテスト",
            confidence=0.85,
            is_vertical=True,
            polygon=np.array([[50, 100], [80, 100], [80, 180], [50, 180]], dtype=np.int32),
        ),
    ]
    mask_full = np.zeros((H, W), dtype=np.uint8)
    mask_full[50:150, 150:230] = 255
    sub_rgba = np.zeros((100, 80, 4), dtype=np.uint8)
    sub_rgba[..., :3] = 150
    sub_rgba[..., 3] = 255
    subjects = [
        Subject(
            rgba=Image.fromarray(sub_rgba),
            mask=mask_full,
            bbox=BBox(150, 50, 80, 100),
            label="subject_1",
        ),
    ]
    return img, regions, subjects


def test_text_mask_built():
    img, regions, _ = _make_fixture()
    W, H = img.size
    mask = _build_text_mask(regions, (W, H), padding=4)
    assert mask.shape == (H, W)
    assert int(mask.max()) == 255
    assert int((mask > 0).sum()) > 0


def test_subject_mask_built():
    img, _, subjects = _make_fixture()
    W, H = img.size
    mask = _build_subject_mask(subjects, (W, H), padding=2)
    assert mask.shape == (H, W)
    assert int((mask > 0).sum()) > 0


def test_text_image_cropped():
    img, regions, _ = _make_fixture()
    crops = [_crop_text_image(img, r) for r in regions]
    assert crops[0].size == (100, 30)
    assert crops[1].size == (30, 80)


def test_psd_roundtrip_with_japanese_names(tmp_path):
    from psd_tools import PSDImage

    img, regions, subjects = _make_fixture()
    text_imgs = [_crop_text_image(img, r) for r in regions]
    spec = PSDBuildSpec(
        background=img, subjects=subjects, text_regions=regions, text_images=text_imgs
    )
    out = write_psd(spec, tmp_path / "out.psd")
    assert out.exists()

    re = PSDImage.open(out)
    names = [l.name for l in re]
    assert "background" in names
    assert "subject_1" in names
    # Japanese layer names round-trip via UNICODE_LAYER_NAME
    assert any("素肌が変わると" in n for n in names)
    assert any("縦書きテスト" in n and "|v|" in n for n in names)


def test_jsx_emitted(tmp_path):
    _, regions, _ = _make_fixture()
    out = write_jsx(regions, tmp_path / "out.jsx")
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert "#target photoshop" in text
    assert "素肌が変わると、" in text
    assert "縦書きテスト" in text
    assert '"vertical": true' in text


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        test_text_mask_built()
        test_subject_mask_built()
        test_text_image_cropped()
        test_psd_roundtrip_with_japanese_names(tmp)
        test_jsx_emitted(tmp)
    print("All tests passed.")
