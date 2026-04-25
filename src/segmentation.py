"""rembg ラッパ（人物/商品マスク抽出）.

rembgで前景全体のマスクを得たあと、連結成分で被写体ごとに分割する。
被写体の人物/商品の自動分類は行わず、単に "subject_1", "subject_2" ... と
番号付けする（必要なら呼び出し側でラベルを付け直す）。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from .utils import BBox, get_logger

logger = get_logger(__name__)


@dataclass
class Subject:
    """切り抜かれた被写体."""

    rgba: Image.Image  # 当該被写体のbboxサイズのRGBA画像
    mask: np.ndarray  # 元画像座標系・元サイズの2次元マスク (uint8 0/255)
    bbox: BBox
    label: str = "subject"


class Segmenter:
    def __init__(
        self,
        model_name: str = "isnet-general-use",
        min_area_ratio: float = 0.005,
    ):
        """
        min_area_ratio: 画像面積に対する最小被写体面積比 (これより小さい連結成分は無視)
        """
        self.model_name = model_name
        self.min_area_ratio = min_area_ratio
        self._session = None

    def _ensure_loaded(self) -> None:
        if self._session is not None:
            return
        from rembg import new_session  # type: ignore

        logger.info("Loading rembg session (model=%s)...", self.model_name)
        self._session = new_session(self.model_name)

    def segment(self, image: Image.Image) -> list[Subject]:
        import cv2
        from rembg import remove  # type: ignore

        self._ensure_loaded()
        rgb = image.convert("RGB")
        rgba_full = remove(rgb, session=self._session)  # PIL.Image (RGBA)
        if not isinstance(rgba_full, Image.Image):
            rgba_full = Image.fromarray(np.array(rgba_full))
        rgba_arr = np.array(rgba_full.convert("RGBA"))
        alpha = rgba_arr[:, :, 3]

        # 2値化 + 連結成分
        _, binary = cv2.threshold(alpha, 32, 255, cv2.THRESH_BINARY)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

        H, W = alpha.shape
        min_area = int(self.min_area_ratio * H * W)
        subjects: list[Subject] = []
        # ラベル0は背景なのでスキップ
        components = []
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            if area < min_area:
                continue
            components.append((area, i, x, y, w, h))
        # 面積降順で安定したインデックスにする
        components.sort(reverse=True, key=lambda c: c[0])

        for idx, (area, label_id, x, y, w, h) in enumerate(components, start=1):
            mask_full = np.zeros((H, W), dtype=np.uint8)
            mask_full[labels == label_id] = 255

            crop_rgba = rgba_arr[y : y + h, x : x + w].copy()
            crop_mask = mask_full[y : y + h, x : x + w]
            # 連結成分のマスクで alpha を上書き（他被写体の漏れ防止）
            crop_rgba[:, :, 3] = crop_mask

            subjects.append(
                Subject(
                    rgba=Image.fromarray(crop_rgba),
                    mask=mask_full,
                    bbox=BBox(int(x), int(y), int(w), int(h)),
                    label=f"subject_{idx}",
                )
            )

        logger.info(
            "Segmentation: %d subjects extracted (>= %d px^2 / %.1f%%)",
            len(subjects),
            min_area,
            self.min_area_ratio * 100,
        )
        return subjects
