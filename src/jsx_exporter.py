"""Photoshop JSX スクリプト生成.

TODO(次フェーズ): 実装を入れる。現状はインタフェース定義のみ。

生成される JSX は、PSD と同じフォルダに置き、Photoshop 側で
「ファイル > スクリプト > 参照」から実行する想定。
実行すると、PSD 内の「Text_*」グループの隣に編集可能テキストレイヤーが
追加される。
"""
from __future__ import annotations

from pathlib import Path

from .ocr import TextRegion
from .utils import get_logger

logger = get_logger(__name__)


def write_jsx(regions: list[TextRegion], output_path: str | Path) -> None:
    raise NotImplementedError("write_jsx は次フェーズで実装します")
