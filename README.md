# lp2psd

LP画像（PNG/JPG）を、**テキスト・人物・商品が別レイヤーに分かれたPSDファイル**へ変換するPythonツールです。
Gradio Web UIで動作します。

## 特徴

- **テキスト領域検出**: PaddleOCRで日本語テキスト（縦書き/横書き）を検出
- **人物・商品の切り抜き**: rembgで被写体マスクを抽出
- **背景復元**: simple-lama-inpaintingでテキスト・被写体を除去した背景を生成
- **PSD出力**: psd-toolsでレイヤー分割したPSDを生成
- **編集可能テキスト化（オプション）**: 併用するJSXスクリプトをPhotoshopで実行すると、テキストレイヤーが真に編集可能な状態になります

## 動作環境

- Windows 10/11（Linux/macOSでも動作可、Photoshop連携のみWindows/Mac）
- Python 3.10 以上
- GPU (NVIDIA CUDA 12.1) 推奨 / CPUでも動作可
- Photoshop 2021 以降（JSX連携を使う場合のみ）

## セットアップ

### Windows

```powershell
git clone https://github.com/yo-star0/ai_lptool.git
cd ai_lptool
setup.bat
```

CUDA環境で高速化したい場合は、`requirements-gpu.txt` のコメントに従って追加インストール。

### Linux / macOS

```bash
git clone https://github.com/yo-star0/ai_lptool.git
cd ai_lptool
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## 使い方

### Gradio UI

```
run.bat           (Windows)
python app.py     (Linux/macOS)
```

ブラウザで `http://127.0.0.1:7860` を開き、LP画像をアップロード → 変換 → PSDダウンロード。

### CLI

```bash
python -m src.pipeline --input path/to/lp.png --output path/to/out.psd
```

### Photoshopで編集可能テキストレイヤーに変換（オプション）

1. 生成された `out.psd` をPhotoshopで開く
2. 生成された `out.jsx` を「ファイル > スクリプト > 参照」で実行
3. 画像化されたテキストレイヤーの隣に、編集可能なテキストレイヤーが追加される

## 処理フロー

```
LP画像
  ├─ PaddleOCR          → テキスト領域 + 認識文字列
  ├─ rembg              → 被写体マスク（人物・商品）
  └─ lama-inpainting    → テキスト/被写体を消した背景
         ↓
   psd-tools            → レイヤー分割PSD
         + jsx_exporter → 編集可能テキスト化スクリプト
```

## プロジェクト構成

```
ai_lptool/
├── app.py                   # Gradio UI エントリポイント
├── setup.bat / run.bat      # Windows 起動スクリプト
├── requirements.txt         # CPU 向け依存
├── requirements-gpu.txt     # CUDA 追加手順
├── src/
│   ├── pipeline.py          # 全体統合
│   ├── ocr.py               # PaddleOCR ラッパ
│   ├── segmentation.py      # rembg ラッパ
│   ├── inpainting.py        # lama ラッパ
│   ├── psd_writer.py        # PSD 出力
│   ├── jsx_exporter.py      # JSX 生成
│   └── utils.py
├── extras/
│   └── text_layers.jsx      # Photoshop 併用スクリプト (雛形)
├── samples/                 # 入力サンプル置き場
└── output/                  # 出力先
```

## 既知の制約

- `psd-tools` は編集可能テキストレイヤーの生成を完全サポートしていません。本ツールは「レンダリング済みテキスト画像レイヤー」＋「JSXスクリプト」の併用で擬似的に編集可能状態を作ります
- 装飾的なテキスト（グラデーション・縁取り・シャドウ）は背景として消去されます。Photoshop側で手動再構築してください
- 人物と商品が大きく重なっている場合、重なり順の完全復元は困難です
- PaddleOCRの初回実行時はモデルのダウンロード（数百MB）が走ります。オフライン環境では事前配置が必要です

## ライセンス

MIT License
