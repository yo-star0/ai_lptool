// text_layers.jsx
// Photoshop 併用スクリプト (雛形)
//
// 使い方:
//   1. pipeline の出力 PSD と同じフォルダに生成される XXX_text.jsx を
//      Photoshop の「ファイル > スクリプト > 参照」から実行する
//   2. このファイル (text_layers.jsx) は、生成される JSX が取り込む
//      共通ユーティリティの置き場です
//
// TODO(次フェーズ): 実装を入れる。現状は骨格のみ。

function addEditableText(doc, text, x, y, w, h, isVertical) {
    var layer = doc.artLayers.add();
    layer.kind = LayerKind.TEXT;
    var t = layer.textItem;
    t.contents = text;
    t.position = [x, y];
    t.size = Math.max(12, Math.floor(h * 0.8));
    if (isVertical) {
        t.direction = Direction.VERTICAL;
    }
    return layer;
}

// このファイル単体では何もしない。生成側 JSX から関数を呼び出す想定。
