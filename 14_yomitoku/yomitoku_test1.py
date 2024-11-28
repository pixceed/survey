import os
import cv2
from datetime import datetime

from yomitoku import DocumentAnalyzer
from yomitoku.data.functions import load_image

if __name__ == "__main__":

    image_path = "input/2021r03h_nw_pm1_qs-2.png"
    output_dir = f"output/{datetime.now().strftime('%Y%m%d%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)

    configs = {
        "ocr": {
            "text_detector": {
                "device": "cuda",
            },
            "text_recognizer": {
                "device": "cuda",
            },
        },
        "layout_analyzer": {
            "layout_parser": {
                "device": "cuda",
            },
            "table_structure_recognizer": {
                "device": "cuda",
            },
        },
    }

    img = load_image(image_path)
    analyzer = DocumentAnalyzer(configs=configs, visualize=True, device="cuda")
    results, ocr_vis, layout_vis = analyzer(img)

    # # HTML形式で解析結果をエクスポート
    # results.to_html(os.path.join(output_dir, "output_ocr.html"))

    # JSON形式で解析結果をエクスポート
    results.to_json(os.path.join(output_dir, "output_ocr.json"))

    # Markdown形式で解析結果をエクスポート
    results.to_markdown(os.path.join(output_dir, "output_ocr.md"))

    # 可視化画像を保存
    cv2.imwrite(os.path.join(output_dir, "output_ocr.jpg"), ocr_vis)
    cv2.imwrite(os.path.join(output_dir, "output_layout.jpg"), layout_vis)