import os
import re
import cv2
import yaml
import shutil
import base64
import argparse
import tempfile
import subprocess
import fitz  # PyMuPDFをインポート

def pdf2image(pdf_path, output_dir):
    """
    pdfを画像に変換し保存する
    """

    file_image_path_set = []

    # PDFを開く
    pdf_document = fitz.open(pdf_path)

    # 各ページを画像として保存
    for page_number in range(len(pdf_document)):
        
        page = pdf_document.load_page(page_number)  # ページをロード

        # 解像度を高めに設定して画像化
        zoom_x = 3.0  # 水平方向のズーム倍率
        zoom_y = 3.0  # 垂直方向のズーム倍率
        mat = fitz.Matrix(zoom_x, zoom_y)  # ズーム倍率を適用したマトリックス
        pix = page.get_pixmap(matrix=mat)  # ページを高解像度画像化

        # 保存ファイル名を決定
        output_path = os.path.join(output_dir, f"page_{str(page_number + 1).zfill(4)}.png")

        # 画像を保存
        pix.save(output_path)      

        print(f"ページ {page_number + 1} を保存しました: {output_path}")
        file_image_path_set.append(output_path)

    # PDFを閉じる
    pdf_document.close()

    return file_image_path_set

def pdf2image_crop(pdf_path, output_dir, crop_info):
    """
    pdfを画像に変換し保存する（画像をクロップする）
    """

    top, bottom, left, right = crop_info  

    file_image_path_set = []

    # PDFを開く
    pdf_document = fitz.open(pdf_path)

    with tempfile.TemporaryDirectory() as temp_dir:
        # 各ページを画像として保存
        for page_number in range(len(pdf_document)):
            
            page = pdf_document.load_page(page_number)  # ページをロード

            # 解像度を高めに設定して画像化
            zoom_x = 3.0  # 水平方向のズーム倍率
            zoom_y = 3.0  # 垂直方向のズーム倍率
            mat = fitz.Matrix(zoom_x, zoom_y)  # ズーム倍率を適用したマトリックス
            pix = page.get_pixmap(matrix=mat)  # ページを高解像度画像化

            # 画像を一時保存
            temp_path = os.path.join(temp_dir, f"temp_{str(page_number + 1).zfill(4)}.png")
            pix.save(temp_path)                

            # ＜画像カッティング＞
            temp_img = cv2.imread(temp_path)
            height, width = temp_img.shape[:2]

            cropped_image = temp_img[top:height-bottom, left:width-right]

            # 保存ファイル名を決定
            output_path = os.path.join(output_dir, f"page_{str(page_number + 1).zfill(4)}.png")

            # 画像を保存
            cv2.imwrite(output_path, cropped_image)

            print(f"ページ {page_number + 1} を保存しました: {output_path}")
            file_image_path_set.append(output_path)

        # PDFを閉じる
        pdf_document.close()

    return file_image_path_set
