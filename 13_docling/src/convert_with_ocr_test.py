from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    EasyOcrOptions,
    TableFormerMode,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.backend.docling_parse_v2_backend import DoclingParseV2DocumentBackend


# pdf_path = "input/1706.03762v7.pdf"
# output_path = "output/1706.03762v7.md"

pdf_path = "input/2021r03h_nw_pm1_qs.pdf"
output_path = "output/2021r03h_nw_pm1_qs_2.md"


# EasyOCRの言語オプションを設定
ocr_options = EasyOcrOptions(lang=["ja"])

pipeline_options = PdfPipelineOptions()
# OCRの使用有無: 使用する
pipeline_options.do_ocr = True
# テーブル構造モデルの使用有無: 使用する
pipeline_options.do_table_structure = True
# セルの認識方法: ビジュアルに基づいてセルを認識させる
pipeline_options.table_structure_options.do_cell_matching = True
# テーブル構造モデルのモード: 正確さを重視
pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
# パイプラインにOCRオプションを設定
pipeline_options.ocr_options = ocr_options

doc_converter = (
    DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
                # PDFバックエンドにDLPARSE_V2を使用
                backend=DoclingParseV2DocumentBackend,
            ),
        }
    )
)


result = doc_converter.convert(pdf_path)

md_text = result.document.export_to_markdown()


with open(output_path, mode="w", encoding="utf-8") as f:
    f.write(md_text)