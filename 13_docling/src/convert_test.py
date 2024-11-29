import os
from docling.document_converter import DocumentConverter



# pdf_path = "input/1706.03762v7.pdf"
# output_path = "output/1706.03762v7.md"

# pdf_path = "input/2021r03h_nw_pm1_qs.pdf"
# output_path = "output/2021r03h_nw_pm1_qs.md"

# pdf_path = "input/140120240521502314.pdf"
# output_path = "output/140120240521502314_docling.md"

pdf_path = "input/会社紹介資料_2024_3.pptx"
output_path = "output/会社紹介資料_2024_docling.md"

# pdf_path = "input/会社紹介資料_2024_libre.pdf"
# output_path = "output/会社紹介資料_2024_libre_docling.md"

converter = DocumentConverter()
result = converter.convert(pdf_path)

md_text = result.document.export_to_markdown()


with open(output_path, mode="w", encoding="utf-8") as f:
    f.write(md_text)