import os
from docling.document_converter import DocumentConverter



# pdf_path = "input/1706.03762v7.pdf"
# output_path = "output/1706.03762v7.md"

pdf_path = "input/2021r03h_nw_pm1_qs.pdf"
output_path = "output/2021r03h_nw_pm1_qs.md"


converter = DocumentConverter()
result = converter.convert(pdf_path)

md_text = result.document.export_to_markdown()


with open(output_path, mode="w", encoding="utf-8") as f:
    f.write(md_text)